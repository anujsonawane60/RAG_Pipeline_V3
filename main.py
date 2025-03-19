import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cohere
import pinecone
from pinecone import ServerlessSpec
import uvicorn
import PyPDF2
import io
import docx
import re
import tempfile
import shutil
import time
from datetime import datetime
from typing import List, Dict
import json
import openai
import together
import requests
from pydantic import BaseModel
import sys

# Load environment variables
load_dotenv()

app = FastAPI(title="Multi-RAG Chatbot API")

# Retrieve API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Config:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    DIMENSION = 1024
    CHUNK_SIZE = 500
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"

    @classmethod
    def validate_env_vars(cls):
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found")
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found")

class QueryRequest(BaseModel):
    model: str
    query: str

    class Config:
        schema_extra = {
            "example": {
                "model": "cohere",  # or "openai" or "togetherai"
                "query": "What is the main topic of the document?"
            }
        }

class ServiceManager:
    def __init__(self):
        self.cohere_client = None
        self.pc = None
        self.index = None

    def initialize_services(self, index_name: str):
        try:
            # Initialize Cohere
            if not Config.COHERE_API_KEY:
                raise ValueError("COHERE_API_KEY not found in environment variables")
            self.cohere_client = cohere.Client(Config.COHERE_API_KEY)

            # Initialize Pinecone
            if not Config.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            print("Initializing Pinecone...")
            # Initialize Pinecone client
            self.pc = pinecone.Pinecone(api_key=Config.PINECONE_API_KEY)

            try:
                # List existing indexes
                existing_indexes = [index.name for index in self.pc.list_indexes()]
                print(f"Existing indexes: {existing_indexes}")

                # Check if index exists and create if it doesn't
                if index_name not in existing_indexes:
                    print(f"Creating new serverless index: {index_name}")
                    # Create index with serverless specification
                    self.pc.create_index(
                        name=index_name,
                        dimension=Config.DIMENSION,
                        metric='cosine',
                        spec={
                            "serverless": {
                                "cloud": Config.PINECONE_CLOUD,
                                "region": Config.PINECONE_REGION
                            }
                        }
                    )
                    print(f"Waiting for index {index_name} to be ready...")
                    # Wait for index to be ready
                    time.sleep(20)
                else:
                    print(f"Index {index_name} already exists")

                # Initialize index
                print(f"Connecting to index: {index_name}")
                self.index = self.pc.Index(index_name)

                # Verify index is accessible
                stats = self.index.describe_index_stats()
                print(f"Successfully connected to index. Stats: {stats}")

            except Exception as e:
                print(f"Error during index operations: {str(e)}")
                raise

        except Exception as e:
            print(f"Service initialization error: {str(e)}")
            raise Exception(f"Service initialization error: {str(e)}")

    def delete_index(self, index_name: str):
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if index_name in existing_indexes:
                print(f"Deleting index: {index_name}")
                self.pc.delete_index(index_name)
                print(f"Successfully deleted index: {index_name}")
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
            raise Exception(f"Failed to delete index: {str(e)}")

class TextProcessor:
    @staticmethod
    def extract_text_from_pdf(file_bytes):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_bytes):
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"DOCX extraction error: {str(e)}")
            return ""

    @staticmethod
    def chunk_text(text, chunk_size=Config.CHUNK_SIZE):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

class ChatbotManager:
    def __init__(self):
        self.chatbots = {}
        self.base_upload_dir = "uploaded_files"
        self.chat_history_dir = "chat_history"
        os.makedirs(self.base_upload_dir, exist_ok=True)
        os.makedirs(self.chat_history_dir, exist_ok=True)
        self.load_existing_chatbots()

    def load_existing_chatbots(self):
        if os.path.exists(self.base_upload_dir):
            for chatbot_name in os.listdir(self.base_upload_dir):
                if os.path.isdir(os.path.join(self.base_upload_dir, chatbot_name)):
                    self.initialize_existing_chatbot(chatbot_name)

    def initialize_existing_chatbot(self, chatbot_name: str):
        index_name = f"rag-chatbot-{chatbot_name.lower()}"
        service_manager = ServiceManager()
        service_manager.initialize_services(index_name)
        
        chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
        files = os.listdir(chatbot_dir)
        
        self.chatbots[chatbot_name] = {
            "index_name": index_name,
            "service_manager": service_manager,
            "files": files,
            "created_date": self.get_creation_date(chatbot_dir)
        }

    def get_creation_date(self, directory: str) -> str:
        timestamp = os.path.getctime(directory)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def create_chatbot(self, chatbot_name: str):
        try:
            if not chatbot_name or not re.match("^[a-zA-Z0-9-_]+$", chatbot_name):
                raise HTTPException(400, "Invalid chatbot name. Use only letters, numbers, hyphens and underscores")

            if chatbot_name in self.chatbots:
                raise HTTPException(400, "Chatbot with this name already exists")

            chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
            os.makedirs(chatbot_dir, exist_ok=True)

            # Create a unique index name for this chatbot
            index_name = f"rag-chatbot-{chatbot_name.lower()}-{int(time.time())}"[:62]
            service_manager = ServiceManager()
            
            try:
                print(f"Initializing services for chatbot: {chatbot_name}")
                service_manager.initialize_services(index_name)
                print(f"Services initialized successfully for chatbot: {chatbot_name}")
            except Exception as e:
                print(f"Error initializing services: {str(e)}")
                if os.path.exists(chatbot_dir):
                    shutil.rmtree(chatbot_dir)
                raise HTTPException(500, f"Failed to initialize services: {str(e)}")

            self.chatbots[chatbot_name] = {
                "index_name": index_name,
                "service_manager": service_manager,
                "files": [],
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return {
                "status": "success",
                "message": f"Chatbot '{chatbot_name}' created successfully with serverless index",
                "name": chatbot_name,
                "index_name": index_name
            }
        except Exception as e:
            print(f"Error creating chatbot: {str(e)}")
            chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
            if os.path.exists(chatbot_dir):
                shutil.rmtree(chatbot_dir)
            if chatbot_name in self.chatbots:
                del self.chatbots[chatbot_name]
            raise HTTPException(500, f"Error creating chatbot: {str(e)}")

    def delete_chatbot(self, chatbot_name: str):
        if chatbot_name not in self.chatbots:
            raise HTTPException(404, "Chatbot not found")

        service_manager = self.chatbots[chatbot_name]["service_manager"]
        index_name = self.chatbots[chatbot_name]["index_name"]
        
        # Delete the index
        service_manager.delete_index(index_name)

        chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
        if os.path.exists(chatbot_dir):
            shutil.rmtree(chatbot_dir)

        history_file = os.path.join(self.chat_history_dir, f"{chatbot_name}.json")
        if os.path.exists(history_file):
            os.remove(history_file)

        del self.chatbots[chatbot_name]

        return {"status": "success", "message": f"Chatbot '{chatbot_name}' deleted"}

    def get_chatbot_info(self, chatbot_name: str) -> Dict:
        if chatbot_name not in self.chatbots:
            raise HTTPException(404, "Chatbot not found")

        return {
            "name": chatbot_name,
            "files": self.chatbots[chatbot_name]["files"],
            "created_date": self.chatbots[chatbot_name]["created_date"]
        }

    def save_chat_history(self, chatbot_name: str, query: str, answer: str):
        history_file = os.path.join(self.chat_history_dir, f"{chatbot_name}.json")
        
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": answer
        })
        
        with open(history_file, 'w') as f:
            json.dump(history, f)

    def get_chat_history(self, chatbot_name: str) -> List[Dict]:
        history_file = os.path.join(self.chat_history_dir, f"{chatbot_name}.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
# Initialize managers
Config.validate_env_vars()
chatbot_manager = ChatbotManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/chatbots")
async def list_chatbots():
    chatbots_info = []
    for name in chatbot_manager.chatbots:
        chatbots_info.append(chatbot_manager.get_chatbot_info(name))
    return {"status": "success", "chatbots": chatbots_info}

@app.post("/chatbot/create")
async def create_chatbot(request: Request):
    try:
        data = await request.json()
        chatbot_name = data.get("name")
        
        if not chatbot_name:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Chatbot name is required"}
            )
        
        result = chatbot_manager.create_chatbot(chatbot_name)
        return JSONResponse(content=result)
        
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": str(he.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )

@app.delete("/chatbot/{chatbot_name}")
async def delete_chatbot(chatbot_name: str):
    return chatbot_manager.delete_chatbot(chatbot_name)

@app.post("/chatbot/{chatbot_name}/upload")
async def upload_file(chatbot_name: str, file: UploadFile = File(...)):
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")

    try:
        # Create directory if it doesn't exist
        chatbot_dir = os.path.join(chatbot_manager.base_upload_dir, chatbot_name)
        os.makedirs(chatbot_dir, exist_ok=True)

        # Validate file size
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        file_content = b''
        
        # Read file in chunks
        while chunk := await file.read(8192):
            file_size += len(chunk)
            file_content += chunk
            if file_size > max_size:
                raise HTTPException(400, "File too large (max 10MB)")

        # Validate file type
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['pdf', 'docx', 'txt']:
            raise HTTPException(400, "Unsupported file format. Only PDF, DOCX, and TXT files are allowed.")

        # Save file
        file_path = os.path.join(chatbot_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Process file content
        if file_extension == 'pdf':
            text = TextProcessor.extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            text = TextProcessor.extract_text_from_docx(file_content)
        else:  # txt file
            text = file_content.decode('utf-8')

        if not text:
            os.remove(file_path)
            raise HTTPException(400, "No text could be extracted from the file")

        # Process chunks
        # In your upload_file function, modify the embedding section:

# Process chunks
        chunks = TextProcessor.chunk_text(text)
        service_manager = chatbot_manager.chatbots[chatbot_name]["service_manager"]

        print(f"Processing {len(chunks)} chunks for file: {file.filename}")
        vectors_to_upsert = []

        # Process in batches of 10 chunks
        BATCH_SIZE = 10
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            filtered_chunks = [chunk for chunk in batch_chunks if chunk.strip()]
            
            if filtered_chunks:
                # Get embeddings for batch
                response = service_manager.cohere_client.embed(
                    texts=filtered_chunks,
                    model='embed-english-v3.0',
                    input_type="search_document"
                )
                
                # Process batch results
                for j, embedding in enumerate(response.embeddings):
                    chunk_index = i + j
                    if chunk_index < len(chunks):  # Safety check
                        vectors_to_upsert.append({
                            'id': f'{chatbot_name}_chunk_{chunk_index}_{os.urandom(4).hex()}',
                            'values': embedding,
                            'metadata': {
                                'text': filtered_chunks[j],
                                'file_name': file.filename,
                                'chatbot': chatbot_name
                            }
                        })
                
                # Add rate limiting delay - stay under 40 calls per minute
                time.sleep(1.5)  # 1.5 seconds delay

        if vectors_to_upsert:
            # Upsert vectors in batches to avoid overwhelming Pinecone
            UPSERT_BATCH_SIZE = 100
            for i in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
                batch = vectors_to_upsert[i:i+UPSERT_BATCH_SIZE]
                service_manager.index.upsert(vectors=batch)
                print(f"Upserted batch {i//UPSERT_BATCH_SIZE + 1} of {len(vectors_to_upsert)//UPSERT_BATCH_SIZE + 1}")
                time.sleep(1)  # Add delay between batches

        # Update chatbot files list
        if file.filename not in chatbot_manager.chatbots[chatbot_name]["files"]:
            chatbot_manager.chatbots[chatbot_name]["files"].append(file.filename)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"File '{file.filename}' uploaded successfully",
                "details": {
                    "filename": file.filename,
                    "size": file_size,
                    "chunks_processed": len(vectors_to_upsert),
                    "text_length": len(text)
                }
            }
        )

    except HTTPException as he:
        print(f"HTTP Exception during file upload: {str(he)}")
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": str(he.detail)}
        )
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Upload failed: {str(e)}"}
        )

@app.post("/chatbot/{chatbot_name}/ask")
async def chat_with_bot(chatbot_name: str, request: QueryRequest):
    """Handles chat requests for different AI models."""
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")

    try:
        # Get relevant context from vector store
        service_manager = chatbot_manager.chatbots[chatbot_name]["service_manager"]
        
        # Get query embedding
        query_embedding = service_manager.cohere_client.embed(
            texts=[request.query],
            model='embed-english-v3.0',
            input_type="search_query"
        ).embeddings[0]

        # Search for similar vectors
        search_results = service_manager.index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={
                'chatbot': {'$eq': chatbot_name}
            }
        )

        # Extract relevant contexts
        contexts = [match.metadata['text'] for match in search_results.matches]
        
        # Prepare prompt with context
        system_prompt = """You are a helpful AI assistant. Using the provided context, 
        answer the user's question. If you cannot find the answer in the context, 
        say "I cannot find the answer in the provided context." 
        Base your answer solely on the context provided."""

        user_prompt = f"""Context: {' '.join(contexts)}

Question: {request.query}

Please provide a clear and concise answer based on the context above."""

        try:
            # Get response from selected model
            if request.model.lower() == "openai":
                if not OPENAI_API_KEY:
                    raise HTTPException(400, "OpenAI API key not configured")
                
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content

            elif request.model.lower() == "cohere":
                if not COHERE_API_KEY:
                    raise HTTPException(400, "Cohere API key not configured")
                
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"

                response = service_manager.cohere_client.chat(
                    message=user_prompt,
                    model="command",
                    temperature=0.7,
                    chat_history=[],
                    prompt_truncation='AUTO'
                )
                answer = response.text

            elif request.model.lower() == "togetherai":
                if not TOGETHERAI_API_KEY:
                    raise HTTPException(400, "TogetherAI API key not configured")
                
                url = "https://api.together.xyz/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {TOGETHERAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-ai/DeepSeek-R1",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
                
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise HTTPException(500, f"TogetherAI API error: {response.text}")
                
                answer = response.json()["choices"][0]["message"]["content"]

            else:
                raise HTTPException(400, f"Unsupported model: {request.model}")

            # Save chat history
            chatbot_manager.save_chat_history(chatbot_name, request.query, answer)

            return JSONResponse(
                content={
                    "status": "success",
                    "response": answer,
                    "context_used": contexts,
                    "model_used": request.model
                }
            )

        except Exception as model_error:
            print(f"Error with {request.model} model: {str(model_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating response with {request.model}: {str(model_error)}"
            )

    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@app.get("/chatbot/{chatbot_name}/history")
async def get_chat_history(chatbot_name: str):
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")
    
    history = chatbot_manager.get_chat_history(chatbot_name)
    return {"status": "success", "history": history}

def test_pinecone_connection():
    try:
        pc = pinecone.Pinecone(api_key=Config.PINECONE_API_KEY)
        indexes = pc.list_indexes()
        print(f"Successfully connected to Pinecone. Available indexes: {[index.name for index in indexes]}")
        return True
    except Exception as e:
        print(f"Failed to connect to Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Validate environment and create necessary directories
        print("starting server")
        Config.validate_env_vars()
        
        # Test Pinecone connection
        if not test_pinecone_connection():
            print("Failed to establish Pinecone connection. Please check your configuration.")
            sys.exit(1)
            
        os.makedirs("uploaded_files", exist_ok=True)
        os.makedirs("chat_history", exist_ok=True)
        
        # Start the application
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")