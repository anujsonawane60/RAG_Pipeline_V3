# Multi-LLM RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot platform that allows users to create chatbots, upload documents, and chat with them using multiple Language Models (LLMs).

## Features

*   Create, manage, and delete chatbots.
*   Upload PDF, DOCX, and TXT files to provide context.
*   Chat with chatbots and receive responses.
*   Choose between OpenAI, Cohere, and TogetherAI language models.
*   Chat history is saved.
* The type and size of the uploaded files are validated.

## Tech Stack

*   FastAPI
*   Pinecone
*   Cohere
*   OpenAI
*   TogetherAI
*   Python 3.10+
*   Uvicorn
* PyPDF2
* docx

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/anujsonawane60/RAG_Pipeline_V3.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd your-repo-name
    ```
3.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv myenv
    ```
4.  Activate the virtual environment:
    ```bash
    myenv\Scripts\activate
    ```
5.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
6. Create a `.env` file and add your API Keys.

OPENAI_API_KEY=your_openai_key 
COHERE_API_KEY=your_cohere_key 
TOGETHERAI_API_KEY=your_togetherai_key 
PINECONE_API_KEY=your_pinecone_key

7. Run the app:

uvicorn main:app --reload


## Usage
1. Go to http://127.0.0.1:8000/
2. Click on `Chatbots` in the menu.
3. Create a chatbot clicking on `Create New` button.
4. Select a chatbot.
5. Upload a file clicking on `Choose File` and then `Upload File`.
6. Select a model and send messages to the chatbot.

## API Endpoints


## Contributing


## License



