# Mgpt
Legal AI Demo
This project is a Streamlit-based web application designed to demystify legal documents using generative AI. It allows users to upload PDF or DOCX files, process them to extract text, generate embeddings, store them in a vector database, and query the documents to retrieve summarized answers with key points, guidance, and citations. The application supports two domains: general and legal, with domain-specific models for text embedding and processing.
Features

File Upload: Upload PDF or DOCX files for text extraction and embedding generation.
Domain Selection: Choose between general or legal domains for tailored processing.
Query Processing: Ask questions about the uploaded document and receive a structured response including a summary, key points, guidance, citations, and a disclaimer.
Embedding Generation: Uses sentence-transformer models to generate embeddings for text chunks.
Vector Database: Stores document chunks and embeddings using ChromaDB.
Generative AI: Integrates with the Grok API for generating answers based on retrieved document chunks.

Prerequisites

Python 3.8+
A Grok API key from xAI
Internet connection for downloading models and accessing the Grok API

Setup Instructions
1. Clone the Repository
git clone <repository-url>
cd legal-ai-demo

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt

4. Set Up Environment Variables
Create a .env file in the project root directory and add the following configuration:

GROQ_API_KEY="your_groq_api_key_here"
VECTOR_DB_DIR=./vector_db
CACHE_DIR=./models
DOMAIN_MODELS={"general": "sentence-transformers/all-MiniLM-L6-v2", "legal": "sentence-transformers/paraphrase-MiniLM-L3-v2"}