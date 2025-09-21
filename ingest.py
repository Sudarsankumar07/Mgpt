import os
import uuid
from typing import Tuple
import chromadb
import mcp
from utils import extract_text_from_pdf, extract_text_from_docx, chunk_text

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")

def ingest_file(content: bytes, filename: str, domain: str, model_context: dict) -> Tuple[str, int]:
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection_name = f"{domain}_docs"
        print(f"Processing collection: {collection_name}")

        if collection_name in [c.name for c in client.list_collections()]:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(name=collection_name)

        print(f"Creating new collection: {collection_name}")
        collection = client.create_collection(name=collection_name)

        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(content)
        elif filename.lower().endswith('.docx'):
            text = extract_text_from_docx(content)
        else:
            text = content.decode("utf-8", errors="ignore")
        print(f"Extracted text length: {len(text)}")

        chunks = chunk_text(text, chunk_size=model_context.get("chunk_size", 500), overlap=0)
        if not chunks:
            raise ValueError("No text extracted from file")
        print(f"Created {len(chunks)} chunks")

        doc_id = str(uuid.uuid4())

        embeddings = mcp.encode_texts(domain, chunks, model_context)
        print(f"Generated embeddings: {len(embeddings)} vectors, dim={len(embeddings[0])}")

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"doc_id": doc_id, "filename": filename}] * len(chunks),
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
        )
        print(f"Added {len(chunks)} documents to collection")

        return doc_id, len(chunks)
    except Exception as e:
        print(f"Ingestion error: {str(e)}")
        raise Exception(f"Ingestion failed: {str(e)}")