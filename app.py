# Main file

import os
from dotenv import load_dotenv
import logging
import streamlit as st

load_dotenv()

import mcp
import ingest
import rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-ai-demo")

st.title("Generative AI for Demystifying Legal Documents")

# Sidebar for configuration
st.sidebar.title("Settings")
domain = st.sidebar.selectbox("Select Domain", ["general", "legal"], index=0)

# Debug: Confirm environment variables
if not os.getenv("GROQ_API_KEY"):
    st.sidebar.error("❌ GROQ_API_KEY not set in .env file!")

# Optional: Button to load model explicitly with context
if st.sidebar.button("Load Model for Domain"):
    try:
        model_context = mcp.get_model_context(domain)
        st.sidebar.success(f"Model loaded for {domain}: {model_context['model_name']}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")

# Main section: File Upload
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
if uploaded_file is not None:
    if st.button("Upload and Ingest"):
        with st.spinner("Processing file and generating embeddings..."):
            try:
                content = uploaded_file.read()
                filename = uploaded_file.name
                model_context = mcp.get_model_context(domain)
                doc_id, chunk_count = ingest.ingest_file(content, filename, domain, model_context, reset_collection=False)
                st.session_state.doc_id = doc_id
                st.success(f"Upload successful! Doc ID: {doc_id}, Chunks: {chunk_count}")
                logger.info(f"Stored doc_id: {doc_id}")
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

# Main section: Query
st.header("Query the Document")
if "doc_id" in st.session_state:
    question = st.text_input("Enter your question:")
    if st.button("Submit Query"):
        with st.spinner("Querying document..."):
            try:
                model_context = mcp.get_model_context(domain)
                response = rag.answer_query(domain, question, st.session_state.doc_id, model_context)
                logger.info(f"Query response: {response}")  # Log full response for debugging
                st.subheader("Summary")
                st.write(response.get("summary", "No summary available"))
                st.subheader("Key Points")
                if response.get("key_points"):
                    for point in response.get("key_points", []):
                        st.write(f"- {point}")
                        if point == "No key points provided due to incomplete response.":
                            st.warning("Key points could not be generated due to an incomplete response from the AI.")
                else:
                    st.write("No key points available")
                    st.warning("Key points could not be generated.")
                st.subheader("Guidance")
                st.write(response.get("guidance", "No guidance available"))
                if response.get("guidance") == "No guidance provided due to incomplete response.":
                    st.warning("Guidance could not be generated due to an incomplete response from the AI.")
                st.subheader("Citations")
                if response.get("citations"):
                    for citation in response.get("citations", []):
                        st.write(f"- {citation}")
                else:
                    st.write("No citations available")
                st.subheader("Disclaimer")
                st.write(response.get("disclaimer", "No disclaimer available"))
                if response.get("error"):
                    st.error(f"Error: {response.get('error')}")
            except Exception as e:
                st.error(f"Query failed: {str(e)}")
else:
    st.info("Please upload a document first to get a Doc ID.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))