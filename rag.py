import os
from typing import Dict, Any, List
import mcp
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ChromaDB setup
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

# Groq setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY in rag.py: {'set' if GROQ_API_KEY else 'not set'}")

DISCLAIMER = "This AI is not a substitute for legal advice; consult a licensed professional."

def retrieve_top_chunks(domain: str, query: str, model_context: dict, top_k: int = 3) -> List[Dict[str, Any]]:
    q_embs = mcp.encode_texts(domain, [query], model_context)
    if not q_embs:
        print("No embeddings generated for query")
        return []

    q_emb = q_embs[0]
    print(f"Query embedding dim: {len(q_emb)}")
    collection_name = f"{domain}_docs"
    print(f"Querying collection: {collection_name}")

    try:
        collection = chroma_client.get_collection(collection_name)
        # Verify collection contents
        collection_count = collection.count()
        print(f"Collection {collection_name} contains {collection_count} documents")
    except Exception as e:
        print(f"Collection {collection_name} not found: {e}")
        return []

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    docs = []
    if results and "documents" in results and len(results["documents"]) > 0:
        for i in range(len(results["documents"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        print(f"Retrieved {len(docs)} documents")
    else:
        print("No documents retrieved")

    return docs

def call_groq_generate(domain: str, question: str, context_chunks: List[str], model_context: dict) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set in call_groq_generate")
        return {"text": "GROQ_API_KEY not set.", "error": True}

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Build the prompt
    prompt = (
        f"As a {domain} expert, using the following context:\n\n"
        f"{''.join(context_chunks) if context_chunks else 'No context available.'}\n\n"
        f"Answer this question concisely and list citations: {question}\n\n"
        f"Provide a response in the following format:\n"
        f"**Summary**: A concise summary of the answer.\n"
        f"**Key Points**: Exactly 3 bullet points starting with '- '.\n"
        f"**Guidance**: Practical guidance or next steps.\n"
        f"Ensure all sections are complete and do not truncate the response."
    )

    try:
        print("Sending request to Groq API (via SDK)")
        completion = client.chat.completions.create(
            model=model_context.get("groq_model", "openai/gpt-oss-20b"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=model_context.get("max_tokens", 2048),
        )

        out_text = completion.choices[0].message.content
        print(f"Raw Groq response: {out_text}")
        # Log token usage if available
        if hasattr(completion, "usage"):
            print(f"Token usage: {completion.usage}")
        response_dict = {"summary": "", "key_points": [], "guidance": ""}

        # Parse response
        lines = out_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Summary:") or line.startswith("**Summary**"):
                current_section = "summary"
                response_dict["summary"] = line.replace("Summary:", "").replace("**Summary**", "").strip()
            elif line.startswith("Key Points:") or line.startswith("**Key Points") or line.startswith("Key Points (3"):
                current_section = "key_points"
            elif line.startswith("Guidance:") or line.startswith("**Guidance**") or line.startswith("**Guidance for Use**"):
                current_section = "guidance"
                response_dict["guidance"] = line.replace("Guidance:", "").replace("**Guidance**", "").replace("**Guidance for Use**", "").strip()
            elif current_section == "key_points" and (line.startswith("-") or line.startswith("*")):
                response_dict["key_points"].append(line[1:].strip())
            elif current_section == "guidance" and line:
                response_dict["guidance"] += "\n" + line
            elif current_section == "summary" and line:
                response_dict["summary"] += " " + line

        # Fallback if no structured response
        if not response_dict["summary"]:
            response_dict["summary"] = out_text or "No response generated."
            print("Warning: No summary found, using raw text as summary")
        if not response_dict["key_points"]:
            response_dict["key_points"] = ["No key points provided due to incomplete response."]
            print("Warning: No key points found, using default message")
        if not response_dict["guidance"]:
            response_dict["guidance"] = "No guidance provided due to incomplete response."
            print("Warning: No guidance found, using default message")

        return {"text": out_text, "parsed": response_dict, "error": False}

    except Exception as e:
        print(f"Groq API error: {e}")
        return {"text": f"Groq failed: {e}", "error": True}

def answer_query(domain: str, question: str, doc_id: str = None, model_context: dict = None) -> Dict[str, Any]:
    model_context = model_context or mcp.get_model_context(domain)

    # Retrieve documents from vector DB
    chunks_info = retrieve_top_chunks(domain, question, model_context, top_k=4)
    context_texts = [c["text"] for c in chunks_info]

    # Call Groq LLM
    groq_resp = call_groq_generate(domain, question, context_texts, model_context)

    response = {
        "summary": groq_resp.get("parsed", {}).get("summary", groq_resp.get("text", "")),
        "key_points": groq_resp.get("parsed", {}).get("key_points", []),
        "guidance": groq_resp.get("parsed", {}).get("guidance", ""),
        "citations": [m.get("doc_id") for m in [ci["metadata"] for ci in chunks_info] if m],
        "disclaimer": DISCLAIMER
    }

    if groq_resp.get("error"):
        response["error"] = groq_resp.get("text")

    return response