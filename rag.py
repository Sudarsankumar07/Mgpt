import os
import requests
from typing import Dict, Any, List
import mcp
import chromadb

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY in rag.py: {'set' if GROQ_API_KEY else 'not set'}")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

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
    except Exception as e:
        print(f"Collection {collection_name} not found: {e}")
        return []
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "documents", "distances"])
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
    prompt = f"As a {domain} expert, using the following context:\n\n{''.join(context_chunks)}\n\nAnswer this question concisely and list citations: {question}\n\nProvide: summary, key_points (3), guidance."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_context.get("groq_model", "openai/gpt-oss-20b"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": model_context.get("max_tokens", 512)
    }
    try:
        print("Sending request to Groq API")
        resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        out_text = body['choices'][0]['message']['content'] if 'choices' in body else str(body)
        lines = out_text.split("\n")
        response_dict = {"summary": "", "key_points": [], "guidance": ""}
        current_section = None
        for line in lines:
            if line.startswith("Summary:"):
                current_section = "summary"
                response_dict["summary"] = line.replace("Summary:", "").strip()
            elif line.startswith("Key Points:"):
                current_section = "key_points"
            elif line.startswith("Guidance:"):
                current_section = "guidance"
                response_dict["guidance"] = line.replace("Guidance:", "").strip()
            elif current_section == "key_points" and line.strip().startswith("-"):
                response_dict["key_points"].append(line.strip()[1:].strip())
            elif current_section and line.strip():
                response_dict[current_section] += " " + line.strip()
        return {"text": out_text, "parsed": response_dict, "error": False}
    except Exception as e:
        print(f"Groq API error: {e}")
        return {"text": f"Groq failed: {e}", "error": True}

def answer_query(domain: str, question: str, doc_id: str = None, model_context: dict = None) -> Dict[str, Any]:
    model_context = model_context or mcp.get_model_context(domain)
    chunks_info = retrieve_top_chunks(domain, question, model_context, top_k=4)
    context_texts = [c["text"] for c in chunks_info]
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