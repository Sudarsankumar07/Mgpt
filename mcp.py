import os
import json
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import torch

CACHE_DIR = os.getenv("CACHE_DIR", "./models")
DOMAIN_MODELS = json.loads(os.getenv("DOMAIN_MODELS", '{"general": "sentence-transformers/all-MiniLM-L6-v2", "legal": "sentence-transformers/paraphrase-MiniLM-L3-v2"}'))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Context Protocol
class ModelContext:
    def __init__(self, domain: str, model_name: str):
        self.domain = domain
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.max_tokens = 512
        self.chunk_size = 500
        self.groq_model = "mixtral-8x7b-32768"
        if domain == "legal":
            self.max_tokens = 1024
            self.chunk_size = 400
            self.groq_model = "mixtral-8x7b-32768"
        elif domain == "general":
            self.max_tokens = 512
            self.chunk_size = 500
            self.groq_model = "mixtral-8x7b-32768"

    def load_model(self):
        if self.model is None:
            print(f"Loading model for {self.domain}: {self.model_name} on {DEVICE}")
            self.model = SentenceTransformer(self.model_name, cache_folder=CACHE_DIR).to(DEVICE)
            test_embedding = self.model.encode(["test"], convert_to_tensor=False)[0]
            self.embedding_dim = len(test_embedding)
            print(f"Embedding dimension for {self.domain}: {self.embedding_dim}")
        else:
            print(f"Using cached model for {self.domain}: {self.model_name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "groq_model": self.groq_model
        }

# Cache for model contexts
_model_contexts = {}

def get_model_context(domain: str) -> Dict[str, Any]:
    if domain not in _model_contexts:
        model_name = DOMAIN_MODELS.get(domain, DOMAIN_MODELS["general"])
        _model_contexts[domain] = ModelContext(domain, model_name)
        _model_contexts[domain].load_model()
    else:
        print(f"Cache hit for {domain} model context")
    return _model_contexts[domain].to_dict()

def load_hf_model(domain: str):
    get_model_context(domain)

def encode_texts(domain: str, texts: List[str], model_context: Dict[str, Any] = None) -> List[List[float]]:
    model_context = model_context or get_model_context(domain)
    model = _model_contexts[domain].model
    if model_context["domain"] == "legal":
        texts = [f"Legal context: {text}" for text in texts]
    embeddings = model.encode(texts, convert_to_tensor=False).tolist()
    print(f"Encoded {len(texts)} texts for {domain}, dim={len(embeddings[0])}")
    return embeddings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="general")
    args = parser.parse_args()
    load_hf_model(args.domain)