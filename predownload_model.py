from sentence_transformers import SentenceTransformer
import os

os.environ["HF_HOME"] = "./models"  # Same as CACHE_DIR
models = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L3-v2"]
for model_name in models:
    print(f"Downloading {model_name}...")
    SentenceTransformer(model_name, cache_folder="./models")
    print(f"Downloaded {model_name} to ./models")