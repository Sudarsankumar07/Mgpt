from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "./models"  # Set cache directory
models = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L3-v2"]

for model_name in models:
    print(f"Downloading {model_name}...")
    try:
        # Explicitly download the model and tokenizer files
        snapshot_download(repo_id=model_name, cache_dir="./models")
        # Initialize the model
        SentenceTransformer(model_name, cache_folder="./models")
        print(f"Downloaded {model_name} to ./models")
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")