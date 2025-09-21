try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader

import io
from typing import List
import docx

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception as e:
        return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_docx(file_bytes: bytes) -> str:
    bio = io.BytesIO(file_bytes)
    doc = docx.Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks