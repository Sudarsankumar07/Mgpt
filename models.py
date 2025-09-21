from pydantic import BaseModel
from typing import Optional, List

class LoadModelRequest(BaseModel):
    domain: str

class UploadResponse(BaseModel):
    doc_id: str
    message: str

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    domain: Optional[str] = "general"

class QueryResponse(BaseModel):
    summary: str
    key_points: List[str] = []
    guidance: str = ""
    citations: List[str] = []
    disclaimer: str