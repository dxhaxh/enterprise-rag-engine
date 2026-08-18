from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentMetadata(BaseModel):
    tenant_id: str = Field(..., description="ID of the organization or tenant")
    allowed_roles: List[str] = Field(..., description="Roles permitted to access this document")
    source: str = Field(default="unknown", description="Origin of the document")
    
class RawDocument(BaseModel):
    id: str
    content: str
    metadata: DocumentMetadata

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class QueryRequest(BaseModel):
    """What the user sends us when asking a question"""
    query: str
    tenant_id: str
    user_role: str
    top_k: int = 2

class QueryResponse(BaseModel):
    """What we send back to the user"""
    answer: str
    sources: List[str]
