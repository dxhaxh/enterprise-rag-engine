import tiktoken
import uuid
from typing import List
from src.api.schemas import RawDocument, DocumentChunk

class TokenAwareChunker:
    """
    Slices raw documents into smaller semantic chunks using LLM-specific tokenization.
    
    Instead of splitting by characters or words, this uses `tiktoken` to split exactly 
    how the OpenAI model processes text. This prevents words or concepts from being 
    arbitrarily sliced in half, which would degrade the quality of vector embeddings.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def chunk_document(self, doc: RawDocument) -> List[DocumentChunk]:
        """
        Processes a single document using a sliding window approach.
        
        The sliding window (driven by the `overlap` parameter) ensures that context 
        at the boundary of two chunks is not lost. The metadata from the parent 
        document is inherited by every child chunk for downstream RBAC enforcement.
        """
        tokens = self.tokenizer.encode(doc.content)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunk = DocumentChunk(
                chunk_id=f"{doc.id}-{uuid.uuid4().hex[:8]}",
                doc_id=doc.id,
                content=chunk_text,
                metadata=doc.metadata
            )
            chunks.append(chunk)
            
        return chunks