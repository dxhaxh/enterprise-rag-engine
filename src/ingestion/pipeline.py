import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from src.api.schemas import RawDocument, DocumentChunk
from src.ingestion.chunker import TokenAwareChunker

class IngestionPipeline:
    """
    Orchestrates the transformation of raw text into embedded vector chunks 
    using a local open-source embedding model for zero-cost, private inference.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # This downloads and loads the model directly into your computer's memory
        self.model = SentenceTransformer(model_name)
        self.chunker = TokenAwareChunker(chunk_size=100, overlap=20)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings locally.
        Returns an ordered list of 384-dimensional float arrays.
        """
        # We tell the model to convert the text into numpy math arrays
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    async def process_documents(self, documents: List[RawDocument]) -> List[DocumentChunk]:
        """
        End-to-end pipeline: chunks documents and attaches vector embeddings in batches.
        """
        all_chunks: List[DocumentChunk] = []
        
        for doc in documents:
            all_chunks.extend(self.chunker.chunk_document(doc))

        BATCH_SIZE = 100
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            texts = [chunk.content for chunk in batch]
            
            # Since local models use your CPU, we use asyncio.to_thread so it doesn't freeze the app
            embeddings = await asyncio.to_thread(self.generate_embeddings, texts)
            
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb
                
        return all_chunks