from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List
import uuid

from src.config import settings
from src.api.schemas import DocumentChunk

class QdrantStore:
    """
    Manages the connection and data ingestion for the Qdrant Vector Database.

    We use Qdrant because it supports complex payload filtering(allows us to filter by metadata)(RBAC) natively 
    during the vector search itself. This means we don't waste memory retrieving 
    unauthorized documents just to filter them out later in Python.
    """
    def __init__(self):
        # Connect to the Qdrant instance running in our Docker container
        #self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.client = QdrantClient(path="local_qdrant_storage") # local storage to bypass need for docker for now
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Ensure the collection exists when the app starts
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Creates the collection if it doesn't already exist.
        We specify 384 dimensions to match our local SentenceTransformer model.
        Cosine distance is used to measure semantic similarity based on vector angles.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION, # 384
                    distance=Distance.DOT
                )
            )

    def upsert_chunks(self, chunks: List[DocumentChunk]):
        """
        Transforms DocumentChunks into Qdrant Points and uploads them.
        'Upsert' means it will insert the chunk, or update it if the ID already exists.
        """
        points = []
        for chunk in chunks:
            # Qdrant requires IDs to be standard UUIDs. 
            # We convert our custom string chunk_id into a proper UUID mathematically.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "tenant_id": chunk.metadata.tenant_id,
                        "allowed_roles": chunk.metadata.allowed_roles,
                        "source": chunk.metadata.source
                    }
                )
            )
        
        # Upload the data to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )