from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.retrieval.reranker import ReRanker

class EnterpriseSearchEngine:
    """
    Executes secure vector searches against Qdrant with integrated RBAC, 
    multi-tenancy, and high-precision Cross-Encoder re-ranking.
    """
    def __init__(self):
        # Connect to our local file-based Qdrant database folder
        self.client = QdrantClient(path="local_qdrant_storage")
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Load the local embedding model and the re-ranker
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.reranker = ReRanker()

    search_top_k: int = 10  # Retrieve more candidates initially for the re-ranker to filter

    def search(self, query_text: str, tenant_id: str, user_role: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a two-stage retrieval pipeline:
        1. Secure Vector Search (Bi-Encoder) to fetch a broader candidate pool.
        2. Cross-Encoder Re-ranking to guarantee maximum semantic precision.
        """
        # 1. Convert search query into a 384-dimensional normalized vector
        query_vector = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True).tolist()

        # 2. Build security filters for multi-tenancy and RBAC
        security_filter = Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="allowed_roles", match=MatchValue(value=user_role))
            ]
        )

        # 3. Execute vector search, pulling a larger pool (e.g., 10) for re-ranking
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=security_filter, 
            limit=10 
        )
        
        search_results = search_result.points

        # 4. Format initial results
        formatted_results = []
        for result in search_results:
            payload = result.payload or {}
            formatted_results.append({
                "chunk_id": payload.get("chunk_id"),
                "content": payload.get("content"),
                "vector_score": result.score, 
                "source": payload.get("source")
            })

        # 5. Apply Phase 4 Cross-Encoder Re-ranking to narrow down to top_k with high precision
        final_results = self.reranker.rerank(query=query_text, results=formatted_results, top_k=top_k)

        return final_results