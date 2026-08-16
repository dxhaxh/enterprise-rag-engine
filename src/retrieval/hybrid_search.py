from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.config import settings

class EnterpriseSearchEngine:
    """
    Executes secure vector searches against Qdrant with integrated RBAC and multi-tenancy.
    """
    def __init__(self):
        # Connect to our local file-based Qdrant database folder
        self.client = QdrantClient(path="local_qdrant_storage")
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Load the local embedding model to convert search queries into math vectors
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def search(self, query_text: str, tenant_id: str, user_role: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a semantic vector search restricted by tenant and user role.
        
        Security Feature: The metadata filters (tenant_id and allowed_roles) are applied 
        DIRECTLY inside the vector search. The database ignores any chunks that 
        do not match the user's security clearance.
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

        # 3. Execute the search using query_points
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=security_filter, 
            limit=top_k
        )
        
        search_results = search_result.points

        # 4. Format the results safely, handling any object structure or tuple variation
        formatted_results = []
        for result in search_results:
            # Handle case where result might be unpacked differently depending on client version
            payload = {}
            score = 0.0
            
            if hasattr(result, "payload"):
                payload = result.payload or {}
                score = getattr(result, "score", 0.0)
            elif isinstance(result, tuple):
                # If it's a tuple, inspect elements for payload/score
                for item in result:
                    if isinstance(item, dict) and "chunk_id" in item:
                        payload = item
                    elif hasattr(item, "payload"):
                        payload = item.payload or {}
                    elif isinstance(item, (int, float)):
                        score = float(item)

            formatted_results.append({
                "chunk_id": payload.get("chunk_id"),
                "content": payload.get("content"),
                "score": score, 
                "source": payload.get("source")
            })

        return formatted_results