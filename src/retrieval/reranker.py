from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class ReRanker:
    """
    Applies Cross-Encoder re-ranking to search results to maximize semantic precision.
    
    While Bi-Encoders (vector search) are great for fast broad filtering, 
    Cross-Encoders evaluate the query and document together to catch subtle contextual relationships.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Load a lightweight, highly efficient cross-encoder model optimized for ranking
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes the raw search results from Qdrant, scores them with the cross-encoder, 
        sorts them by true relevance, and returns the top_k best chunks.
        """
        if not results:
            return []

        # Prepare pairs of [query, document_content] for the cross-encoder to evaluate together
        pairs = [[query, r["content"]] for r in results]

        # Compute precision scores
        scores = self.model.predict(pairs)

        # Attach the new cross-encoder score to each result
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort descending by the new re-rank score
        sorted_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Return only the top_k highest precision results
        return sorted_results[:top_k]