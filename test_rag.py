from src.retrieval.hybrid_search import EnterpriseSearchEngine
from src.generation.generator import RAGGenerator

def test_rag_pipeline():
    print("1. Initializing Enterprise Search Engine and Local LLM Generator...")
    search_engine = EnterpriseSearchEngine()
    generator = RAGGenerator()
    
    query = "How do backend architectures process latency-sensitive requests?"
    
    print(f"\n2. Searching database for query: '{query}'...")
    # Perform secure search & re-ranking for an authorized user
    chunks = search_engine.search(
        query_text=query, 
        tenant_id="tenant-Alpha", 
        user_role="quant", 
        top_k=2
    )
    print(f"   Found and re-ranked {len(chunks)} precise chunks.")

    print("\n3. Generating natural language answer using local Llama 3...")
    answer = generator.generate_answer(query=query, chunks=chunks)
    
    print("\n================ AI GENERATED ANSWER ================")
    print(answer)
    print("=====================================================")

if __name__ == "__main__":
    test_rag_pipeline()