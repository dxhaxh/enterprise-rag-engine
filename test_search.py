from src.retrieval.hybrid_search import EnterpriseSearchEngine

def test_search_engine():
    print("1. Initializing Enterprise Search Engine...")
    search_engine = EnterpriseSearchEngine()
    
    query = "How do backend architectures process requests?"
    
    # Test Scenario 1: Authorized User (tenant-Alpha, role 'quant')
    print(f"\n--- Scenario 1: Authorized User (Alpha / quant) ---")
    results = search_engine.search(
        query_text=query, 
        tenant_id="tenant-Alpha", 
        user_role="quant", 
        top_k=2
    )
    print(f"Found {len(results)} authorized chunks:")
    for r in results:
        print(f"  - Re-rank Score: {r['rerank_score']:.4f} | Vector Score: {r['vector_score']:.4f} | Snippet: {r['content'][:40]}...")

    # Test Scenario 2: Unauthorized Role (tenant-Alpha, role 'intern' - not in allowed roles)
    print(f"\n--- Scenario 2: Unauthorized Role (Alpha / intern) ---")
    results_unauthorized = search_engine.search(
        query_text=query, 
        tenant_id="tenant-Alpha", 
        user_role="intern", 
        top_k=2
    )
    print(f"Found {len(results_unauthorized)} chunks (Should be 0 due to RBAC block).")

    # Test Scenario 3: Wrong Tenant (tenant-Beta, role 'quant')
    print(f"\n--- Scenario 3: Wrong Tenant (Beta / quant) ---")
    results_wrong_tenant = search_engine.search(
        query_text=query, 
        tenant_id="tenant-Beta", 
        user_role="quant", 
        top_k=2
    )
    print(f"Found {len(results_wrong_tenant)} chunks (Should be 0 due to multi-tenant isolation).")

if __name__ == "__main__":
    test_search_engine()