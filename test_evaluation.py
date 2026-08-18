from src.retrieval.hybrid_search import EnterpriseSearchEngine
from src.generation.generator import RAGGenerator
from src.evaluation.evaluator import RAGEvaluator

def run_evaluation_suite():
    print("1. Booting Pipeline & Evaluator...")
    search_engine = EnterpriseSearchEngine()
    generator = RAGGenerator()
    evaluator = RAGEvaluator()
    
    query = "How does QuantumStrike achieve sub-millisecond latency?"
    tenant_id = "tenant-Alpha"
    user_role = "quant"
    
    print(f"\n2. Executing RAG Pipeline for query: '{query}'")
    
    # Run Retrieval
    chunks = search_engine.search(query_text=query, tenant_id=tenant_id, user_role=user_role, top_k=2)
    context_text = "\n\n".join([c['content'] for c in chunks])
    
    # Run Generation
    answer = generator.generate_answer(query=query, chunks=chunks)
    
    print("\n--- GENERATED ANSWER ---")
    print(answer)
    print("------------------------")
    
    print("\n3. Running LLM-as-a-Judge Evaluation...")
    scorecard = evaluator.evaluate(query=query, context=context_text, answer=answer)
    
    print("\n================ PIPELINE SCORECARD ================")
    print(f"Faithfulness Score : {scorecard.get('faithfulness_score')}/10")
    print(f"Reasoning          : {scorecard.get('faithfulness_reason')}")
    print(f"Relevance Score    : {scorecard.get('relevance_score')}/10")
    print(f"Reasoning          : {scorecard.get('relevance_reason')}")
    print("====================================================")

if __name__ == "__main__":
    run_evaluation_suite()