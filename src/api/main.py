from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.schemas import RawDocument, QueryRequest, QueryResponse
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import QdrantStore
from src.retrieval.hybrid_search import EnterpriseSearchEngine
from src.generation.generator import RAGGenerator

# 1. Define our global AI and Database variables
ingestion_pipeline = None
vector_store = None
search_engine = None
generator = None

# 2. Use FastAPI 'lifespan' to safely load everything once when the server boots
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_pipeline, vector_store, search_engine, generator
    print("\n--- BOOTING ENTERPRISE RAG ENGINE ---")
    
    ingestion_pipeline = IngestionPipeline()
    vector_store = QdrantStore()
    
    # THE FIX: Pass the exact same database connection to the search engine!
    search_engine = EnterpriseSearchEngine(qdrant_client=vector_store.client)
    generator = RAGGenerator()
    
    print("--- ALL SYSTEMS GO: API IS LIVE ---\n")
    yield  # The API runs during this yield
    print("Shutting down engine safely...")

# Initialize the API application
app = FastAPI(title="Enterprise Secure RAG API", lifespan=lifespan)

@app.post("/ingest")
async def ingest_document(doc: RawDocument):
    """Receives a document, chunks it, embeds it, and saves it to the database."""
    try:
        chunks = await ingestion_pipeline.process_documents([doc])
        vector_store.upsert_chunks(chunks)
        return {"status": "success", "message": f"Ingested {len(chunks)} chunks for {doc.metadata.tenant_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    """Receives a question, securely searches, and generates an AI answer."""
    try:
        chunks = search_engine.search(
            query_text=req.query,
            tenant_id=req.tenant_id,
            user_role=req.user_role,
            top_k=req.top_k
        )
        answer = generator.generate_answer(query=req.query, chunks=chunks)
        unique_sources = list(set([c["source"] for c in chunks]))
        return QueryResponse(answer=answer, sources=unique_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))