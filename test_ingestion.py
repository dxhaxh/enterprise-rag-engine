import asyncio
from src.api.schemas import RawDocument, DocumentMetadata
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import QdrantStore  # New import for the database

async def main():
    print("1. Initializing pipeline and Vector DB connection...")
    pipeline = IngestionPipeline()
    vector_store = QdrantStore()  # Connect to the database running in Docker

    # fake document that is long enough to trigger the chunking logic
    fake_text = "Modern backend architectures require highly scalable distributed systems to process latency-sensitive requests. " * 30

    # apply our strict metadata rules (tenant_id and roles)
    doc = RawDocument(
        id="doc-001",
        content=fake_text,
        metadata=DocumentMetadata(
            tenant_id="tenant-Squarepoint",
            allowed_roles=["admin", "quant"],
            source="internal_wiki"
        )
    )
    
    print(f"2. Processing document {doc.id} for {doc.metadata.tenant_id}...")

    # This calls our pipeline: it will chunk the text, then use our local AI model to generate the math vectors
    chunks = await pipeline.process_documents([doc])
    
    print(f"\nSuccess! Generated {len(chunks)} chunks.")

    # print out the results to prove it worked (just showing the first 2 chunks to keep the terminal clean)
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Content snippet: {chunk.content[:50]}...")
        print(f"Embedding length: {len(chunk.embedding)} dimensions")

    print("\n3. Saving chunks permanently to Qdrant...")
    
    # This takes our chunks (text + metadata + math vectors) and saves them into the Docker database
    vector_store.upsert_chunks(chunks)
    
    print("   Success! Data is now safely stored in the vector database.")

if __name__ == "__main__":
    asyncio.run(main())