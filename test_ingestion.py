import asyncio
from src.api.schemas import RawDocument, DocumentMetadata
from src.ingestion.pipeline import IngestionPipeline

async def main():
    print("1. Initializing pipeline...")
    pipeline = IngestionPipeline()

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

    # This calls our pipeline: it will chunk the text, then ask OpenAI for the math vectors
    chunks = await pipeline.process_documents([doc])
    
    print(f"\nSuccess! Generated {len(chunks)} chunks.")

    # print out the results to prove it worked
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Content snippet: {chunk.content[:50]}...")
        print(f"Embedding length: {len(chunk.embedding)} dimensions")

if __name__ == "__main__":
    asyncio.run(main())