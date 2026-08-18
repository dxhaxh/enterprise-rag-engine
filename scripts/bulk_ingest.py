import asyncio
import aiohttp
from datasets import load_dataset
from tqdm.asyncio import tqdm

# Configuration
API_URL = "http://127.0.0.1:8000/ingest"
BATCH_SIZE = 25   # Concurrency bouncer
MAX_DOCS = 1000   

async def send_document(session: aiohttp.ClientSession, content: str, doc_id: str, semaphore: asyncio.Semaphore):
    """Sends a single document matching the exact FastAPI RawDocument schema."""
    async with semaphore:
        payload = {
            "id": f"fin_doc_{doc_id}",
            "content": content,
            "metadata": {
                "tenant_id": "tenant-Finance",
                "allowed_roles": ["quant", "analyst", "admin"],
                "source": f"hf-financial-news-{doc_id}"
            }
        }
        
        try:
            async with session.post(API_URL, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}: {await response.text()}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

async def main():
    print("1. Downloading Financial Dataset from Hugging Face...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    
    docs_to_process = dataset.select(range(min(MAX_DOCS, len(dataset))))
    print(f"2. Loaded {len(docs_to_process)} financial documents. Starting Async Ingestion...")

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    connector = aiohttp.TCPConnector(limit=100)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, row in enumerate(docs_to_process):
            text = row.get('text', '').strip()
            if text:
                tasks.append(send_document(session, text, str(i), semaphore))

        results = await tqdm.gather(*tasks, desc="Ingesting Documents")
        
        successes = sum(1 for r in results if r.get("status") == "success")
        print(f"\n3. Ingestion Complete! Successfully ingested {successes}/{len(tasks)} documents.")
        
        if successes != len(tasks) and len(results) > 0:
            errors = [r for r in results if r.get("status") != "success"]
            print(f"🚨 Error sample: {errors[0]}")

if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())