from openai import OpenAI
from typing import List, Dict, Any

class RAGGenerator:
    """
    Takes retrieved and re-ranked document chunks and uses a local LLM (via Ollama) 
    to synthesize a secure, context-aware answer.
    """
    def __init__(self):
        # Point the OpenAI client to our local Ollama server
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama doesn't require a real API key, but the client expects a string
        )
        self.model_name = "llama3"

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs a strict prompt incorporating the retrieved chunks and queries the LLM.
        """
        if not chunks:
            return "I'm sorry, but I couldn't find any authorized documents matching your request."

        # Format the retrieved context chunks into a clean text block
        context_text = "\n\n".join([f"Source ({c['source']}):\n{c['content']}" for c in chunks])

        # Craft a strict enterprise system prompt to prevent hallucinations
        system_prompt = (
            "You are a secure enterprise AI assistant. Answer the user's question "
            "strictly using only the provided context below. If the answer cannot be found "
            "in the context, state that you do not have permission or information to answer it."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        # Call the local model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1  # Low temperature to minimize hallucinations and keep it factual
        )

        return response.choices[0].message.content