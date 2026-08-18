import json
from openai import OpenAI
from typing import List, Dict, Any

class RAGEvaluator:
    """
    Acts as an automated judge to score the quality of the RAG pipeline's outputs.
    Evaluates based on Faithfulness (no hallucinations) and Relevance.
    """
    def __init__(self):
        # We reuse our local Ollama setup to act as the judge
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model_name = "llama3"

    def evaluate(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Passes the query, retrieved context, and generated answer to the LLM judge
        and forces it to return a structured JSON scorecard.
        """
        system_prompt = (
            "You are an impartial, strict enterprise AI grader. Your job is to evaluate "
            "a Retrieval-Augmented Generation (RAG) system. You will be given a user's QUERY, "
            "the retrieved CONTEXT, and the AI's generated ANSWER.\n\n"
            "You must score two metrics from 0 to 10:\n"
            "1. 'faithfulness': How well is the ANSWER supported by the CONTEXT? (0 = totally made up, 10 = strictly derived from context)\n"
            "2. 'relevance': How well does the ANSWER address the QUERY? (0 = completely irrelevant, 10 = perfect and direct answer)\n\n"
            "You MUST return ONLY a valid JSON object in this exact format, with no additional text:\n"
            "{\"faithfulness_score\": 10, \"faithfulness_reason\": \"...\", \"relevance_score\": 10, \"relevance_reason\": \"...\"}"
        )

        user_prompt = f"QUERY: {query}\n\nCONTEXT: {context}\n\nANSWER: {answer}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # Zero creativity. Just strict grading.
                response_format={"type": "json_object"} # Force the model to output strict JSON
            )
            
            # Parse the JSON string returned by the LLM into a Python dictionary
            scorecard = json.loads(response.choices[0].message.content)
            return scorecard
        except Exception as e:
            return {"error": f"Failed to evaluate: {str(e)}"}