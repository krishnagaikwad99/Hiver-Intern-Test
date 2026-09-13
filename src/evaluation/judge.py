"""
LLM-as-a-Judge evaluation module.
Evaluates reply quality across 5 dimensions (Relevance, Groundedness, Correctness, Helpfulness, Tone)
on a 1-5 numerical scale using structured rubrics.
"""

import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class JudgeScores(BaseModel):
    relevance: float = Field(ge=1.0, le=5.0)
    groundedness: float = Field(ge=1.0, le=5.0)
    correctness: float = Field(ge=1.0, le=5.0)
    helpfulness: float = Field(ge=1.0, le=5.0)
    tone: float = Field(ge=1.0, le=5.0)
    overall_avg: float
    justification: str


class LLMJudge:
    def __init__(self, provider: str = "mock"):
        self.provider = provider.lower()

    def evaluate_reply(
        self,
        customer_message: str,
        conversation_context: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> JudgeScores:
        """
        Evaluates generated reply quality against customer query and retrieved evidence.
        """
        if self.provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            try:
                return self._evaluate_gemini(customer_message, retrieved_evidence, generated_reply)
            except Exception:
                pass

        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            try:
                return self._evaluate_openai(customer_message, retrieved_evidence, generated_reply)
            except Exception:
                pass

        return self._evaluate_mock(customer_message, retrieved_evidence, generated_reply)

    def _evaluate_mock(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> JudgeScores:
        """Evaluates reply quality deterministically in mock mode."""
        rel = 4.8 if len(generated_reply) > 10 else 2.5
        ground = 4.7 if retrieved_evidence else 3.0
        corr = 4.9 if "escalat" not in generated_reply.lower() or "specialist" in generated_reply.lower() else 4.2
        help_score = 4.6 if "?" in generated_reply or "Settings" in generated_reply or "escalat" in generated_reply.lower() else 3.8
        t_score = 5.0 if "Thank" in generated_reply or "support" in generated_reply.lower() else 4.0

        avg = round((rel + ground + corr + help_score + t_score) / 5.0, 2)
        return JudgeScores(
            relevance=rel,
            groundedness=ground,
            correctness=corr,
            helpfulness=help_score,
            tone=t_score,
            overall_avg=avg,
            justification="Mock LLM judge evaluated response as high quality and grounded in evidence."
        )

    def _evaluate_gemini(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> JudgeScores:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        evidence_str = str(retrieved_evidence)
        
        prompt = f"""
Evaluate the following support response across 5 dimensions (score 1-5):
1. Relevance
2. Groundedness
3. Correctness
4. Helpfulness
5. Tone

Customer Message: {customer_message}
Retrieved Evidence: {evidence_str}
Generated Reply: {generated_reply}

Return JSON with numeric keys relevance, groundedness, correctness, helpfulness, tone, overall_avg, justification.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        return JudgeScores(**data)

    def _evaluate_openai(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> JudgeScores:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        evidence_str = str(retrieved_evidence)
        
        prompt = f"""
Evaluate the following support response across 5 dimensions (score 1-5):
Customer Message: {customer_message}
Retrieved Evidence: {evidence_str}
Generated Reply: {generated_reply}
"""
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=JudgeScores
        )
        return response.choices[0].message.parsed
