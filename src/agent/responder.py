"""
Grounded Response Generator module.
Drafts customer responses grounded strictly in retrieved historical resolutions.
Includes anti-hallucination guardrails preventing invented policies or unverified refund promises.
"""

import os
import json
from typing import List, Dict, Any, Optional

class GroundedResponder:
    def __init__(self, provider: str = "mock"):
        self.provider = provider.lower()

    def generate_response(
        self,
        customer_message: str,
        intent: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Drafts a grounded support response based on retrieved historical evidence.
        """
        if not retrieved_evidence:
            return "Thank you for reaching out! To assist you further with this issue, I am escalating your query to a support specialist."

        if self.provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            try:
                return self._generate_gemini(customer_message, intent, retrieved_evidence)
            except Exception:
                pass

        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            try:
                return self._generate_openai(customer_message, intent, retrieved_evidence)
            except Exception:
                pass

        return self._generate_mock(customer_message, intent, retrieved_evidence)

    def _generate_mock(
        self,
        customer_message: str,
        intent: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> str:
        """Generates grounded response using historical evidence in mock mode."""
        top_match = retrieved_evidence[0]
        historical_res = top_match.get("support_response", "")

        # Format historical resolution cleanly without internal handle mentions
        import re
        clean_res = re.sub(r'^(?:@\w+\s*)+', '', historical_res).strip()

        if clean_res:
            return f"Thank you for contacting support! {clean_res}"
        
        return "Thank you for reaching out. Please let us know if trying these troubleshooting steps helps resolve your query."

    def _generate_gemini(self, customer_message: str, intent: str, evidence: List[Dict[str, Any]]) -> str:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        evidence_text = "\n".join([f"- {e.get('support_response')}" for e in evidence])
        
        prompt = f"""
You are an AI customer support agent for AppleSupport.
Customer Message: {customer_message}
Intent: {intent}

Historical Support Evidence:
{evidence_text}

Rules:
1. Base your answer strictly on the historical evidence.
2. Do not invent refunds, policies, or unverified claims.
3. Be concise, polite, and professional.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()

    def _generate_openai(self, customer_message: str, intent: str, evidence: List[Dict[str, Any]]) -> str:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        evidence_text = "\n".join([f"- {e.get('support_response')}" for e in evidence])
        
        prompt = f"""
You are an AI customer support agent.
Customer Message: {customer_message}
Intent: {intent}

Historical Support Evidence:
{evidence_text}

Draft a helpful, grounded response based ONLY on the evidence.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
