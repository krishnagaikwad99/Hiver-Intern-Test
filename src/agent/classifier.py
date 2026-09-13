"""
Intent Classifier module.
Classifies incoming customer messages into intent taxonomy classes with confidence scores.
Supports Gemini API, OpenAI API, and deterministic fallback classification.
"""

import os
import json
import yaml
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

class IntentClassificationResult(BaseModel):
    intent: str
    confidence: float
    reasoning: str


class IntentClassifier:
    def __init__(self, taxonomy_path: str = "config/intents.yaml", provider: str = "mock"):
        self.provider = provider.lower()
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = yaml.safe_load(f)
        self.intents = self.taxonomy.get("intents", {})
        self.intent_list = list(self.intents.keys())

    def classify(self, message: str, context: str = "") -> IntentClassificationResult:
        """Classifies customer message using configured provider."""
        if self.provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            try:
                return self._classify_gemini(message, context)
            except Exception:
                pass

        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            try:
                return self._classify_openai(message, context)
            except Exception:
                pass

        return self._classify_mock(message)

    def _classify_mock(self, message: str) -> IntentClassificationResult:
        """Deterministic keyword/rule-based intent classifier for mock/offline mode."""
        lower = message.lower()

        if any(k in lower for k in ["battery", "screen", "earbud", "hardware", "charging", "mic", "display"]):
            return IntentClassificationResult(
                intent="device_hardware_issue",
                confidence=0.92,
                reasoning="Matched hardware component keywords."
            )
        elif any(k in lower for k in ["ios", "update", "freeze", "restart", "boot", "bug", "macos", "os"]):
            return IntentClassificationResult(
                intent="software_update_bug",
                confidence=0.89,
                reasoning="Matched software update/OS bug keywords."
            )
        elif any(k in lower for k in ["lock", "apple id", "password", "sign in", "2fa", "login", "auth"]):
            return IntentClassificationResult(
                intent="account_access_login",
                confidence=0.94,
                reasoning="Matched account login/authentication keywords."
            )
        elif any(k in lower for k in ["charge", "refund", "subscription", "double", "invoice", "receipt"]):
            return IntentClassificationResult(
                intent="billing_refund_query",
                confidence=0.91,
                reasoning="Matched billing and subscription refund keywords."
            )
        elif any(k in lower for k in ["shipping", "track", "delivery", "ups", "fedex", "order"]):
            return IntentClassificationResult(
                intent="order_delivery_delay",
                confidence=0.88,
                reasoning="Matched order tracking and delivery keywords."
            )
        elif any(k in lower for k in ["crack", "damage", "broken", "crushed", "box"]):
            return IntentClassificationResult(
                intent="damaged_defective_item",
                confidence=0.90,
                reasoning="Matched damaged/defective package keywords."
            )
        elif any(k in lower for k in ["app", "crash", "app store"]):
            return IntentClassificationResult(
                intent="app_crash_performance",
                confidence=0.87,
                reasoning="Matched application performance keywords."
            )
        elif any(k in lower for k in ["return", "policy", "exchange", "trade-in"]):
            return IntentClassificationResult(
                intent="return_exchange_policy",
                confidence=0.89,
                reasoning="Matched return policy and trade-in keywords."
            )
        elif any(k in lower for k in ["how to", "transfer", "setup", "enable", "how do i"]):
            return IntentClassificationResult(
                intent="feature_how_to",
                confidence=0.86,
                reasoning="Matched how-to feature configuration keywords."
            )
        else:
            return IntentClassificationResult(
                intent="other_general_query",
                confidence=0.55,
                reasoning="No strong keyword match found; assigned general query."
            )

    def _classify_gemini(self, message: str, context: str) -> IntentClassificationResult:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = f"""
Classify the customer message into exactly one of these intents: {self.intent_list}.
Message: {message}

Return JSON with keys: intent, confidence (0.0 to 1.0), reasoning.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        return IntentClassificationResult(**data)

    def _classify_openai(self, message: str, context: str) -> IntentClassificationResult:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = f"""
Classify the customer message into exactly one of these intents: {self.intent_list}.
Message: {message}
"""
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=IntentClassificationResult
        )
        return response.choices[0].message.parsed
