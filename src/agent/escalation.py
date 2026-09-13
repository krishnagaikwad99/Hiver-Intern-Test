"""
Deterministic Escalation Policy Engine.
Evaluates multi-trigger risk factors (intent confidence, retrieval similarity, sensitive keywords, policy boundaries)
and decides whether to auto-handle or escalate to human agent.
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel

class EscalationDecision(BaseModel):
    action: str                        # "auto_handle" or "escalate"
    should_escalate: bool
    escalation_reason: Optional[str] = None
    risk_score: float                  # 0.0 to 1.0


class EscalationEngine:
    def __init__(
        self,
        min_intent_confidence: float = 0.65,
        min_retrieval_similarity: float = 0.55,
        sensitive_keywords: Optional[List[str]] = None
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity
        self.sensitive_keywords = sensitive_keywords or [
            "password", "pin", "ssn", "credit card", "hack", "compromised",
            "fraud", "legal", "lawsuit", "double charge", "stolen", "account locked"
        ]

    def evaluate(
        self,
        message: str,
        intent: str,
        confidence: float,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> EscalationDecision:
        """
        Evaluates escalation rules in strict order.
        """
        lower = message.lower()

        # Trigger 1: Sensitive / Security / Financial keyword detection
        for kw in self.sensitive_keywords:
            if kw in lower:
                return EscalationDecision(
                    action="escalate",
                    should_escalate=True,
                    escalation_reason=f"Sensitive topic detected: '{kw}' requires human security/billing authorization.",
                    risk_score=0.95
                )

        # Trigger 2: High-risk intents (account access, physical damage claims)
        if intent == "account_access_login":
            return EscalationDecision(
                action="escalate",
                should_escalate=True,
                escalation_reason="Account authentication and security recovery requires human verification.",
                risk_score=0.90
            )

        # Trigger 3: Low Intent Classification Confidence
        if confidence < self.min_intent_confidence:
            return EscalationDecision(
                action="escalate",
                should_escalate=True,
                escalation_reason=f"Low intent classification confidence ({confidence:.2f} < {self.min_intent_confidence:.2f}).",
                risk_score=0.75
            )

        # Trigger 4: No sufficiently relevant historical precedent found
        if not retrieved_evidence:
            return EscalationDecision(
                action="escalate",
                should_escalate=True,
                escalation_reason="No sufficiently similar historical resolution precedent was found in retrieval index.",
                risk_score=0.80
            )

        top_similarity = retrieved_evidence[0].get("similarity_score", 0.0)
        if top_similarity < self.min_retrieval_similarity:
            return EscalationDecision(
                action="escalate",
                should_escalate=True,
                escalation_reason=f"Top retrieval similarity score ({top_similarity:.2f} < {self.min_retrieval_similarity:.2f}) below threshold.",
                risk_score=0.70
            )

        # Trigger 5: Short ambiguous message check
        if len(message.strip()) < 10 or intent == "other_general_query":
            return EscalationDecision(
                action="escalate",
                should_escalate=True,
                escalation_reason="Ambiguous or incomplete query lacking technical context.",
                risk_score=0.65
            )

        # Passed all escalation checks -> Safe to Auto-Handle
        return EscalationDecision(
            action="auto_handle",
            should_escalate=False,
            escalation_reason=None,
            risk_score=0.10
        )
