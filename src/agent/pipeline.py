"""
Unified AI Support Agent Pipeline.
Orchestrates Intent Classification -> Evidence Retrieval -> Escalation Evaluation -> Response Generation.
Returns structured Pydantic / JSON response.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.agent.classifier import IntentClassifier
from src.agent.responder import GroundedResponder
from src.agent.escalation import EscalationEngine
from src.retrieval.retrieve import retrieve_similar_resolutions

class EvidenceItem(BaseModel):
    conversation_id: str
    similarity: float
    summary: str


class AgentResponse(BaseModel):
    intent: str
    confidence: float
    action: str                        # "auto_handle" or "escalate"
    evidence: List[EvidenceItem] = []
    reply: str
    escalation_reason: Optional[str] = None


class HiverSupportAgent:
    def __init__(
        self,
        taxonomy_path: str = "config/intents.yaml",
        index_path: str = "data/processed/retrieval_index.pkl",
        provider: str = "mock"
    ):
        self.classifier = IntentClassifier(taxonomy_path=taxonomy_path, provider=provider)
        self.responder = GroundedResponder(provider=provider)
        self.escalation_engine = EscalationEngine()
        self.index_path = index_path

    def process_message(self, message: str, conversation_context: str = "") -> AgentResponse:
        """
        Processes an incoming customer message end-to-end.
        """
        # Step 1: Classify Intent
        cls_result = self.classifier.classify(message, conversation_context)
        
        # Step 2: Retrieve Historical Resolution Evidence
        evidence_matches = retrieve_similar_resolutions(
            query=message,
            top_k=3,
            threshold=0.50,
            index_path=self.index_path
        )

        formatted_evidence = []
        for em in evidence_matches:
            formatted_evidence.append(EvidenceItem(
                conversation_id=em.get("conversation_id", "conv_unknown"),
                similarity=em.get("similarity_score", 0.0),
                summary=em.get("support_response", "")[:120] + "..."
            ))

        # Step 3: Evaluate Escalation Policy
        esc_decision = self.escalation_engine.evaluate(
            message=message,
            intent=cls_result.intent,
            confidence=cls_result.confidence,
            retrieved_evidence=evidence_matches
        )

        # Step 4: Generate Grounded Response or Escalation Notice
        if esc_decision.should_escalate:
            reply = f"Your query regarding '{cls_result.intent}' has been escalated to a support specialist. Reason: {esc_decision.escalation_reason}"
        else:
            reply = self.responder.generate_response(
                customer_message=message,
                intent=cls_result.intent,
                retrieved_evidence=evidence_matches
            )

        return AgentResponse(
            intent=cls_result.intent,
            confidence=round(cls_result.confidence, 4),
            action=esc_decision.action,
            evidence=formatted_evidence,
            reply=reply,
            escalation_reason=esc_decision.escalation_reason
        )
