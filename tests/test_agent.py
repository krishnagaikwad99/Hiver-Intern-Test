"""
Unit tests for AI Support Agent pipeline, classifier, responder, and escalation rules.
"""

import unittest
from src.agent.classifier import IntentClassifier
from src.agent.escalation import EscalationEngine
from src.agent.pipeline import HiverSupportAgent

class TestAgentModule(unittest.TestCase):
    def test_intent_classifier(self):
        classifier = IntentClassifier(provider="mock")
        res = classifier.classify("My battery dies in 30 minutes")
        self.assertEqual(res.intent, "device_hardware_issue")
        self.assertGreater(res.confidence, 0.8)

    def test_escalation_engine_sensitive_trigger(self):
        engine = EscalationEngine()
        decision = engine.evaluate(
            message="I was charged twice on my credit card!",
            intent="billing_refund_query",
            confidence=0.9,
            retrieved_evidence=[]
        )
        self.assertTrue(decision.should_escalate)
        self.assertEqual(decision.action, "escalate")

    def test_support_agent_end_to_end(self):
        agent = HiverSupportAgent(provider="mock")
        response = agent.process_message("My iPhone battery dies fast")
        self.assertIn(response.intent, ["device_hardware_issue", "software_update_bug"])
        self.assertIn(response.action, ["auto_handle", "escalate"])
        self.assertIsInstance(response.reply, str)

if __name__ == "__main__":
    unittest.main()
