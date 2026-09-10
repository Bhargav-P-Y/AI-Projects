import sys
from pathlib import Path
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import unittest
# pyrefly: ignore [missing-import]
from llm_router.llm_router import ApiKeyRotator, LLMRouter, RoutingDecision
from context_builder.context_builder import AssembledContext
from safety_filter.safety_filter import RiskSignals

class TestApiKeyRotator(unittest.TestCase):
    def test_key_rotation(self):
        keys = ["key1", "key2", "key3"]
        rotator = ApiKeyRotator(keys)
        # Verify round-robin order
        self.assertEqual(rotator.get_next_key(), "key1")
        self.assertEqual(rotator.get_next_key(), "key2")
        self.assertEqual(rotator.get_next_key(), "key3")
        self.assertEqual(rotator.get_next_key(), "key1")

    def test_rate_limiting_cooldown(self):
        keys = ["key1", "key2"]
        rotator = ApiKeyRotator(keys)
        self.assertEqual(rotator.get_next_key(), "key1")
        # Mark key2 as rate limited
        rotator.mark_rate_limited("key2", cooldown_seconds=5)
        # Next key should bypass key2 and return key1 again
        self.assertEqual(rotator.get_next_key(), "key1")
        self.assertEqual(rotator.get_next_key(), "key1")


class TestLLMRouterParsing(unittest.TestCase):
    def setUp(self):
        self.router = LLMRouter(api_keys=["dummy_key"])

    def test_clean_json_array(self):
        raw = '[{"action": "notify", "message_type": "urgent", "reason": "test", "confidence": 0.9, "evidence_message_ids": "none"}]'
        parsed = self.router._parse_response(raw, 1)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["action"], "notify")

    def test_json_with_code_fences(self):
        raw = '```json\n[{"action": "digest", "message_type": "promotion", "reason": "promo", "confidence": 0.8, "evidence_message_ids": "none"}]\n```'
        parsed = self.router._parse_response(raw, 1)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["action"], "digest")

    def test_json_embedded_in_garbage(self):
        raw = 'Random text here [\n  {"action": "mute", "message_type": "scam", "reason": "scam text", "confidence": 0.95, "evidence_message_ids": "none"}\n] some more random text'
        parsed = self.router._parse_response(raw, 1)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["action"], "mute")

    def test_invalid_json_fallback(self):
        raw = "completely invalid json string"
        parsed = self.router._parse_response(raw, 2)
        self.assertEqual(parsed, [None, None])


class TestLLMRouterValidation(unittest.TestCase):
    def setUp(self):
        self.router = LLMRouter(api_keys=["dummy_key"])
        self.dummy_context = AssembledContext(
            message_id="msg_001",
            user_id="u_001",
            prompt_text="hello",
            evidence_message_ids_str="hist_1",
            risk_signals=RiskSignals(),
            has_dnd_active=False
        )

    def test_validation_success(self):
        raw = {
            "action": "notify",
            "message_type": "urgent",
            "reason": "Clear urgency signal",
            "confidence": 0.85,
            "evidence_message_ids": "hist_2"
        }
        dec = self.router._validate_decision(raw, self.dummy_context)
        self.assertEqual(dec.action, "notify")
        self.assertEqual(dec.message_type, "urgent")
        self.assertEqual(dec.reason, "Clear urgency signal")
        self.assertEqual(dec.confidence, 0.85)
        self.assertEqual(dec.evidence_message_ids, "hist_2")

    def test_validation_fallbacks(self):
        raw = {
            "action": "invalid_action",
            "message_type": "invalid_type",
            "reason": "",
            "confidence": "invalid_float",
            "evidence_message_ids": "none"
        }
        dec = self.router._validate_decision(raw, self.dummy_context)
        self.assertEqual(dec.action, "digest")  # FALLBACK_ACTION
        self.assertEqual(dec.message_type, "unknown")  # FALLBACK_MESSAGE_TYPE
        self.assertEqual(dec.confidence, 0.30)  # FALLBACK_CONFIDENCE
        # Fall back to context's evidence
        self.assertEqual(dec.evidence_message_ids, "hist_1")


if __name__ == "__main__":
    unittest.main()
