import logging
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from data_pipeline.config import load_env_file
load_env_file(str(code_dir / ".env"))

from data_pipeline.data_loader import DataLoader
from data_pipeline.profile_builder import UserProfile
from llm_router.llm_router import RoutingDecision
from safety_filter.safety_filter import RiskSignals
# pyrefly: ignore [missing-import]
from output_writer.confidence_calibrator import ConfidenceCalibrator
# pyrefly: ignore [missing-import]
from output_writer.output_writer import OutputWriter, REQUIRED_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestPhase6")


class TestConfidenceCalibrator(unittest.TestCase):
    def setUp(self):
        self.base_dec = RoutingDecision(
            message_id="msg_001",
            action="notify",
            message_type="urgent",
            reason="Test reason",
            confidence=0.80,
            evidence_message_ids="hist_101",
        )

    def test_scam_risk_boost(self):
        dec = RoutingDecision(
            message_id="msg_002",
            action="mute",
            message_type="scam",
            reason="Scam attempt",
            confidence=0.85,
            evidence_message_ids="hist_102",
        )
        risk = RiskSignals(computed_scam_risk_score=0.85)
        calibrated = ConfidenceCalibrator.calibrate(dec, risk_signals=risk)
        self.assertEqual(calibrated.confidence, 0.90)

    def test_verified_business_boost(self):
        biz_info = {
            "verified": 1,
            "official_domain": "hdfcbank.com",
            "domain_used_by_sender": "hdfcbank.com",
        }
        calibrated = ConfidenceCalibrator.calibrate(self.base_dec, business_info=biz_info)
        self.assertEqual(calibrated.confidence, 0.83)

    def test_muted_group_penalty(self):
        grp_info = {"group_muted_by_user": True}
        calibrated = ConfidenceCalibrator.calibrate(self.base_dec, group_info=grp_info)
        self.assertEqual(calibrated.confidence, 0.70)

    def test_dnd_penalty(self):
        calibrated = ConfidenceCalibrator.calibrate(self.base_dec, has_dnd_active=True)
        self.assertEqual(calibrated.confidence, 0.72)

    def test_user_profile_open_rate_boost(self):
        prof = UserProfile(
            user_id="u_001", dnd_window_str="", has_dnd=False,
            dnd_start_hour=None, dnd_start_minute=None, dnd_end_hour=None, dnd_end_minute=None,
            messages_opened_30d=10, messages_replied_30d=5, notifications_dismissed_30d=2, messages_reported_30d=0,
            open_rate_30d=0.85, reply_rate_30d=0.5, dismiss_rate_30d=0.2
        )
        calibrated = ConfidenceCalibrator.calibrate(self.base_dec, user_profile=prof)
        self.assertEqual(calibrated.confidence, 0.83)

    def test_user_profile_dismiss_rate_penalty(self):
        prof = UserProfile(
            user_id="u_002", dnd_window_str="", has_dnd=False,
            dnd_start_hour=None, dnd_start_minute=None, dnd_end_hour=None, dnd_end_minute=None,
            messages_opened_30d=2, messages_replied_30d=0, notifications_dismissed_30d=10, messages_reported_30d=0,
            open_rate_30d=0.2, reply_rate_30d=0.0, dismiss_rate_30d=0.70
        )
        calibrated = ConfidenceCalibrator.calibrate(self.base_dec, user_profile=prof)
        self.assertEqual(calibrated.confidence, 0.75)

    def test_missing_evidence_penalty(self):
        dec = RoutingDecision(
            message_id="msg_003",
            action="digest",
            message_type="personal",
            reason="Personal text",
            confidence=0.70,
            evidence_message_ids="none",
        )
        calibrated = ConfidenceCalibrator.calibrate(dec)
        self.assertEqual(calibrated.confidence, 0.65)

    def test_clamping(self):
        # High confidence clamped to 0.95 max
        high_dec = RoutingDecision(
            message_id="msg_004",
            action="mute",
            message_type="scam",
            reason="Scam text",
            confidence=0.98,
            evidence_message_ids="hist_1",
        )
        calibrated_high = ConfidenceCalibrator.calibrate(high_dec)
        self.assertEqual(calibrated_high.confidence, 0.95)

        # Low confidence clamped to 0.30 min
        low_dec = RoutingDecision(
            message_id="msg_005",
            action="notify",
            message_type="urgent",
            reason="Low confidence text",
            confidence=0.20,
            evidence_message_ids="none",
        )
        calibrated_low = ConfidenceCalibrator.calibrate(low_dec)
        self.assertEqual(calibrated_low.confidence, 0.30)


class TestOutputWriter(unittest.TestCase):
    def setUp(self):
        # Load dataset messages for expected_messages DataFrame
        self.data_bundle = DataLoader("dataset").load_all()
        self.expected_messages = self.data_bundle.messages

    def test_valid_decisions_pass(self):
        decisions = {}
        for msg_id in self.expected_messages["message_id"]:
            decisions[msg_id] = RoutingDecision(
                message_id=msg_id,
                action="digest",
                message_type="personal",
                reason="Valid test decision reason.",
                confidence=0.80,
                evidence_message_ids="none",
            )

        errors = OutputWriter.validate_decisions(decisions, self.expected_messages)
        self.assertEqual(len(errors), 0)

    def test_invalid_action_detected(self):
        decisions = {}
        for msg_id in self.expected_messages["message_id"]:
            decisions[msg_id] = RoutingDecision(
                message_id=msg_id,
                action="invalid_action",
                message_type="personal",
                reason="Valid reason.",
                confidence=0.80,
                evidence_message_ids="none",
            )

        errors = OutputWriter.validate_decisions(decisions, self.expected_messages)
        self.assertGreater(len(errors), 0)

    def test_row_count_mismatch_detected(self):
        # Pass empty dict
        errors = OutputWriter.validate_decisions({}, self.expected_messages)
        self.assertGreater(len(errors), 0)

    def test_write_csv_creation(self):
        decisions = {}
        for msg_id in self.expected_messages["message_id"]:
            decisions[msg_id] = RoutingDecision(
                message_id=msg_id,
                action="notify" if msg_id.endswith("1") else "digest",
                message_type="urgent" if msg_id.endswith("1") else "personal",
                reason="Auto-generated test decision.",
                confidence=0.85,
                evidence_message_ids="hist_001" if msg_id.endswith("1") else "none",
            )

        temp_csv_path = code_dir / "cache" / "test_output.csv"
        df = OutputWriter.write_csv(decisions, self.expected_messages, temp_csv_path)

        self.assertTrue(temp_csv_path.exists())
        self.assertEqual(len(df), len(self.expected_messages))
        self.assertEqual(list(df.columns), REQUIRED_COLUMNS)

        # Read back and verify row count & schema
        read_df = pd.read_csv(temp_csv_path)
        self.assertEqual(len(read_df), 110)
        self.assertEqual(list(read_df.columns), REQUIRED_COLUMNS)


def run_phase6_tests():
    logger.info("=== Starting Phase 6: Confidence Calibrator & Output Writer Tests ===")
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n[SUCCESS] PHASE 6 TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n[FAIL] Phase 6 tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_phase6_tests()
