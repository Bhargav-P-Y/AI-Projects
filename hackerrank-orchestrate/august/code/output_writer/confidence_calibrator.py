import logging
from typing import Dict, List, Optional

from llm_router.llm_router import RoutingDecision
from safety_filter.safety_filter import RiskSignals
from data_pipeline.profile_builder import UserProfile

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.30
MAX_CONFIDENCE = 0.95


class ConfidenceCalibrator:
    """Post-processor for calibrating LLM routing decision confidence scores.

    Applies empirical domain-signal adjustments based on safety risk signals, business verification,
    group mute settings, active DND state, and historical evidence availability.
    """

    @staticmethod
    def calibrate(
        decision: RoutingDecision,
        risk_signals: Optional[RiskSignals] = None,
        user_profile: Optional[UserProfile] = None,
        business_info: Optional[dict] = None,
        has_dnd_active: bool = False,
        group_info: Optional[dict] = None,
    ) -> RoutingDecision:
        """Calibrates confidence score for a single RoutingDecision."""
        conf = decision.confidence

        # 1. Boost: High Scam Risk for Mute decisions
        if decision.action == "mute" and risk_signals and risk_signals.computed_scam_risk_score >= 0.70:
            conf += 0.05

        # 2. Boost: Verified Business with matching domain
        if business_info and (business_info.get("verified") == 1 or business_info.get("verified") is True):
            official_domain = str(business_info.get("official_domain", "")).strip().lower()
            used_domain = str(business_info.get("domain_used_by_sender", "")).strip().lower()
            if official_domain and used_domain and official_domain == used_domain:
                conf += 0.03

        # 3. Boost: User with high historical open rate (>80%) on notify actions
        open_rate = getattr(user_profile, "open_rate_30d", getattr(user_profile, "global_open_rate", 0.0)) if user_profile else 0.0
        if decision.action == "notify" and open_rate >= 0.80:
            conf += 0.03

        # 4. Penalty: Notify action on a group muted by the user
        if decision.action == "notify" and group_info and group_info.get("group_muted_by_user"):
            conf -= 0.10

        # 5. Penalty: Notify action during active DND hours
        if decision.action == "notify" and has_dnd_active:
            conf -= 0.08

        # 6. Penalty: User with high notification fatigue (>65% dismiss rate) on notify actions
        dismiss_rate = getattr(user_profile, "dismiss_rate_30d", getattr(user_profile, "global_dismiss_rate", 0.0)) if user_profile else 0.0
        if decision.action == "notify" and dismiss_rate >= 0.65:
            conf -= 0.05

        # 7. Penalty: Missing historical evidence
        if decision.evidence_message_ids.strip().lower() == "none":
            conf -= 0.05

        # 8. Strict Clamping
        calibrated_conf = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, round(conf, 2)))

        return RoutingDecision(
            message_id=decision.message_id,
            action=decision.action,
            message_type=decision.message_type,
            reason=decision.reason,
            confidence=calibrated_conf,
            evidence_message_ids=decision.evidence_message_ids,
            source=decision.source,
        )

    @classmethod
    def calibrate_all(
        cls,
        decisions: Dict[str, RoutingDecision],
        risk_signals_map: Optional[Dict[str, RiskSignals]] = None,
        user_profiles: Optional[Dict[str, UserProfile]] = None,
        data_bundle: Optional[object] = None,
        contexts_map: Optional[Dict[str, object]] = None,
    ) -> Dict[str, RoutingDecision]:
        """Calibrates confidence scores for all routing decisions in batch."""
        calibrated_decisions: Dict[str, RoutingDecision] = {}

        # Pre-index messages if data_bundle is provided
        msg_map = {}
        if data_bundle and hasattr(data_bundle, "messages"):
            msg_map = {m["message_id"]: m for m in data_bundle.messages.to_dict("records")}

        for msg_id, dec in decisions.items():
            risk_sig = risk_signals_map.get(msg_id) if risk_signals_map else None
            ctx = contexts_map.get(msg_id) if contexts_map else None

            user_profile = None
            business_info = None
            group_info = None
            has_dnd = False

            if ctx:
                user_profile = getattr(ctx, "user_profile", None)
                has_dnd = getattr(ctx, "has_dnd_active", False)

            # Lookup via data_bundle if provided
            msg_row = msg_map.get(msg_id)
            if msg_row:
                user_id = msg_row.get("user_id")
                group_id = msg_row.get("group_id")
                business_id = msg_row.get("business_id")

                if not user_profile and user_profiles and user_id:
                    user_profile = user_profiles.get(user_id)

                if data_bundle and hasattr(data_bundle, "business_map") and business_id:
                    business_info = data_bundle.business_map.get(business_id)

                if user_profile and group_id and hasattr(user_profile, "group_memberships"):
                    group_info = user_profile.group_memberships.get(group_id)

            calibrated_decisions[msg_id] = cls.calibrate(
                decision=dec,
                risk_signals=risk_sig,
                user_profile=user_profile,
                business_info=business_info,
                has_dnd_active=has_dnd,
                group_info=group_info,
            )

        return calibrated_decisions
