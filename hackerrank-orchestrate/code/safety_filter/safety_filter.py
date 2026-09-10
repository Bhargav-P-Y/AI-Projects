import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import pandas as pd
from data_pipeline.data_loader import DataBundle
from .semantic_injection import SemanticInjectionDetector


@dataclass
class FastTrackDecision:
    message_id: str
    action: str  # "mute", "digest", "notify"
    message_type: str  # "scam", "spam", "forward", "greeting", "promotion"
    reason: str
    confidence: float
    evidence_message_ids: str = "none"
    filter_layer: str = ""


@dataclass
class RiskSignals:
    """Pre-computed structured risk signals fed to ContextBuilder and LLM Prompt."""

    has_otp_pattern: bool = False
    has_urgency_pressure: bool = False
    has_suspicious_url: bool = False
    has_chain_forward_pattern: bool = False
    forwarded_count: int = 0
    suspicious_url_matches: List[str] = field(default_factory=list)
    computed_scam_risk_score: float = 0.0


class SafetyFilter:
    """Hybrid Safety Engine combining Ultra-High Precision Hard Pre-Filtering with Structured Risk Signal Enrichment.

    - Hard Pre-Filters ONLY indisputable security threats (System Prompt Injections & Domain Identity Spoofing).
    - Computes RiskSignals (OTP patterns, Urgency threats, Suspicious URLs, Forward counts) for all other
      messages and injects them into the LLM context prompt for retrieval-grounded LLM reasoning.
    """

    STRUCTURAL_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+(routing\s+)?rules",
        r"(system|router|internal)\s+(note|metadata|instruction)",
        r"mark\s+(this\s+)?(message\s+)?as\s+notify",
        r"set\s+action\s*=\s*(notify|digest|mute)",
        r"verified_business\s*=\s*true",
        r"user_priority\s*=\s*high",
        r"assistant\s+instruction",
        r"routing\s+override",
        r"system\s+directive",
    ]

    SCAM_OTP_PATTERNS = [
        r"\b(OTP|otp|PIN|pin)\b",
        r"\b(verification|security|one-time|one time)\s+(code|passcode|otp)\b",
        r"\b(share|send|verify|confirm|enter|give)\b.*\b(OTP|otp|code|PIN|password)\b",
        r"\b(do not share|never share)\b",
    ]

    SCAM_URGENCY_PATTERNS = [
        r"account\s+block",
        r"profile\s+will\s+be\s+blocked",
        r"profile\s+band\s+ho\s+jayega",
        r"access\s+card\s+penalty",
        r"access\s+will\s+expire",
        r"immediate\s+action",
        r"immediately",
        r"urgent",
    ]

    SCAM_URL_PATTERNS = [
        r"account-login\.in",
        r"account-help\.in",
        r"pay-check-secure\.com",
        r"chase-secure-alert\.com",
        r"amazonpay-delivery\.in",
        r"bit\.ly/",
        r"lucky-draw-result\.in",
    ]

    FORWARD_PATTERNS = [
        r"forward\s+(this\s+)?to\s+\d+\s+people",
        r"share\s+(this\s+)?(with|in)\s+(all|family|groups|\d+)",
        r"do\s+not\s+(break|ignore)\s+(the\s+)?chain",
        r"bhagwan\s+sabka\s+bhala",
        r"positive\s+energy\s+failao",
        r"share\s+blessings",
    ]

    def __init__(self, data_bundle: DataBundle, semantic_detector: Optional[SemanticInjectionDetector] = None):
        self.data = data_bundle
        self.business_map = data_bundle.business_map
        self.user_map = data_bundle.user_map
        self.semantic_detector = semantic_detector or SemanticInjectionDetector()

    def compute_risk_signals(self, message_row: Dict[str, Any]) -> RiskSignals:
        """Computes rich risk signals to enrich the LLM context prompt."""
        text = str(message_row.get("message_text", ""))
        forwarded_count = int(message_row.get("forwarded_count", 0))

        has_otp = any(re.search(p, text, re.IGNORECASE) for p in self.SCAM_OTP_PATTERNS)
        has_urgency = any(re.search(p, text, re.IGNORECASE) for p in self.SCAM_URGENCY_PATTERNS)
        url_matches = [m.group(0) for p in self.SCAM_URL_PATTERNS for m in [re.search(p, text, re.IGNORECASE)] if m]
        has_chain_fwd = any(re.search(p, text, re.IGNORECASE) for p in self.FORWARD_PATTERNS)

        scam_score = min(1.0, 0.35 * int(has_otp) + 0.35 * int(has_urgency) + 0.40 * int(len(url_matches) > 0))

        return RiskSignals(
            has_otp_pattern=has_otp,
            has_urgency_pressure=has_urgency,
            has_suspicious_url=len(url_matches) > 0,
            has_chain_forward_pattern=has_chain_fwd,
            forwarded_count=forwarded_count,
            suspicious_url_matches=url_matches,
            computed_scam_risk_score=scam_score,
        )

    def check_hard_threats(self, message_row: Dict[str, Any], user_profile: Any = None) -> Optional[FastTrackDecision]:
        """Hard pre-filters ONLY 100% indisputable security threats (Prompt Injections & Domain Identity Spoofing).

        Returns FastTrackDecision for hard security threats. Returns None for all other messages
        so they can be evaluated by the LLM using computed risk signals and historical evidence.
        """
        msg_id = str(message_row["message_id"])
        text = str(message_row.get("message_text", ""))
        business_id = str(message_row.get("business_id", "")).strip()

        # Layer 1: Structural Prompt Injection (Regex Fast-Path)
        for pattern in self.STRUCTURAL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return FastTrackDecision(
                    message_id=msg_id,
                    action="mute",
                    message_type="scam",
                    reason="Message contains an explicit structural prompt injection directive attempting to override router logic.",
                    confidence=0.92,
                    evidence_message_ids="none",
                    filter_layer="Layer1_StructuralPromptInjection",
                )

        # Layer 2: Semantic Prompt Injection (Dense Embedding Similarity)
        is_inj, sim_score = self.semantic_detector.check_injection(text)
        if is_inj:
            return FastTrackDecision(
                message_id=msg_id,
                action="mute",
                message_type="scam",
                reason=f"Semantic embedding similarity ({sim_score:.2f}) matches known prompt injection attack exemplars.",
                confidence=0.90,
                evidence_message_ids="none",
                filter_layer="Layer2_SemanticPromptInjection",
            )

        # Layer 3: Domain Identity Spoofing (Cryptographic/Domain Mismatch)
        if business_id and business_id in self.business_map:
            biz = self.business_map[business_id]
            is_verified = int(biz.get("verified", 0)) == 1
            official_domain = str(biz.get("official_domain", "")).strip().lower()
            used_domain = str(biz.get("domain_used_by_sender", "")).strip().lower()

            if not is_verified and official_domain and used_domain:
                if official_domain != used_domain and ("amazon" in used_domain or "chase" in used_domain or "talabat" in used_domain or "hdfc" in used_domain):
                    return FastTrackDecision(
                        message_id=msg_id,
                        action="mute",
                        message_type="scam",
                        reason=f"Unverified sender spoofing official domain ({used_domain} vs official {official_domain}).",
                        confidence=0.91,
                        evidence_message_ids="none",
                        filter_layer="Layer3_DomainIdentitySpoofing",
                    )

        return None

    def check(self, message_row: Dict[str, Any], user_profile: Any = None) -> Optional[FastTrackDecision]:
        """Alias for check_hard_threats for backward compatibility."""
        return self.check_hard_threats(message_row, user_profile)
