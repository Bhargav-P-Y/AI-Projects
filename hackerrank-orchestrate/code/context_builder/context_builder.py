import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from data_pipeline.data_loader import DataBundle
from data_pipeline.profile_builder import UserProfile
from safety_filter.safety_filter import RiskSignals
from .hybrid_retriever import EvidenceCandidate, HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class AssembledContext:
    message_id: str
    user_id: str
    prompt_text: str
    evidence_message_ids_str: str
    risk_signals: RiskSignals
    has_dnd_active: bool


class ContextBuilder:
    """Assembles structured, XML-sandboxed per-message context for the Gemini 3.6 Flash LLM router.

    Merges:
    - Message metadata & raw content (sandboxed in <untrusted_user_message>)
    - Multimodal content (OCR / ASR sandboxed in <untrusted_media_content>)
    - Receiver UserProfile (DND state relative to timestamp, engagement metrics, daily load)
    - Relationship & Trust context (Business verification, domain match, group mute, opt-in/opt-out)
    - Safety RiskSignals (OTP, urgency, URL, chain forwards, composite scam score)
    - Retrieved Historical Evidence (candidates + user reaction history)
    """

    def __init__(
        self,
        data_bundle: DataBundle,
        profiles: Dict[str, UserProfile],
        media_cache: Optional[Dict[str, str]] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.data_bundle = data_bundle
        self.profiles = profiles
        self.media_cache = media_cache or {}
        self.retriever = retriever

    def build(
        self,
        message: dict,
        risk_signals: Optional[RiskSignals] = None,
    ) -> AssembledContext:
        msg_id = message["message_id"]
        user_id = message["user_id"]
        conv_type = message.get("conversation_type", "")
        msg_text = message.get("message_text", "") or ""
        media_type = message.get("media_type", "") or ""
        media_id = message.get("media_id", "") or ""
        forwarded_count = int(message.get("forwarded_count", 0) or 0)
        group_id = message.get("group_id", "") or ""
        business_id = message.get("business_id", "") or ""
        sender_user_id = message.get("sender_user_id", "") or ""

        # Parse message timestamp
        msg_dt = message.get("created_at_dt")
        if not isinstance(msg_dt, pd.Timestamp) and not isinstance(msg_dt, datetime):
            try:
                msg_dt = pd.to_datetime(message.get("created_at", ""))
            except Exception:
                msg_dt = datetime.now()
        created_at_str = str(message.get("created_at", ""))

        # 1. Receiver User Profile Context
        profile = self.profiles.get(user_id)
        if not profile:
            # Fallback empty profile
            profile = UserProfile(
                user_id=user_id,
                dnd_window_str="",
                has_dnd=False,
                dnd_start_hour=None,
                dnd_start_minute=None,
                dnd_end_hour=None,
                dnd_end_minute=None,
                messages_opened_30d=0,
                messages_replied_30d=0,
                notifications_dismissed_30d=0,
                messages_reported_30d=0,
                open_rate_30d=0.5,
                reply_rate_30d=0.1,
                dismiss_rate_30d=0.3,
            )

        is_dnd_active = profile.is_in_dnd(msg_dt)

        # 2. Multimodal extracted content (OCR / ASR)
        media_content_text = ""
        if media_id and media_id in self.media_cache:
            raw_media = self.media_cache[media_id]
            if isinstance(raw_media, dict):
                media_content_text = raw_media.get("extracted_text", "") or raw_media.get("summary", "") or str(raw_media)
            else:
                media_content_text = str(raw_media)

        # 3. Retrieve Historical Evidence
        evidence_ids_str = "none"
        evidence_candidates: List[EvidenceCandidate] = []
        if self.retriever:
            evidence_ids_str, evidence_candidates = self.retriever.retrieve(
                message=message,
                user_profile=profile,
                top_k=3,
                min_score_threshold=0.35,
                media_text=media_content_text,
            )

        # Default RiskSignals if not passed
        if risk_signals is None:
            risk_signals = RiskSignals()

        # Build prompt sections
        sections: List[str] = []

        # SECTION 1: Header & Message Metadata
        sections.append(f"### MESSAGE HEADER: {msg_id}")
        sections.append(f"- Recipient User ID: {user_id}")
        sections.append(f"- Conversation Type: {conv_type}")
        sections.append(f"- Message Timestamp: {created_at_str}")
        sections.append(f"- Forwarded Count: {forwarded_count}")

        if conv_type == "business":
            sections.append(f"- Sender Business ID: {business_id}")
            biz_info = self.data_bundle.business_map.get(business_id, {})
            if biz_info:
                sections.append(f"  * Brand Name: {biz_info.get('brand_name', '')}")
                sections.append(f"  * Category: {biz_info.get('category', '')}")
                sections.append(f"  * Account Verified: {bool(biz_info.get('verified', False))}")
                sections.append(f"  * Official Domain: {biz_info.get('official_domain', 'N/A')}")
                sections.append(f"  * Sender Domain: {biz_info.get('domain_used_by_sender', 'N/A')}")
        elif conv_type == "group":
            sections.append(f"- Group ID: {group_id}")
            sections.append(f"- Sender User ID: {sender_user_id}")
            grp_info = self.data_bundle.group_map.get(group_id, {})
            if grp_info:
                sections.append(f"  * Group Name: {grp_info.get('group_name', '')}")
                sections.append(f"  * Group Type: {grp_info.get('group_type', '')}")
                sections.append(f"  * Member Count: {grp_info.get('member_count', 0)}")
        else:
            sections.append(f"- Sender User ID: {sender_user_id}")

        # SECTION 2: Receiver Profile & Behavior
        sections.append("\n### RECEIVER PROFILE & CONTEXT")
        sections.append(f"- DND Window Configured: '{profile.dnd_window_str}'")
        sections.append(f"- DND Active at Message Time: {'YES (Quiet Hours)' if is_dnd_active else 'NO'}")
        sections.append(
            f"- 30-Day Activity: {profile.messages_opened_30d} opened, {profile.messages_replied_30d} replied, "
            f"{profile.notifications_dismissed_30d} dismissed, {profile.messages_reported_30d} reported"
        )
        sections.append(
            f"- Engagement Ratios: Open Rate {profile.open_rate_30d:.0%}, Reply Rate {profile.reply_rate_30d:.0%}, "
            f"Dismiss Rate {profile.dismiss_rate_30d:.0%}"
        )
        sections.append(
            f"- Daily Notification Load: avg {profile.avg_daily_notifications:.1f} msgs/day (dismiss rate {profile.avg_daily_dismiss_rate:.0%})"
        )

        # SECTION 3: Relationship Specific Context
        sections.append("\n### RELATIONSHIP & TRUST CONTEXT")
        if conv_type == "business" and business_id:
            biz_rel = profile.business_relationships.get(business_id, {})
            if biz_rel:
                sections.append(f"- History with Business: '{biz_rel.get('why_user_knows_account', 'none')}'")
                sections.append(f"- Allows Marketing Promotions: {bool(biz_rel.get('allows_promotions', True))}")
                sections.append(
                    f"- 30-Day Opens/Dismisses/Replies: {biz_rel.get('messages_opened_30d', 0)} opened, "
                    f"{biz_rel.get('messages_dismissed_30d', 0)} dismissed, {biz_rel.get('messages_replied_30d', 0)} replied"
                )
            else:
                sections.append("- History with Business: NO PRIOR RELATIONSHIP RECORDED")
        elif conv_type == "group" and group_id:
            grp_mem = profile.group_memberships.get(group_id, {})
            if grp_mem:
                sections.append(f"- User Group Role: {grp_mem.get('role', 'member')}")
                sections.append(f"- Group Muted by User: {'YES' if grp_mem.get('group_muted_by_user') else 'NO'}")
                sections.append(
                    f"- Group Engagement (30d): {grp_mem.get('messages_read_30d', 0)} read, "
                    f"{grp_mem.get('replies_sent_30d', 0)} replies, {grp_mem.get('notifications_dismissed_30d', 0)} dismissed"
                )
            else:
                sections.append("- User Group Role: member (default)")
        else:
            sections.append("- Personal Direct Message context active")

        # SECTION 4: Pre-Computed Risk Signals
        sections.append("\n### PRE-COMPUTED RISK SIGNALS")
        sections.append(f"- OTP Pattern Detected: {risk_signals.has_otp_pattern}")
        sections.append(f"- Urgency Threat Pressure: {risk_signals.has_urgency_pressure}")
        sections.append(f"- Suspicious URL Present: {risk_signals.has_suspicious_url}")
        if risk_signals.suspicious_url_matches:
            sections.append(f"  * Flagged Domains: {', '.join(risk_signals.suspicious_url_matches)}")
        sections.append(f"- Chain Forward Pattern: {risk_signals.has_chain_forward_pattern}")
        sections.append(f"- Composite Scam Risk Score: {risk_signals.computed_scam_risk_score:.2f} / 1.00")

        # SECTION 5: Multimodal Content (ASR / OCR)
        if media_type and isinstance(media_type, str) and media_type.strip():
            sections.append(f"\n### MULTIMODAL MEDIA ATTACHMENT ({media_type.upper()}: {media_id})")
            if media_content_text:
                sections.append("<untrusted_media_content>")
                sections.append(media_content_text.strip())
                sections.append("</untrusted_media_content>")
            else:
                sections.append("[Media file present but no text extracted]")

        # SECTION 6: Untrusted Message Content (XML Sandboxed)
        sections.append("\n### INCOMING MESSAGE CONTENT")
        sections.append("<untrusted_user_message>")
        sections.append(msg_text.strip() if msg_text and isinstance(msg_text, str) else "[No text content]")
        sections.append("</untrusted_user_message>")

        # SECTION 7: Historical Evidence Retrieved
        sections.append("\n### HISTORICAL EVIDENCE RETRIEVED")
        sections.append(f"- Evidence Message IDs: {evidence_ids_str}")
        if evidence_candidates:
            for idx, cand in enumerate(evidence_candidates, 1):
                sections.append(
                    f"  {idx}. [{cand.message_id}] (Score: {cand.score:.2f}, Reaction: {cand.user_action.upper()})"
                )
                sections.append(f"     Text: \"{cand.message_text}\"")
        else:
            sections.append("  No historical messages met the relevance threshold (evidence_message_ids = none).")

        full_prompt_text = "\n".join(sections)

        return AssembledContext(
            message_id=msg_id,
            user_id=user_id,
            prompt_text=full_prompt_text,
            evidence_message_ids_str=evidence_ids_str,
            risk_signals=risk_signals,
            has_dnd_active=is_dnd_active,
        )
