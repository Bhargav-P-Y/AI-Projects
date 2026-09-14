import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from .data_loader import DataBundle

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    user_id: str
    dnd_window_str: str
    has_dnd: bool
    dnd_start_hour: Optional[int]
    dnd_start_minute: Optional[int]
    dnd_end_hour: Optional[int]
    dnd_end_minute: Optional[int]
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int
    open_rate_30d: float
    reply_rate_30d: float
    dismiss_rate_30d: float
    group_memberships: Dict[str, dict] = field(default_factory=dict)
    business_relationships: Dict[str, dict] = field(default_factory=dict)
    avg_daily_notifications: float = 0.0
    avg_daily_dismiss_rate: float = 0.0

    def is_in_dnd(self, dt: datetime) -> bool:
        """Checks if a datetime falls within the user's Do-Not-Disturb window."""
        if not self.has_dnd or self.dnd_start_hour is None or self.dnd_end_hour is None:
            return False

        msg_time = dt.time()
        start = time(self.dnd_start_hour, self.dnd_start_minute or 0)
        end = time(self.dnd_end_hour, self.dnd_end_minute or 0)

        if start <= end:
            # Daytime window e.g. 09:00 to 17:00
            return start <= msg_time <= end
        else:
            # Overnight window e.g. 22:00 to 07:00
            return msg_time >= start or msg_time <= end


class ProfileBuilder:
    """Pre-computes rich personalized user profiles for the notification router."""

    @staticmethod
    def _parse_dnd(dnd_str: Any) -> Optional[Tuple[int, int, int, int]]:
        """Parses DND window string like '22:00-07:00' or '21:30-06:30'.

        Returns None if missing, empty, or invalid, logging explicit warnings
        instead of returning arbitrary default values.
        """
        if pd.isna(dnd_str) or not dnd_str or str(dnd_str).strip().lower() in ("none", "disabled", "off", "nan", ""):
            return None

        clean_str = str(dnd_str).strip()
        try:
            parts = clean_str.split("-")
            if len(parts) != 2:
                logger.warning("Invalid DND window format '%s': expected 'HH:MM-HH:MM'", clean_str)
                return None

            start_parts = [int(x) for x in parts[0].split(":")]
            end_parts = [int(x) for x in parts[1].split(":")]

            sh, sm = (start_parts[0], start_parts[1]) if len(start_parts) >= 2 else (start_parts[0], 0)
            eh, em = (end_parts[0], end_parts[1]) if len(end_parts) >= 2 else (end_parts[0], 0)

            if not (0 <= sh <= 23 and 0 <= sm <= 59 and 0 <= eh <= 23 and 0 <= em <= 59):
                logger.warning("Out of bounds time in DND window '%s'", clean_str)
                return None

            return sh, sm, eh, em
        except Exception as err:
            logger.warning("Failed to parse DND window '%s': %s", clean_str, err)
            return None

    @classmethod
    def build_profile(cls, user_id: str, data: DataBundle) -> UserProfile:
        user_row = data.user_map.get(user_id)
        if user_row is None:
            raise ValueError(f"User ID {user_id} not found in users.csv")

        raw_dnd = user_row.get("do_not_disturb_window", "")
        parsed_dnd = cls._parse_dnd(raw_dnd)

        if parsed_dnd is not None:
            has_dnd = True
            sh, sm, eh, em = parsed_dnd
            dnd_str = str(raw_dnd).strip()
        else:
            has_dnd = False
            sh, sm, eh, em = None, None, None, None
            dnd_str = "None"

        opened = int(user_row.get("messages_opened_30d", 0))
        replied = int(user_row.get("messages_replied_30d", 0))
        dismissed = int(user_row.get("notifications_dismissed_30d", 0))
        reported = int(user_row.get("messages_reported_30d", 0))

        total_interacted = opened + dismissed
        open_rate = opened / max(1, total_interacted)
        reply_rate = replied / max(1, opened)
        dismiss_rate = dismissed / max(1, total_interacted)

        # Build group memberships dict
        memberships: Dict[str, dict] = {}
        user_groups = data.group_members_by_user.get(user_id, {})
        for gid, grow in user_groups.items():
            memberships[gid] = {
                "group_id": gid,
                "role": str(grow.get("role", "member")),
                "joined_at": str(grow.get("joined_at", "")),
                "messages_sent_30d": int(grow.get("messages_sent_30d", 0)),
                "messages_read_30d": int(grow.get("messages_read_30d", 0)),
                "replies_sent_30d": int(grow.get("replies_sent_30d", 0)),
                "notifications_dismissed_30d": int(grow.get("notifications_dismissed_30d", 0)),
                "group_muted_by_user": bool(grow.get("group_muted_by_user", 0)),
            }

        # Build business relationships dict
        biz_rel: Dict[str, dict] = {}
        user_biz = data.business_history_by_user.get(user_id, {})
        for bid, brow in user_biz.items():
            biz_rel[bid] = {
                "business_id": bid,
                "why_user_knows_account": str(brow.get("why_user_knows_account", "")),
                "last_activity_at": str(brow.get("last_activity_at", "")),
                "allows_promotions": bool(brow.get("allows_promotions", 1)),
                "promotions_opted_out_at": str(brow.get("promotions_opted_out_at", "")),
                "activity_count_180d": int(brow.get("activity_count_180d", 0)),
                "messages_opened_30d": int(brow.get("messages_opened_30d", 0)),
                "messages_dismissed_30d": int(brow.get("messages_dismissed_30d", 0)),
                "messages_replied_30d": int(brow.get("messages_replied_30d", 0)),
                "last_reply_at": str(brow.get("last_reply_at", "")),
            }

        # Compute daily summary load
        daily_df = data.daily_notification_summary[
            data.daily_notification_summary["user_id"] == user_id
        ]
        avg_daily_sent = float(daily_df["notifications_sent"].mean()) if not daily_df.empty else 0.0
        avg_daily_dismissed = float(daily_df["notifications_dismissed"].mean()) if not daily_df.empty else 0.0
        daily_dismiss_rate = (avg_daily_dismissed / max(1.0, avg_daily_sent)) if avg_daily_sent > 0 else 0.0

        return UserProfile(
            user_id=user_id,
            dnd_window_str=dnd_str,
            has_dnd=has_dnd,
            dnd_start_hour=sh,
            dnd_start_minute=sm,
            dnd_end_hour=eh,
            dnd_end_minute=em,
            messages_opened_30d=opened,
            messages_replied_30d=replied,
            notifications_dismissed_30d=dismissed,
            messages_reported_30d=reported,
            open_rate_30d=open_rate,
            reply_rate_30d=reply_rate,
            dismiss_rate_30d=dismiss_rate,
            group_memberships=memberships,
            business_relationships=biz_rel,
            avg_daily_notifications=avg_daily_sent,
            avg_daily_dismiss_rate=daily_dismiss_rate,
        )

    @classmethod
    def build_all(cls, data: DataBundle) -> Dict[str, UserProfile]:
        """Pre-computes UserProfile for all users in the dataset."""
        profiles = {}
        for uid in data.user_map:
            profiles[uid] = cls.build_profile(uid, data)
        return profiles
