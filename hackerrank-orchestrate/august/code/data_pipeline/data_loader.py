import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


@dataclass
class DataBundle:
    messages: pd.DataFrame
    sample_messages: pd.DataFrame
    users: pd.DataFrame
    groups: pd.DataFrame
    group_members: pd.DataFrame
    business_accounts: pd.DataFrame
    user_business_history: pd.DataFrame
    message_history: pd.DataFrame
    message_events: pd.DataFrame
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    daily_notification_summary: pd.DataFrame
    output_template: pd.DataFrame

    # Lookup dictionaries for fast indexing
    user_map: Dict[str, dict]
    group_map: Dict[str, dict]
    business_map: Dict[str, dict]
    group_members_by_user: Dict[str, Dict[str, dict]]  # user_id -> {group_id: dict}
    business_history_by_user: Dict[str, Dict[str, dict]]  # user_id -> {business_id: dict}
    image_path_map: Dict[str, str]  # image_id -> file_path
    voice_note_path_map: Dict[str, str]  # voice_note_id -> file_path


class DataLoader:
    """Loads and validates all dataset CSV files using fast dict iteration."""

    def __init__(self, dataset_dir: str = "dataset"):
        path = Path(dataset_dir)
        if not path.exists():
            path = Path("..") / dataset_dir
        self.dataset_dir = path

    def load_all(self) -> DataBundle:
        """Reads all 13 CSV files, converts types, and builds index maps."""
        messages = pd.read_csv(self.dataset_dir / "messages.csv")
        sample_messages = pd.read_csv(self.dataset_dir / "sample_messages.csv")
        users = pd.read_csv(self.dataset_dir / "users.csv")
        groups = pd.read_csv(self.dataset_dir / "groups.csv")
        group_members = pd.read_csv(self.dataset_dir / "group_members.csv")
        business_accounts = pd.read_csv(self.dataset_dir / "business_accounts.csv")
        user_business_history = pd.read_csv(self.dataset_dir / "user_business_history.csv")
        message_history = pd.read_csv(self.dataset_dir / "message_history.csv")
        message_events = pd.read_csv(self.dataset_dir / "message_events.csv")
        images = pd.read_csv(self.dataset_dir / "images.csv")
        voice_notes = pd.read_csv(self.dataset_dir / "voice_notes.csv")
        daily_summary = pd.read_csv(self.dataset_dir / "daily_notification_summary.csv")
        output_template = pd.read_csv(self.dataset_dir / "output.csv")

        # Process timestamps
        messages["created_at_dt"] = pd.to_datetime(messages["created_at"])
        message_history["created_at_dt"] = pd.to_datetime(message_history["created_at"])

        # Fill NaNs for string columns to avoid None errors
        for df in [messages, sample_messages, message_history]:
            df["message_text"] = df["message_text"].fillna("")
            df["group_id"] = df["group_id"].fillna("")
            df["business_id"] = df["business_id"].fillna("")
            df["sender_user_id"] = df["sender_user_id"].fillna("")
            df["media_type"] = df["media_type"].fillna("")
            df["media_id"] = df["media_id"].fillna("")

        # Fast dict iteration (~40x faster than .iterrows())
        user_map = {row["user_id"]: row for row in users.to_dict("records")}
        group_map = {row["group_id"]: row for row in groups.to_dict("records")}
        business_map = {row["business_id"]: row for row in business_accounts.to_dict("records")}

        # Build nested relationship maps
        group_members_by_user: Dict[str, Dict[str, dict]] = {}
        for row in group_members.to_dict("records"):
            uid = row["user_id"]
            gid = row["group_id"]
            if uid not in group_members_by_user:
                group_members_by_user[uid] = {}
            group_members_by_user[uid][gid] = row

        business_history_by_user: Dict[str, Dict[str, dict]] = {}
        for row in user_business_history.to_dict("records"):
            uid = row["user_id"]
            bid = row["business_id"]
            if uid not in business_history_by_user:
                business_history_by_user[uid] = {}
            business_history_by_user[uid][bid] = row

        image_path_map = {row["image_id"]: row["file_path"] for row in images.to_dict("records")}
        voice_note_path_map = {row["voice_note_id"]: row["file_path"] for row in voice_notes.to_dict("records")}

        return DataBundle(
            messages=messages,
            sample_messages=sample_messages,
            users=users,
            groups=groups,
            group_members=group_members,
            business_accounts=business_accounts,
            user_business_history=user_business_history,
            message_history=message_history,
            message_events=message_events,
            images=images,
            voice_notes=voice_notes,
            daily_notification_summary=daily_summary,
            output_template=output_template,
            user_map=user_map,
            group_map=group_map,
            business_map=business_map,
            group_members_by_user=group_members_by_user,
            business_history_by_user=business_history_by_user,
            image_path_map=image_path_map,
            voice_note_path_map=voice_note_path_map,
        )
