import os
import sys
from datetime import datetime

# Add parent directory ('code') to sys.path so package imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.config import get_api_keys
from data_pipeline.data_loader import DataLoader
from data_pipeline.profile_builder import ProfileBuilder


def run_phase1_tests():
    print("=== Running Data Pipeline Package (Phase 1) Verification Tests ===")

    # 1. Test Config & API Key loader
    keys = get_api_keys()
    print(f"[1] Config API Keys Loaded: {len(keys)} keys found.")
    assert len(keys) >= 1, "Expected at least 1 API key in .env"

    # 2. Test DataLoader
    loader = DataLoader("dataset")
    data = loader.load_all()

    print(f"[2] Data Loading Completed:")
    print(f"    - Messages to route: {len(data.messages)} rows")
    print(f"    - Sample benchmark messages: {len(data.sample_messages)} rows")
    print(f"    - Users: {len(data.users)} users")
    print(f"    - Groups: {len(data.groups)} groups")
    print(f"    - Business accounts: {len(data.business_accounts)} businesses")
    print(f"    - Historical messages: {len(data.message_history)} messages")
    print(f"    - Media images: {len(data.images)} images")
    print(f"    - Media voice notes: {len(data.voice_notes)} voice notes")

    assert len(data.messages) > 0, "messages.csv is empty"
    assert len(data.users) == 54, f"Expected 54 users, got {len(data.users)}"
    assert len(data.images) == 20, f"Expected 20 images, got {len(data.images)}"
    assert len(data.voice_notes) == 13, f"Expected 13 voice notes, got {len(data.voice_notes)}"

    # 3. Test ProfileBuilder & Robust DND Error Handling
    profiles = ProfileBuilder.build_all(data)
    print(f"[3] Built {len(profiles)} UserProfiles successfully.")

    # Test DND window logic for u_001 (DND: 22:00-07:00)
    u1 = profiles["u_001"]
    assert u1.has_dnd == True
    dnd_night = datetime(2026, 7, 30, 23, 15)  # 23:15 -> In DND
    dnd_morning = datetime(2026, 7, 30, 5, 30)  # 05:30 -> In DND
    awake_afternoon = datetime(2026, 7, 30, 14, 30)  # 14:30 -> Awake

    assert u1.is_in_dnd(dnd_night) == True, "u_001 at 23:15 should be in DND"
    assert u1.is_in_dnd(dnd_morning) == True, "u_001 at 05:30 should be in DND"
    assert u1.is_in_dnd(awake_afternoon) == False, "u_001 at 14:30 should NOT be in DND"

    # Test DND window with half-hours: u_003 (DND: 21:30-07:30)
    u3 = profiles["u_003"]
    assert u3.has_dnd == True
    dnd_halfhour = datetime(2026, 7, 30, 21, 45)  # 21:45 -> In DND
    awake_earlynight = datetime(2026, 7, 30, 21, 15)  # 21:15 -> Awake

    assert u3.is_in_dnd(dnd_halfhour) == True, "u_003 at 21:45 should be in DND"
    assert u3.is_in_dnd(awake_earlynight) == False, "u_003 at 21:15 should NOT be in DND"

    # Test explicit DND error / missing cases
    assert ProfileBuilder._parse_dnd(None) is None
    assert ProfileBuilder._parse_dnd("") is None
    assert ProfileBuilder._parse_dnd("disabled") is None
    assert ProfileBuilder._parse_dnd("invalid_string") is None
    assert ProfileBuilder._parse_dnd("25:00-30:00") is None

    # Test relationship mapping
    assert "group_001" in u1.group_memberships, "u_001 should belong to group_001"
    assert u1.group_memberships["group_001"]["group_muted_by_user"] == True, "u_001 has muted group_001"

    print("=== Data Pipeline Package (Phase 1) Verification Passed Cleanly! ===")


if __name__ == "__main__":
    run_phase1_tests()
