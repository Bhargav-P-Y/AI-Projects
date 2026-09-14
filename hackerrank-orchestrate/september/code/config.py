# code/config.py
"""
Configuration management for Buy or Wait financial decision system.
Loads settings and environment variables securely from .env.
Supports round-robin API key pooling across multiple keys to prevent rate limits.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv

# Base paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_IMAGES_DIR = DATASET_DIR / "media" / "images"
CODE_DIR = REPO_ROOT / "code"
EVALUATION_DIR = CODE_DIR / "evaluation"

# Target output files
ROOT_OUTPUT_CSV = REPO_ROOT / "output.csv"
USAGE_REPORT_MD = EVALUATION_DIR / "usage_report.md"
OCR_CACHE_FILE = DATASET_DIR / "extracted_image_amounts.json"

# Load .env from repo root
load_dotenv(dotenv_path=REPO_ROOT / ".env")


def _get_api_keys() -> List[str]:
    """Collects all configured Gemini / Google API keys from .env environment."""
    keys: List[str] = []
    # Primary key
    primary = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if primary:
        keys.append(primary.strip())

    # Check numbered keys: GEMINI_API_KEY_1..10, GOOGLE_API_KEY_1..10, API_KEY_1..10
    for prefix in ["GEMINI_API_KEY_", "GOOGLE_API_KEY_", "API_KEY_"]:
        for i in range(1, 20):
            val = os.getenv(f"{prefix}{i}", "").strip()
            if val and val not in keys:
                keys.append(val)

    # Also check if keys are comma-separated in GEMINI_API_KEYS
    multi = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GOOGLE_API_KEYS", "")
    if multi:
        for k in multi.split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

    return keys


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path = REPO_ROOT
    dataset_dir: Path = DATASET_DIR
    media_images_dir: Path = MEDIA_IMAGES_DIR
    ocr_cache_file: Path = OCR_CACHE_FILE
    root_output_csv: Path = ROOT_OUTPUT_CSV
    usage_report_md: Path = USAGE_REPORT_MD
    
    # API Configurations: Standardized to latest gemini-3.8-flash
    gemini_api_keys: List[str] = field(default_factory=_get_api_keys)
    gemini_model_name: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    
    # Financial Engine Defaults
    forecast_days: int = 90
    max_spending_changes: int = 3

    @property
    def gemini_api_key(self) -> str:
        """Returns the first available key or empty string."""
        return self.gemini_api_keys[0] if self.gemini_api_keys else ""


config = AppConfig()
