import os
import re
from pathlib import Path
from typing import List


def load_env_file(env_path: str = ".env") -> None:
    """Parses a local .env file into os.environ."""
    path = Path(env_path)
    if not path.is_file():
        # Try looking up one level if called from a subpackage
        path = Path("..") / env_path
        if not path.is_file():
            return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match KEY=VALUE where VALUE can span multiple lines enclosed in quotes
    pattern = re.compile(
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\n#]+))',
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(content):
        key = match.group(1)
        val = match.group(2) or match.group(3) or match.group(4) or ""
        os.environ[key] = val.strip()


def get_api_keys() -> List[str]:
    """Returns a list of clean Gemini API keys from environment."""
    load_env_file()
    raw = os.environ.get("GEMINI_API_KEYS", "")
    if not raw:
        raw = os.environ.get("GEMINI_API_KEY", "")

    # Replace newlines, carriage returns, and split by comma
    cleaned_raw = raw.replace("\n", ",").replace("\r", ",")
    keys = [k.strip().strip('"').strip("'") for k in cleaned_raw.split(",") if k.strip()]
    return keys
