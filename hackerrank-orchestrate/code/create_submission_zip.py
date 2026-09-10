"""Submission Packaging Script.

Zips the code/ directory into code.zip excluding cache, __pycache__,
virtualenvs, build artifacts, dataset/, and data/.
"""

import os
import zipfile
from pathlib import Path

repo_root = Path(__file__).parent.parent
code_dir = repo_root / "code"
zip_output = repo_root / "code.zip"

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", "dataset", "data", "cache"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".zip"}

print(f"Creating submission zip: {zip_output}")
with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            full_path = Path(root) / file
            arcname = full_path.relative_to(code_dir)
            zf.write(full_path, arcname)
            print(f"  + Added: {arcname}")

print(f"\n[SUCCESS] Successfully packaged submission zip to '{zip_output}'!")
