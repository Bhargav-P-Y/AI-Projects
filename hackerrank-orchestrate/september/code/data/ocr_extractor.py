# code/data/ocr_extractor.py
"""
Multimodal OCR Extractor using direct HTTP REST API calls to Google Gemini API.
Standardized on gemini-3.8-flash with exponential backoff retry logic (max 3 retries)
and round-robin distribution across the configured API key pool.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import requests

from code.config import config

logger = logging.getLogger(__name__)


class GeminiOCRExtractor:
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        cache_path: Optional[Path] = None,
        images_csv_path: Optional[Path] = None,
        media_dir: Optional[Path] = None,
    ):
        self.api_keys = api_keys or config.gemini_api_keys
        self.key_index = 0
        # Standardized strictly on gemini-3.8-flash
        self.model_name = model_name or config.gemini_model_name or "gemini-3.8-flash"
        self.cache_path = cache_path or config.ocr_cache_file
        self.images_csv_path = images_csv_path or (config.dataset_dir / "images.csv")
        self.media_dir = media_dir or config.media_images_dir

    def _get_next_api_key(self) -> str:
        if not self.api_keys:
            raise ValueError(
                "No GEMINI_API_KEY found in environment or .env. Please configure API keys."
            )
        key = self.api_keys[self.key_index % len(self.api_keys)]
        self.key_index += 1
        return key

    def extract_and_cache_all(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Processes all 16 images in images.csv. If cached already, loads and returns cache.
        Otherwise, calls Gemini REST API for each image and saves to cache_path.
        """
        if not force_refresh and self.cache_path.exists():
            logger.info(f"Loading existing OCR cache from {self.cache_path}")
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read existing cache, will re-extract: {e}")

        if not self.images_csv_path.exists():
            raise FileNotFoundError(f"images.csv not found at {self.images_csv_path}")

        images_df = pd.read_csv(self.images_csv_path, keep_default_na=False)
        events_df = pd.read_csv(config.dataset_dir / "financial_events.csv", keep_default_na=False)
        events_map = {r.event_id: r for r in events_df.itertuples(index=False)}

        cache_data: Dict[str, Any] = {}

        for r in images_df.itertuples(index=False):
            img_id = str(r.image_id).strip()
            event_id = str(r.related_event_id).strip()
            user_id = str(r.user_id).strip()
            req_id = str(r.request_id).strip()

            ev_info = events_map.get(event_id)
            ev_desc = ev_info.description if ev_info else ""
            ev_cat = ev_info.category if ev_info else ""
            ev_curr = ev_info.currency if ev_info else ""

            img_file = self.media_dir / f"{img_id}.png"
            if not img_file.exists():
                logger.error(f"Image file {img_file} does not exist!")
                continue

            logger.info(f"Extracting amount from {img_id} for {event_id} ({ev_desc}) using {self.model_name}...")
            extracted = self._extract_single_image(img_file, event_id, ev_desc, ev_cat, ev_curr)
            cache_data[img_id] = {
                "image_id": img_id,
                "event_id": event_id,
                "user_id": user_id,
                "request_id": req_id,
                "description": ev_desc,
                "amount": extracted.get("amount", 0.0),
                "currency": extracted.get("currency", ev_curr),
                "confidence": extracted.get("confidence", "high"),
                "notes": extracted.get("notes", ""),
            }

        # Write cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Saved {len(cache_data)} extracted image amounts to {self.cache_path}")
        return cache_data

    def _extract_single_image(
        self,
        image_path: Path,
        event_id: str,
        description: str,
        category: str,
        expected_currency: str,
    ) -> Dict[str, Any]:
        """Direct HTTP REST call to Google Gemini endpoint with exponential backoff (max 3 retries)."""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""You are a precise financial OCR auditor.
Analyze this receipt/invoice/payslip image for financial event '{event_id}'.
Context:
- Event Description: {description}
- Category: {category}
- Expected Currency: {expected_currency}

Instructions:
1. Extract the EXACT final payable amount (or net salary / total invoice balance due).
2. If this is a payslip, extract the Net Salary (take-home pay), not the gross earnings.
3. If this is an invoice/bill/receipt, extract the Grand Total / Total Amount Due.
4. Extract the exact currency (e.g., IDR, ZAR, EUR, USD, INR).

Return ONLY valid JSON matching this schema:
{{
  "amount": <numeric float without commas>,
  "currency": "<3-letter uppercase currency code>",
  "confidence": "<high|medium|low>",
  "notes": "<short explanation of what line item was extracted>"
}}
"""

        max_retries = 5
        backoff_seconds = 3.0
        last_error = None

        for attempt in range(max_retries):
            key = self._get_next_api_key()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": img_b64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            }

            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 200:
                    resp_json = response.json()
                    text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text)
                    return {
                        "amount": float(parsed.get("amount", 0.0)),
                        "currency": str(parsed.get("currency", expected_currency)).strip().upper(),
                        "confidence": str(parsed.get("confidence", "high")),
                        "notes": str(parsed.get("notes", "")),
                    }
                elif response.status_code in [429, 500, 503]:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    wait_time = backoff_seconds * (1.5 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{max_retries} on {self.model_name} got {response.status_code}. Retrying in {wait_time:.1f}s with rotated key...")
                    time.sleep(wait_time)
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Unexpected status {response.status_code}: {response.text}")
                    time.sleep(backoff_seconds)
            except Exception as e:
                last_error = str(e)
                wait_time = backoff_seconds * (1.5 ** attempt)
                logger.warning(f"Attempt {attempt+1}/{max_retries} network error: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"All {max_retries} retries failed for {self.model_name} on image {image_path.name}. Last error: {last_error}")
