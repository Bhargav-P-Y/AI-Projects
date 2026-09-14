# code/data/evidence_resolver.py
"""
Entity-Scoped Evidence & Message Resolver using Gemini 3.8 Flash with Model Fallback and JSON caching.
Extracts structured financial semantics from asynchronous messages:
- Confirmed salary amounts and effective dates
- Contract terminations / household income losses
- Rent changes
- Failed debits still outstanding
- Pending vs settled refunds / commissions / prizes (flags unconfirmed credits to ignore)
- Internal account transfers (prevent double-counting)

Persists output to dataset/extracted_message_evidence.json.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import pandas as pd
import requests

from code.config import config
from code.data.models import MessageItem
from code.data.repository import MessageRepository

logger = logging.getLogger(__name__)

MESSAGE_EVIDENCE_CACHE = config.dataset_dir / "extracted_message_evidence.json"


@dataclass
class ResolvedMessageDTO:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    source_type: str
    is_confirmed_income: bool = False
    confirmed_income_amount: Optional[float] = None
    confirmed_income_currency: Optional[str] = None
    confirmed_income_date: Optional[str] = None
    is_contract_ended: bool = False
    rent_increase_pct: Optional[float] = None
    is_failed_debit_still_owed: bool = False
    is_internal_transfer: bool = False
    is_unconfirmed_credit_to_ignore: bool = False
    summary: str = ""


@dataclass
class ResolvedUserEvidence:
    user_id: str
    confirmed_salary_override: Optional[float] = None
    salary_effective_date: Optional[str] = None
    contract_ended: bool = False
    rent_increase_pct: Optional[float] = None
    failed_debits_still_owed: List[str] = field(default_factory=list)
    ignored_unconfirmed_credits: List[str] = field(default_factory=list)
    messages_summary: List[str] = field(default_factory=list)


class EvidenceResolver:
    def __init__(
        self,
        message_repo: MessageRepository,
        cache_path: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        self.message_repo = message_repo
        self.cache_path = cache_path or MESSAGE_EVIDENCE_CACHE
        self.model_name = model_name or config.gemini_model_name or "gemini-3.8-flash"
        self.api_keys = config.gemini_api_keys
        self.key_index = 0
        self._cache: Dict[str, ResolvedMessageDTO] = {}
        self._load_cache()

    def _get_next_api_key(self) -> str:
        if not self.api_keys:
            raise ValueError("No Gemini API keys found in environment.")
        key = self.api_keys[self.key_index % len(self.api_keys)]
        self.key_index += 1
        return key

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for mid, item in data.items():
                        self._cache[mid] = ResolvedMessageDTO(**item)
            except Exception as e:
                logger.warning(f"Failed to load message evidence cache: {e}")

    def extract_and_cache_all_messages(self, force_refresh: bool = False) -> Dict[str, ResolvedMessageDTO]:
        """
        Parses all 215 messages in dataset/messages.csv with Gemini in optimal batches of 10,
        caching structured semantic output to dataset/extracted_message_evidence.json.
        """
        if not force_refresh and self._cache and len(self._cache) >= 215:
            logger.info(f"Loaded all {len(self._cache)} message evidence items from cache.")
            return self._cache

        messages_df = pd.read_csv(config.dataset_dir / "messages.csv", keep_default_na=False)
        total_msgs = len(messages_df)
        logger.info(f"Extracting semantic evidence from {total_msgs} messages using {self.model_name}...")

        batch_size = 10
        records = messages_df.to_dict("records")

        for i in range(0, total_msgs, batch_size):
            batch = records[i : i + batch_size]
            # Check if all in batch are already cached
            if not force_refresh and all(r["message_id"] in self._cache for r in batch):
                continue

            batch_prompt = """You are a meticulous financial auditor classifying asynchronous evidence messages.
For each message, analyze its financial impact:
1. Is it a confirmed regular base salary / recurring income update with an exact amount and effective date?
2. Is it an unconfirmed credit (quarterly bonus pending review, commission awaiting deal close, prize in verification, pending refund) that MUST NOT be counted as cash yet?
3. Has employment / seasonal contract ended (no more salary)?
4. Is it a failed debit that is still owed / outstanding?
5. Is it an internal transfer between user's own accounts (not new external income)?
6. Is it a rent increase percentage?

Classify each message in the batch and return a JSON list matching:
[
  {
    "message_id": "<id>",
    "is_confirmed_income": <bool>,
    "confirmed_income_amount": <float or null>,
    "confirmed_income_currency": "<EUR|IDR|INR|USD|ZAR or null>",
    "confirmed_income_date": "<YYYY-MM-DD or null>",
    "is_contract_ended": <bool>,
    "rent_increase_pct": <float or null>,
    "is_failed_debit_still_owed": <bool>,
    "is_internal_transfer": <bool>,
    "is_unconfirmed_credit_to_ignore": <bool>,
    "summary": "<1 concise sentence explanation>"
  }
]

Messages to analyze:
"""
            for m in batch:
                batch_prompt += f"\n--- ID: {m['message_id']} | Source: {m['source_type']} | User: {m['user_id']} ---\n{m['message_text']}\n"

            extracted_items = self._call_gemini_json(batch_prompt)
            for item in extracted_items:
                mid = item.get("message_id")
                orig = next((r for r in batch if r["message_id"] == mid), None)
                if orig:
                    dto = ResolvedMessageDTO(
                        message_id=mid,
                        user_id=orig["user_id"],
                        request_id=orig["request_id"] or None,
                        related_event_id=orig["related_event_id"] or None,
                        source_type=orig["source_type"],
                        is_confirmed_income=bool(item.get("is_confirmed_income", False)),
                        confirmed_income_amount=float(item["confirmed_income_amount"]) if item.get("confirmed_income_amount") else None,
                        confirmed_income_currency=item.get("confirmed_income_currency"),
                        confirmed_income_date=item.get("confirmed_income_date"),
                        is_contract_ended=bool(item.get("is_contract_ended", False)),
                        rent_increase_pct=float(item["rent_increase_pct"]) if item.get("rent_increase_pct") else None,
                        is_failed_debit_still_owed=bool(item.get("is_failed_debit_still_owed", False)),
                        is_internal_transfer=bool(item.get("is_internal_transfer", False)),
                        is_unconfirmed_credit_to_ignore=bool(item.get("is_unconfirmed_credit_to_ignore", False)),
                        summary=item.get("summary", ""),
                    )
                    self._cache[mid] = dto

            # Persist incrementally
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self._cache.items()}, f, indent=2)

        logger.info(f"Saved {len(self._cache)} resolved message evidence records to {self.cache_path}")
        return self._cache

    def _call_gemini_json(self, prompt: str) -> List[Dict[str, Any]]:
        # Models to try in order (falls back to stable 3.7 / 3.6 flash if 3.8 has temporary spike)
        models_to_try = [self.model_name, "gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest"]
        seen = set()
        models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        max_retries = 8
        backoff_seconds = 2.0
        last_error = None

        for attempt in range(max_retries):
            key = self._get_next_api_key()
            model = models_to_try[attempt % len(models_to_try)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            }

            try:
                response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
                if response.status_code == 200:
                    resp_json = response.json()
                    text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "messages" in parsed:
                        return parsed["messages"]
                    return [parsed]
                elif response.status_code in [429, 500, 503]:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    wait_time = backoff_seconds * (1.3 ** attempt)
                    logger.warning(f"Batch attempt {attempt+1}/{max_retries} on {model} got {response.status_code}. Retrying in {wait_time:.1f}s with rotated key...")
                    time.sleep(wait_time)
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    time.sleep(backoff_seconds)
            except Exception as e:
                last_error = str(e)
                wait_time = backoff_seconds * (1.3 ** attempt)
                logger.warning(f"Batch attempt {attempt+1}/{max_retries} error: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to process message batch after {max_retries} attempts. Last error: {last_error}")

    def resolve_evidence_for_user(self, user_id: str, request_date: str) -> ResolvedUserEvidence:
        """Retrieves and merges cached evidence for a specific user."""
        user_msgs = self.message_repo.get_messages_for_user(user_id)
        evidence = ResolvedUserEvidence(user_id=user_id)

        for m in user_msgs:
            cached = self._cache.get(m.message_id)
            if not cached:
                continue

            if cached.is_confirmed_income and cached.confirmed_income_amount:
                evidence.confirmed_salary_override = cached.confirmed_income_amount
                if cached.confirmed_income_date:
                    evidence.salary_effective_date = cached.confirmed_income_date

            if cached.is_contract_ended:
                evidence.contract_ended = True

            if cached.rent_increase_pct:
                evidence.rent_increase_pct = cached.rent_increase_pct

            if cached.is_failed_debit_still_owed and cached.related_event_id:
                evidence.failed_debits_still_owed.append(cached.related_event_id)

            if cached.is_unconfirmed_credit_to_ignore and cached.related_event_id:
                evidence.ignored_unconfirmed_credits.append(cached.related_event_id)

            if cached.summary:
                evidence.messages_summary.append(f"[{cached.source_type}] {cached.summary}")

        # Deterministic fallback: extract salary amount from employer messages if missed
        if evidence.confirmed_salary_override is None:
            for m in user_msgs:
                if m.source_type == "employer" and any(k in m.message_text.lower() for k in ["salary", "pay", "payroll"]):
                    match = re.search(r'(?:EUR|USD|ZAR|INR|IDR)\s*([\d\.,]+)', m.message_text)
                    if match:
                        try:
                            clean_str = match.group(1).rstrip('.').rstrip(',').replace(',', '')
                            val = float(clean_str)
                            if val > 0:
                                evidence.confirmed_salary_override = val
                                break
                        except Exception:
                            pass

        return evidence
