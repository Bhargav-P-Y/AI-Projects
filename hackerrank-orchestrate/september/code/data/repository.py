# code/data/repository.py
"""
Interface-Segregated Repositories for Profiles, Events, Options, and Messages.
Adheres strictly to the Law of Demeter, ISP, and high-performance in-memory indexing.
Uses fast tuple/dict-based parsing instead of slow row-by-row iteration.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

from code.config import config
from code.data.models import (
    UserFinancialProfile,
    FinancialEvent,
    RequestPaymentOption,
    RequestItem,
    MessageItem,
    EventFlexibility,
)
from code.data.fx_converter import FXConverter


class ProfileRepository:
    def __init__(self, profiles_csv: Optional[Path] = None):
        self.csv_path = profiles_csv or (config.dataset_dir / "financial_profiles.csv")
        self._profiles: Dict[str, UserFinancialProfile] = {}
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        for r in df.itertuples(index=False):
            uid = str(r.user_id).strip()
            priorities = [p.strip() for p in str(r.financial_priorities).split("|") if p.strip()]
            protect = set(p.strip() for p in str(r.expense_categories_to_protect).split("|") if p.strip())
            reduce_cat = set(p.strip() for p in str(r.expense_categories_user_is_willing_to_reduce).split("|") if p.strip())
            stop_cat = set(p.strip() for p in str(r.expense_categories_user_is_willing_to_stop).split("|") if p.strip())
            consider = set(p.strip() for p in str(r.payment_methods_user_will_consider).split("|") if p.strip())
            
            raw_months = str(r.max_installment_months).strip()
            max_months = int(raw_months) if raw_months and raw_months.isdigit() else None

            self._profiles[uid] = UserFinancialProfile(
                user_id=uid,
                home_currency=str(r.home_currency).strip(),
                current_available_balance=float(r.current_available_balance),
                minimum_balance_to_keep=float(r.minimum_balance_to_keep),
                financial_priorities=priorities,
                expense_categories_to_protect=protect,
                expense_categories_user_is_willing_to_reduce=reduce_cat,
                expense_categories_user_is_willing_to_stop=stop_cat,
                payment_methods_user_will_consider=consider,
                max_installment_months=max_months,
            )

    def get_profile(self, user_id: str) -> Optional[UserFinancialProfile]:
        return self._profiles.get(user_id)


class EventRepository:
    def __init__(
        self,
        events_csv: Optional[Path] = None,
        ocr_cache_path: Optional[Path] = None,
        fx_converter: Optional[FXConverter] = None,
    ):
        self.csv_path = events_csv or (config.dataset_dir / "financial_events.csv")
        self.ocr_cache_path = ocr_cache_path or config.ocr_cache_file
        self.fx = fx_converter or FXConverter()
        self._user_events: Dict[str, List[FinancialEvent]] = {}
        self._load()

    def _load(self):
        # Load OCR cached amounts if present
        cached_amounts: Dict[str, float] = {}
        if self.ocr_cache_path.exists():
            try:
                with open(self.ocr_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.values():
                        if isinstance(item, dict) and "event_id" in item and "amount" in item:
                            cached_amounts[item["event_id"]] = float(item["amount"])
            except json.JSONDecodeError as e:
                import logging
                logging.getLogger(__name__).error(f"Corrupted OCR cache JSON file at {self.ocr_cache_path}: {e}")
                raise ValueError(f"Failed to parse OCR cache file {self.ocr_cache_path}: {e}")
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Unexpected error reading OCR cache: {e}")
                raise

        df = pd.read_csv(self.csv_path, keep_default_na=False)
        # Columns: event_id, user_id, event_type, description, category, direction, amount, currency, event_date, settlement_date, status, linked_event_id, flexibility, minimum_allowed_amount
        for r in df.itertuples(index=False):
            eid = str(r.event_id).strip()
            uid = str(r.user_id).strip()
            raw_amt = str(r.amount).strip()
            
            if raw_amt:
                amt = float(raw_amt)
            elif eid in cached_amounts:
                amt = cached_amounts[eid]
            else:
                amt = 0.0

            raw_min = str(r.minimum_allowed_amount).strip()
            min_amt = float(raw_min) if raw_min else None

            flex_str = str(r.flexibility).strip()
            try:
                flex = EventFlexibility(flex_str)
            except ValueError:
                flex = EventFlexibility.FIXED

            ev = FinancialEvent(
                event_id=eid,
                user_id=uid,
                event_type=str(r.event_type).strip(),
                description=str(r.description).strip(),
                category=str(r.category).strip(),
                direction=str(r.direction).strip(),
                amount=amt,
                currency=str(r.currency).strip(),
                event_date=str(r.event_date).strip(),
                settlement_date=str(r.settlement_date).strip(),
                status=str(r.status).strip(),
                linked_event_id=str(r.linked_event_id).strip() or None,
                flexibility=flex,
                minimum_allowed_amount=min_amt,
            )
            self._user_events.setdefault(uid, []).append(ev)

    def get_events_for_user(self, user_id: str) -> List[FinancialEvent]:
        return self._user_events.get(user_id, [])


class PaymentOptionRepository:
    def __init__(self, options_csv: Optional[Path] = None):
        self.csv_path = options_csv or (config.dataset_dir / "request_payment_options.csv")
        self._options: Dict[str, List[RequestPaymentOption]] = {}
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        for r in df.itertuples(index=False):
            p_oid = str(r.payment_option_id).strip()
            req_id = str(r.request_id).strip()
            raw_freq = str(r.payment_frequency_days).strip()
            freq = int(raw_freq) if raw_freq and raw_freq.isdigit() else None

            opt = RequestPaymentOption(
                payment_option_id=p_oid,
                request_id=req_id,
                payment_method=str(r.payment_method).strip(),
                payment_amount=float(r.payment_amount),
                number_of_payments=int(r.number_of_payments),
                first_payment_date=str(r.first_payment_date).strip(),
                payment_frequency_days=freq,
                financing_fee=float(r.financing_fee) if r.financing_fee else 0.0,
                total_payable_amount=float(r.total_payable_amount),
            )
            self._options.setdefault(req_id, []).append(opt)

    def get_options_for_request(self, request_id: str) -> List[RequestPaymentOption]:
        return self._options.get(request_id, [])


class MessageRepository:
    def __init__(self, messages_csv: Optional[Path] = None):
        self.csv_path = messages_csv or (config.dataset_dir / "messages.csv")
        self._user_messages: Dict[str, List[MessageItem]] = {}
        self._request_messages: Dict[str, List[MessageItem]] = {}
        self._event_messages: Dict[str, List[MessageItem]] = {}
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        for r in df.itertuples(index=False):
            mid = str(r.message_id).strip()
            uid = str(r.user_id).strip()
            req_id = str(r.request_id).strip() or None
            eid = str(r.related_event_id).strip() or None

            msg = MessageItem(
                message_id=mid,
                user_id=uid,
                request_id=req_id,
                related_event_id=eid,
                sent_at=str(r.sent_at).strip(),
                source_type=str(r.source_type).strip(),
                message_text=str(r.message_text).strip(),
            )
            self._user_messages.setdefault(uid, []).append(msg)
            if req_id:
                self._request_messages.setdefault(req_id, []).append(msg)
            if eid:
                self._event_messages.setdefault(eid, []).append(msg)

    def get_messages_for_user(self, user_id: str) -> List[MessageItem]:
        return self._user_messages.get(user_id, [])

    def get_messages_for_request(self, request_id: str) -> List[MessageItem]:
        return self._request_messages.get(request_id, [])

    def get_messages_for_event(self, event_id: str) -> List[MessageItem]:
        return self._event_messages.get(event_id, [])


class RequestRepository:
    def __init__(self, requests_csv: Optional[Path] = None):
        self.csv_path = requests_csv or (config.dataset_dir / "requests.csv")
        self._requests: List[RequestItem] = []
        self._req_map: Dict[str, RequestItem] = {}
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        for r in df.itertuples(index=False):
            rid = str(r.request_id).strip()
            partial = str(r.allows_partial_payment).strip().lower() in ["true", "1", "t"]
            item = RequestItem(
                request_id=rid,
                user_id=str(r.user_id).strip(),
                request_date=str(r.request_date).strip(),
                request_type=str(r.request_type).strip(),
                requested_amount=float(r.requested_amount),
                desired_completion_date=str(r.desired_completion_date).strip(),
                allows_partial_payment=partial,
                request_text=str(r.request_text).strip(),
            )
            self._requests.append(item)
            self._req_map[rid] = item

    def get_all_requests(self) -> List[RequestItem]:
        return self._requests

    def get_request(self, request_id: str) -> Optional[RequestItem]:
        return self._req_map.get(request_id)
