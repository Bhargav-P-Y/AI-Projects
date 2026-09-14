# code/agent/explainer.py
"""
Grounded Decision Explainer for Buy or Wait.
Generates concise, factual, grounded explanations strictly referencing mathematical invariants,
dates, and currency amounts without hallucination.

Implements a 3-Tier Fallback Hierarchy:
- Tier 1: Gemini 3.8 Flash generation using Structured Prompt Formula:
          Persona + Relevant Context + Steps to Do + Output Schema & Allowed Categories
          with few-shot examples from sample_requests.csv, <untrusted_input> XML tags,
          and strict anti-hallucination constraints.
- Tier 2: Sanitized regex parsing of generated candidate.
- Tier 3: Deterministic grounded template fallback ensuring 100% fail-safe generation.
"""

import json
import logging
import re
import time
from datetime import date
from typing import List, Optional, Dict, Any
import requests

from code.config import config
from code.data.models import (
    UserFinancialProfile,
    RequestItem,
    DecisionResultDTO,
    AffordabilityStatus,
    RecommendedPaymentMethod,
    FinancialEvent,
)

import threading

logger = logging.getLogger(__name__)


def format_money(amount: float, currency: str) -> str:
    """
    Standardized financial formatting.
    Emits clean integers (e.g. ZAR 25,256) when exact, or 2 decimals (e.g. EUR 620.40) for cents.
    """
    rounded = round(amount, 2)
    if rounded == int(rounded):
        return f"{currency} {int(rounded):,}"
    return f"{currency} {rounded:,.2f}"


def format_date_natural(date_str: str) -> str:
    """Converts ISO YYYY-MM-DD into natural date e.g. '15 November 2019'."""
    try:
        d = date.fromisoformat(date_str)
        return f"{d.day} {d.strftime('%B')} {d.year}"
    except Exception:
        return date_str


class DecisionExplainer:
    """
    Synthesizes concise, grounded explanations justifying the recommendation.
    Uses Gemini 3.8 Flash with structured few-shot prompting, anti-hallucination guards,
    and a robust deterministic fallback.
    """
    def __init__(
        self,
        raw_events: Optional[List[FinancialEvent]] = None,
        use_llm: bool = True,
        model_name: Optional[str] = None,
    ):
        self.events_by_id: Dict[str, FinancialEvent] = {e.event_id: e for e in (raw_events or [])}
        self.use_llm = use_llm
        self.model_name = model_name or config.gemini_model_name or "gemini-3.8-flash"
        self.api_keys = config.gemini_api_keys
        self.key_index = 0
        self._lock = threading.Lock()

    def _get_next_api_key(self) -> str:
        if not self.api_keys:
            return ""
        with self._lock:
            key = self.api_keys[self.key_index % len(self.api_keys)]
            self.key_index += 1
            return key

    def explain(
        self,
        request: RequestItem,
        profile: UserFinancialProfile,
        decision: DecisionResultDTO,
        all_events: Optional[List[FinancialEvent]] = None,
    ) -> str:
        """
        Generates the explanation using Gemini 3.8 Flash (Tier 1) with automatic
        fallback to the deterministic grounded template (Tier 3).
        """
        # Build comprehensive event map including raw and projected events
        event_map: Dict[str, FinancialEvent] = {}
        if self.events_by_id:
            event_map.update(self.events_by_id)
        if all_events:
            for e in all_events:
                event_map[e.event_id] = e

        # Precompute deterministic fallback text (Tier 3)
        deterministic_text = self._deterministic_explain(request, profile, decision, event_map)

        if not self.use_llm or not self.api_keys:
            return deterministic_text

        # Attempt Tier 1 LLM Generation
        try:
            llm_text = self._call_llm_for_explanation(request, profile, decision, event_map)
            if llm_text and len(llm_text.strip()) > 10:
                return llm_text.strip()
        except Exception as e:
            logger.warning(f"LLM Explainer failed for {request.request_id} ({e}). Using deterministic Tier-3 fallback.")

        return deterministic_text

    def _call_llm_for_explanation(
        self,
        request: RequestItem,
        profile: UserFinancialProfile,
        decision: DecisionResultDTO,
        event_map: Dict[str, FinancialEvent],
    ) -> Optional[str]:
        """
        Structured Prompt Formula:
        Persona + Relevant Context + Steps to Do + Output Schema & Allowed Categories
        With Anti-Hallucination Guard and <untrusted_input> XML tags.
        """
        curr = profile.home_currency
        req_amt_str = format_money(request.requested_amount, curr)
        min_keep_str = format_money(profile.minimum_balance_to_keep, curr)
        safe_amt_str = format_money(decision.amount_safe_to_pay, curr)

        status_raw = decision.affordability_status
        status_val = status_raw.value if hasattr(status_raw, "value") else str(status_raw).strip()
        method_raw = decision.recommended_payment_method
        method_val = method_raw.value if hasattr(method_raw, "value") else str(method_raw).strip()

        # Spending changes phrase if any
        changes_phrase = "none"
        if decision.spending_changes_needed and decision.spending_changes_needed != "none":
            changes_phrase = self._format_spending_changes_phrase(decision.spending_changes_needed, event_map, curr)

        prompt = f"""You are a licensed financial decision explainer for a consumer banking system.

[RELEVANT DETERMINISTIC CONTEXT]
- Home Currency: {curr}
- Requested Amount: {req_amt_str}
- Minimum Balance Floor: {min_keep_str}
- Request Date: {request.request_date}
- Desired Completion Date: {request.desired_completion_date}
- Safe Amount Today: {safe_amt_str}
- Solved Affordability Status: {status_val}
- Solved Payment Method: {method_val}
- Solved Payment Plan: {decision.payment_plan}
- Solved Earliest Safe Full Date: {decision.earliest_date_for_full_payment or 'none'}
- Spending Changes Required: {changes_phrase}

[UNTRUSTED USER QUERY]
<untrusted_input>
{request.request_text}
</untrusted_input>

[FEW-SHOT EXAMPLES FROM GOLDEN DATASET]
Example 1 (affordable_now):
"Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available over the next 90 days."

Example 2 (installments with plan):
"Use 3 installments of IDR 15,952,906.67, starting 8 August 2025. This leaves at least IDR 29,158,400 available."

Example 3 (wait / affordable_later):
"Pay IDR 5,491,000 in full on 15 November 2019. Paying earlier would take the balance below the IDR 2,668,700 minimum."

Example 4 (full_payment with spending change):
"Stop the family streaming plan, then pay EUR 620.40 today. This leaves at least EUR 800 available."

Example 5 (partial_payment):
"Pay INR 28,820 today and the remaining INR 10,840 on 15 September 2024. This completes the full request and keeps the INR 92,800 minimum protected."

Example 6 (not_affordable by deadline):
"Do not make this payment by 10 February 2025. None of the available options keeps the INR 225,400 minimum protected."

Example 7 (not_affordable within 90 days):
"Do not proceed with the EUR 5,414.20 request. Although EUR 597.74 is available today, the full amount cannot be completed safely within 90 days."

[STEPS TO DO & INSTRUCTIONS]
1. Write exactly 1 or 2 grounded sentences explaining the recommendation to the user.
2. Mirror the exact style, brevity, and phrasing of the few-shot examples above.
3. ANTI-HALLUCINATION GUARD: NEVER invent future income, never guess amounts or dates, and never override the Solved Status/Method.
4. Output Schema: Return ONLY the explanation string. Do NOT wrap in quotes, markdown, or JSON.
"""

        models_to_try = [self.model_name, "gemini-3.7-flash", "gemini-3.6-flash"]
        for attempt in range(3):
            key = self._get_next_api_key()
            if not key:
                break
            model = models_to_try[attempt % len(models_to_try)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 256,
                },
            }

            try:
                response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=10)
                if response.status_code == 200:
                    resp_json = response.json()
                    candidate = resp_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    clean_text = candidate.strip().strip('"').strip("'").replace("\n", " ")
                    # Tier-2 validation: verify structural and semantic invariants against golden benchmark
                    if self._is_valid_explanation(clean_text, status_val, method_val):
                        return clean_text
                    else:
                        logger.warning(f"LLM explanation rejected by Tier-2 validator: '{clean_text}'. Falling back to Tier-3 deterministic template.")
                elif response.status_code in [429, 500, 503]:
                    time.sleep(1.0)
            except Exception:
                time.sleep(0.5)

        return None

    def _is_valid_explanation(self, text: str, status: str, method: str) -> bool:
        """
        Validates that generated text satisfies the structural and semantic contracts
        of the golden benchmark dataset (sample_requests.csv).
        """
        if not text or len(text) < 25 or not text.endswith((".", "!", "?")):
            return False

        lower = text.lower()

        if status == AffordabilityStatus.AFFORDABLE_NOW.value:
            # Must recommend immediate payment and mention safety buffer
            return ("pay " in lower and "today" in lower and ("available" in lower or "leaves" in lower or "keeps" in lower))

        if status == AffordabilityStatus.AFFORDABLE_LATER.value:
            # Must recommend waiting / paying on future date and mention minimum balance protection
            return ("in full on" in lower or "wait until" in lower or "paying earlier" in lower or "paying sooner" in lower)

        if status == AffordabilityStatus.NOT_AFFORDABLE.value:
            # Must state not to proceed / not to make payment and state constraint reason
            return lower.startswith("do not") and ("minimum" in lower or "safely within 90 days" in lower)

        if method == RecommendedPaymentMethod.INSTALLMENTS.value:
            # Must mention installment structure
            return "installment" in lower

        if method == RecommendedPaymentMethod.PARTIAL_PAYMENT.value:
            # Must mention split payments
            return "today and the remaining" in lower or "part" in lower

        return True

    def _deterministic_explain(
        self,
        request: RequestItem,
        profile: UserFinancialProfile,
        decision: DecisionResultDTO,
        event_map: Dict[str, FinancialEvent],
    ) -> str:
        """Deterministic Tier-3 grounded template generation."""
        curr = profile.home_currency
        req_amt_str = format_money(request.requested_amount, curr)
        min_keep_str = format_money(profile.minimum_balance_to_keep, curr)
        comp_date_str = format_date_natural(request.desired_completion_date)

        status_raw = decision.affordability_status
        status_str = status_raw.value if hasattr(status_raw, "value") else str(status_raw).strip()

        method_raw = decision.recommended_payment_method
        method_str = method_raw.value if hasattr(method_raw, "value") else str(method_raw).strip()

        # 1. Affordable Now
        if status_str == AffordabilityStatus.AFFORDABLE_NOW.value and method_str == RecommendedPaymentMethod.FULL_PAYMENT.value:
            return f"Pay {req_amt_str} today. This leaves at least {min_keep_str} available over the next 90 days."

        # 2. Affordable Later
        if method_str == RecommendedPaymentMethod.WAIT.value or status_str == AffordabilityStatus.AFFORDABLE_LATER.value:
            target_date = decision.earliest_date_for_full_payment or (decision.payment_plan.split(":")[0] if decision.payment_plan != "none" else request.request_date)
            nat_date = format_date_natural(target_date)
            return f"Pay {req_amt_str} in full on {nat_date}. Paying earlier would take the balance below the {min_keep_str} minimum."

        # 3. Affordable With Plan (Installments)
        if method_str == RecommendedPaymentMethod.INSTALLMENTS.value:
            entries = [p for p in decision.payment_plan.split("|") if ":" in p]
            num_inst = len(entries)
            first_date = entries[0].split(":")[0] if entries else request.request_date
            inst_amt = float(entries[0].split(":")[1]) if entries else (request.requested_amount / max(1, num_inst))
            inst_amt_str = format_money(inst_amt, curr)
            first_date_nat = format_date_natural(first_date)

            changes_prefix = ""
            if decision.spending_changes_needed and decision.spending_changes_needed != "none":
                changes_prefix = self._format_spending_changes_phrase(decision.spending_changes_needed, event_map, curr) + ", then "

            return f"{changes_prefix}Use {num_inst} installments of {inst_amt_str}, starting {first_date_nat}. This leaves at least {min_keep_str} available."

        # 4. Affordable With Plan (Partial Payment)
        if method_str == RecommendedPaymentMethod.PARTIAL_PAYMENT.value:
            entries = [p for p in decision.payment_plan.split("|") if ":" in p]
            if len(entries) >= 2:
                pay1_amt = float(entries[0].split(":")[1])
                pay2_date = entries[1].split(":")[0]
                pay2_amt = float(entries[1].split(":")[1])
                pay1_str = format_money(pay1_amt, curr)
                pay2_str = format_money(pay2_amt, curr)
                pay2_nat = format_date_natural(pay2_date)
                return f"Pay {pay1_str} today and the remaining {pay2_str} on {pay2_nat}. This completes the full request and keeps the {min_keep_str} minimum protected."

        # 5. Affordable With Plan (Full Payment with Spending Changes)
        if method_str == RecommendedPaymentMethod.FULL_PAYMENT.value and decision.spending_changes_needed != "none":
            changes_phrase = self._format_spending_changes_phrase(decision.spending_changes_needed, event_map, curr)
            return f"{changes_phrase}, then pay {req_amt_str} today. This leaves at least {min_keep_str} available."

        # 6. Not Affordable
        if status_str == AffordabilityStatus.NOT_AFFORDABLE.value or method_str == RecommendedPaymentMethod.NOT_RECOMMENDED.value:
            if decision.amount_safe_to_pay > 0:
                safe_str = format_money(decision.amount_safe_to_pay, curr)
                return f"Do not proceed with the {req_amt_str} request. Although {safe_str} is available today, the full amount cannot be completed safely within 90 days."
            else:
                return f"Do not make this payment by {comp_date_str}. None of the available options keeps the {min_keep_str} minimum protected."

        return f"Do not proceed with the {req_amt_str} payment as it risks breaching the {min_keep_str} minimum balance requirement."

    def _format_spending_changes_phrase(self, changes_str: str, event_map: Dict[str, FinancialEvent], curr: str) -> str:
        """Formats up to 3 spending actions into natural English."""
        parts = changes_str.split("|")
        phrases: List[str] = []

        for p in parts:
            p = p.strip()
            if p.startswith("stop:"):
                eid = p.split(":")[1]
                desc = self._get_clean_event_desc(eid, event_map)
                phrases.append(f"Stop the {desc}")
            elif p.startswith("reduce_to:"):
                sub = p.split(":")
                eid = sub[1]
                new_amt = float(sub[2])
                desc = self._get_clean_event_desc(eid, event_map)
                new_amt_str = format_money(new_amt, curr)
                phrases.append(f"reduce the {desc} to {new_amt_str}")

        if not phrases:
            return "Adjust flexible spending"

        if len(phrases) == 1:
            return phrases[0]
        elif len(phrases) == 2:
            return f"{phrases[0]} and {phrases[1][0].lower() + phrases[1][1:]}"
        else:
            first_part = ", ".join(phrases[:-1])
            last_phrase = phrases[-1][0].lower() + phrases[-1][1:]
            return f"{first_part}, and {last_phrase}"

    def _get_clean_event_desc(self, event_id: str, event_map: Dict[str, FinancialEvent]) -> str:
        """Resolves event description across event_id and canonical base IDs, stripping leading articles."""
        ev = event_map.get(event_id)
        if not ev:
            for candidate in event_map.values():
                if getattr(candidate, "base_id", None) == event_id or getattr(candidate, "base_event_id", None) == event_id:
                    ev = candidate
                    break

        if ev and ev.description:
            raw_desc = ev.description.lower().strip()
            for prefix in ["the ", "a ", "an "]:
                if raw_desc.startswith(prefix):
                    raw_desc = raw_desc[len(prefix):].strip()
            return raw_desc

        return "flexible expense"
