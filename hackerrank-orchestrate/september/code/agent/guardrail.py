# code/agent/guardrail.py
"""
Strict Output Schema Guardrail for Buy or Wait.
Enforces all mathematical, structural, and semantic invariants specified in SPEC.md §6.2
and AGENTS.md §6.2 before any decision row is written to output.csv.
"""

from datetime import date
from typing import Any, List, Optional, Set
from code.data.models import (
    AffordabilityStatus,
    RecommendedPaymentMethod,
    DecisionResultDTO,
    RequestItem,
)


VALID_STATUSES: Set[str] = {
    AffordabilityStatus.AFFORDABLE_NOW.value,
    AffordabilityStatus.AFFORDABLE_WITH_PLAN.value,
    AffordabilityStatus.AFFORDABLE_LATER.value,
    AffordabilityStatus.NOT_AFFORDABLE.value,
}

VALID_METHODS: Set[str] = {
    RecommendedPaymentMethod.FULL_PAYMENT.value,
    RecommendedPaymentMethod.PARTIAL_PAYMENT.value,
    RecommendedPaymentMethod.INSTALLMENTS.value,
    RecommendedPaymentMethod.WAIT.value,
    RecommendedPaymentMethod.NOT_RECOMMENDED.value,
}


class SchemaGuardrailError(ValueError):
    """
    Raised when an output decision violates project contract invariants.
    Provides structured diagnostic attributes for automated error handling and logging.
    """
    def __init__(
        self,
        field_name: str,
        invalid_value: Any,
        expected_invariant: str,
        request_id: Optional[str] = None,
    ):
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_invariant = expected_invariant
        self.request_id = request_id
        req_prefix = f"[{request_id}] " if request_id else ""
        super().__init__(
            f"{req_prefix}Contract Invariant Violation on field '{field_name}': "
            f"got {repr(invalid_value)}, expected {expected_invariant}"
        )


class OutputGuardrail:
    """
    Validates and sanitizes DecisionResultDTO records to guarantee 100% compliance
    with HackerRank Orchestrate evaluation requirements and prevents runtime crashes.
    """
    @staticmethod
    def validate_and_sanitize(
        decision: DecisionResultDTO,
        request: RequestItem,
    ) -> DecisionResultDTO:
        req_id = str(request.request_id).strip()

        # 1. request_id validation
        if not decision.request_id or not str(decision.request_id).strip():
            decision.request_id = req_id

        # 2. amount_safe_to_pay invariant: 0 <= amount_safe_to_pay <= requested_amount
        try:
            safe_amt = round(float(decision.amount_safe_to_pay), 2)
        except (ValueError, TypeError):
            safe_amt = 0.0

        if safe_amt < 0.0:
            safe_amt = 0.0
        if safe_amt > request.requested_amount:
            safe_amt = request.requested_amount
        decision.amount_safe_to_pay = safe_amt

        # 3. Enum normalization & canonical string extraction
        status_raw = decision.affordability_status
        status_val = status_raw.value if hasattr(status_raw, "value") else str(status_raw).strip()
        if status_val not in VALID_STATUSES:
            raise SchemaGuardrailError("affordability_status", status_val, f"one of {VALID_STATUSES}", req_id)

        method_raw = decision.recommended_payment_method
        method_val = method_raw.value if hasattr(method_raw, "value") else str(method_raw).strip()
        if method_val not in VALID_METHODS:
            raise SchemaGuardrailError("recommended_payment_method", method_val, f"one of {VALID_METHODS}", req_id)

        # 4. Cross-Field Semantic Coherence Reconciliation
        # Enforce contract couplings between affordability_status and recommended_payment_method
        if method_val == RecommendedPaymentMethod.NOT_RECOMMENDED.value:
            status_val = AffordabilityStatus.NOT_AFFORDABLE.value
        elif method_val == RecommendedPaymentMethod.WAIT.value:
            status_val = AffordabilityStatus.AFFORDABLE_LATER.value
        elif status_val == AffordabilityStatus.NOT_AFFORDABLE.value:
            method_val = RecommendedPaymentMethod.NOT_RECOMMENDED.value
        elif status_val == AffordabilityStatus.AFFORDABLE_LATER.value and method_val != RecommendedPaymentMethod.WAIT.value:
            method_val = RecommendedPaymentMethod.WAIT.value
        elif status_val == AffordabilityStatus.AFFORDABLE_NOW.value and method_val not in [
            RecommendedPaymentMethod.FULL_PAYMENT.value,
            RecommendedPaymentMethod.PARTIAL_PAYMENT.value,
            RecommendedPaymentMethod.INSTALLMENTS.value,
        ]:
            method_val = RecommendedPaymentMethod.FULL_PAYMENT.value

        decision.affordability_status = AffordabilityStatus(status_val)
        decision.recommended_payment_method = RecommendedPaymentMethod(method_val)

        # 5. earliest_date_for_full_payment
        # Contract: Equals request_date for affordable_now.
        # If unaffordable throughout the entire 90 days, empty string "".
        # If unaffordable by user's deadline but affordable later within 90 days, preserves the true date!
        earliest = str(decision.earliest_date_for_full_payment or "").strip()
        if status_val == AffordabilityStatus.AFFORDABLE_NOW.value:
            earliest = request.request_date
        elif earliest:
            try:
                date.fromisoformat(earliest)
            except (ValueError, TypeError):
                earliest = ""
        decision.earliest_date_for_full_payment = earliest

        # 6. payment_plan invariant
        plan = str(decision.payment_plan or "none").strip()
        if method_val == RecommendedPaymentMethod.NOT_RECOMMENDED.value or status_val == AffordabilityStatus.NOT_AFFORDABLE.value:
            plan = "none"
        elif plan != "none":
            # Sanitize pipe-separated entries YYYY-MM-DD:amt with crash protection
            entries = plan.split("|")
            cleaned_entries = []
            for entry in entries:
                if ":" in entry:
                    try:
                        d_str, amt_str = entry.split(":", 1)
                        date.fromisoformat(d_str.strip())
                        val = float(amt_str.strip())
                        clean_amt = f"{val:.2f}".rstrip("0").rstrip(".") or "0"
                        cleaned_entries.append(f"{d_str.strip()}:{clean_amt}")
                    except (ValueError, TypeError):
                        continue
            plan = "|".join(cleaned_entries) if cleaned_entries else "none"
        decision.payment_plan = plan

        # 7. spending_changes_needed invariant (none or up to 3 valid actions)
        changes = str(decision.spending_changes_needed or "none").strip()
        if changes != "none":
            actions = changes.split("|")
            if len(actions) > 3:
                actions = actions[:3]
            valid_actions = []
            seen_events = set()
            for a in actions:
                parts = a.split(":")
                try:
                    if parts[0] == "stop" and len(parts) == 2:
                        eid = parts[1].strip()
                        if eid and eid not in seen_events:
                            valid_actions.append(f"stop:{eid}")
                            seen_events.add(eid)
                    elif parts[0] == "reduce_to" and len(parts) == 3:
                        eid = parts[1].strip()
                        val = float(parts[2].strip())
                        amt_clean = f"{val:.2f}".rstrip("0").rstrip(".") or "0"
                        if eid and eid not in seen_events:
                            valid_actions.append(f"reduce_to:{eid}:{amt_clean}")
                            seen_events.add(eid)
                except (ValueError, TypeError):
                    continue
            changes = "|".join(valid_actions) if valid_actions else "none"
        decision.spending_changes_needed = changes

        # 8. decision_explanation invariant
        expl = str(decision.decision_explanation or "").strip()
        if not expl:
            expl = f"Recommendation for {req_id} conforms to financial limits."
        decision.decision_explanation = expl

        return decision
