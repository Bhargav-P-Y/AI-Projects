# code/data/models.py
"""
Strongly typed domain models and Data Transfer Objects (DTOs) for the Buy or Wait system.
Adheres strictly to the ER schema and tool contracts defined in SPEC.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Set, Dict


class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class RecommendedPaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class EventFlexibility(str, Enum):
    FIXED = "fixed"
    STOPPABLE = "stoppable"
    REDUCIBLE = "reducible"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"


@dataclass
class UserFinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str] = field(default_factory=list)
    expense_categories_to_protect: Set[str] = field(default_factory=set)
    expense_categories_user_is_willing_to_reduce: Set[str] = field(default_factory=set)
    expense_categories_user_is_willing_to_stop: Set[str] = field(default_factory=set)
    payment_methods_user_will_consider: Set[str] = field(default_factory=set)
    max_installment_months: Optional[int] = None


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # 'debit', 'credit', 'non_cash'
    amount: float
    currency: str
    event_date: str  # YYYY-MM-DD
    settlement_date: str  # YYYY-MM-DD
    status: str  # 'settled', 'pending', 'scheduled', 'cancelled', 'failed', 'unrealized'
    linked_event_id: Optional[str] = None
    flexibility: EventFlexibility = EventFlexibility.FIXED
    minimum_allowed_amount: Optional[float] = None
    base_event_id: Optional[str] = None

    @property
    def base_id(self) -> str:
        """Returns the canonical base event ID, stripping projection prefixes/suffixes."""
        if self.base_event_id:
            return self.base_event_id
        if self.event_id.startswith("proj_"):
            return self.event_id[5:].rsplit("_", 1)[0]
        return self.event_id


@dataclass
class RequestPaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # 'full_payment' or 'installments'
    payment_amount: float
    number_of_payments: int
    first_payment_date: str  # YYYY-MM-DD
    payment_frequency_days: Optional[int] = None
    financing_fee: float = 0.0
    total_payable_amount: float = 0.0


@dataclass
class RequestItem:
    request_id: str
    user_id: str
    request_date: str  # YYYY-MM-DD
    request_type: str
    requested_amount: float
    desired_completion_date: str  # YYYY-MM-DD
    allows_partial_payment: bool
    request_text: str


@dataclass
class MessageItem:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str
    source_type: str
    message_text: str


@dataclass
class PaymentItemDTO:
    date: str  # YYYY-MM-DD
    amount: float


@dataclass
class SpendingChangeDTO:
    action_type: str  # 'stop' or 'reduce_to'
    event_id: str
    new_amount: Optional[float] = None

    def to_contract_str(self) -> str:
        if self.action_type == "stop":
            return f"stop:{self.event_id}"
        elif self.action_type == "reduce_to":
            # Formatted clean float/int
            amt_str = f"{self.new_amount:.2f}".rstrip("0").rstrip(".")
            return f"reduce_to:{self.event_id}:{amt_str}"
        return "none"


@dataclass
class SimulationResultDTO:
    is_safe: bool
    minimum_projected_balance: float
    balance_deficit_below_floor: float  # max(0.0, minimum_balance_to_keep - min_balance)
    daily_balance_trajectory: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidatePlan:
    method: RecommendedPaymentMethod
    payments: List[PaymentItemDTO]
    total_payable_amount: float
    payment_option_id: Optional[str] = None
    first_payment_date: str = ""
    completion_date: str = ""
    number_of_payments: int = 0
    spending_changes: List[SpendingChangeDTO] = field(default_factory=list)
    is_safe: bool = False


@dataclass
class DecisionResultDTO:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payment_plan: str  # YYYY-MM-DD:amount|... or 'none'
    earliest_date_for_full_payment: str  # YYYY-MM-DD or ""
    spending_changes_needed: str  # pipe-separated or 'none'
    decision_explanation: str
