# code/agent/tools.py
"""
Atomic Agent Tools for Buy or Wait Financial Decision Agent.
Compliant with Interface Segregation Principle (ISP) and Law of Demeter.
Each tool provides a single, isolated capability with strongly-typed interfaces.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from code.data.models import (
    UserFinancialProfile,
    FinancialEvent,
    RequestPaymentOption,
    RequestItem,
    DecisionResultDTO,
    PaymentItemDTO,
    SpendingChangeDTO,
    SimulationResultDTO,
)
from code.data.repository import (
    ProfileRepository,
    EventRepository,
    PaymentOptionRepository,
)
from code.data.evidence_resolver import EvidenceResolver, ResolvedUserEvidence
from code.engine.cadence import CadenceEngine
from code.engine.cash_flow import CashFlowSimulator
from code.engine.spending_optimizer import SpendingOptimizer
from code.engine.plan_solver import PlanSolver


@dataclass
class UserFinancialContextDTO:
    """Strongly-typed DTO encapsulating the user's financial profile and timeline context."""
    profile: UserFinancialProfile
    raw_events: List[FinancialEvent]
    projected_events: List[FinancialEvent]
    evidence: ResolvedUserEvidence


class FinancialAgentTools:
    """
    Encapsulates all atomic tools exposed to the agent layer.
    Strictly decouples repositories and engines from agent orchestration.
    """
    def __init__(
        self,
        profile_repo: ProfileRepository,
        event_repo: EventRepository,
        options_repo: PaymentOptionRepository,
        evidence_resolver: EvidenceResolver,
        cadence_engine: CadenceEngine,
        cash_flow_sim: CashFlowSimulator,
        spending_opt: SpendingOptimizer,
        plan_solver: PlanSolver,
    ):
        self.profile_repo = profile_repo
        self.event_repo = event_repo
        self.options_repo = options_repo
        self.evidence_resolver = evidence_resolver
        self.cadence_engine = cadence_engine
        self.cash_flow_sim = cash_flow_sim
        self.spending_opt = spending_opt
        self.plan_solver = plan_solver

    def get_user_financial_context(self, user_id: str, request_date: str) -> UserFinancialContextDTO:
        """
        Tool 1: Retrieves user profile, historical/projected cash events, and evidence.
        Fails fast with descriptive error if user_id is missing.
        """
        profile = self.profile_repo.get_profile(user_id)
        if profile is None:
            raise ValueError(f"Profile for user_id '{user_id}' not found in ProfileRepository.")

        raw_events = self.event_repo.get_events_for_user(user_id)
        evidence = self.evidence_resolver.resolve_evidence_for_user(user_id, request_date)

        projected_events = self.cadence_engine.project_recurring_events(
            events=raw_events,
            profile=profile,
            request_date_str=request_date,
            confirmed_salary_override=evidence.confirmed_salary_override,
            salary_effective_date=evidence.salary_effective_date,
            contract_ended=evidence.contract_ended,
        )

        return UserFinancialContextDTO(
            profile=profile,
            raw_events=raw_events,
            projected_events=projected_events,
            evidence=evidence,
        )

    def get_payment_options(self, request_id: str) -> List[RequestPaymentOption]:
        """
        Tool 2: Retrieves available merchant payment options (installments, etc.) for a request.
        """
        if not request_id or not str(request_id).strip():
            return []
        return self.options_repo.get_options_for_request(str(request_id).strip())

    def simulate_cash_flow(
        self,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        request_date: str,
        payments: Optional[List[PaymentItemDTO]] = None,
        spending_changes: Optional[List[SpendingChangeDTO]] = None,
        forecast_days: int = 90,
    ) -> SimulationResultDTO:
        """
        Tool 3: Simulates continuous daily balance trajectory for any proposed payment schedule.
        Returns a strongly-typed SimulationResultDTO.
        """
        return self.cash_flow_sim.simulate(
            profile=profile,
            events=events,
            request_date_str=request_date,
            payments=payments,
            spending_changes=spending_changes,
            forecast_days=forecast_days,
        )

    def find_spending_changes(
        self,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        request_date_str: str,
        payments: Optional[List[PaymentItemDTO]] = None,
        max_changes: int = 3,
    ) -> Tuple[bool, List[SpendingChangeDTO]]:
        """
        Tool 4: Standalone budget-relief optimizer evaluating what-if spending cuts.
        """
        return self.spending_opt.find_spending_changes(
            profile=profile,
            events=events,
            request_date_str=request_date_str,
            payments=payments,
            max_changes=max_changes,
        )

    def solve_optimal_plan(
        self,
        request: RequestItem,
        profile: UserFinancialProfile,
        projected_events: List[FinancialEvent],
        options: List[RequestPaymentOption],
    ) -> DecisionResultDTO:
        """
        Tool 5: Solves the optimal financial decision and payment plan deterministically.
        """
        return self.plan_solver.solve(
            request=request,
            profile=profile,
            events=projected_events,
            payment_options=options,
        )
