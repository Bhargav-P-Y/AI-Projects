# code/agent/financial_agent.py
"""
Financial Decision Agent Orchestrator.
Orchestrates data retrieval, reasoning, deterministic plan solving, grounded explanation generation,
and strict schema guardrail validation with error isolation and an Observable Bounded ReAct Loop.
"""

import logging
from typing import List, Optional, Dict, Any
from code.data.models import (
    RequestItem,
    DecisionResultDTO,
    AffordabilityStatus,
    RecommendedPaymentMethod,
    PaymentItemDTO,
    FinancialEvent,
)
from code.data.repository import (
    ProfileRepository,
    EventRepository,
    PaymentOptionRepository,
    MessageRepository,
)
from code.data.evidence_resolver import EvidenceResolver
from code.engine.cadence import CadenceEngine
from code.engine.cash_flow import CashFlowSimulator
from code.engine.spending_optimizer import SpendingOptimizer
from code.engine.plan_solver import PlanSolver
from code.agent.tools import FinancialAgentTools, UserFinancialContextDTO
from code.agent.explainer import DecisionExplainer
from code.agent.guardrail import OutputGuardrail, SchemaGuardrailError

logger = logging.getLogger(__name__)


class FinancialAgent:
    """
    Main Agent Orchestrator implementing an Observable Bounded ReAct Execution Loop (max_steps = 3),
    integrating atomic tools with strict guardrails, grounded LLM explainer, and Tier-3 safety fallback.
    """
    def __init__(
        self,
        profile_repo: Optional[ProfileRepository] = None,
        event_repo: Optional[EventRepository] = None,
        options_repo: Optional[PaymentOptionRepository] = None,
        msg_repo: Optional[MessageRepository] = None,
        use_llm_explainer: bool = True,
    ):
        self.profile_repo = profile_repo or ProfileRepository()
        self.event_repo = event_repo or EventRepository()
        self.options_repo = options_repo or PaymentOptionRepository()
        self.msg_repo = msg_repo or MessageRepository()
        self.evidence_resolver = EvidenceResolver(self.msg_repo)

        # Unified single-instance dependency injection
        self.cadence_engine = CadenceEngine()
        self.cash_flow_sim = CashFlowSimulator()
        self.spending_opt = SpendingOptimizer(self.cash_flow_sim)
        self.plan_solver = PlanSolver(self.cash_flow_sim, self.spending_opt)

        self.tools = FinancialAgentTools(
            profile_repo=self.profile_repo,
            event_repo=self.event_repo,
            options_repo=self.options_repo,
            evidence_resolver=self.evidence_resolver,
            cadence_engine=self.cadence_engine,
            cash_flow_sim=self.cash_flow_sim,
            spending_opt=self.spending_opt,
            plan_solver=self.plan_solver,
        )
        self.explainer = DecisionExplainer(use_llm=use_llm_explainer)
        self.guardrail = OutputGuardrail()

    def process_request(self, request: RequestItem, max_retries: int = 3) -> DecisionResultDTO:
        """
        Processes a single financial request through an Observable Bounded ReAct Execution Loop (max_steps = 3)
        with multi-tier error handling and self-correction.
        """
        last_error = None

        for retry_attempt in range(max_retries):
            try:
                # Observable ReAct Execution Context
                react_memory: Dict[str, Any] = {
                    "request_id": request.request_id,
                    "retry_attempt": retry_attempt,
                    "steps": [],
                }

                context: Optional[UserFinancialContextDTO] = None
                options: List[Any] = []
                decision: Optional[DecisionResultDTO] = None

                # Observable Bounded ReAct Loop: max_steps = 3
                max_steps = 3
                for step in range(max_steps):
                    step_log = {"step": step + 1}

                    if step == 0:
                        # Step 1: Perceive & Ingest Context via Native Tools
                        # Thought: Retrieve user profile, baseline cash trajectory, and payment options.
                        context = self.tools.get_user_financial_context(request.user_id, request.request_date)
                        options = self.tools.get_payment_options(request.request_id)
                        step_log["thought"] = "Retrieved financial profile, recurring timeline, and candidate options."
                        step_log["action"] = "get_user_financial_context + get_payment_options"
                        step_log["observation"] = f"Balance: {context.profile.current_available_balance}, Options: {len(options)}"

                    elif step == 1:
                        # Step 2: Reason & Simulate Liquidity Bounds
                        # Thought: Evaluate whether full payment today breaches the minimum balance floor.
                        if context is None:
                            context = self.tools.get_user_financial_context(request.user_id, request.request_date)
                        
                        # Dynamic simulation tool invocation
                        baseline_sim = self.tools.simulate_cash_flow(
                            profile=context.profile,
                            events=context.projected_events,
                            request_date=request.request_date,
                            payments=[PaymentItemDTO(date=request.request_date, amount=request.requested_amount)],
                        )
                        
                        step_log["thought"] = "Tested liquidity floor invariant for immediate full payment."
                        step_log["action"] = "simulate_cash_flow"
                        step_log["observation"] = (
                            f"is_safe={baseline_sim.is_safe}, min_balance={baseline_sim.minimum_projected_balance}, "
                            f"deficit={baseline_sim.balance_deficit_below_floor}"
                        )

                        # If deficit exists, test spending changes tool dynamically
                        if not baseline_sim.is_safe:
                            relief_ok, changes = self.tools.find_spending_changes(
                                profile=context.profile,
                                events=context.projected_events,
                                request_date_str=request.request_date,
                                payments=[PaymentItemDTO(date=request.request_date, amount=request.requested_amount)],
                            )
                            step_log["spending_relief_tested"] = f"relief_found={relief_ok}, count={len(changes)}"

                    elif step == 2:
                        # Step 3: Solve Optimal Plan & Guardrail Validation
                        # Thought: Formulate candidate plans across 6-level hierarchy and validate invariants.
                        if context is None:
                            context = self.tools.get_user_financial_context(request.user_id, request.request_date)
                        if not options:
                            options = self.tools.get_payment_options(request.request_id)

                        decision = self.tools.solve_optimal_plan(
                            request=request,
                            profile=context.profile,
                            projected_events=context.projected_events,
                            options=options,
                        )

                        # Generate grounded explanation (Tier 1 LLM with Tier 3 deterministic fallback)
                        all_events = context.raw_events + context.projected_events
                        explanation = self.explainer.explain(request, context.profile, decision, all_events)
                        decision.decision_explanation = explanation

                        step_log["thought"] = "Solved optimal plan and synthesized grounded explanation."
                        step_log["action"] = "solve_optimal_plan + explain"
                        step_log["observation"] = f"Method: {decision.recommended_payment_method}, Status: {decision.affordability_status}"

                    react_memory["steps"].append(step_log)

                # Post-ReAct Guardrail Validation & Sanitization
                if decision is not None:
                    sanitized_decision = self.guardrail.validate_and_sanitize(decision, request)
                    return sanitized_decision

            except SchemaGuardrailError as sge:
                last_error = sge
                logger.warning(f"Guardrail discrepancy on {request.request_id} (retry {retry_attempt+1}/{max_retries}): {sge}")
                if retry_attempt == max_retries - 1:
                    break

            except Exception as ex:
                last_error = ex
                logger.error(f"Error processing {request.request_id} (retry {retry_attempt+1}/{max_retries}): {ex}")
                if retry_attempt == max_retries - 1:
                    break

        # Tier-3 Safe Fallback Decision if an unexpected exception persisted
        logger.error(f"Engaging safe fallback for {request.request_id} due to: {last_error}")
        return DecisionResultDTO(
            request_id=request.request_id,
            amount_safe_to_pay=0.0,
            affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
            recommended_payment_method=RecommendedPaymentMethod.NOT_RECOMMENDED,
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation="Do not proceed with this request as it risks breaching the minimum balance requirement.",
        )

    def process_batch(self, requests: List[RequestItem], max_workers: int = 8) -> List[DecisionResultDTO]:
        """
        Processes a batch of requests with individual error isolation and high-performance concurrency.
        Guarantees that a corrupted record cannot crash the evaluation run, while preserving canonical sequence.
        """
        from concurrent.futures import ThreadPoolExecutor

        results_dict = {}

        def _worker(req: RequestItem):
            try:
                dec = self.process_request(req)
                return req.request_id, dec
            except Exception as e:
                logger.critical(f"Critical error on {req.request_id} during batch run: {e}")
                fallback = DecisionResultDTO(
                    request_id=req.request_id,
                    amount_safe_to_pay=0.0,
                    affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
                    recommended_payment_method=RecommendedPaymentMethod.NOT_RECOMMENDED,
                    payment_plan="none",
                    earliest_date_for_full_payment="",
                    spending_changes_needed="none",
                    decision_explanation="Do not proceed with this request.",
                )
                return req.request_id, fallback

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_results = executor.map(_worker, requests)
            for req_id, dec in future_results:
                results_dict[req_id] = dec

        # Return results in the exact original request sequence
        return [results_dict[req.request_id] for req in requests]
