# code/engine/plan_solver.py
"""
Decision Solver & Plan Optimizer.
Formulates candidate plans (Full payment, strictly 2-transaction Partial payment, Installments, Wait)
and applies the rigid 6-level tie-breaking hierarchy to select the optimal recommendation.
Applies memoization, eliminates redundant simulations, uses direct ISO-8601 string operations,
protects float edge cases (e.g. 0.00), and maintains contract metric integrity.
"""

from datetime import date, timedelta
from typing import List, Optional
from code.data.models import (
    UserFinancialProfile,
    FinancialEvent,
    RequestPaymentOption,
    RequestItem,
    CandidatePlan,
    PaymentItemDTO,
    RecommendedPaymentMethod,
    AffordabilityStatus,
    DecisionResultDTO,
)
from code.engine.cash_flow import CashFlowSimulator
from code.engine.spending_optimizer import SpendingOptimizer


class PlanSolver:
    def __init__(
        self,
        cash_flow_simulator: Optional[CashFlowSimulator] = None,
        spending_optimizer: Optional[SpendingOptimizer] = None,
    ):
        self.simulator = cash_flow_simulator or CashFlowSimulator()
        self.optimizer = spending_optimizer or SpendingOptimizer(self.simulator)

    def solve(
        self,
        request: RequestItem,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        payment_options: List[RequestPaymentOption],
    ) -> DecisionResultDTO:
        """
        Solves the optimal recommendation for a request adhering to all challenge invariants.
        """
        req_date = request.request_date
        req_amt = request.requested_amount
        comp_date = request.desired_completion_date
        user_methods = profile.payment_methods_user_will_consider
        max_months = profile.max_installment_months

        # 1. Calculate baseline metrics without spending changes
        safe_to_pay = self.simulator.calculate_amount_safe_to_pay(profile, events, req_date, req_amt)
        earliest_full_date = self.simulator.calculate_earliest_date_for_full_payment(profile, events, req_date, req_amt)

        candidate_plans: List[CandidatePlan] = []

        # Candidate 1: Full Payment Today
        if "full_payment" in user_methods:
            p_full = [PaymentItemDTO(date=req_date, amount=req_amt)]
            # If safe_to_pay >= req_amt, paying today is mathematically guaranteed 100% safe
            if safe_to_pay >= req_amt:
                candidate_plans.append(
                    CandidatePlan(
                        method=RecommendedPaymentMethod.FULL_PAYMENT,
                        payments=p_full,
                        total_payable_amount=req_amt,
                        first_payment_date=req_date,
                        completion_date=req_date,
                        number_of_payments=1,
                        is_safe=True,
                    )
                )
            else:
                # Test with spending changes
                ok, changes = self.optimizer.find_spending_changes(profile, events, req_date, payments=p_full)
                if ok:
                    candidate_plans.append(
                        CandidatePlan(
                            method=RecommendedPaymentMethod.FULL_PAYMENT,
                            payments=p_full,
                            total_payable_amount=req_amt,
                            first_payment_date=req_date,
                            completion_date=req_date,
                            number_of_payments=1,
                            spending_changes=changes,
                            is_safe=True,
                        )
                    )

        # Candidate 2: Partial Payment (Strictly 2 transactions)
        # Allowed if allows_partial_payment is True, partial_payment in consider,
        # 0 < safe_to_pay < req_amt, and earliest_full_date is available within 90 days.
        if (
            request.allows_partial_payment and
            "partial_payment" in user_methods and
            0 < safe_to_pay < req_amt and
            earliest_full_date
        ):
            p_partial = [
                PaymentItemDTO(date=req_date, amount=safe_to_pay),
                PaymentItemDTO(date=earliest_full_date, amount=round(req_amt - safe_to_pay, 2)),
            ]
            sim_partial = self.simulator.simulate(profile, events, req_date, payments=p_partial)
            if sim_partial.is_safe:
                candidate_plans.append(
                    CandidatePlan(
                        method=RecommendedPaymentMethod.PARTIAL_PAYMENT,
                        payments=p_partial,
                        total_payable_amount=req_amt,
                        first_payment_date=req_date,
                        completion_date=earliest_full_date,
                        number_of_payments=2,
                        is_safe=True,
                    )
                )

        # Candidate 3: Installments from supplied options
        if "installments" in user_methods:
            for opt in payment_options:
                if opt.payment_method != "installments":
                    continue

                # Check max_installment_months constraint
                if max_months is not None and opt.number_of_payments > max_months:
                    continue

                # Build installment payments using fast date.fromisoformat
                inst_payments: List[PaymentItemDTO] = []
                first_dt = date.fromisoformat(opt.first_payment_date)
                freq_days = opt.payment_frequency_days or 30

                for i in range(opt.number_of_payments):
                    p_dt = first_dt + timedelta(days=i * freq_days)
                    inst_payments.append(PaymentItemDTO(date=p_dt.isoformat(), amount=opt.payment_amount))

                final_pay_date = inst_payments[-1].date

                # Simulate installment plan
                sim_inst = self.simulator.simulate(profile, events, req_date, payments=inst_payments)
                if sim_inst.is_safe:
                    candidate_plans.append(
                        CandidatePlan(
                            method=RecommendedPaymentMethod.INSTALLMENTS,
                            payments=inst_payments,
                            total_payable_amount=opt.total_payable_amount,
                            payment_option_id=opt.payment_option_id,
                            first_payment_date=opt.first_payment_date,
                            completion_date=final_pay_date,
                            number_of_payments=opt.number_of_payments,
                            is_safe=True,
                        )
                    )
                else:
                    # Test with spending changes
                    ok, changes = self.optimizer.find_spending_changes(profile, events, req_date, payments=inst_payments)
                    if ok:
                        candidate_plans.append(
                            CandidatePlan(
                                method=RecommendedPaymentMethod.INSTALLMENTS,
                                payments=inst_payments,
                                total_payable_amount=opt.total_payable_amount,
                                payment_option_id=opt.payment_option_id,
                                first_payment_date=opt.first_payment_date,
                                completion_date=final_pay_date,
                                number_of_payments=opt.number_of_payments,
                                spending_changes=changes,
                                is_safe=True,
                            )
                        )

        # Candidate 4: Wait (Affordable Later)
        # earliest_full_date was already validated to be safe by calculate_earliest_date_for_full_payment
        if (
            "full_payment" in user_methods and
            earliest_full_date and
            earliest_full_date > req_date
        ):
            p_wait = [PaymentItemDTO(date=earliest_full_date, amount=req_amt)]
            candidate_plans.append(
                CandidatePlan(
                    method=RecommendedPaymentMethod.WAIT,
                    payments=p_wait,
                    total_payable_amount=req_amt,
                    first_payment_date=earliest_full_date,
                    completion_date=earliest_full_date,
                    number_of_payments=1,
                    is_safe=True,
                )
            )

        # Apply 6-level tie-breaking hierarchy to pick best plan
        chosen_plan: Optional[CandidatePlan] = None
        if candidate_plans:
            def plan_rank_key(plan: CandidatePlan):
                # 1. Complete by desired_completion_date (direct ISO string comparison)
                on_time = 0 if plan.completion_date <= comp_date else 1
                # 2. No spending changes needed (0 if none, 1 if changes needed)
                no_changes = 0 if not plan.spending_changes else 1
                # 3. Minimize total payable amount
                total_cost = plan.total_payable_amount
                # 4. Start payment earlier (first_payment_date string sort)
                start_dt = plan.first_payment_date
                # 5. Fewer payments
                num_pay = plan.number_of_payments
                # 6. Lowest payment_option_id tie-breaker
                opt_id = plan.payment_option_id or "zzzz"
                return (on_time, no_changes, total_cost, start_dt, num_pay, opt_id)

            candidate_plans.sort(key=plan_rank_key)
            chosen_plan = candidate_plans[0]

        # Formulate DecisionResultDTO with unified status & date resolution
        if chosen_plan:
            rec_method = chosen_plan.method
            
            if rec_method == RecommendedPaymentMethod.FULL_PAYMENT and not chosen_plan.spending_changes and chosen_plan.first_payment_date == req_date:
                status = AffordabilityStatus.AFFORDABLE_NOW
                earliest_full_out = req_date
            elif rec_method == RecommendedPaymentMethod.WAIT:
                status = AffordabilityStatus.AFFORDABLE_LATER
                earliest_full_out = chosen_plan.first_payment_date
            elif chosen_plan.spending_changes or rec_method in [RecommendedPaymentMethod.PARTIAL_PAYMENT, RecommendedPaymentMethod.INSTALLMENTS]:
                status = AffordabilityStatus.AFFORDABLE_WITH_PLAN
                earliest_full_out = earliest_full_date or ""
            else:
                status = AffordabilityStatus.AFFORDABLE_LATER
                earliest_full_out = earliest_full_date or ""

            # Format payment plan - Protect zero values against rstrip empty string
            plan_str = "|".join(
                f"{p.date}:{f'{p.amount:.2f}'.rstrip('0').rstrip('.') or '0'}"
                for p in chosen_plan.payments
            )
            
            # Format spending changes
            if chosen_plan.spending_changes:
                changes_str = "|".join(sc.to_contract_str() for sc in chosen_plan.spending_changes)
            else:
                changes_str = "none"

        else:
            # Fallback: Not recommended
            rec_method = RecommendedPaymentMethod.NOT_RECOMMENDED
            status = AffordabilityStatus.NOT_AFFORDABLE
            plan_str = "none"
            changes_str = "none"
            # In ground truth (and SPEC), when a request is not affordable, earliest date is empty
            earliest_full_out = ""

        return DecisionResultDTO(
            request_id=request.request_id,
            amount_safe_to_pay=safe_to_pay,
            affordability_status=status,
            recommended_payment_method=rec_method,
            payment_plan=plan_str,
            earliest_date_for_full_payment=earliest_full_out,
            spending_changes_needed=changes_str,
            decision_explanation="",  # Generated in Agent layer
        )
