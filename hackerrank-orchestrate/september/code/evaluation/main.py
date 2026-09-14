# code/evaluation/main.py
"""
Evaluation & Benchmarking Harness for Buy or Wait.
Runs the complete deterministic decision pipeline against the 25 ground-truth requests
in dataset/sample_requests.csv and calculates exact accuracy across all target fields.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.config import config
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
from code.data.models import RequestItem


def run_sample_benchmark():
    print("================================================================")
    print("RUNNING EVALUATION-DRIVEN BENCHMARK ON dataset/sample_requests.csv")
    print("================================================================")

    # 1. Load ground truth
    sample_df = pd.read_csv(config.dataset_dir / "sample_requests.csv", keep_default_na=False)
    
    # 2. Initialize repositories & engines
    profile_repo = ProfileRepository()
    event_repo = EventRepository()
    options_repo = PaymentOptionRepository()
    msg_repo = MessageRepository()
    evidence_resolver = EvidenceResolver(msg_repo)
    
    cadence_engine = CadenceEngine()
    cash_flow_sim = CashFlowSimulator()
    spending_opt = SpendingOptimizer(cash_flow_sim)
    plan_solver = PlanSolver(cash_flow_sim, spending_opt)

    total_samples = len(sample_df)
    matches = {
        "amount_safe_to_pay": 0,
        "affordability_status": 0,
        "recommended_payment_method": 0,
        "payment_plan": 0,
        "earliest_date_for_full_payment": 0,
        "spending_changes_needed": 0,
    }

    for row in sample_df.itertuples(index=False):
        req_id = str(row.request_id).strip()
        user_id = str(row.user_id).strip()
        req_date = str(row.request_date).strip()
        req_amt = float(row.requested_amount)
        comp_date = str(row.desired_completion_date).strip()
        partial_ok = str(row.allows_partial_payment).strip().lower() in ["true", "1", "t"]

        req_item = RequestItem(
            request_id=req_id,
            user_id=user_id,
            request_date=req_date,
            request_type=str(row.request_type).strip(),
            requested_amount=req_amt,
            desired_completion_date=comp_date,
            allows_partial_payment=partial_ok,
            request_text=str(row.request_text).strip(),
        )

        profile = profile_repo.get_profile(user_id)
        raw_events = event_repo.get_events_for_user(user_id)
        options = options_repo.get_options_for_request(req_id)
        evidence = evidence_resolver.resolve_evidence_for_user(user_id, req_date)

        # Project recurring events forward
        projected_events = cadence_engine.project_recurring_events(
            events=raw_events,
            profile=profile,
            request_date_str=req_date,
            confirmed_salary_override=evidence.confirmed_salary_override,
            salary_effective_date=evidence.salary_effective_date,
            contract_ended=evidence.contract_ended,
        )

        # Solve optimal plan
        decision = plan_solver.solve(req_item, profile, projected_events, options)

        # Compare against ground truth
        gt_safe = float(row.amount_safe_to_pay)
        gt_status = str(row.affordability_status).strip()
        gt_method = str(row.recommended_payment_method).strip()
        gt_plan = str(row.payment_plan).strip()
        gt_earliest = str(row.earliest_date_for_full_payment).strip()
        gt_changes = str(row.spending_changes_needed).strip()

        # Check tolerances
        is_safe_match = abs(decision.amount_safe_to_pay - gt_safe) <= 1.0 or (decision.amount_safe_to_pay == gt_safe)
        is_status_match = decision.affordability_status.value == gt_status
        is_method_match = decision.recommended_payment_method.value == gt_method
        is_plan_match = decision.payment_plan == gt_plan
        is_earliest_match = decision.earliest_date_for_full_payment == gt_earliest
        is_changes_match = decision.spending_changes_needed == gt_changes

        if is_safe_match: matches["amount_safe_to_pay"] += 1
        if is_status_match: matches["affordability_status"] += 1
        if is_method_match: matches["recommended_payment_method"] += 1
        if is_plan_match: matches["payment_plan"] += 1
        if is_earliest_match: matches["earliest_date_for_full_payment"] += 1
        if is_changes_match: matches["spending_changes_needed"] += 1

        print(f"\n[{req_id} - {user_id}] Req: {req_amt} by {comp_date} (Partial: {partial_ok})")
        print(f"  Status:  Pred='{decision.affordability_status.value}' vs GT='{gt_status}' [{'OK' if is_status_match else 'DIFF'}]")
        print(f"  Method:  Pred='{decision.recommended_payment_method.value}' vs GT='{gt_method}' [{'OK' if is_method_match else 'DIFF'}]")
        print(f"  SafeAmt: Pred={decision.amount_safe_to_pay} vs GT={gt_safe} [{'OK' if is_safe_match else 'DIFF'}]")
        print(f"  Earliest: Pred='{decision.earliest_date_for_full_payment}' vs GT='{gt_earliest}' [{'OK' if is_earliest_match else 'DIFF'}]")
        print(f"  Changes: Pred='{decision.spending_changes_needed}' vs GT='{gt_changes}' [{'OK' if is_changes_match else 'DIFF'}]")
        print(f"  Plan:    Pred='{decision.payment_plan}' vs GT='{gt_plan}' [{'OK' if is_plan_match else 'DIFF'}]")

    print("\n================================================================")
    print("BENCHMARK SUMMARY RESULTS:")
    for k, v in matches.items():
        pct = (v / total_samples) * 100
        print(f"  {k:30s}: {v:2d}/{total_samples} ({pct:5.1f}%)")
    print("================================================================")


if __name__ == "__main__":
    run_sample_benchmark()
