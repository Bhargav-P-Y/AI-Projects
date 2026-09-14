# code/engine/cash_flow.py
"""
90-Day Daily Continuous Liquidity Simulator.
Enforces the core safety invariant:
  Balance(t) >= minimum_balance_to_keep, for all t in [0, 90d].
Calculates amount_safe_to_pay and earliest_date_for_full_payment deterministically.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from code.data.models import (
    UserFinancialProfile,
    FinancialEvent,
    PaymentItemDTO,
    SpendingChangeDTO,
    SimulationResultDTO,
)
from code.data.fx_converter import FXConverter


class CashFlowSimulator:
    def __init__(self, fx_converter: Optional[FXConverter] = None):
        self.fx = fx_converter or FXConverter()

    def simulate(
        self,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        request_date_str: str,
        payments: Optional[List[PaymentItemDTO]] = None,
        spending_changes: Optional[List[SpendingChangeDTO]] = None,
        forecast_days: int = 90,
    ) -> SimulationResultDTO:
        """
        Executes a day-by-day cash flow simulation over the 90-day window.
        Returns SimulationResultDTO with is_safe, min_balance, deficit, and daily trajectory.
        """
        req_dt = datetime.strptime(request_date_str, "%Y-%m-%d").date()
        end_dt = req_dt + timedelta(days=forecast_days)
        end_dt_str = end_dt.strftime("%Y-%m-%d")
        home_curr = profile.home_currency
        min_bal_floor = profile.minimum_balance_to_keep

        # Map spending changes
        stopped_events: Set[str] = set()
        reduced_events: Dict[str, float] = {}
        if spending_changes:
            for sc in spending_changes:
                if sc.action_type == "stop":
                    stopped_events.add(sc.event_id)
                elif sc.action_type == "reduce_to" and sc.new_amount is not None:
                    reduced_events[sc.event_id] = sc.new_amount

        # Map proposed payments by date
        plan_payments_by_date: Dict[str, float] = {}
        if payments:
            for p in payments:
                plan_payments_by_date[p.date] = plan_payments_by_date.get(p.date, 0.0) + p.amount

        # Aggregate daily net cash flows
        daily_inflows: Dict[str, float] = {}
        daily_outflows: Dict[str, float] = {}

        for ev in events:
            # Only consider events within the forecast period
            if not ev.settlement_date:
                continue
            if ev.settlement_date < request_date_str or ev.settlement_date > end_dt_str:
                continue

            date_str = ev.settlement_date

            # Rule: Ignore cancelled, failed, or unrealized/non-cash events
            if ev.status in ["cancelled", "failed", "unrealized"] or ev.direction == "non_cash":
                continue

            # Check spending changes using canonical base_id matching (no prefix collisions)
            if ev.base_id in stopped_events or ev.event_id in stopped_events:
                continue

            amt = ev.amount
            if ev.base_id in reduced_events:
                amt = reduced_events[ev.base_id]
            elif ev.event_id in reduced_events:
                amt = reduced_events[ev.event_id]

            # Convert to user's home currency
            amt_home = self.fx.convert(amt, ev.currency, home_curr, date_str)

            if ev.direction == "debit":
                # Reserve all scheduled and pending debits
                daily_outflows[date_str] = daily_outflows.get(date_str, 0.0) + amt_home
            elif ev.direction == "credit":
                # Rule: Only count confirmed salary or settled credits; ignore unconfirmed pending credits
                if ev.status in ["settled", "scheduled"] or ev.category == "salary":
                    daily_inflows[date_str] = daily_inflows.get(date_str, 0.0) + amt_home

        # Run day-by-day continuous simulation using precomputed date strings
        current_bal = profile.current_available_balance
        min_projected_bal = current_bal
        trajectory: Dict[str, float] = {}

        curr_dt = req_dt
        for _ in range(forecast_days + 1):
            d_str = curr_dt.strftime("%Y-%m-%d")
            
            # Apply daily cash flows: inflows clear before outflows and proposed payments
            inflow = daily_inflows.get(d_str, 0.0)
            outflow = daily_outflows.get(d_str, 0.0)
            proposed_pay = plan_payments_by_date.get(d_str, 0.0)

            current_bal += inflow
            current_bal -= (outflow + proposed_pay)
            # Guard against IEEE-754 micro-float precision errors
            current_bal = round(current_bal, 4)

            trajectory[d_str] = current_bal
            if current_bal < min_projected_bal:
                min_projected_bal = current_bal

            curr_dt += timedelta(days=1)

        deficit = max(0.0, round(min_bal_floor - min_projected_bal, 2))
        is_safe = (min_projected_bal >= min_bal_floor - 1e-5)

        return SimulationResultDTO(
            is_safe=is_safe,
            minimum_projected_balance=min_projected_bal,
            balance_deficit_below_floor=deficit,
            daily_balance_trajectory=trajectory,
        )

    def calculate_amount_safe_to_pay(
        self,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        request_date_str: str,
        requested_amount: float,
        forecast_days: int = 90,
    ) -> float:
        """
        Calculates the maximum amount safe to pay today (request_date) before optional
        spending changes, subject to 0 <= amount_safe_to_pay <= requested_amount.
        """
        # Run baseline simulation without any request payment
        base_res = self.simulate(profile, events, request_date_str, payments=None, forecast_days=forecast_days)
        
        # Max headroom above the minimum balance floor across the entire 90 days
        headroom = base_res.minimum_projected_balance - profile.minimum_balance_to_keep
        safe_amt = max(0.0, min(headroom, requested_amount))
        return round(safe_amt, 2)

    def calculate_earliest_date_for_full_payment(
        self,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        request_date_str: str,
        requested_amount: float,
        forecast_days: int = 90,
    ) -> Optional[str]:
        """
        Finds the first date within the 90-day forecast where paying the full requested_amount
        as a single payment satisfies the safety check without optional spending changes.
        Uses Suffix Minima (Dynamic Programming) in a single backward pass over the baseline
        trajectory to eliminate nested simulations completely.
        """
        base_res = self.simulate(profile, events, request_date_str, payments=None, forecast_days=forecast_days)
        
        # Trajectory dates were inserted sequentially by day, so list(keys) preserves chronological order at zero cost
        dates_list = list(base_res.daily_balance_trajectory.keys())
        if not dates_list:
            return None
            
        n = len(dates_list)
        balances = [base_res.daily_balance_trajectory[d] for d in dates_list]
        min_floor = profile.minimum_balance_to_keep
        
        # Compute Suffix Minima: suff_min[i] = min(balances[i], balances[i+1], ..., balances[n-1])
        suff_min = [0.0] * n
        suff_min[-1] = balances[-1]
        for i in range(n - 2, -1, -1):
            suff_min[i] = min(balances[i], suff_min[i + 1])
            
        # Single forward check:
        # 1. On all days t < k, baseline balance must remain >= min_floor.
        # 2. On all days t >= k, balance after paying requested_amount must remain >= min_floor.
        prefix_safe = True
        for k in range(n):
            if k > 0 and balances[k - 1] < min_floor - 1e-5:
                prefix_safe = False
                break
                
            if prefix_safe and (suff_min[k] - requested_amount >= min_floor - 1e-5):
                return dates_list[k]
                
        return None
