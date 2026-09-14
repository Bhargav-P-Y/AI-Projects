# code/engine/spending_optimizer.py
"""
Spending Optimizer Engine.
Finds optimal, permitted spending adjustments (stoppable / reducible recurring expenses)
to resolve liquidity deficits when an eligible plan breaches the minimum balance floor.
Adheres strictly to the max 3 actions, mutual exclusivity, and protected categories rules.
Filters strictly for future events, uses canonical ev.base_id, and uses generalized combinatorial search.
"""

from typing import List, Optional, Set, Tuple, Dict
from itertools import combinations, product
from code.data.models import (
    UserFinancialProfile,
    FinancialEvent,
    SpendingChangeDTO,
    PaymentItemDTO,
    EventFlexibility,
)
from code.engine.cash_flow import CashFlowSimulator


class SpendingOptimizer:
    def __init__(self, cash_flow_simulator: CashFlowSimulator):
        self.simulator = cash_flow_simulator

    def find_spending_changes(
        self,
        profile: UserFinancialProfile,
        events: List[FinancialEvent],
        request_date_str: str,
        payments: List[PaymentItemDTO],
        max_changes: int = 3,
    ) -> Tuple[bool, List[SpendingChangeDTO]]:
        """
        Attempts to eliminate the deficit by stopping or reducing flexible recurring expenses.
        Returns (is_successful, list_of_spending_changes).
        Searches combinations of 1 up to max_changes actions sorted by financial relief.
        """
        # Baseline simulation with proposed payments
        sim = self.simulator.simulate(profile, events, request_date_str, payments=payments)
        if sim.is_safe:
            return True, []

        protect_cats = profile.expense_categories_to_protect
        stop_cats = profile.expense_categories_user_is_willing_to_stop
        reduce_cats = profile.expense_categories_user_is_willing_to_reduce

        # 1. Identify eligible FUTURE debit events that are stoppable or reducible
        # Filter strictly for events in the current forecast window (settlement_date >= request_date_str)
        eligible_candidates: Dict[str, Tuple[FinancialEvent, List[SpendingChangeDTO], float]] = {}

        for ev in events:
            if not ev.settlement_date or ev.settlement_date < request_date_str:
                continue
            if ev.direction != "debit":
                continue
            if ev.category in protect_cats:
                continue

            base_id = ev.base_id
            if base_id in eligible_candidates:
                continue

            # Check stoppable
            can_stop = (
                ev.flexibility in [EventFlexibility.STOPPABLE, EventFlexibility.REDUCIBLE_OR_STOPPABLE]
                and ev.category in stop_cats
            )
            # Check reducible
            can_reduce = (
                ev.flexibility in [EventFlexibility.REDUCIBLE, EventFlexibility.REDUCIBLE_OR_STOPPABLE]
                and ev.category in reduce_cats
                and ev.minimum_allowed_amount is not None
                and ev.minimum_allowed_amount < ev.amount
            )

            if not (can_stop or can_reduce):
                continue

            # Precompute candidate actions for this event (guaranteeing mutual exclusivity)
            actions: List[SpendingChangeDTO] = []
            max_relief = 0.0

            if can_stop:
                actions.append(SpendingChangeDTO(action_type="stop", event_id=base_id))
                max_relief = max(max_relief, ev.amount)

            if can_reduce and ev.minimum_allowed_amount is not None:
                actions.append(
                    SpendingChangeDTO(
                        action_type="reduce_to",
                        event_id=base_id,
                        new_amount=ev.minimum_allowed_amount,
                    )
                )
                max_relief = max(max_relief, ev.amount - ev.minimum_allowed_amount)

            eligible_candidates[base_id] = (ev, actions, max_relief)

        if not eligible_candidates:
            return False, []

        # 2. Sort candidate base events by financial relief (descending)
        sorted_candidates = sorted(
            eligible_candidates.values(),
            key=lambda item: item[2],
            reverse=True,
        )

        # 3. Generalized Combinatorial Search over k in [1 .. max_changes]
        # Using combinations(candidates, k) and product(*actions) strictly enforces:
        # - Exactly k distinct base events modified
        # - Mutual exclusivity (at most 1 action per base event)
        # - Greedily evaluates highest financial relief cuts first
        for k in range(1, min(max_changes, len(sorted_candidates)) + 1):
            for candidate_group in combinations(sorted_candidates, k):
                action_lists = [c[1] for c in candidate_group]
                for action_tuple in product(*action_lists):
                    action_list = list(action_tuple)
                    res = self.simulator.simulate(
                        profile,
                        events,
                        request_date_str,
                        payments=payments,
                        spending_changes=action_list,
                    )
                    if res.is_safe:
                        return True, action_list

        return False, []
