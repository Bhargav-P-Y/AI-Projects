# code/engine/cadence.py
"""
Cadence & Recurrence Engine.
Identifies historical recurring debit and credit patterns (salary, rent, utilities, subscriptions,
and cadenced variable expenses like groceries and transport) from settled financial events
and projects them forward conservatively across the 90-day forecast window.
Enforces salary deduplication, employment termination detection, base event ID linkage,
and exact cadence interval projection.
"""

import calendar
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from code.data.models import FinancialEvent, UserFinancialProfile, EventFlexibility


CONTRACTUAL_MONTHLY_CATEGORIES: Set[str] = {
    "rent",
    "utilities",
    "subscription",
    "debt_repayment",
    "insurance",
    "cloud_storage",
    "streaming",
    "education",
    "music_subscription",
    "delivery_membership",
    "gym",
    "housing",
    "family_support",
    "healthcare",
}

CADENCED_VARIABLE_CATEGORIES: Set[str] = {
    "groceries",
    "transport",
    "dining",
}


class CadenceEngine:
    def __init__(self):
        pass

    def project_recurring_events(
        self,
        events: List[FinancialEvent],
        profile: UserFinancialProfile,
        request_date_str: str,
        confirmed_salary_override: Optional[float] = None,
        salary_effective_date: Optional[str] = None,
        contract_ended: bool = False,
        forecast_days: int = 90,
    ) -> List[FinancialEvent]:
        """
        Projects verified recurring income and contractual expenses forward across the 90-day window.
        """
        req_d = date.fromisoformat(request_date_str)
        end_d = req_d + timedelta(days=forecast_days)
        end_date_str = end_d.isoformat()

        future_events: List[FinancialEvent] = []
        past_events: List[FinancialEvent] = []

        for e in events:
            if not e.settlement_date:
                continue
            if e.settlement_date >= request_date_str:
                future_events.append(e)
            else:
                past_events.append(e)

        projected: List[FinancialEvent] = list(future_events)

        # 1. Salary Processing & Single Primary Payroll Stream Enforcement
        has_final_payroll = any(
            "final" in e.description.lower()
            for e in past_events
            if e.category == "salary" and e.direction == "credit"
        )
        employment_active = not contract_ended and not has_final_payroll

        existing_future_salary_months = {
            (int(e.settlement_date[:4]), int(e.settlement_date[5:7]))
            for e in future_events
            if e.category == "salary" and e.direction == "credit"
        }

        if employment_active:
            salary_amount = None
            salary_day = None
            salary_curr = profile.home_currency

            # Priority 1: Confirmed salary override from messages/evidence
            if confirmed_salary_override is not None and confirmed_salary_override > 0:
                salary_amount = confirmed_salary_override

            # Priority 2: Upcoming scheduled salary in future_events
            for e in future_events:
                if e.category == "salary" and e.direction == "credit" and e.amount and e.amount > 0:
                    if salary_amount is None:
                        salary_amount = e.amount
                    salary_day = int(e.settlement_date[8:10])
                    salary_curr = e.currency
                    break

            # Priority 3: Historical settled salaries (using mode of regular payments)
            past_salaries = [
                e for e in past_events
                if e.category == "salary" and e.direction == "credit" and e.status == "settled" and e.amount and e.amount > 0
            ]
            if past_salaries:
                past_salaries.sort(key=lambda x: x.settlement_date)
                last_sal = past_salaries[-1]
                if salary_amount is None:
                    sal_amts = [s.amount for s in past_salaries]
                    salary_amount = Counter(sal_amts).most_common(1)[0][0]
                if salary_day is None:
                    salary_day = int(last_sal.settlement_date[8:10])
                salary_curr = last_sal.currency

            if salary_amount and salary_day:
                curr_y = req_d.year
                curr_m = req_d.month
                for _ in range(4):  # Project up to 4 consecutive future months
                    max_d = calendar.monthrange(curr_y, curr_m)[1]
                    t_day = min(salary_day, max_d)
                    p_date = f"{curr_y:04d}-{curr_m:02d}-{t_day:02d}"

                    if request_date_str <= p_date <= end_date_str:
                        if (curr_y, curr_m) not in existing_future_salary_months:
                            projected.append(
                                FinancialEvent(
                                    event_id=f"proj_salary_{curr_y}_{curr_m}",
                                    user_id=profile.user_id,
                                    event_type="income",
                                    description="Projected payroll credit",
                                    category="salary",
                                    direction="credit",
                                    amount=salary_amount,
                                    currency=salary_curr,
                                    event_date=p_date,
                                    settlement_date=p_date,
                                    status="scheduled",
                                    flexibility=EventFlexibility.FIXED,
                                    base_event_id="salary",
                                )
                            )

                    # Advance one month
                    if curr_m == 12:
                        curr_y += 1
                        curr_m = 1
                    else:
                        curr_m += 1

        # 2. Contractual Monthly Bills (rent, utilities, loans, subscriptions, insurance)
        monthly_groups: Dict[Tuple[str, str], List[FinancialEvent]] = {}
        for e in past_events:
            if e.status == "settled" and e.direction == "debit":
                if e.category in CONTRACTUAL_MONTHLY_CATEGORIES:
                    monthly_groups.setdefault((e.category, e.description), []).append(e)

        for (cat, desc), ev_list in monthly_groups.items():
            ev_list.sort(key=lambda x: x.settlement_date)
            last_ev = ev_list[-1]
            last_d = int(last_ev.settlement_date[8:10])

            existing_months = {
                (int(e.settlement_date[:4]), int(e.settlement_date[5:7]))
                for e in future_events
                if e.category == cat and e.description == desc
            }

            curr_y = int(last_ev.settlement_date[:4])
            curr_m = int(last_ev.settlement_date[5:7])

            while True:
                if curr_m == 12:
                    curr_y += 1
                    curr_m = 1
                else:
                    curr_m += 1

                max_d = calendar.monthrange(curr_y, curr_m)[1]
                t_day = min(last_d, max_d)
                p_date = f"{curr_y:04d}-{curr_m:02d}-{t_day:02d}"

                if p_date > end_date_str:
                    break

                if p_date >= request_date_str:
                    if (curr_y, curr_m) not in existing_months:
                        projected.append(
                            FinancialEvent(
                                event_id=f"proj_{last_ev.event_id}_{p_date.replace('-', '')}",
                                user_id=profile.user_id,
                                event_type=last_ev.event_type,
                                description=last_ev.description,
                                category=last_ev.category,
                                direction=last_ev.direction,
                                amount=last_ev.amount,
                                currency=last_ev.currency,
                                event_date=p_date,
                                settlement_date=p_date,
                                status="scheduled",
                                flexibility=last_ev.flexibility,
                                minimum_allowed_amount=last_ev.minimum_allowed_amount,
                                base_event_id=last_ev.event_id,
                            )
                        )

        # 3. Cadenced High-Frequency Variable Expenses (groceries, transport, dining)
        for cat in CADENCED_VARIABLE_CATEGORIES:
            cat_evs = [
                e for e in past_events
                if e.category == cat and e.status == "settled" and e.direction == "debit"
            ]
            if len(cat_evs) >= 2:
                cat_evs.sort(key=lambda x: x.settlement_date)
                dates = [date.fromisoformat(e.settlement_date) for e in cat_evs]
                diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                common_diff, _ = Counter(diffs).most_common(1)[0]

                if common_diff in [5, 7, 10, 14, 21, 28]:
                    last_dt = dates[-1]
                    last_ev = cat_evs[-1]
                    # Median amount for stability against one-off grocery/transport spikes
                    amounts = [e.amount for e in cat_evs]
                    median_amt = sorted(amounts)[len(amounts) // 2]

                    curr_dt = last_dt + timedelta(days=common_diff)
                    while curr_dt <= end_d:
                        p_date = curr_dt.isoformat()
                        if p_date >= request_date_str:
                            projected.append(
                                FinancialEvent(
                                    event_id=f"proj_{cat}_{p_date.replace('-', '')}",
                                    user_id=profile.user_id,
                                    event_type="expense",
                                    description=f"Projected {cat}",
                                    category=cat,
                                    direction="debit",
                                    amount=round(median_amt, 2),
                                    currency=last_ev.currency,
                                    event_date=p_date,
                                    settlement_date=p_date,
                                    status="scheduled",
                                    flexibility=last_ev.flexibility,
                                    minimum_allowed_amount=last_ev.minimum_allowed_amount,
                                    base_event_id=last_ev.event_id,
                                )
                            )
                        curr_dt += timedelta(days=common_diff)

        return projected
