from typing import Tuple
from models import RouterState

def _clamp_score(score: float) -> float:
    """Strictly clamps scores between 0.01 and 0.99 to pass Phase 2."""
    return max(0.01, min(0.99, float(score)))

def grade_task_1_keyword(state: RouterState) -> Tuple[float, str]:
    if state.budget < 0:
        return _clamp_score(0.1), "Failed: Bankrupt."
    if len(state.queue) > 0:
        return _clamp_score(0.1), "Failed: Queue not cleared."

    avg_quality = state.total_quality / max(1, state.queries_routed)

    if avg_quality > 0.7:
        reason = "Success: High quality routing achieved."
    else:
        reason = "Partial Success: Queue cleared but quality was suboptimal."

    return _clamp_score(avg_quality), reason

def grade_task_2_budget(state: RouterState) -> Tuple[float, str]:
    if state.budget < 0:
        return _clamp_score(0.1), "Failed: Bankrupt. Poor budget management."
    if len(state.queue) > 0:
        return _clamp_score(0.1), "Failed: Queue not cleared."

    avg_quality = state.total_quality / max(1, state.queries_routed)
    money_spent = 1.0 - state.budget
    efficiency_bonus = 0.0 if money_spent == 0 else (avg_quality / money_spent) * 0.1

    raw_score = avg_quality + efficiency_bonus
    return _clamp_score(raw_score), "Evaluated budget efficiency."

def grade_task_3_latency(state: RouterState) -> Tuple[float, str]:
    if state.budget < 0:
        return _clamp_score(0.1), "Failed: Bankrupt under heavy load."
    if len(state.queue) > 0:
        return _clamp_score(0.1), "Failed: System crashed before clearing queue."

    avg_quality = state.total_quality / max(1, state.queries_routed)
    raw_score = avg_quality - state.latency_penalty

    if raw_score >= 0.8:
        reason = "Exceptional: Perfect dynamic load balancing."
    elif raw_score >= 0.5:
        reason = "Passable: Survived the queue but latency/quality suffered."
    else:
        reason = "Poor: Route logic failed to adapt to query complexity."

    return _clamp_score(raw_score), reason
