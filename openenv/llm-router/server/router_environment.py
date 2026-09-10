from typing import Tuple, List, Dict
from models import RouterAction, RouterObservation, RouterReward, RouterState
from server.graders import grade_task_1_keyword, grade_task_2_budget, grade_task_3_latency

def _clamp_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))

ENDPOINTS = {
    "FAST_CHEAP": {"cost": 0.01, "quality_cap": 0.6, "heat": -0.2},
    "BALANCED": {"cost": 0.05, "quality_cap": 0.85, "heat": 0.15},
    "EXPENSIVE_REASONER": {"cost": 0.20, "quality_cap": 1.0, "heat": 0.4}
}

def generate_queue(task_id: str) -> List[Dict]:
    if task_id == "task_1_keyword":
        return [{"id": f"q_{i}", "complexity": 0.2} for i in range(10)]
    elif task_id == "task_2_budget":
        return [{"id": f"q_{i}", "complexity": 0.5 if i % 2 == 0 else 0.8} for i in range(20)]
    elif task_id == "task_3_latency":
        return [{"id": f"q_{i}", "complexity": 0.9 if i % 5 == 0 else 0.4} for i in range(50)]
    return []

class LLMRouterEnvironment:
    def __init__(self):
        self.state = RouterState()
        self.max_steps = 100
        self.current_step = 0

    def reset(self, task_id: str = "task_1_keyword") -> RouterObservation:
        budget = 10.0 if task_id == "task_1_keyword" else (1.0 if task_id == "task_2_budget" else 5.0)
        self.state = RouterState(
            current_task_id=task_id,
            budget=budget,
            queue=generate_queue(task_id)
        )
        self.current_step = 0
        return self._get_observation("Environment initialized.")

    def _get_observation(self, status: str) -> RouterObservation:
        avg_q = self.state.total_quality / max(1, self.state.queries_routed)
        return RouterObservation(
            last_action_status=status,
            remaining_budget=round(self.state.budget, 3),
            queries_remaining=len(self.state.queue),
            average_quality=round(avg_q, 3),
            server_load=round(self.state.server_load, 3),
            queue_preview=self.state.queue[:3] if self.state.queue else None
        )

    def step(self, action: RouterAction) -> Tuple[RouterObservation, RouterReward, bool, dict]:
        self.current_step += 1
        done = False
        status = ""
        reward_val = 0.0
        reason = ""

        if action.command == "submit":
            done = True
            status = "Submission Evaluated."

        elif action.command in ["check_budget", "inspect_queue"]:
            self.state.server_load = max(0.0, self.state.server_load - 0.3)
            status = f"{action.command} executed. Server cooling."
            reward_val = _clamp_score(0.15)
            reason = "Meta-action executed."

        elif action.command == "route":
            if not self.state.queue:
                status, reward_val, reason = "Error: Queue empty.", 0.01, "Empty queue."
            elif action.endpoint not in ENDPOINTS:
                status, reward_val, reason = "Error: Invalid endpoint.", 0.01, "Invalid endpoint."
            else:
                query = self.state.queue.pop(0)
                ep_data = ENDPOINTS[action.endpoint]

                self.state.budget -= ep_data["cost"]
                self.state.server_load = max(0.0, min(1.0, self.state.server_load + ep_data["heat"]))
                quality_achieved = min(ep_data["quality_cap"], 1.0 - max(0, query["complexity"] - ep_data["quality_cap"]))

                if self.state.server_load >= 0.8:
                    quality_achieved *= 0.5
                    self.state.latency_penalty += 0.05
                    throttle_warning = " THROTTLED!"
                else:
                    throttle_warning = ""

                self.state.queries_routed += 1
                self.state.total_quality += quality_achieved
                status = f"Routed to {action.endpoint}. Quality: {quality_achieved:.2f}. Load: {self.state.server_load:.2f}.{throttle_warning}"

                if self.state.budget < 0:
                    reward_val, reason, done = 0.01, "Bankrupt!", True
                else:
                    raw_step_reward = (quality_achieved / ep_data["cost"]) * 0.005
                    if self.state.server_load >= 0.8:
                        raw_step_reward -= 0.1
                    reward_val = _clamp_score(raw_step_reward)
                    reason = "Successful route."

        if self.current_step >= self.max_steps and not done:
            done, reward_val, reason, status = True, 0.01, "Timeout.", "Environment timeout."

        # THE FIX: If the episode terminates, force the final grader to run so current_score is NEVER 0.0
        if done:
            if self.state.current_task_id == "task_1_keyword":
                final_score, grade_reason = grade_task_1_keyword(self.state)
            elif self.state.current_task_id == "task_2_budget":
                final_score, grade_reason = grade_task_2_budget(self.state)
            else:
                final_score, grade_reason = grade_task_3_latency(self.state)

            self.state.current_score = final_score
            reward_val = final_score
            reason = f"Episode Concluded. {grade_reason}"

        return self._get_observation(status), RouterReward(value=_clamp_score(reward_val), reason=reason), done, {}

    def get_state(self) -> RouterState:
        return self.state
