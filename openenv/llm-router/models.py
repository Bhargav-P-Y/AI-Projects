from typing import Literal, Optional, Dict, List
from pydantic import BaseModel, Field

class RouterAction(BaseModel):
    command: Literal["route", "check_budget", "inspect_queue", "submit"] = Field(..., description="The action to perform.")
    query_id: Optional[str] = Field(None, description="The ID of the query to route.")
    endpoint: Optional[Literal["FAST_CHEAP", "BALANCED", "EXPENSIVE_REASONER"]] = Field(None, description="The target LLM endpoint.")

class RouterObservation(BaseModel):
    last_action_status: str = Field(..., description="Success or Error message.")
    remaining_budget: float = Field(..., description="The remaining dollar budget.")
    queries_remaining: int = Field(..., description="Number of unrouted queries left.")
    average_quality: float = Field(0.0, description="Rolling average quality.")
    server_load: float = Field(0.0, description="Current server heat/latency risk (0.0 to 1.0).") # NEW
    queue_preview: Optional[List[Dict]] = Field(None, description="Preview of upcoming queries.")

class RouterReward(BaseModel):
    value: float = Field(..., description="Fractional reward.")
    reason: str = Field(..., description="Reason for the given reward.")

class RouterState(BaseModel):
    current_task_id: str = Field(default="task_1_keyword")
    budget: float = Field(default=10.0)
    queue: List[Dict] = Field(default_factory=list)
    queries_routed: int = Field(default=0)
    total_quality: float = Field(default=0.0)
    history: List[Dict] = Field(default_factory=list)
    current_score: float = Field(default=0.0)
    latency_penalty: float = Field(default=0.0)
    server_load: float = Field(default=0.0)
