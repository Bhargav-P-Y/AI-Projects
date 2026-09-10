from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict

class MLOpsAction(BaseModel):
    command: Literal["read_file", "list_directory", "search_and_replace", "write_file", "test_deploy", "submit"] = Field(
        ..., description="The action to execute."
    )
    filepath: Optional[str] = Field(None, description="Path to the target file.")
    old_text_block: Optional[str] = Field(None, description="Exact text block to replace.")
    new_text_block: Optional[str] = Field(None, description="New text to insert.")

class MLOpsObservation(BaseModel):
    task_objective: str = Field(..., description="The current bug to fix.")
    system_instructions: str = Field(..., description="Formatting rules for zero-shot LLMs.")
    terminal_output: str = Field(..., description="stdout/stderr from the last command.")
    last_command_status: str = Field(..., description="Success, Error, Timeout, or Corrupted.")

class MLOpsReward(BaseModel):
    value: float = Field(..., description="Fractional reward (-1.0 to 1.0).")
    reason: str = Field(..., description="Reason for reward allocation.")

class MLOpsState(BaseModel):
    task_id: int = Field(default=1)
    milestones_achieved: Dict[str, bool] = Field(default_factory=dict)
    is_corrupted: bool = Field(default=False)
    current_score: float = Field(default=0.0)
