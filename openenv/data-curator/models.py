from typing import Literal, Optional, Dict
from pydantic import BaseModel, Field

class DataCuratorAction(BaseModel):
    """The tools available to the AI Alignment Engineer."""
    command: Literal[
        "read_file", 
        "list_directory", 
        "search_and_replace", 
        "execute_pipeline",
        "submit"
    ] = Field(..., description="The command to execute.")
    filepath: Optional[str] = Field(None, description="Target file for reading/writing.")
    old_text: Optional[str] = Field(None, description="Exact text to replace (for search_and_replace).")
    new_text: Optional[str] = Field(None, description="Text to insert (for search_and_replace).")

class DataCuratorObservation(BaseModel):
    """The feedback loop provided to the agent after every step."""
    task_objective: str = Field(..., description="The current alignment task to solve.")
    last_command_status: str = Field(..., description="'Success' or 'Error'.")
    terminal_output: str = Field(..., description="Stdout/Stderr from the last executed command.")
    dataset_head: Optional[str] = Field(None, description="Preview of the first 2 rows of the output dataset, if compilation succeeded.")

class DataCuratorReward(BaseModel):
    """The structured reward signal."""
    value: float = Field(..., description="Fractional reward (0.0 to 1.0).")
    reason: str = Field(..., description="Reason for the given reward.")

class DataCuratorState(BaseModel):
    """The internal state of the environment."""
    current_task_id: str = Field(default="task_1_format")
    milestones_achieved: Dict[str, bool] = Field(default_factory=dict)
    current_score: float = Field(default=0.0)
    pipeline_crashes: int = Field(default=0, description="Number of times the pipeline failed to compile.")
