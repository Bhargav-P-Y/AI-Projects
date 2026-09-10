import os
import shutil
import subprocess
import sys
import ast 
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DataCuratorAction, DataCuratorObservation, DataCuratorReward, DataCuratorState
from server.graders import grade_task_1_bias, grade_task_2_format, grade_task_3_pii

WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspace")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

class DataCuratorEnvironment:
    def __init__(self):
        self.state = DataCuratorState()
        self.max_steps = 15
        self.current_step = 0

    def reset(self, task_id: str = "task_1_bias") -> DataCuratorObservation:
        self.state = DataCuratorState(current_task_id=task_id)
        self.current_step = 0

        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR)
        shutil.copytree(TEMPLATES_DIR, WORKSPACE_DIR)

        if task_id == "task_1_bias":
            objective = "EASY: Debug the naive 'in' check in filter_logic.py using regex word boundaries to stop over-censorship."
        elif task_id == "task_2_format":
            objective = "MEDIUM: Fix the formatting logic in pipeline.py to output strict Llama 3 special tokens."
        elif task_id == "task_3_pii":
            objective = "HARD: Implement regex in pipeline.py to redact phone numbers to [REDACTED] and drop duplicate rows."
        else:
            objective = "Unknown Task"

        return DataCuratorObservation(
            task_objective=objective,
            last_command_status="Success",
            terminal_output="Environment initialized. Workspace reset with broken pipeline files.",
            dataset_head=None
        )

    def step(self, action: DataCuratorAction) -> Tuple[DataCuratorObservation, DataCuratorReward, bool, dict]:
        self.current_step += 1
        done = False
        obs_status = "Success"
        obs_output = ""
        dataset_head = None
        reward_val = 0.1 
        reward_reason = "Standard step."

        try:
            if action.command == "read_file":
                filepath = os.path.join(WORKSPACE_DIR, action.filepath)
                with open(filepath, 'r') as f:
                    obs_output = f.read()

            elif action.command == "list_directory":
                obs_output = "\n".join(os.listdir(WORKSPACE_DIR))

            elif action.command == "search_and_replace":
                if action.filepath.endswith(('.jsonl', '.csv', '.json', '.txt')):
                    obs_status = "Error"
                    obs_output = "Action blocked."
                    reward_val = -0.1
                    reward_reason = "Attempted to cheat by editing output artifact directly."
                else:
                    filepath = os.path.join(WORKSPACE_DIR, action.filepath)
                    with open(filepath, 'r') as f:
                        content = f.read()

                    if action.old_text in content:
                        new_content = content.replace(action.old_text, action.new_text)
                        with open(filepath, 'w') as f:
                            f.write(new_content)

                        obs_output = f"Successfully replaced text in {action.filepath}."
                        reward_val = 0.2
                        reward_reason = "Successfully modified code."

                        if action.filepath.endswith('.py'):
                            try:
                                subprocess.run(
                                    ["autopep8", "--in-place", "--select=E1,E2,E3,W1,W2,W3", filepath], 
                                    capture_output=True, 
                                    check=False
                                )
                                with open(filepath, 'r') as f:
                                    formatted_content = f.read()
                                ast.parse(formatted_content)
                            except SyntaxError as e:
                                obs_status = "Error"
                                obs_output = f"SyntaxError: {e.msg} at line {e.lineno}."
                                reward_val = -0.05
                                reward_reason = "Invalid Python syntax."
                            except FileNotFoundError:
                                pass
                    else:
                        obs_status = "Error"
                        obs_output = f"old_text not found in {action.filepath}."
                        reward_val = -0.05
                        reward_reason = "Failed search_and_replace."

            elif action.command == "execute_pipeline":
                result = subprocess.run(
                    ["python", "pipeline.py"],
                    cwd=WORKSPACE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                obs_output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                if result.returncode == 0:
                    reward_val = 0.2
                    reward_reason = "Pipeline compiled and ran successfully."
                    out_path = os.path.join(WORKSPACE_DIR, "processed_dataset.jsonl")
                    if os.path.exists(out_path):
                        with open(out_path, 'r') as f:
                            dataset_head = "".join(f.readlines()[:2])
                else:
                    obs_status = "Error"
                    self.state.pipeline_crashes += 1
                    reward_val = -0.1
                    reward_reason = "Pipeline crashed."

            elif action.command == "submit":
                done = True
                if self.state.current_task_id == "task_1_bias":
                    final_score, grade_reason = grade_task_1_bias(WORKSPACE_DIR)
                elif self.state.current_task_id == "task_2_format":
                    final_score, grade_reason = grade_task_2_format(WORKSPACE_DIR)
                elif self.state.current_task_id == "task_3_pii":
                    final_score, grade_reason = grade_task_3_pii(WORKSPACE_DIR)
                else:
                    final_score, grade_reason = 0.1, "Unknown Task ID submitted." 

                obs_output = f"Submission Evaluated.\nResult: {grade_reason}\nFinal Score: {final_score}"
                reward_val = final_score
                reward_reason = f"Episode concluded. Grader evaluation: {grade_reason}"
                self.state.current_score = final_score

        except subprocess.TimeoutExpired:
            obs_status = "Error"
            obs_output = "Execution timed out! Infinite loop detected."
            reward_val = -0.3
            reward_reason = "Severe Penalty: Infinite loop timeout."
        except Exception as e:
            obs_status = "Error"
            obs_output = str(e)
            reward_val = -0.1
            reward_reason = "Action failed due to an exception."

        # --- THE FIX: AUTO-GRADE ON TIMEOUT ---
        if self.current_step >= self.max_steps and not done:
            done = True
            if self.state.current_task_id == "task_1_bias":
                final_score, grade_reason = grade_task_1_bias(WORKSPACE_DIR)
            elif self.state.current_task_id == "task_2_format":
                final_score, grade_reason = grade_task_2_format(WORKSPACE_DIR)
            elif self.state.current_task_id == "task_3_pii":
                final_score, grade_reason = grade_task_3_pii(WORKSPACE_DIR)
            else:
                final_score, grade_reason = 0.1, "Unknown Task ID." 

            self.state.current_score = final_score
            reward_val = final_score
            obs_output += f"\nMax steps reached. Auto-grading... Final Score: {final_score}"
            reward_reason = f"Max steps reached. Auto-grade: {grade_reason}"

        obs = DataCuratorObservation(
            task_objective=self.state.current_task_id,
            last_command_status=obs_status,
            terminal_output=obs_output,
            dataset_head=dataset_head
        )
        reward = DataCuratorReward(value=reward_val, reason=reward_reason)

        return obs, reward, done, {}

    def get_state(self) -> DataCuratorState:
        return self.state
