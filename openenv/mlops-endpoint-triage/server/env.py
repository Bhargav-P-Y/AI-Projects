import os
import json
import tempfile
import shutil
import subprocess
from typing import Any, Dict, Tuple
from models import MLOpsAction, MLOpsObservation, MLOpsReward, MLOpsState

class MLOpsEnvironment:
    """The core OpenEnv implementation for the MLOps Triage simulation."""

    def __init__(self):
        self._state = MLOpsState()
        self.base_workspace = "/app/workspace" if os.path.exists("/app") else os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.base_workspace, exist_ok=True)
        self.current_workspace = None

    def _generate_task_files(self, task_id: int):
        """Generates mock files to simulate real-world MLOps failures."""
        if task_id == 1:
            with open(os.path.join(self.current_workspace, "config.json"), "w") as f:
                f.write('{\n  "model_type": "llama",\n  "vocal_size": 32000,\n  "hidden_size": 4096\n}')
            with open(os.path.join(self.current_workspace, "test_deploy.py"), "w") as f:
                f.write('import json\nwith open("config.json") as f:\n    cfg = json.load(f)\nif "vocab_size" not in cfg:\n    raise KeyError("Missing required key: vocab_size. Found: vocal_size")\nprint("DEPLOYMENT SUCCESSFUL")')

        elif task_id == 2:
            with open(os.path.join(self.current_workspace, "requirements.txt"), "w") as f:
                f.write('torch==2.1.0\ntransformers==4.0.0\n')
            with open(os.path.join(self.current_workspace, "test_deploy.py"), "w") as f:
                f.write('with open("requirements.txt") as f:\n    reqs = f.read()\nif "transformers==4.0.0" in reqs:\n    raise RuntimeError("transformers 4.0.0 is incompatible. Requires >= 4.30.0")\nprint("DEPLOYMENT SUCCESSFUL")')

        elif task_id == 3:
            with open(os.path.join(self.current_workspace, "model_loader.py"), "w") as f:
                f.write('import os\ndef load_model():\n    device = "cuda"\n    return f"Model loaded to {device}"\n')
            with open(os.path.join(self.current_workspace, "test_deploy.py"), "w") as f:
                f.write('import model_loader\nres = model_loader.load_model()\nif "cuda" in res:\n    raise RuntimeError("CUDA error: no GPU available (Server is CPU-only)")\nprint("DEPLOYMENT SUCCESSFUL")')

        elif task_id == 4:
            with open(os.path.join(self.current_workspace, "model.safetensors.index.json"), "w") as f:
                f.write('{\n  "weight_map": {\n    "attention.wq": "model-00001.safetensors"\n  }\n}')
            with open(os.path.join(self.current_workspace, "test_deploy.py"), "w") as f:
                f.write('import json\nwith open("model.safetensors.index.json") as f:\n    index = f.read()\nif "attention.wq" in index and "attention.q_proj" not in index:\n    raise ValueError("State dict mismatch. Expected attention.q_proj")\nprint("DEPLOYMENT SUCCESSFUL")')

        elif task_id == 5:
            with open(os.path.join(self.current_workspace, "tokenizer_config.json"), "w") as f:
                f.write('{\n  "bos_token": "<s>",\n  "eos_token": "</s>"\n}')
            with open(os.path.join(self.current_workspace, "test_deploy.py"), "w") as f:
                f.write('import json\nwith open("tokenizer_config.json") as f:\n    cfg = json.load(f)\nif "chat_template" not in cfg:\n    raise ValueError("Jinja2 chat_template is missing.")\nprint("DEPLOYMENT SUCCESSFUL")')

    def reset(self, task_id: int = 1) -> MLOpsObservation:
        self._state = MLOpsState(task_id=task_id)
        if self.current_workspace and os.path.exists(self.current_workspace):
            shutil.rmtree(self.current_workspace, ignore_errors=True)
        self.current_workspace = tempfile.mkdtemp(dir=self.base_workspace)
        self._generate_task_files(task_id)

        return MLOpsObservation(
            task_objective=f"Task {task_id}: Fix the broken deployment. Use 'test_deploy' to check work.",
            system_instructions="Output ONLY valid JSON matching the MLOpsAction schema.",
            terminal_output=f"Workspace initialized for Task {task_id}.",
            last_command_status="Success"
        )

    def _execute_action(self, action: MLOpsAction) -> Tuple[str, str, bool]:
        if not self.current_workspace:
            return "Error: Workspace not initialized.", "Error", False
        if action.command in ["write_file", "search_and_replace"] and action.filepath == "test_deploy.py":
            return "Permission Denied: Cannot modify grader script.", "Corrupted", True

        try:
            if action.command == "list_directory":
                return f"Files: {', '.join(os.listdir(self.current_workspace))}", "Success", False
            elif action.command == "read_file":
                with open(os.path.join(self.current_workspace, action.filepath), "r") as f:
                    return f.read(), "Success", False
            elif action.command == "write_file":
                with open(os.path.join(self.current_workspace, action.filepath), "w") as f:
                    f.write(action.new_text_block)
                return f"Successfully wrote to {action.filepath}", "Success", False
            elif action.command == "search_and_replace":
                with open(os.path.join(self.current_workspace, action.filepath), "r") as f:
                    content = f.read()
                if action.old_text_block not in content:
                    return f"Error: '{action.old_text_block}' not found.", "Error", False
                with open(os.path.join(self.current_workspace, action.filepath), "w") as f:
                    f.write(content.replace(action.old_text_block, action.new_text_block))
                return f"Successfully replaced text in {action.filepath}", "Success", False
            elif action.command == "test_deploy":
                result = subprocess.run(["python", "test_deploy.py"], cwd=self.current_workspace, capture_output=True, text=True, timeout=5)
                return (result.stdout + result.stderr).strip(), ("Success" if result.returncode == 0 else "Error"), False
            elif action.command == "submit":
                return "Task submitted.", "Success", True
        except Exception as e:
            return f"Execution Error: {str(e)}", "Error", False

    def _grade_artifact(self) -> float:
        """Deterministic Grader: Evaluates the physical files to produce a 0.0 to 1.0 score."""

        try:
            task_id = self._state.task_id
            if task_id == 1:
                with open(os.path.join(self.current_workspace, "config.json")) as f:
                    data = json.load(f)
                return 0.9 if "vocab_size" in data and "vocal_size" not in data else 0.1

            elif task_id == 2:
                with open(os.path.join(self.current_workspace, "requirements.txt")) as f:
                    reqs = f.read()
                return 0.9 if "transformers==4.0.0" not in reqs else 0.1

            elif task_id == 3:
                with open(os.path.join(self.current_workspace, "model_loader.py")) as f:
                    code = f.read()
                return 0.9 if "cuda" not in code else 0.1

            elif task_id == 4:
                with open(os.path.join(self.current_workspace, "model.safetensors.index.json")) as f:
                    data = json.load(f)
                return 0.9 if "attention.q_proj" in data.get("weight_map", {}) else 0.1

            elif task_id == 5:
                with open(os.path.join(self.current_workspace, "tokenizer_config.json")) as f:
                    data = json.load(f)
                return 0.9 if "chat_template" in data else 0.1

        except Exception:
            return 0.1 # Return 0.1 on file parsing failure as well

    def step(self, action: MLOpsAction) -> Tuple[MLOpsObservation, MLOpsReward, bool, Dict[str, Any]]:
        terminal_out, status, is_terminal_action = self._execute_action(action)

        # --- PHASE 4: DENSE REWARD SHAPING ---
        reward_val = 0.0
        reason = f"Executed {action.command}."
        done = is_terminal_action

        # Penalty 1: Destructive Behavior (-0.5)
        if status == "Corrupted":
            self._state.is_corrupted = True
            reward_val = -0.5
            reason = "PENALTY: Destructive action (Tried to cheat grader)."
            done = True

        # Penalty 2: Syntax or Execution Errors (-0.1)
        elif status == "Error":
            reward_val = -0.1
            reason = "PENALTY: Command caused an error."

        # Milestone: Reading a file (+0.1)
        if action.command == "read_file" and not self._state.milestones_achieved.get("read_file"):
            self._state.milestones_achieved["read_file"] = True
            reward_val += 0.1
            reason = "MILESTONE: Successfully read a file."

        # Milestone: Successful file modification (+0.2)
        if action.command in ["write_file", "search_and_replace"] and status == "Success" and not self._state.milestones_achieved.get("edited_file"):
            self._state.milestones_achieved["edited_file"] = True
            reward_val += 0.2
            reason = "MILESTONE: Successfully edited a file."

        # Milestone: Successful test_deploy ends the episode (+0.5 base)
        if action.command == "test_deploy" and "DEPLOYMENT SUCCESSFUL" in terminal_out:
            done = True

        # --- PHASE 4: DETERMINISTIC GRADING ---
        if done:
            final_score = self._grade_artifact()
            self._state.current_score = final_score
            # If they fixed it, they get the remainder of the reward pie up to 1.0
            if final_score == 1.0 and status != "Corrupted":
                reward_val += 0.5
            reason += f" | EPISODE COMPLETE. Final Artifact Score: {final_score}"

        obs = MLOpsObservation(
            task_objective=f"Task {self._state.task_id}: Fix the deployment.",
            system_instructions="Output ONLY valid JSON matching the MLOpsAction schema.",
            terminal_output=terminal_out,
            last_command_status=status
        )
        reward = MLOpsReward(value=round(reward_val, 2), reason=reason)

        return obs, reward, done, self._state.model_dump()

    def state(self) -> MLOpsState:
        return self._state
