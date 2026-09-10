import os
import json
import asyncio
from typing import List, Optional
from openai import OpenAI # FIX: Strictly use standard synchronous OpenAI client

from models import DataCuratorAction
from server.env import DataCuratorEnvironment

# --- STRICT CHECKLIST COMPLIANCE VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct") 
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = "openenv-alignment-data-curator"
BENCHMARK = "data_curator"

MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.8 

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ").replace("\r", "") if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "") 
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_eval_loop(env=None):
    if env is None:
        env = DataCuratorEnvironment()

    # FIX: Use synchronous client as per sample
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    action_schema = json.dumps(DataCuratorAction.model_json_schema(), indent=2)

    results = {}

    tasks = ["task_1_bias", "task_2_format", "task_3_pii"]

    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        current_task_name = f"{TASK_NAME}_{task_id}"

        log_start(task=current_task_name, env=BENCHMARK, model=MODEL_NAME)

        messages = [
            {"role": "system", "content": f"You are an expert Data Alignment Engineer. Output raw JSON only, matching this schema:\n{action_schema}"}
        ]

        rewards = []
        step_count = 0
        done = False
        success = False
        final_score = 0.0 

        try:
            while not done and step_count < MAX_STEPS:
                step_count += 1

                messages.append({
                    "role": "user", 
                    "content": f"Objective: {obs.task_objective}\nTerminal Output: {obs.terminal_output}\nLast Status: {obs.last_command_status}\nDataset Head: {obs.dataset_head}\nWhat is your next action?"
                })

                try:
                    # FIX: Synchronous call
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.0,
                        seed=42,
                        response_format={"type": "json_object"}
                    )

                    raw_action = response.choices[0].message.content
                    action_dict = json.loads(raw_action)
                    action = DataCuratorAction(**action_dict)

                    obs, raw_reward, done, info = env.step(action)

                    if hasattr(raw_reward, 'value'):
                        extracted_reward = float(raw_reward.value)
                    elif isinstance(raw_reward, dict):
                        extracted_reward = float(raw_reward.get("value", 0.0))
                    else:
                        extracted_reward = float(raw_reward)

                    messages.append({"role": "assistant", "content": raw_action})

                    error_msg = None if obs.last_command_status == "Success" else str(obs.terminal_output)
                    action_str = json.dumps(action_dict)

                except Exception as e:
                    extracted_reward = -0.1
                    done = False
                    error_msg = f"Parsing failed: {str(e)}"
                    action_str = "invalid_action()"

                rewards.append(extracted_reward)
                log_step(step=step_count, action=action_str, reward=rewards[-1], done=done, error=error_msg)

            final_score = env.get_state().current_score
            success = final_score >= SUCCESS_SCORE_THRESHOLD
            results[task_id] = final_score

        finally:
            log_end(success=success, steps=step_count, score=final_score, rewards=rewards)

    return results

if __name__ == "__main__":
    asyncio.run(run_eval_loop())
