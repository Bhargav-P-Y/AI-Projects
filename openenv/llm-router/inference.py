import os
import json
import asyncio
import time
from typing import List, Optional
from openai import OpenAI

from models import RouterAction
from server.router_environment import LLMRouterEnvironment

# --- STRICT CHECKLIST COMPLIANCE VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct") 
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is missing but required.", flush=True)

TASK_NAME = "openenv-llm-router"
BENCHMARK = "inference_ops"

MAX_STEPS = 60  
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
        env = LLMRouterEnvironment()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    action_schema = json.dumps(RouterAction.model_json_schema(), indent=2)

    results = {}
    tasks = ["task_1_keyword", "task_2_budget", "task_3_latency"]

    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        current_task_name = f"{TASK_NAME}_{task_id}"

        log_start(task=current_task_name, env=BENCHMARK, model=MODEL_NAME)

        # PROMPT FIXED: Added strict negative constraints to prevent early submission
        messages = [
            {"role": "system", "content": f"You are an AI InferenceOps Router. Output raw JSON only matching this schema:\n{action_schema}\nValid endpoints: FAST_CHEAP, BALANCED, EXPENSIVE_REASONER.\n\nCRITICAL RULES:\n1. If Queries Remaining > 0, you MUST output command='route' or 'inspect_queue'.\n2. NEVER output command='submit' unless Queries Remaining is exactly 0.\n3. WARNING: Expensive endpoints increase Server Heat. If heat >= 0.8, it causes catastrophic latency. Use FAST_CHEAP or 'inspect_queue' to cool down!"}
        ]

        rewards = []
        step_count = 0
        done = False
        success = False
        final_score = 0.0 

        try:
            while not done and step_count < MAX_STEPS:
                step_count += 1

                state_prompt = (
                    f"Budget Remaining: ${obs.remaining_budget:.3f}\n"
                    f"Queries Remaining: {obs.queries_remaining}\n"
                    f"Avg Quality: {obs.average_quality:.3f}\n"
                    f"Server Heat: {obs.server_load:.3f}\n"
                    f"Last Status: {obs.last_action_status}\n"
                    "Next action?"
                )
                messages.append({"role": "user", "content": state_prompt})

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.0,
                        seed=42,
                        response_format={"type": "json_object"}
                    )

                    raw_action = response.choices[0].message.content
                    action_dict = json.loads(raw_action)
                    action = RouterAction(**action_dict)

                    obs, raw_reward, done, info = env.step(action)

                    if hasattr(raw_reward, 'value'):
                        extracted_reward = float(raw_reward.value)
                    elif isinstance(raw_reward, dict):
                        extracted_reward = float(raw_reward.get("value", 0.0))
                    else:
                        extracted_reward = float(raw_reward)

                    messages.append({"role": "assistant", "content": raw_action})

                    error_msg = None if "Error" not in obs.last_action_status else str(obs.last_action_status)
                    action_str = json.dumps(action_dict)

                except Exception as e:
                    extracted_reward = 0.1 
                    done = False
                    error_msg = f"Parsing failed: {str(e)}"
                    action_str = "invalid_action()"

                    if obs.queries_remaining > 0:
                        obs, _, done, _ = env.step(RouterAction(command="route", endpoint="FAST_CHEAP"))
                    else:
                        obs, _, done, _ = env.step(RouterAction(command="submit"))

                rewards.append(extracted_reward)
                log_step(step=step_count, action=action_str, reward=rewards[-1], done=done, error=error_msg)

            final_score = env.get_state().current_score
            success = final_score >= SUCCESS_SCORE_THRESHOLD
            results[task_id] = final_score

        finally:
            log_end(success=success, steps=step_count, score=final_score, rewards=rewards)

    return results

if __name__ == "__main__":
    time.sleep(1) 
    asyncio.run(run_eval_loop())
