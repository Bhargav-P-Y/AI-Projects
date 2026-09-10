from fastapi import FastAPI, Request
from models import MLOpsAction
from server.env import MLOpsEnvironment
import asyncio
import uvicorn
import traceback

# Initialize the core app and our global environment state
app = FastAPI(title="MLOps Triage Environment")
global_env = MLOpsEnvironment()

# --- 1. MANDATORY OPENENV SPEC ENDPOINTS (PROXY-PROOFED) ---
# Multiple decorators ensure the HF Dashboard can reach these endpoints 
# regardless of how its internal proxy formats the URL path.

@app.post("/reset")
@app.post("/spaces/Bhargav-P-Y/mlops-endpoint-triage/reset")
@app.post("/api/reset")
async def api_reset(request: Request):
    """OpenEnv standard reset endpoint. Initializes a task."""
    # Handle empty or JSON payloads gracefully
    payload = await request.json() if await request.body() else {}
    task_id = payload.get("task_id", 1)

    obs = global_env.reset(task_id=task_id)
    return obs

@app.post("/step")
@app.post("/spaces/Bhargav-P-Y/mlops-endpoint-triage/step")
@app.post("/api/step")
def api_step(action: MLOpsAction):
    """OpenEnv standard step endpoint. Takes an action, returns trajectory."""
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
@app.get("/spaces/Bhargav-P-Y/mlops-endpoint-triage/state")
@app.get("/api/state")
def api_state():
    """OpenEnv standard state endpoint. Returns current score and status."""
    return global_env.state()

# --- 2. COMPETITION DIAGNOSTIC ENDPOINTS ---

@app.get("/")
@app.get("/spaces/Bhargav-P-Y/mlops-endpoint-triage")
def ping_root():
    """Phase 1 Automated Ping Test Gatekeeper."""
    return {"status": "200 OK", "environment": "mlops-endpoint-triage"}

@app.get("/tasks")
def get_tasks():
    """Returns the task list AND the Pydantic schema."""
    return {
        "tasks": [
            {"id": 1, "difficulty": "easy", "description": "Fix the JSON config typo."},
            {"id": 2, "difficulty": "medium", "description": "Resolve the requirements.txt version clash."},
            {"id": 3, "difficulty": "medium", "description": "Fix the hardcoded PyTorch .cuda() call."},
            {"id": 4, "difficulty": "hard", "description": "Fix the safetensors key mismatch."},
            {"id": 5, "difficulty": "hard", "description": "Inject the missing Hugging Face Jinja2 chat_template."}
        ],
        "action_schema": MLOpsAction.model_json_schema()
    }

@app.get("/grader")
def get_grader():
    return {"score": global_env.state().current_score}

@app.get("/baseline")
async def trigger_baseline():
    """Triggers the in-memory OpenAI script, bypassing HTTP timeouts."""
    try:
        from inference import run_eval_loop
        results = await run_eval_loop(global_env)
        return {"status": "success", "scores": results}
    except Exception as e:
        return {
            "status": "error", 
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# --- OPENENV VALIDATOR REQUIREMENT ---
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
