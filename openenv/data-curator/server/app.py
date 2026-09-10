from fastapi import FastAPI, Request
from models import DataCuratorAction
from server.env import DataCuratorEnvironment
import uvicorn
import traceback

# Initialize the core app and our global environment state
app = FastAPI(title="Data Curator Alignment Environment")
global_env = DataCuratorEnvironment()

# --- 1. MANDATORY OPENENV SPEC ENDPOINTS (PROXY-PROOFED) ---
@app.post("/reset")
@app.post("/spaces/Bhargav-P-Y/openenv-alignment-data-curator/reset")
@app.post("/api/reset")
async def api_reset(request: Request):
    """OpenEnv standard reset endpoint. Initializes a task."""
    payload = await request.json() if await request.body() else {}
    # Default to the first task if none is provided
    task_id = payload.get("task_id", "task_1_bias")

    obs = global_env.reset(task_id=task_id)
    return obs

@app.post("/step")
@app.post("/spaces/Bhargav-P-Y/openenv-alignment-data-curator/step")
@app.post("/api/step")
def api_step(action: DataCuratorAction):
    """OpenEnv standard step endpoint. Takes an action, returns trajectory."""
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
@app.get("/spaces/Bhargav-P-Y/openenv-alignment-data-curator/state")
@app.get("/api/state")
def api_state():
    """OpenEnv standard state endpoint. Returns current score and status."""
    return global_env.get_state()

# --- 2. COMPETITION DIAGNOSTIC ENDPOINTS ---
@app.get("/")
@app.get("/spaces/Bhargav-P-Y/openenv-alignment-data-curator")
def ping_root():
    """Phase 1 Automated Ping Test Gatekeeper."""
    return {"status": "200 OK", "environment": "openenv-alignment-data-curator"}

@app.get("/baseline")
async def trigger_baseline():
    """Triggers the in-memory OpenAI script."""
    try:
        from inference import run_eval_loop
        results = await run_eval_loop()
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
