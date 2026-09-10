from fastapi import FastAPI, Request
from models import RouterAction
from server.router_environment import LLMRouterEnvironment
import uvicorn
import traceback

# Initialize the core app and our global environment state
app = FastAPI(title="OpenEnv LLM Router")
global_env = LLMRouterEnvironment()

# --- 1. MANDATORY OPENENV SPEC ENDPOINTS (PROXY-PROOFED) ---
@app.post("/reset")
@app.post("/spaces/Bhargav-P-Y/openenv-llm-router/reset")
@app.post("/api/reset")
async def api_reset(request: Request):
    """OpenEnv standard reset endpoint. Initializes a task."""
    try:
        payload = await request.json()
    except:
        payload = {}

    # Check for task_id in query params first, then payload, then default
    task_id = request.query_params.get("task_id", payload.get("task_id", "task_1_keyword"))

    obs = global_env.reset(task_id=task_id)
    return obs

@app.post("/step")
@app.post("/spaces/Bhargav-P-Y/openenv-llm-router/step")
@app.post("/api/step")
def api_step(action: RouterAction):
    """OpenEnv standard step endpoint. Takes an action, returns trajectory."""
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
@app.get("/spaces/Bhargav-P-Y/openenv-llm-router/state")
@app.get("/api/state")
def api_state():
    """OpenEnv standard state endpoint. Returns current score and status."""
    return global_env.get_state()

# --- 2. COMPETITION DIAGNOSTIC ENDPOINTS ---
@app.get("/")
@app.get("/spaces/Bhargav-P-Y/openenv-llm-router")
def ping_root():
    """Phase 1 Automated Ping Test Gatekeeper."""
    return {"status": "200 OK", "environment": "openenv-llm-router"}

# --- OPENENV VALIDATOR REQUIREMENT ---
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
