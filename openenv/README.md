# 🌐 OpenEnv Suite: Autonomous RL & Agentic Evaluation Environments

A production collection of specialized **OpenEnv Reinforcement Learning Environments** designed for training and benchmarking autonomous LLM agents against real-world AI infrastructure and engineering workflows.

All environments strictly adhere to the **OpenEnv Pydantic Quad-Spec** (`Action`, `Observation`, `Reward`, `State`), run as unprivileged non-root users (`UID 1000`) for containerized Hugging Face Spaces deployment, and feature mathematically calibrated difficulty tiers.

---

## 🏛️ Environment Portfolio Overview

| Environment | Primary Domain | Core Challenge | Key Mechanistic Feature | Verified Baseline Model |
| :--- | :--- | :--- | :--- | :--- |
| [**`llm-router/`**](./llm-router) | **InferenceOps & Budget Routing** | Dynamic queue dispatching across tiered LLM endpoints under strict financial budgets | Simulated API economy, thermal throttling cooling loops, context explosion traps | `Llama-3.3-70B-Instruct` |
| [**`data-curator/`**](./data-curator) | **Data Alignment & SFT Curation** | Autonomous codebase surgery on broken data preparation pipelines | AST-validated search-and-replace, anti-cheating syntax filters, PII regex | `Llama-3.3-70B-Instruct` |
| [**`mlops-endpoint-triage/`**](./mlops-endpoint-triage) | **MLOps & Infrastructure Debugging** | Triage and repair failed containerized LLM deployment endpoints | Lightweight mock tracebacks (<150MB RAM), Jinja2 chat templates, Safetensor key repair | `Qwen/Qwen2.5-72B-Instruct` |

---

## 🚦 1. InferenceOps LLM Router (`openenv/llm-router`)
- **Real-World Problem**: Dispatching every request to frontier models (e.g., GPT-4 / Claude 3.5 Sonnet) is economically unsustainable. Routing everything to small models leads to severe reasoning degradation. InferenceOps routers dynamically classify complexity and dispatch queries to the Pareto-optimal endpoint.
- **Action Grammar**: `RouterAction` emitting strict JSON commands (`route`, `check_budget`, `inspect_queue`, `submit`) targeting `FAST_CHEAP`, `BALANCED`, or `EXPENSIVE_REASONER`.
- **Environment Dynamics**: Continuous dollar budget tracking, rolling quality scoring, and thermal throttling traps requiring cooling actions.
- **Baseline Evaluation**: `Llama-3.3-70B` achieved **0.900** on constraint optimization (Task 2) and surfaced the context-window explosion challenge in long-horizon 50+ step routing (Task 3).

## 🧹 2. Data Curator Alignment (`openenv/data-curator`)
- **Real-World Problem**: Curating datasets for Supervised Fine-Tuning (SFT) and RLHF requires rigorous PII scrubbing, tokenization boundary alignment, and heuristic bias filter auditing.
- **Action Grammar**: `DataCuratorAction` issuing AST-validated edits (`read_file`, `list_directory`, `search_and_replace`, `execute_pipeline`, `submit`).
- **Safety & Guardrails**: Direct manipulation of output datasets is blocked. Code changes must pass `autopep8` and `ast.parse` syntax checks, penalizing malformed code with shaped negative rewards.
- **Task Progression**: Heuristic bias filter regex -> Llama-3 special token `<|start_header_id|>` format clash -> Stateful PII redaction and deduplication.

## 🛠️ 3. MLOps Endpoint Triage (`openenv/mlops-endpoint-triage`)
- **Real-World Problem**: Production LLM endpoints frequently crash due to missing chat templates, CUDA out-of-memory errors on CPU hosts, Safetensor key naming divergence, and version pin conflicts.
- **Action Grammar**: SWE-Agent style bounded toolset (`read_file`, `list_directory`, `search_and_replace`, `write_file`, `test_deploy`, `submit`).
- **Resource Constraints**: Engineered specifically to run within `2 vCPU` and `8GB Memory` by mocking heavy PyTorch weight allocations while generating authentic stack traces.
- **Deterministic Grader**: State validator inspecting deployment exit codes and runtime configs. Evaluated with `Qwen 2.5-72B`, demonstrating rapid single-turn dependency resolution (Task 2 score: **0.9**).

---

## 🚀 Quickstart & Container Execution

Each environment contains a dedicated `Dockerfile` and `pyproject.toml` managed via `uv`:

```bash
# Navigate to any environment
cd openenv/llm-router

# Build and run locally
docker build -t openenv-llm-router .
docker run -p 7860:7860 openenv-llm-router

# Run evaluation baseline
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_hf_token"
python inference.py
```
