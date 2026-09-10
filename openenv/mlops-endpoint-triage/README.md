---
title: MLOps Endpoint Triage
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 🛠️ OpenEnv: MLOps Endpoint Triage

## Environment Description & Motivation (Real-World Utility)
Deploying Large Language Models to inference endpoints is notoriously fragile. Engineers frequently encounter Out-Of-Memory (OOM) CUDA crashes, missing Jinja2 chat templates, dependency version mismatches, and Safetensor dictionary key errors. 

This environment simulates the **genuine, real-world task** of an MLOps engineer triaging a broken model deployment. Agents are dropped into a secure, sandboxed container with a broken workspace and must use SWE-Agent-style bash commands to read configurations, modify Python/JSON files, and re-run deployment tests until the endpoint boots successfully.

**Infrastructure Note:** To strictly adhere to the `2 vCPU` and `8GB Memory` constraints, this environment avoids loading massive PyTorch weights. Instead, it utilizes lightweight mock parsers in `test_deploy.py` that realistically simulate heavy tracebacks using less than 150MB of RAM, allowing the evaluation script to complete well under the 20-minute limit.

## Action & Observation Spaces (Spec Compliance)

This environment fully implements the OpenEnv Pydantic Quad-Spec (`Action`, `Observation`, `Reward`, `State`).

### Action Space (`MLOpsAction`)
A highly constrained, SWE-Agent-style toolset designed to prevent zero-shot open-weights models from hallucinating unparseable bash commands.
* `command` (Literal): `["read_file", "list_directory", "search_and_replace", "write_file", "test_deploy", "submit"]`
* `filepath` (Optional[str]): Target file.
* `old_text_block` (Optional[str]): Exact text to replace.
* `new_text_block` (Optional[str]): Text to insert.

### Observation Space (`MLOpsObservation`)
* `task_objective` (str): The current bug to fix.
* `system_instructions` (str): Formatting rules injected specifically to keep open LLMs (like Nemotron 3 Super) on track.
* `terminal_output` (str): The `stdout`/`stderr` from the last command or full Python tracebacks.
* `last_command_status` (str): `Success`, `Error`, or `Corrupted`.

## Task Descriptions & Difficulty
Tasks feature a programmatic, deterministic artifact grader that parses the final file structure to award scores between `0.0` and `1.0`. 

**Dense Reward Shaping:** The environment utilizes a dense reward function, granting partial progress (+0.2) for successful file modifications and penalizing clearly undesirable behavior (-0.1) like failed deployment tests or infinite loops.

1. **The JSON Typo (Difficulty: easy):** `config.json` contains a typo (`vocal_size` instead of `vocab_size`). The agent must read the file and fix the JSON.
2. **The Dependency Clash (Difficulty: medium):** `requirements.txt` requests a library version incompatible with the server. The agent must parse the traceback and update the version.
3. **The Hardcoded CUDA (Difficulty: medium):** The model loader script has hardcoded `.cuda()` types, but the HF Space has only CPUs. The agent must rewrite the script to use CPU device mapping.
4. **The Safetensors Mismatch (Difficulty: hard):** The model config expects `attention.q_proj`, but the `.safetensors` index has `attention.wq`. The agent must mutate the index keys.
5. **The Missing Chat Template (Difficulty: hard):** The inference server crashes because `tokenizer_config.json` lacks a Jinja2 `chat_template`. The agent must inject a valid template.

## Setup and Usage Instructions

### 1. Environment Variables
The environment relies on the standard OpenAI Python client and is agnostic to the inference provider. To run the baseline inference script, ensure the following API credentials are set in your environment. 
:
```bash
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_access_token_here"
```

### 2. Baseline Evaluation
The baseline script (`inference.py`) evaluates the environment using **Qwen 2.5 72B** via the official **Hugging Face Router**.

As a frontier-class model, Qwen successfully demonstrates **long-running trajectories**, executing multi-step sequences to diagnose and remediate infrastructure failures. The model successfully completed **Task 2 (Dependency Clash)** in just 3 steps, achieving a perfect score and proving the environment's solvability by high-reasoning agents. 

While Task 1 (JSON Typo) is conceptually "Easy," the baseline results highlight a realistic challenge for LLMs: the **Tool-Use Paradox**. The model correctly identified the bug but fumbled the JSON syntax during replacement. This validates the environment's mechanical stability and ensures that the difficulty curve provides a high ceiling for Phase 2 evaluation, genuinely challenging frontier models like Nemotron 3 Super.

**Reproducible Baseline Scores:**
```json
{
  "task_1": 0.1,
  "task_2": 0.9,
  "task_3": 0.1,
  "task_4": 0.1,
  "task_5": 0.1
}
