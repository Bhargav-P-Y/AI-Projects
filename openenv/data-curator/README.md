---
title: OpenEnv Data Curator Alignment
emoji: 🧹
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# 🧹 OpenEnv: Data Curator Alignment

## Environment Description & Motivation
**Real-World Utility (Fills a real gap for the RL community):** Preparing high-quality datasets for Supervised Fine-Tuning (SFT) and alignment is one of the most critical, labor-intensive tasks in modern AI development. Human data alignment engineers spend countless hours scrubbing Personally Identifiable Information (PII), resolving tokenizer formatting clashes, and debugging overly aggressive heuristic toxicity filters that cause model refusal behavior. 

**Creativity & Novelty:** Moving beyond generic SWE-bench tasks, this OpenEnv environment simulates the bleeding-edge daily workflow of a Data Alignment Engineer. The autonomous agent is dropped into a workspace with a broken data pipeline and a raw `.jsonl` dataset. To succeed, the agent must read the Python pipeline code, identify logical flaws (like leaking PII or over-censoring safe medical terms), and surgically edit the codebase using an AST-validated search-and-replace tool to produce a clean, compliant dataset.

## Action and Observation Spaces

### Action Space (`DataCuratorAction`)
The agent interacts with the environment by emitting JSON objects matching this Pydantic schema:
* `command` (str): The tool to use (`read_file`, `list_directory`, `search_and_replace`, `execute_pipeline`, `submit`).
* `filepath` (str, optional): The target file (e.g., `pipeline.py`).
* `old_text` (str, optional): The exact text block to be replaced (requires exact whitespace matching).
* `new_text` (str, optional): The replacement text block.

> **Clever Mechanic & Environment Design:** The environment features AST-Validated **Anti-Cheating Guardrails**. It actively blocks attempts to manually edit the output `processed_dataset.jsonl` file. The agent *must* fix the underlying Python pipeline to succeed. To combat hallucinated text appends, the `search_and_replace` tool pipes edits through `autopep8` and `ast.parse`. Severe `SyntaxError`s result in immediate negative reward shaping, forcing the agent to write syntactically valid Python.


### Observation Space (`DataCuratorObservation`)
* `task_objective` (str): The specific curation goal for the current task.
* `last_command_status` (str): "Success" or "Error" feedback from the previous action.
* `terminal_output` (str): Standard output, stderr tracebacks, or AST syntax errors.
* `dataset_head` (str, optional): A preview of the first two lines of the compiled dataset after a successful pipeline run.

## Task Descriptions & Expected Difficulty

This environment features a mathematically verified difficulty progression based on *Agentic Complexity* (the combination of logical reasoning and multi-line mechanical execution limits of LLMs).

* **EASY (`task_1_bias`): Heuristic Bias Filter Debugging**
  * *Objective:* Fix an over-aggressive toxicity filter that drops safe medical terms ("painkiller") while correctly dropping actual threats.
  * *Difficulty:* **Easy**. Requires finding a naive substring `in` check and upgrading it to a single-line regex word boundary `re.search(rf"\b{word}\b")`.

* **MEDIUM (`task_2_format`): Llama 3 Format Clash**
  * *Objective:* Refactor a dataset formatter to output strictly compliant Llama 3 special tokens (`<|start_header_id|>`).
  * *Difficulty:* **Medium**. Requires multi-line string manipulation and basic awareness of Python indentation.

* **HARD (`task_3_pii`): PII Leakage and Deduplication**
  * *Objective:* Implement regex logic to redact phone numbers to `[REDACTED]` and implement a stateful deduplication filter to drop duplicate rows.
  * *Difficulty:* **Hard**. Requires complex, multi-line refactoring, managing scope (`seen = set()`), and injecting logic without triggering variable reference errors or syntax crashes.

## Setup and Usage Instructions

### 1. Environment Variables
The environment relies on the standard OpenAI Python client. To run the baseline inference script, ensure the following API credentials are set in your environment as per the competition guidelines:

```bash
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_access_token_here"
```

### 2. Local Containerized Execution
This environment is fully containerized and runs as a non-root user (UID 1000) for strict Hugging Face Spaces compliance.

**Build the Docker Image:**
```bash
docker build -t openenv-data-curator .
```

**Run the Container:**
```bash
docker run -p 7860:7860 openenv-data-curator
```

### 3. Validating the Environment
Ensure you have the `openenv-core` package installed, then run the official pre-submission validator in the root directory:
```bash
openenv validate
```

## Baseline Scores

The baseline agent was evaluated using **Llama-3.3-70B-Instruct** (via HF Router) using the strictly compliant `inference.py` script included in the root directory.

The agent's performance mathematically validates the difficulty progression—excelling at isolated logical fixes but struggling significantly with the mechanical syntax required for complex multi-line pipeline refactoring.

* **Task 1 (Easy):** 0.9 (Success in 4 steps)
* **Task 2 (Medium):** 0.1 (Failed due to persistent AST SyntaxErrors)
* **Task 3 (Hard):** 0.1 (Failed due to variable scope reference errors)

**Overall Baseline Score:** 0.366
