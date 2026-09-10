---
title: OpenEnv InferenceOps LLM Router
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 🚦 OpenEnv: InferenceOps LLM Router

## Environment Description & Motivation

**Real-World Utility (Fills a real gap for the RL community):** In modern production AI, routing every user query to a frontier model (like GPT-4 or Claude 3.5 Sonnet) is financially catastrophic and introduces unnecessary latency. Conversely, routing everything to small, cheap models results in unacceptable quality drops for complex reasoning tasks. The holy grail of modern AI infrastructure—**InferenceOps**—is dynamic API routing: intelligently classifying incoming queries and routing them to the optimal endpoint to maximize quality while strictly adhering to budget and latency constraints. 

**Creativity & Novelty:** Moving beyond standard coding or web-navigation tasks, this OpenEnv environment simulates a live production server economy. An autonomous agent acts as the API Router. It must inspect a backlog of incoming queries (each with a hidden semantic complexity score), monitor a degrading financial budget, and dynamically route each query to one of three simulated endpoints: `FAST_CHEAP`, `BALANCED`, or `EXPENSIVE_REASONER`. By training agents on this environment, the RL community can develop infrastructure-management agents capable of autonomously lowering AI cloud bills without sacrificing application quality.

## Action and Observation Spaces

### Action Space (`RouterAction`)
The agent manages the queue by emitting JSON objects matching this strict Pydantic schema:
* `command` (str): The tool to use (`route`, `check_budget`, `inspect_queue`, `submit`).
* `endpoint` (str, optional): The target LLM for routing (`FAST_CHEAP`, `BALANCED`, `EXPENSIVE_REASONER`). Required if the command is `route`.
* `query_id` (str, optional): The specific ID of the query being processed.

> **Clever Mechanic & Environment Design:** The environment features a **Simulated API Economy**. Agents cannot simply brute-force the best model. Every route action immediately deducts from a strict, task-specific budget. If an agent routes a low-complexity keyword search to the `EXPENSIVE_REASONER`, it wastes critical funds, risking bankruptcy before the queue is cleared. 

### Observation Space (`RouterObservation`)
Instead of sparse end-of-episode signals, the agent receives high-fidelity telemetry after every step:
* `last_action_status` (str): Detailed feedback, including exactly how much the last route cost and the simulated quality score achieved.
* `remaining_budget` (float): The exact dollar amount left in the server's budget.
* `queries_remaining` (int): The size of the current backlog.
* `average_quality` (float): A rolling average of response quality, forcing the agent to monitor its overall performance.
* `queue_preview` (list, optional): Metadata about upcoming queries (accessed via `inspect_queue`).

## Task Descriptions & Expected Difficulty

This environment features three distinct tiers of difficulty, challenging the agent's ability to balance short-term routing logic with long-term resource management.

* **EASY (`task_1_keyword`): Heuristic Routing**
  * *Objective:* Route 10 low-complexity queries with an effectively unlimited budget ($10.00). 
  * *Difficulty:* **Easy**. Tests if the agent understands the basic `route` and `submit` action grammar and can successfully clear the queue while maintaining a basic baseline of quality.

* **MEDIUM (`task_2_budget`): Constraint-Aware Optimization**
  * *Objective:* Route 20 queries of mixed complexity with a severely restricted budget ($1.00).
  * *Difficulty:* **Medium**. Requires state-tracking and resource conservation over a longer horizon. The agent is scored heavily on *budget efficiency*. If it spams the `EXPENSIVE_REASONER`, it will instantly go bankrupt. It must actively use `FAST_CHEAP` for easy queries to survive.

* **HARD (`task_3_latency`): Dynamic Load Balancing**
  * *Objective:* Clear a massive backlog of 50 queries with dynamic, wildly fluctuating complexities and a hidden budget constraint.
  * *Difficulty:* **Hard**. Challenges frontier models to maintain logical consistency and avoid context-window degradation over 50+ continuous routing decisions. To achieve an "Exceptional" grade (score > 0.8), the agent must perfectly map the inferred complexity of the queue to the exact quality cap of the endpoints.

## Setup and Usage Instructions

### 1. Environment Variables
The environment relies on the standard OpenAI Python client. To run the baseline inference script, ensure the following API credentials are set in your environment as per the competition guidelines:

```bash
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_access_token_here"
```

### 2. Local Containerized Execution
This environment is fully containerized and strictly designed to run as a non-root user (UID 1000) to ensure seamless Hugging Face Spaces deployment.

**Build the Docker Image:**
```bash
docker build -t openenv-llm-router .
```

**Run the Container:**
```bash
docker run -p 7860:7860 openenv-llm-router
```

### 3. Validating the Environment
Ensure you have the `openenv-core` package installed, then run the official pre-submission validator in the root directory:
```bash
openenv validate
```

## Baseline Scores

The baseline agent was evaluated using **Llama-3.3-70B-Instruct** using the strictly compliant `inference.py` script. The agent successfully demonstrated advanced long-horizon planning by utilizing `inspect_queue` as a cooling mechanic to manage the environment's thermal throttling traps.

* **Task 1 (Easy): 0.600 (Partial Success)**
    * **Result:** Completed in **11 steps**.
    * **Reasoning:** The agent successfully cleared the queue but prioritized **Budget Frugality** over **Quality Maxing**. By routing all queries to the `FAST_CHEAP` endpoint, the quality was capped at **0.6**, demonstrating the grader's ability to distinguish between basic completion and high-performance routing.
* **Task 2 (Medium): 0.900 (Success)**
    * **Result:** Completed in **21 steps**.
    * **Reasoning:** The agent demonstrated **Constraint-Aware Optimization**. Recognizing the severe $1.00 budget restriction, it successfully routed all 20 queries sequentially to the cheap endpoint to avoid bankruptcy, earning a high score for financial efficiency.
* **Task 3 (Hard): 0.100 (Failed)**
    * **Result:** System Failure at **Step 44**.
    * **Reasoning:** The agent identified the thermal trap and successfully deployed a brilliant interleaving strategy (routing `BALANCED` twice, then calling `inspect_queue` to cool the server). However, maintaining this 40+ step trajectory caused a **Context Window Explosion**. The sheer length of the required ReAct history exhausted the agent's token API limit before it could clear the 50-query queue, resulting in the failsafe score of 0.100. This proves Task 3 is a genuine frontier challenge for context-heavy agentic loops.

**Overall Baseline Score: 0.533**
