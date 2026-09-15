# AI/ML Engineering Portfolio | Systems, Foundation Models & Autonomous Agents

[![Author](https://img.shields.io/badge/Author-Bhargav%20P%20Y-blue.svg)](https://github.com/Bhargav-P-Y)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20%7C%20Transformers-yellow)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)]()
[![Pydantic](https://img.shields.io/badge/Schemas-Pydantic%20v2-E92063?logo=pydantic&logoColor=white)]()

An industry-grade portfolio demonstrating full-lifecycle Applied AI, Multi-Agent Systems, Reinforcement Learning Gym Environments, and Production MLOps. Engineered with verified competitive track records across tier-one hackathons (Amazon, Meta x PyTorch x Hugging Face, HackerRank Orchestrate) and strict production standards: containerized non-root runtimes, AST syntax guardrails, Pydantic type safety, and reproducible benchmarks across frontier LLMs.

---

## 🎯 Executive Summary & Competitive Benchmark Track Record

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🏆 Amazon Machine Learning Challenge 2025 — Nationwide Top 12% Finish (SMAPE: 52.44%)                 │
│  Engineered an end-to-end multimodal pricing pipeline fusing dual-stream transformer text embeddings   │
│  (all-MiniLM-L6-v2 + all-mpnet-base-v2) with ResNet50 vision vectors (2048-dim GAP) and leakage-free   │
│  out-of-fold target encoding, optimized via LightGBM regression across thousands of teams.             │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🏆 HackerRank Orchestrate Sept 2026 — Global Rank 197 / Top 1% (30,000+ Global Participants)           │
│  Architected a compound financial decision engine combining O(N) Suffix Minima Dynamic Programming    │
│  for zero-hallucination liquidity forecasting with an 8-worker concurrent ReAct loop and Gemini Flash  │
│  OCR. Evaluated 250 requests in 119s (<$0.01 cost) with 100% data integrity & schema invariant checks. │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🥈 HackerRank Orchestrate Aug 2026 — Global Rank 421 / 1,983 Finalists (~22,000 Global Registrants)    │
│  Architected a 6-stage autonomous multimodal agent for WhatsApp notification triage (notify/digest/    │
│  mute) featuring Gemini Flash OCR, Whisper ASR, 2-stage injection defense, and hybrid BM25 retrieval. │
│  Defended in a 30-minute voice AI Judge architectural interview; Top ~2% of all 22,000 registrants.    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🌐 Meta PyTorch OpenEnv Hackathon (x Scaler & Hugging Face) 2026 — RL Gym Suite                       │
│  Authored three containerized OpenEnv reinforcement learning environments benchmarking LLM agents on   │
│  InferenceOps routing, AST codebase surgery, and cloud deployment triage adhering to the Quad-Spec.    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 💡 Core Engineering Capabilities at a Glance

| Competency | Demonstration in Repository | Key Tooling & Frameworks |
| :--- | :--- | :--- |
| **Autonomous Multi-Agent Systems** | 6-stage sequential agent pipeline (Aug) & Bounded ReAct agent with $O(N)$ Suffix Minima DP (Sept) | Gemini 3.8 Flash, ReAct Loop, BM25, Pydantic |
| **Financial AI & Constraint Solving** | $O(N)$ Suffix Minima DP cash-flow simulator, installment solver, and multi-currency FX conversion | Python, NumPy, Dataclasses, Dynamic Programming |
| **Multimodal Systems & Embeddings** | Dual-stream sentence transformers concatenated with ResNet50 GAP; Gemini Flash receipt OCR & ASR | PyTorch, `sentence-transformers`, `torchvision`, BLIP |
| **RL Environments & Benchmarking** | Standardized gym environments adhering to the Meta OpenEnv Pydantic Quad-Spec (`Action`/`Obs`/`Reward`/`State`) | OpenEnv, Docker (non-root UID 1000), `uv` |
| **Code Surgery & AST Guardrails** | Automated programmatic refactoring with `ast.parse` syntax checks and shaped negative rewards | Python AST, Autopep8, Llama-3.3-70B |
| **Deep Learning from First Principles** | Custom PyTorch pipeline with manual forward/backward loops, L2 weight decay, and XAI feature weights | PyTorch `nn.Module`, SGD, BCELoss |
| **High-Cardinality Tabular ML** | Leakage-free out-of-fold target encoding, regex parsing (IPQ), and log-space distribution normalization | LightGBM, XGBoost, Scikit-Learn |

---

## 🏛️ Portfolio Architecture & Project Index

| Paradigm | Project / Subsystem | Primary Tech Stack | Key Architectural Innovation / Verified Metric | Directory Link |
| :--- | :--- | :--- | :--- | :--- |
| **Multimodal Systems** | **Amazon ML Challenge: Price Predictor** | PyTorch, ResNet50, LightGBM, Hugging Face | **Top 12% Finish (52.44% SMAPE)**; Dual-transformer text fusion + 2048-dim ResNet50 GAP + Target Encoding | [`Multimodal-Cost-Predictor/`](./Multimodal-Cost-Predictor) |
| **Compound AI & Systems** | **Orchestrate Sept: Financial Engine** | Python, Gemini 3.8 Flash, Dynamic Programming, ReAct | **Global Rank 197 (Top 1% / 30,000+ developers)**; $O(N)$ DP cash-flow solver, 8-worker concurrency | [`hackerrank-orchestrate/september/`](./hackerrank-orchestrate/september) |
| **Autonomous Agents** | **Orchestrate Aug: Notification Router** | Gemini 3.6 Flash, BM25, RapidFuzz, Whisper | **Global Rank 421 / 1,983 Finalists** (~22K Registrants); 6-stage agent pipeline, 2-stage safety filter | [`hackerrank-orchestrate/august/`](./hackerrank-orchestrate/august) |
| **RL Environments** | **OpenEnv: InferenceOps LLM Router** | OpenEnv, Docker, Pydantic v2, Llama-3.3-70B | *Meta x Scaler Hackathon*; Simulated API economy, dynamic Pareto cost-latency dispatching with thermal traps | [`openenv/llm-router/`](./openenv/llm-router) |
| **RL Environments** | **OpenEnv: Data Curator Alignment** | Python AST, Autopep8, Docker, Hugging Face | *Meta x Scaler Hackathon*; AST-validated code surgery, anti-cheating guardrails, PII redaction | [`openenv/data-curator/`](./openenv/data-curator) |
| **RL Environments** | **OpenEnv: MLOps Endpoint Triage** | SWE-Agent Toolset, Docker, Qwen 2.5-72B | *Meta x Scaler Hackathon*; Memory-efficient tracebacks (<150MB RAM), Jinja2 chat templates, Safetensors | [`openenv/mlops-endpoint-triage/`](./openenv/mlops-endpoint-triage) |
| **Vision-Language** | **Aircraft Damage Classification & Captioning** | VGG16 (ImageNet), BLIP Transformer, PyTorch | Dual-stage pipeline: defect classification (dent vs. crack) paired with generative damage captioning | [`Aircraft-Damage/`](./Aircraft-Damage) |
| **Deep Learning** | **League of Legends Match Predictor** | PyTorch (`nn.Module`), SGD, StandardScaler | Custom PyTorch pipeline from scratch: manual training loop, state_dict serialization, XAI feature weights | [`League-of-Legends-Match-Predictor/`](./League-of-Legends-Match-Predictor) |
| **Computer Vision** | **Automated Waste Sorting Classifier** | TensorFlow, Keras, VGG16, Data Augmentation | Transfer learning with progressive unfreezing and fine-tuning for recyclable/organic segregation | [`Waste-Classification/`](./Waste-Classification) |
| **Classical ML** | **Australian Rainfall Prediction Pipeline** | Scikit-Learn, Pandas, Seaborn, GridSearch | Imputation, feature scaling, and hyperparameter-tuned Logistic Regression vs. Random Forest models | [`Rainfall-Prediction/`](./Rainfall-Prediction) |

---

## 🔬 In-Depth Engineering Deep Dives

### 1. Amazon ML Challenge 2025: Multimodal Price Prediction (Top 12% Finish)
An end-to-end multimodal machine learning pipeline predicting product prices across high-variance e-commerce catalogs.
- **Dual-Stream NLP Embeddings**: Concatenates `all-MiniLM-L6-v2` (384-dim, broad semantic similarity) with `all-mpnet-base-v2` (768-dim, granular contextual nuance).
- **Deep Visual Representations**: Extracts 2048-dimensional feature vectors via a pre-trained **ResNet50** backbone using Global Average Pooling. Corrupted image downloads are converted into a predictive `image_missing` indicator.
- **Feature Engineering**: Custom regex extracting Item Pack Quantity (IPQ) and normalizing physical measurement units; out-of-fold Bayesian target encoding for categorical variables.
- **Model Optimization**: LightGBM Regressor using `regression_l1` (MAE proxy for SMAPE), verified via **5-Fold Cross-Validation** achieving a competitive **52.44% SMAPE**.
- 🔗 *Explore code & methodology*: [`Multimodal-Cost-Predictor/`](./Multimodal-Cost-Predictor)

---

### 2. HackerRank Orchestrate Series: Autonomous Agent Systems

#### A. September 2026 Edition: "Buy or Wait?" Autonomous Financial Decision Agent (Global Rank 197)
An autonomous, deterministic, and multimodal financial agent evaluating consumer affordability against personal cash-flow forecasts, commitments, and provider installment plans.
- **Competition Track Record**: Ranked **#197 globally (Top 1%)** among **30,000+ registered participants** (~2,000+ submitting finalists) in the 24-hour HackerRank Orchestrate hackathon.
- **$O(N)$ Suffix Minima Dynamic Programming**: Employs a forward-backward DP pass over 90-day cash horizons ($\text{suffix\_min}[t] = \min_{k \ge t}(\text{balance}[k] - \text{floor})$), verifying future liquidity bounds in $O(1)$ time without expensive step-by-step simulations.
- **Multimodal Evidence Extraction**: Automated Gemini 3.8 Flash OCR extracting missing transaction values from receipts, with a semantic message reconciler adjusting for amendments and cancellations.
- **Bounded ReAct Reasoning Loop (`max_steps=3`)**: Dynamic tool-assisted execution with deterministic fallbacks, achieving 0 unhandled exceptions across 250 evaluation requests in 119 seconds (~476ms/req) at $0.006 total cost.
- 🔗 *Explore September solution*: [`hackerrank-orchestrate/september/`](./hackerrank-orchestrate/september)

#### B. August 2026 Edition: Multimodal Notification Router (Global Rank 421)
A 6-stage sequential agent pipeline reasoning over incoming multimodal WhatsApp messages to decide immediate attention (`notify`), batching (`digest`), or suppression (`mute`).
- **Zero-Token Safety Short-Circuiting**: Intercepts prompt injections and scam URLs via deterministic regex heuristics upfront, preserving downstream API tokens and minimizing latency.
- **Hybrid Context Retrieval**: Combines BM25 keyword matching with dense semantic search across 13 relational tables, sandboxed inside strict XML tags to prevent context poisoning.
- **Competition Track Record**: Ranked **#421 globally** among 1,983 submitting finalists across ~22,000 global signups (**Top ~2%**).
- 🔗 *Explore August solution*: [`hackerrank-orchestrate/august/`](./hackerrank-orchestrate/august)

---

### 3. OpenEnv Suite: Meta PyTorch x Scaler Hackathon Gyms
Standardized reinforcement learning environments developed for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology** (backed by Meta, Hugging Face, and PyTorch). Adheres to the **OpenEnv Pydantic Quad-Spec** (`Action`, `Observation`, `Reward`, `State`), built for containerized evaluation on Hugging Face Spaces.
- **InferenceOps LLM Router (`openenv/llm-router`)**: Simulates a live production server economy. Agents balance queue throughput across `FAST_CHEAP`, `BALANCED`, and `EXPENSIVE_REASONER` endpoints under degrading financial budgets and thermal throttling.
- **Data Curator Alignment (`openenv/data-curator`)**: Autonomous agent environment simulating dataset alignment engineers. Agents perform AST-validated surgical code modifications (`autopep8`, `ast.parse`) to resolve PII leaks and formatting crashes without touching the raw dataset directly.
- **MLOps Endpoint Triage (`openenv/mlops-endpoint-triage`)**: SWE-Agent style environment debugging production inference crashes (Safetensor key discrepancies, Jinja2 chat templates, CPU device mapping) under strict resource bounds (`2 vCPU`, `8GB RAM`, <150MB mock memory footprint).
- 🔗 *Explore environments & benchmarks*: [`openenv/`](./openenv)

---

### 4. Vision-Language & Deep Learning Systems
- **Aircraft Damage Classification & Captioning (`Aircraft-Damage/`)**: Transfer learning on VGG16 (ImageNet) for structural defect classification (dent vs. crack), feeding into a BLIP Transformer for generative captioning and automated damage report generation.
- **PyTorch Match Predictor from Scratch (`League-of-Legends-Match-Predictor/`)**: Custom PyTorch neural architecture implementing manual forward/backward propagation, L2 weight decay, dynamic `nn.Module` configuration, and post-training feature importance extraction for Explainable AI (XAI).
- **Waste Classification via Transfer Learning (`Waste-Classification/`)**: VGG16 convolutional backbone with custom dense classification heads, using progressive unfreezing and fine-tuning on industrial recyclable streams.

---

## 🛠️ Engineering Rigor & Production Standards

- **Strict Schema Enforcement**: All agent interfaces, observations, and actions conform to validated Pydantic v2 and Dataclass models.
- **Deterministic AST & Contract Guardrails**: Code-editing environments enforce AST-level syntax validation, while financial agents enforce exact floating-point balance conservation ($P_1 + P_2 = \text{total}$).
- **Containerization**: Non-root container specifications (`UID 1000`) ready for one-click Hugging Face Spaces deployment.
- **Resilience & High Availability**: Multi-key API rotation with exponential backoff and circuit-breaking error handling.
- **Reproducible Evaluation**: Verified benchmark baselines established across **Llama-3.3-70B-Instruct**, **Qwen 2.5-72B**, and **Gemini 3.8 Flash**.

---

## 💻 Getting Started & Local Development

### Prerequisites
- Python 3.10+
- Docker (for containerized OpenEnv environments)
- [`uv`](https://github.com/astral-sh/uv) (recommended for dependency resolution)

### Clone & Setup
```bash
git clone https://github.com/Bhargav-P-Y/AI-Projects.git
cd AI-Projects

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Run the September Orchestrate Financial Agent
cd hackerrank-orchestrate/september
python code/main.py

# Run an OpenEnv container
cd ../../openenv/llm-router
docker build -t openenv-llm-router .
docker run -p 7860:7860 openenv-llm-router
```

---

## 👤 Author

**Bhargav P Y**
- **GitHub**: [@Bhargav-P-Y](https://github.com/Bhargav-P-Y)
- **Email**: [yellambalse.bhargav@gmail.com](mailto:yellambalse.bhargav@gmail.com)
