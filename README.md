# AI/ML Engineering Portfolio | Systems, Foundation Models & Autonomous Agents

[![Author](https://img.shields.io/badge/Author-Bhargav%20P%20Y-blue.svg)](https://github.com/Bhargav-P-Y)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20%7C%20Transformers-yellow)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)]()
[![Pydantic](https://img.shields.io/badge/Schemas-Pydantic%20v2-E92063?logo=pydantic&logoColor=white)]()

An industry-grade portfolio demonstrating full-lifecycle Applied AI, Multi-Agent Systems, Reinforcement Learning Gym Environments, and Production MLOps. Engineered with verified competitive track records across tier-one hackathons (Amazon, Meta x PyTorch x Hugging Face, HackerRank) and strict production standards: containerized non-root runtimes, AST syntax guardrails, Pydantic type safety, and reproducible benchmarks across frontier LLMs.

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
│  🥈 HackerRank Orchestrate 2026 — Global Rank 421 / 1,983 Finalists (~22,000 Global Registrants)       │
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
| **Autonomous Multi-Agent Systems** | 6-stage sequential agent pipeline with OCR/ASR ingestion, safety gating, and context retrieval | Gemini 3.6 Flash, BM25, Pydantic, RapidFuzz |
| **Multimodal Systems & Embeddings** | Dual-stream sentence transformers concatenated with ResNet50 Global Average Pooling | PyTorch, `sentence-transformers`, `torchvision`, BLIP |
| **RL Environments & Benchmarking** | Standardized gym environments adhering to the Meta OpenEnv Pydantic Quad-Spec (`Action`/`Obs`/`Reward`/`State`) | OpenEnv, Docker (non-root UID 1000), `uv` |
| **Code Surgery & AST Guardrails** | Automated programmatic refactoring with `ast.parse` syntax checks and shaped negative rewards | Python AST, Autopep8, Llama-3.3-70B |
| **Deep Learning from First Principles** | Custom PyTorch pipeline with manual forward/backward loops, L2 weight decay, and XAI feature weights | PyTorch `nn.Module`, SGD, BCELoss |
| **High-Cardinality Tabular ML** | Leakage-free out-of-fold target encoding, regex parsing (IPQ), and log-space distribution normalization | LightGBM, XGBoost, Scikit-Learn |

---

## 🏛️ Portfolio Architecture & Project Index

| Paradigm | Project / Subsystem | Primary Tech Stack | Key Architectural Innovation / Verified Metric | Directory Link |
| :--- | :--- | :--- | :--- | :--- |
| **Multimodal Systems** | **Amazon ML Challenge: Price Predictor** | PyTorch, ResNet50, LightGBM, Hugging Face | **Top 12% Finish (52.44% SMAPE)**; Dual-transformer text fusion + 2048-dim ResNet50 GAP + Target Encoding | [`Multimodal-Cost-Predictor/`](./Multimodal-Cost-Predictor) |
| **Autonomous Agents** | **HackerRank Orchestrate: Notification Router** | Gemini 3.6 Flash, BM25, RapidFuzz, Whisper | **Global Rank 421 / 1,983 Finalists** (~22K Registrants); 6-stage agent pipeline, 2-stage safety filter | [`hackerrank-orchestrate/`](./hackerrank-orchestrate) |
| **RL Environments** | **OpenEnv: InferenceOps LLM Router** | OpenEnv, Docker, Pydantic v2, Llama-3.3-70B | *Meta x Scaler Hackathon*; Simulated API economy, dynamic Pareto cost-latency dispatching with thermal traps | [`openenv/llm-router/`](./openenv/llm-router) |
| **RL Environments** | **OpenEnv: Data Curator Alignment** | Python AST, Autopep8, Docker, Hugging Face | *Meta x Scaler Hackathon*; AST-validated code surgery, anti-cheating guardrails, PII redaction | [`openenv/data-curator/`](./openenv/data-curator) |
| **RL Environments** | **OpenEnv: MLOps Endpoint Triage** | SWE-Agent Toolset, Docker, Qwen 2.5-72B | *Meta x Scaler Hackathon*; Memory-efficient tracebacks (<150MB RAM), Jinja2 chat templates, Safetensors | [`openenv/mlops-endpoint-triage/`](./openenv/mlops-endpoint-triage) |
| **Vision-Language** | **Aircraft Damage Classification & Captioning** | VGG16 (ImageNet), BLIP Transformer, PyTorch | Dual-stage pipeline: defect classification (dent vs. crack) paired with generative damage captioning | [`Aircraft-Damage/`](./Aircraft-Damage) |
| **Deep Learning** | **League of Legends Match Predictor** | PyTorch (`nn.Module`), SGD, StandardScaler | Custom PyTorch pipeline from scratch: manual training loop, state_dict serialization, XAI feature weights | [`League-of-Legends-Match-Predictor/`](./League-of-Legends-Match-Predictor) |
| **Computer Vision** | **Automated Waste Sorting Classifier** | TensorFlow, Keras, VGG16, Data Augmentation | Transfer learning with progressive unfreezing and fine-tuning for recyclable/organic segregation | [`Waste-Classification/`](./Waste-Classification) |
| **Classical ML** | **Australian Rainfall Prediction Pipeline** | Scikit-Learn, Pandas, Seaborn, GridSearch | Imputation, feature scaling, and hyperparameter-tuned Logistic Regression vs. Random Forest models | [`Rainfall-Prediction/`](./Rainfall-Prediction) |
| **Algorithmic AI** | **Adversarial Minimax Game Engine** | Python, Pygame, NumPy | Adversarial game tree search with optimal Minimax decision-making and real-time GUI | [`Tic-Tac-Toe Game.py`](./Tic-Tac-Toe%20Game.py) |
| **Information Filtering** | **Collaborative Filtering Recommender** | Pandas, NumPy, Matplotlib | Item-item Pearson correlation matrix with minimum interaction thresholds | [`Recommendation System.py`](./Recommendation%20System.py) |
| **Conversational AI** | **Context-Augmented Rule Chatbot** | NLTK, Wikipedia API, Regex | Rule-based dialogue system augmented with dynamic Wikipedia API live article synthesis | [`Rule-Based Chatbot.py`](./Rule-Based%20Chatbot.py) |

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

### 2. Autonomous Multimodal Notification Router (HackerRank Orchestrate — Global Rank 421)
A 6-stage sequential agent pipeline reasoning over incoming multimodal WhatsApp messages to decide immediate attention (`notify`), batching (`digest`), or suppression (`mute`). Built for the 24-hour HackerRank Orchestrate hackathon (August 2026).
```
Incoming Message (Text / Image / Voice)
   │
   ▼
[Phase 1: Data Loader & Profile Builder] ──► Builds O(1) in-memory UserProfile maps across 13 relational CSVs
   │
   ▼
[Phase 2: Media Extractor] ───────────────► Gemini Flash Vision OCR for images + ASR for voice notes
   │
   ▼
[Phase 3: 2-Stage Safety Filter] ─────────► Fast-tracks hard threats (prompt injection, scam domains) -> MUTE
   │ (Safe messages proceed)
   ▼
[Phase 4: Hybrid Retriever & Context] ───► BM25 + Dense Semantic search over message history in XML sandbox
   │
   ▼
[Phase 5: LLM Router Engine] ─────────────► Gemini 3.6 Flash batching (3-msg batches, 4-key rotation, 429 failover)
   │
   ▼
[Phase 6: Confidence Calibrator] ────────► Domain signal adjustments -> Final submission output.csv
```
- **Competition Track Record**: Ranked **#421 globally** among **1,983 submitting finalists** who completed the build and the 30-minute AI Judge interview, out of **~22,000 global signups** (Top ~2% of all registrants).
- 🔗 *Explore source & test suites*: [`hackerrank-orchestrate/`](./hackerrank-orchestrate)

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

- **Strict Schema Enforcement**: All agent interfaces, observations, and actions conform to validated Pydantic v2 models.
- **Deterministic AST Guardrails**: Code-editing environments enforce AST-level syntax and compliance validation before runtime execution.
- **Containerization**: Non-root container specifications (`UID 1000`) ready for one-click Hugging Face Spaces deployment.
- **Resilience & High Availability**: Multi-key API rotation with exponential backoff and circuit-breaking error handling.
- **Reproducible Evaluation**: Verified benchmark baselines established across **Llama-3.3-70B-Instruct**, **Qwen 2.5-72B**, and **Gemini 3.6 Flash**.

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

# Run an OpenEnv container
cd openenv/llm-router
docker build -t openenv-llm-router .
docker run -p 7860:7860 openenv-llm-router
```

---

## 👤 Author

**Bhargav P Y**
- **GitHub**: [@Bhargav-P-Y](https://github.com/Bhargav-P-Y)
- **Email**: [yellambalse.bhargav@gmail.com](mailto:yellambalse.bhargav@gmail.com)
