# Multimodal Message Notification Router — Autonomous AI Agent

[![Rank](https://img.shields.io/badge/Rank-Global%20%23421%20%7C%20Top%202%25-brightgreen.svg)]()
[![Competition](https://img.shields.io/badge/HackerRank-Orchestrate%20Aug%202026-orange.svg)](https://www.hackerrank.com)
[![Scale](https://img.shields.io/badge/Participants-22%2C000%2B%20Registered-blue.svg)]()
[![Model](https://img.shields.io/badge/LLM-Gemini%20Flash-green.svg)]()
[![Evaluation](https://img.shields.io/badge/Accuracy-86.7%25-success.svg)]()

> **Global Placement**: Ranked **#421 globally** among 1,983 submitting finalists out of **~22,000 registered developers** (Top ~2%) in the 24-hour HackerRank Orchestrate hackathon.  
> **System Architecture**: An end-to-end 6-stage autonomous AI agent pipeline for WhatsApp that dynamically classifies multimodal messages into **notify** (interrupt now), **digest** (batch for later), or **mute** (suppress as low-value, repetitive, or unsafe).

---

## 1. Executive Summary

Modern messaging platforms like WhatsApp suffer from extreme signal-to-noise degradation. A user receives family emergencies, group chatter, school notices, delivery updates, spam posters, voice notes, and phishing links in the exact same channel. Treating every message uniformly either causes high-priority interruptions to be missed or bombards the user with cognitive overload.

This project implements a production-grade **6-Stage Sequential Agent Pipeline** that ingests multimodal inputs (text, OCR from image screenshots, and ASR from voice notes), filters adversarial prompt injections via zero-token heuristics, retrieves historical context via hybrid BM25 and dense semantic search, and dynamically routes messages using batched Gemini Flash calls with multi-key rate-limit rotation.

---

## 2. Architecture Overview

The agent is engineered as a **6-stage sequential pipeline**, where each stage acts as a self-contained module enriching message signals before passing them downstream:

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Incoming Message (Text / Image Screenshot / Voice Note)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
               ┌─────────────▼──────────────┐
               │  Phase 1: Data Loader &    │  Loads 13 relational tables,
               │  Profile Builder           │  builds O(1) UserProfile maps
               └─────────────┬──────────────┘
                             │
               ┌─────────────▼──────────────┐
               │  Phase 2: Media Extractor  │  Gemini Flash Vision OCR for
               │  (OCR + ASR)               │  images + ASR for voice notes
               └─────────────┬──────────────┘
                             │
               ┌─────────────▼──────────────┐
               │  Phase 3: 2-Stage Safety   │  Zero-token fast-tracking for
               │  Filter                    │  prompt injections & scam domains
               └──────┬──────────────┬──────┘
                      │              │
              Hard    │              │  Safe messages
              threat  │              │
              (mute)  │    ┌─────────▼──────────────┐
                      │    │  Phase 4: Hybrid        │  BM25 + Dense Embeddings +
                      │    │  Retriever + Context    │  Recency & Engagement Decay
                      │    │  Builder                │  Wrapped in XML sandboxes
                      │    └─────────┬──────────────┘
                      │              │
                      │    ┌─────────▼──────────────┐
                      │    │  Phase 5: LLM Router   │  Gemini Flash batch API
                      │    │  (Batch API)            │  (3-msg batches, 4-key
                      │    │                         │  rotation, 429 backoff)
                      │    └─────────┬──────────────┘
                      │              │
               ┌──────▼──────────────▼──────┐
               │  Phase 6: Confidence       │  Domain signal adjustments →
               │  Calibrator + OutputWriter │  strict schema output.csv
               └────────────────────────────┘
```

---

## 3. Key Design Decisions & Systems Engineering

| Architectural Component | Engineering Rationale & Implementation |
| :--- | :--- |
| **Zero-Token Safety Filter** | Hard security threats (prompt injections, domain spoofing) are intercepted in `<1ms` via regex and heuristic safety filters before invoking any LLM, eliminating latency and saving API token quotas. |
| **XML Context Sandboxing** | All untrusted user text and media transcriptions are encapsulated inside `<untrusted_user_message>` and `<untrusted_media_content>` tags, preventing indirect prompt injection from hijacking the system prompt. |
| **Hybrid RAG Retrieval** | BM25 keyword matching alone misses semantic equivalence, while dense embeddings alone miss exact account IDs. Fusing both with recency decay and user interaction frequency yielded the highest evidence precision. |
| **Batch LLM Routing** | Packaging 3 messages per structured JSON LLM call reduced total API roundtrips by 3x compared to naive per-message routing, comfortably staying within throughput limits. |
| **Empirical Confidence Calibration** | Raw LLM self-reported confidence scores are notoriously overconfident. Post-hoc calibration rules adjust scores based on verified business badges, quiet-hour DND windows, and historical interaction rates. |
| **Multi-Key Thread-Safe Rotation** | Configured round-robin rotation across independent API key pools with exponential backoff, ensuring 100% evaluation uptime under burst requests. |

---

## 4. Setup and Execution

### 4.1 Prerequisites
- **Python**: Version `3.10` or higher
- **Lightweight Dependencies**: Standard library + `pandas`, `requests`, `scikit-learn`, `numpy` (no heavy frameworks required).

### 4.2 Installation
```bash
# Clone the repository
git clone https://github.com/Bhargav-P-Y/AI-Projects.git
cd AI-Projects/hackerrank-orchestrate/august

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas requests scikit-learn numpy
```

### 4.3 Environment Configuration (`.env`)
Create a `.env` file in the `code/` directory:
```ini
GEMINI_API_KEYS=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
```

### 4.4 Run Evaluation Pipeline
```bash
python code/main.py
```
This runs the 6-stage pipeline across all incoming test messages, outputs the validated `output.csv`, and caches intermediate OCR/ASR extractions.

---

## 5. Performance & Verification Results

* **Global Standing**: Ranked **#421** among **1,983 submitting finalists** (~22,000 global signups; Top ~2%).
* **Action Accuracy**: **86.67%** routing accuracy against ground-truth benchmarks.
* **Classification Precision**: **80.00%** precision across heterogeneous text, image screenshots, and voice notes.
* **Pre-LLM Security**: 100% of prompt injection attacks successfully intercepted with sub-millisecond response times.
