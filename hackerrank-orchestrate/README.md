# 🤖 HackerRank Orchestrate Series: Autonomous AI Agents

[![Challenge Series](https://img.shields.io/badge/Competition-HackerRank%20Orchestrate-orange.svg)](https://www.hackerrank.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)]()
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash-green.svg)]()
[![Status](https://img.shields.io/badge/Benchmark-Production%20Grade-brightgreen.svg)]()

This directory contains solutions developed for the prestigious **HackerRank Orchestrate 24-hour hackathon series** (August & September 2026), showcasing end-to-end autonomous multi-agent architectures, dynamic programming financial solvers, multimodal evidence reconciliation, and deterministic safety guardrails.

---

## 🏛️ Hackathon Editions Overview

| Edition | Challenge Title & Domain | Core Architecture & Algorithms | Verified Placement / Benchmark | Subfolder Link |
| :--- | :--- | :--- | :--- | :--- |
| **September 2026** | **"Buy or Wait?" Autonomous Financial Decision Agent** | $O(N)$ Suffix Minima DP cash-flow simulator, Bounded ReAct loop (`max_steps=3`), Gemini 3.8 Flash OCR receipt extraction, message reconciler | 250 requests processed in 119s (~476ms/req) at $0.006 total cost; 100% schema invariant compliance | [`september/`](./september) |
| **August 2026** | **Multimodal Message Notification Router** | 6-stage sequential agent pipeline (WhatsApp triage: `notify`/`digest`/`mute`), Gemini Flash OCR + Whisper ASR, 2-stage injection filter, BM25 retrieval | **Global Rank 421 / 1,983 Finalists** (~22,000 Global Registrants; Top ~2%) | [`august/`](./august) |

---

## 🔬 Deep Dive: September 2026 — "Buy or Wait?" Financial Decision Agent

### Problem Formulation
In consumer finance, deciding **"Can I afford this?"** cannot be answered by current account balance alone. A safe decision requires reconstructing future liquidity across recurring obligations, pending card authorizations, essential living expenses, confirmed salary schedules, and personal savings reserve floors.

### Architectural Highlights
```
                                 [INPUT DATASETS]
       ┌──────────────────┬──────────────────────┬──────────────────────┐
       │   requests.csv   │   financial_events   │  financial_profiles  │
       │ payment_options  │   exchange_rates     │  receipt images/msgs │
       └─────────┬────────┴──────────┬───────────┴──────────┬───────────┘
                 │                   │                      │
                 ▼                   ▼                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                  DATA INGESTION & NORMALIZATION                  │
     │      (Typed Dataclasses, FX Matrix, ISO Date Normalization)      │
     └───────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │           MULTIMODAL EVIDENCE RESOLVER (PHASES 2 & 3)            │
     │   - Gemini 3.8 Flash OCR: Resolves missing transaction amounts   │
     │   - Message Reconciler: Reconciles amendments & cancellations    │
     │   - Persistent JSON Cache: dataset/extracted_*.json              │
     └───────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │       DETERMINISTIC CASH FLOW ENGINE ($O(N)$ SUFFIX MINIMA DP)   │
     │   - Computes exact forward cumulative liquidity over 90 days     │
     │   - O(1) balance floor verification: suffix_min[t] >= floor      │
     │   - Evaluates: Affordable Now, Installments, Partials, Wait, Cut │
     └───────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │            OBSERVABLE BOUNDED ReAct AGENT (max_steps=3)          │
     │   - Tool-driven reasoning over financial context & evidence      │
     │   - Structural Semantic Invariant Validator                      │
     │   - Strict 8-Column Contract Guardrail & Dual-Payment Sum Checks │
     └───────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
                      [FINAL OUTPUT: output.csv]
```

1. **$O(N)$ Suffix Minima Dynamic Programming**:
   Instead of re-simulating 90-day cash flows for every candidate payment or installment plan, the `CashFlowSimulator` runs a forward cumulative pass followed by a backward suffix minimum pass:
   $$\text{suffix\_min}[t] = \min_{k \ge t} \left( \text{balance}[k] - \text{min\_balance\_to\_keep} \right)$$
   This enables exact, $O(1)$ verification of whether any payment on date $t$ breaches the cash reserve at any point in the future.
2. **Multimodal Evidence Resolution**:
   Automated OCR extraction for receipts with missing transaction amounts and semantic message reconciliation for financial amendments, backed by persistent entity-scoped JSON caches.
3. **Bounded ReAct Agent (`max_steps=3`)**:
   Dynamic tool-driven reasoning preventing infinite loops and context blowup, with deterministic fallbacks ensuring 0 unhandled exceptions across all 250 evaluation requests.
4. **Token Usage & Economic Efficiency**:
   Processed all 250 evaluation instances in **119.12 seconds** using 56,050 total tokens for a total run cost of **$0.00642** (< $0.01 total).

- 🔗 *Explore source code and tests*: [`september/`](./september)

---

## 🔬 Deep Dive: August 2026 — Multimodal Notification Router

### Problem Formulation
WhatsApp is overloaded with personal chats, work threads, apartment society notices, marketing posters, voice memos, and scams. The agent dynamically decides whether each incoming message should interrupt the user now (`notify`), be batched into a daily summary (`digest`), or be suppressed (`mute`).

### Architectural Highlights
- **6-Stage Sequential Agent Pipeline**: Enriches signals progressively across Data Loading, Media Extraction, Safety Gating, Hybrid Retrieval, LLM Batch Routing, and Confidence Calibration.
- **Zero-Token Safety Short-Circuiting**: Drops prompt injections and scam URLs via deterministic regex filters before invoking any LLM, saving critical API quota and eliminating latency.
- **Hybrid Context Retrieval**: Combines BM25 keyword matching with dense semantic embeddings across 13 relational tables, sandboxing context inside XML delimiters to prevent context poisoning.
- **Fault-Tolerant Multi-Key Rotation**: 4-key round-robin rotation with exponential backoff handling 429 rate limits seamlessly during evaluation.
- **Verified Benchmark**: Ranked **#421 globally** out of 1,983 submitting finalists across ~22,000 global signups (**Top ~2%**).

- 🔗 *Explore source code and datasets*: [`august/`](./august)
