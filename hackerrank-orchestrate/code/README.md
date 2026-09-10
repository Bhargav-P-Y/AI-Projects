# Message Notification Router — AI Agent

An end-to-end autonomous AI agent for WhatsApp that classifies every incoming multimodal message into one of three routing decisions: **notify** (interrupt now), **digest** (batch for later), or **mute** (suppress as low-value, repetitive, or unsafe).

Built for the HackerRank Orchestrate August 2026 hackathon challenge.

---

## Architecture Overview

The agent is a **6-stage sequential pipeline**, where each stage is a self-contained module that enriches the message signal before passing it downstream.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Incoming Message (from dataset/messages.csv)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
               ┌─────────────▼──────────────┐
               │  Phase 1: Data Loader &    │  Loads 13 CSVs, builds
               │  Profile Builder           │  O(1) UserProfile maps
               └─────────────┬──────────────┘
                             │
               ┌─────────────▼──────────────┐
               │  Phase 2: Media Extractor  │  Gemini Flash OCR for
               │  (OCR + ASR)               │  images, ASR for voice
               └─────────────┬──────────────┘
                             │
               ┌─────────────▼──────────────┐
               │  Phase 3: 2-Stage Safety   │  Fast-tracks hard threats
               │  Filter                    │  (prompt injection, scam
               │                            │  domains) without LLM call
               └──────┬──────────────┬──────┘
                      │              │
              Hard    │              │  Safe messages
              threat  │              │
              (mute)  │    ┌─────────▼──────────────┐
                      │    │  Phase 4: Hybrid        │  BM25 + Dense Embed +
                      │    │  Retriever + Context    │  Recency + Engagement
                      │    │  Builder                │  XML-sandboxed prompt
                      │    └─────────┬──────────────┘
                      │              │
                      │    ┌─────────▼──────────────┐
                      │    │  Phase 5: LLM Router   │  Gemini 3.6 Flash,
                      │    │  (Batch API)            │  3-msg batches, 4-key
                      │    │                         │  rotation, 429 failover
                      │    └─────────┬──────────────┘
                      │              │
               ┌──────▼──────────────▼──────┐
               │  Phase 6: Confidence       │  Domain signal boosts/
               │  Calibrator + OutputWriter │  penalties → output.csv
               └────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pre-LLM Safety Filter** | Hard threats (prompt injection, domain spoofing) are intercepted in <1ms before any LLM call, preventing token waste and jailbreaking |
| **XML Sandboxing** | All untrusted user text and media transcriptions are wrapped in `<untrusted_user_message>` / `<untrusted_media_content>` tags to prevent system prompt hijacking |
| **Hybrid Retrieval** | BM25 alone misses semantic matches; dense embeddings alone miss exact account numbers. The combination with recency decay and engagement rates gives the best evidence quality |
| **Batch LLM Routing** | Routing 3 messages per LLM call with structured JSON output reduces API calls by 3x compared to per-message routing |
| **Confidence Calibration** | LLM self-reported confidence is overconfident. Empirical post-processing adjustments based on verified business status, user DND windows, muted groups, and scam risk produce more reliable scores |
| **Project-Isolated API Keys** | Google AI Studio Free Tier limits `gemini-3.6-flash` to 20 RPD per project. One key per project gives independent quota pools |

---

## Directory Structure

```
code/
├── main.py                         # Entry point — runs full pipeline
├── .env                            # API keys (NOT committed to git)
│
├── data_pipeline/
│   ├── config.py                   # .env loader, API key reader
│   ├── data_loader.py              # Loads all 13 CSVs into DataBundle
│   ├── profile_builder.py          # Builds UserProfile objects (DND, relationships)
│   └── test_phase1.py              # Unit tests for Phase 1
│
├── media_extractor/
│   ├── media_extractor.py          # Parallel OCR (images) + ASR (voice notes)
│   └── test_phase2.py              # Integration tests for Phase 2
│
├── safety_filter/
│   ├── safety_filter.py            # 2-stage hard threat + risk signal engine
│   ├── semantic_injection.py       # Embedding-based injection detector
│   └── test_phase3.py              # Unit tests for Phase 3
│
├── context_builder/
│   ├── hybrid_retriever.py         # BM25 + Dense + Recency + Engagement retrieval
│   ├── context_builder.py          # Assembles XML-sandboxed prompt context
│   └── test_phase4.py              # Unit tests for Phase 4
│
├── llm_router/
│   ├── llm_router.py               # Batch Gemini API caller, key rotation
│   └── test_phase5.py              # Integration tests for Phase 5
│
├── output_writer/
│   ├── confidence_calibrator.py    # Domain signal post-processing
│   ├── output_writer.py            # Schema validator + CSV writer
│   └── test_phase6.py              # Unit tests for Phase 6
│
├── evaluation/
│   ├── eval.py                     # Benchmark evaluation against sample_messages.csv
│   └── eval_report.json            # Saved benchmark metrics (generated at runtime)
│
└── cache/
    └── media_cache.json            # Cached OCR/ASR results (auto-generated)
```

---

## Dependencies

The agent uses **Python 3.10+** with no heavy ML frameworks. All dependencies are from the standard library or lightweight packages.

Install dependencies:

```bash
pip install pandas requests scikit-learn numpy
```

| Package | Usage |
|---------|-------|
| `pandas` | CSV loading and output writing |
| `requests` | Gemini REST API calls |
| `scikit-learn` | TF-IDF vectorizer for BM25-style retrieval |
| `numpy` | Cosine similarity for dense retrieval |

---

## Environment Variables

Create a `.env` file inside the `code/` directory (same level as `main.py`):

```env
GEMINI_API_KEYS="
YOUR_KEY_1_HERE,
YOUR_KEY_2_HERE,
YOUR_KEY_3_HERE,
YOUR_KEY_4_HERE
"
```

**Important notes:**
- Each key must be a Google AI Studio API key (`AQ.` prefix format — the current Google authorization key format).
- For best throughput without rate limiting, create one key **per distinct Google Cloud project**. Each project has an independent 20 requests-per-day free tier quota for `gemini-3.6-flash`.
- **Never commit `.env` to git.** It is in `.gitignore`.

---

## Dataset Layout

The pipeline expects the following files under `dataset/` at the **repository root** (one level above `code/`):

```
dataset/
├── messages.csv              # Incoming messages to classify (input)
├── output.csv                # Agent predictions (output — written by pipeline)
├── sample_messages.csv       # Annotated benchmark for evaluation workflow
├── users.csv
├── groups.csv
├── group_members.csv
├── business_accounts.csv
├── user_business_history.csv
├── message_history.csv
├── message_events.csv
├── images.csv
├── voice_notes.csv
├── daily_notification_summary.csv
└── media/
    ├── images/               # Image files referenced in images.csv
    └── audio/                # Audio files referenced in voice_notes.csv
```

---

## How To Run

### Full Pipeline (produces `dataset/output.csv`)

Run from the **repository root**:

```bash
python code/main.py
```

Expected output:
```
=== Starting Message Notification Router Pipeline ===
[Setup] Loaded 4 API key(s) for rotation.
[Phase 1] Loading dataset CSVs and building User Profiles...
[Phase 1] Loaded 110 incoming messages, 412 historical messages, and built 54 user profiles in 0.06s.
[Phase 2] Loading/extracting multimodal content (OCR & ASR)...
[Phase 3] Running 2-Stage Safety Engine...
[Phase 4] Retrieving historical evidence and assembling XML-sandboxed contexts...
[Phase 5] Routing 99 contexts via Gemini 3.6 Flash batch API...
[Phase 6] Calibrating decision confidence scores & validating schema...
Successfully wrote 110 validated rows to 'dataset/output.csv'.
=== END-TO-END PIPELINE EXECUTION SUMMARY ===
Total Messages Processed: 110
Fast-Track Security Mutes: 11
LLM-Routed Messages: 99
Output File Written: dataset/output.csv
Total Execution Time: ~816 seconds
```

### Evaluation Workflow (benchmarks against ground truth)

```bash
python code/evaluation/eval.py
```

Evaluates action accuracy, message type accuracy, and evidence retrieval recall against `dataset/sample_messages.csv` and saves a JSON report to `code/evaluation/eval_report.json`.

**Benchmark results on `sample_messages.csv` (30 annotated messages):**
- Action Accuracy: **86.67%**
- Message Type Accuracy: **80.00%**
- Evidence Retrieval Recall: **63.33%**
- Confidence Mean: **0.88** (range: 0.78–0.95)

### Run Individual Phase Tests

```bash
python code/data_pipeline/test_phase1.py
python code/media_extractor/test_phase2.py
python code/safety_filter/test_phase3.py
python code/context_builder/test_phase4.py
python code/llm_router/test_phase5.py
python code/output_writer/test_phase6.py
```

---

## Output Schema

`dataset/output.csv` must conform to the following schema (enforced by `OutputWriter`):

| Column | Type | Valid Values |
|--------|------|-------------|
| `message_id` | string | Matches `messages.csv` |
| `action` | string | `notify`, `digest`, `mute` |
| `message_type` | string | `urgent`, `personal`, `event`, `payment`, `business_update`, `promotion`, `scam`, `greeting`, `forward`, `media`, `other` |
| `reason` | string | Non-empty human-readable explanation |
| `confidence` | float | `[0.30, 0.95]` |
| `evidence_message_ids` | string | Semicolon-separated historical IDs or `none` |

---

## Safety Architecture

The agent implements a **defense-in-depth** strategy against malicious messages:

1. **Layer 1 — Structural Prompt Injection**: Regex detection of explicit system override patterns (`IGNORE ALL PREVIOUS INSTRUCTIONS`, `{{jailbreak}}`, etc.)
2. **Layer 2 — Semantic Injection**: Embedding similarity against a corpus of known injection phrases, catching paraphrased attacks
3. **Layer 3 — Domain Identity Spoofing**: Domain Levenshtein distance check against known trusted domains (e.g., `talabat-refund.com` vs. `talabat.com`)
4. **Layer 4 — XML Sandboxing**: All user-controlled text in the LLM prompt is wrapped in `<untrusted_user_message>` tags with explicit system instructions to treat content as data, never as commands

---

## Evaluation Workflow

The evaluation script (`code/evaluation/eval.py`) is a fully self-contained, non-hardcoded benchmark runner:

- Loads ground truth labels from `dataset/sample_messages.csv`
- Runs the **identical 6-stage pipeline** as `main.py` (no shortcuts or hardcoded answers)
- Computes action accuracy, message type accuracy, and evidence recall
- Saves a detailed per-message breakdown to `code/evaluation/eval_report.json`

This satisfies the hackathon requirement: *"Must include an evaluation workflow. Must avoid hardcoded test labels or file-specific answers."*
