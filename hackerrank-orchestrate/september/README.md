# HackerRank Orchestrate (September 2026) — Buy or Wait?

[![Rank](https://img.shields.io/badge/Rank-Global%20%23197%20%7C%20Top%201%25-brightgreen.svg)]()
[![Hackathon](https://img.shields.io/badge/Competition-HackerRank%20Orchestrate-orange.svg)](https://www.hackerrank.com)
[![Scale](https://img.shields.io/badge/Participants-30%2C000%2B%20Registered-blue.svg)]()
[![Status](https://img.shields.io/badge/Invariants-100%25%20Verified-success.svg)]()

> **Global Placement**: Ranked **#197 globally (Top 1%)** out of **30,000+ registered developers** (~2,000+ submitting finalists) in the 24-hour hackathon.  
> **An Autonomous, Deterministic & Multimodal AI Financial Decision Agent**  
> Evaluates consumer purchase and payment requests against user financial position, commitments, liquidity forecasts, dated exchange rates, provider payment options, and multimodal evidence (receipts, statements, and messages treated as untrusted financial evidence per `AGENTS.md` §1).

---

## 1. Executive Summary

In consumer finance, deciding **"Can I afford this?"** cannot be answered by current balance alone. A safe decision requires reconstructing future liquidity across recurring obligations, pending card debits, essential expenses, confirmed salary dates, and personal financial priorities, while strictly protecting the user's minimum cash reserve floor.

This repository contains the complete, production-grade implementation of the **Buy or Wait? Financial Decision Agent**. Built on a **High-Performance Concurrent Hybrid Architecture**, the system combines:
1. **100% Deterministic Financial Mathematics**: $O(N)$ Suffix Minima Dynamic Programming for instantaneous, zero-hallucination cash flow forecasting and constraint satisfaction.
2. **Multimodal Evidence Resolution**: Automated OCR extraction for missing transaction amounts and semantic message reconciliation with entity-scoped persistent caching.
3. **Observable Bounded ReAct Execution Loop (`max_steps=3`)**: Dynamic tool-driven reasoning and multi-tier self-correcting error handling.
4. **Grounded Natural Language Synthesis**: Live Gemini 3.8 Flash generation with strict few-shot prompting, anti-hallucination guards, and a structural Tier-2 semantic validator.
5. **Strict Schema Guardrails**: Enforces 100% invariant adherence against the project contract defined in `AGENTS.md` §6.2.

---

## 2. System Architecture

```text
                                 [INPUT DATASETS]
        ┌───────────────────┬───────────────────┬───────────────────┐
        │  requests.csv     │ financial_events  │ financial_profiles│
        │  payment_options  │ exchange_rates    │ images/ + msgs    │
        └─────────┬─────────┴─────────┬─────────┴─────────┬─────────┘
                  │                   │                   │
                  ▼                   ▼                   ▼
     ┌─────────────────────────────────────────────────────────────┐
     │               DATA INGESTION & REPOSITORIES                 │
     │      (Typed Dataclasses, FX Matrix, ISO Date Normalizer)     │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │           MULTIMODAL EVIDENCE RESOLVER (PHASE 2 & 3)        │
     │   - Gemini 3.8 Flash OCR: Resolves 16 missing event amounts  │
     │   - Message Reconciler: Handles amendments, cancellations   │
     │   - Persistent JSON Cache: dataset/extracted_*.json         │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │         DETERMINISTIC FINANCIAL ENGINE (PHASE 4)            │
     │  ┌─────────────────────────┐   ┌──────────────────────────┐ │
     │  │      CadenceEngine      │   │    CashFlowSimulator     │ │
     │  │  (Recurrence Detection) │   │ ($O(N)$ Suffix Minima DP)│ │
     │  └────────────┬────────────┘   └─────────────┬────────────┘ │
     │               │                              │              │
     │  ┌────────────▼────────────┐   ┌─────────────▼────────────┐ │
     │  │    SpendingOptimizer    │   │        PlanSolver        │ │
     │  │  (Combinatorial Search) │   │(Optimal Schedule Decider)│ │
     │  └─────────────────────────┘   └──────────────────────────┘ │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │       OBSERVABLE BOUNDED ReAct AGENT (max_steps = 3)        │
     │       - Contextual State Assembly                           │
     │       - Dynamic Tool Execution (CashFlow, Plan, Options)    │
     │       - Self-Correction & Multi-Tier Error Recovery         │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │               3-TIER GROUNDED EXPLAINER                     │
     │   Tier 1: Gemini 3.8 Flash with Structured Prompt Formula   │
     │           (<untrusted_input> XML tags & anti-hallucination) │
     │   Tier 2: Semantic & Structural Invariant Validator         │
     │           (Rejects truncated/non-standard candidate strings)│
     │   Tier 3: Deterministic Golden Template Fallback            │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │                 OUTPUT SCHEMA GUARDRAIL                     │
     │  - Strict 8-Column Contract Validation                      │
     │  - Floating-point Sum Verification (Partial Payments)       │
     │  - Date Chronology & Formatting Enforcement                 │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
                          [FINAL EVALUATION OUTPUTS]
                  ├── output.csv (and dataset/output.csv)
                  ├── evaluation/usage_report.md
                  └── log.txt (Verbatim Chat Transcript)
```

---

## 3. Core Technical Pillars & Financial Decision Logic

### 3.1 $O(N)$ Suffix Minima Dynamic Programming
Rather than re-simulating 90-day cash flows for every trial payment, our `CashFlowSimulator` runs a forward cumulative pass followed by a backward suffix minimum pass:
$$\text{suffix\_min}[t] = \min_{k \ge t} \left( \text{balance}[k] - \text{min\_balance\_to\_keep} \right)$$
This enables exact, $O(1)$ verification of whether any payment on date $t$ breaches the cash reserve at any point in the future.

### 3.2 Strict Cash Reserve & Conservative Invariant Rules
- **Reserved Pending Debits**: Pending card authorizations and settlements are reserved immediately.
- **Conservative Future Credits**: Unsettled bonuses, refunds, lottery gains, and investment gains are excluded until finalized. Confirmed salary is recognized strictly on its stated settlement date.
- **Safety Floor**: After any payment or essential expense, the user balance must remain strictly $\ge \text{minimum\_balance\_to\_keep}$.
- **Decision Hierarchy**:
  1. `affordable_now` (`full_payment`): Safe immediately on `request_date` without spending changes.
  2. `affordable_with_plan`: Full amount completed by `desired_completion_date` via:
     - **Installments**: Follows provider options, respects `max_installment_months`.
     - **Partial Payment**: Exactly 2 payments ($P_1$ today, $P_2$ on earliest safe date), $P_1 + P_2 = \text{requested\_amount}$.
     - **Spending Changes**: Up to 3 `stop:<id>` or `reduce_to:<id>:<amount>` modifications targeting flexible, non-protected categories only.
  3. `affordable_later` (`wait`): Full payment is safe on a single future date after confirmed income.
  4. `not_affordable` (`not_recommended`): Request cannot be safely completed within 90 days.

---

## 4. Setup and Installation

### 4.1 Prerequisites
- **Python**: Version `3.10` or higher
- **Operating System**: macOS, Linux, or Windows (PowerShell/CMD)

### 4.2 Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/interviewstreet/hackerrank-orchestrate-september26.git
cd hackerrank-orchestrate-september26

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 4.3 Environment Configuration (`.env`)
Create a `.env` file in the repository root containing your Google Gemini API keys (key rotation is handled automatically with thread safety):
```ini
GEMINI_API_KEYS=YOUR_API_KEY_1,YOUR_API_KEY_2
GEMINI_MODEL_NAME=gemini-3.8-flash
```
*(Note: If no API keys are provided, the system operates seamlessly in 100% Deterministic Fallback Mode without crashing).*

---

## 5. Execution Instructions

### 5.1 Run Full Production Pipeline
Execute the full evaluation across all 250 requests in `dataset/requests.csv`:
```bash
python code/main.py
```
This executes the multi-threaded agent, validates all schema invariants via `OutputGuardrail`, generates the final submission file `output.csv` in the repository root (as required by `AGENTS.md` §6.2 and competition specifications), mirrors the result to `dataset/output.csv` for full evaluator compatibility, and generates `evaluation/usage_report.md`.

### 5.2 Automated Judicial Verification Scripts
Run the comprehensive verification test suite:
```bash
# 1. Verify schema invariants, date ordering, and data integrity (0 violations)
python code/audit_output.py

# 2. Verify exact two-payment floating-point sums for partial payments (0 discrepancies)
python code/audit_partials.py

# 3. Inspect status distributions and explanation formatting
python code/inspect_distribution.py
```

---

## 6. Benchmark Evaluation Metrics

### 6.1 Performance Benchmarks
* **Evaluation Requests**: 250 rows (`request_26` to `request_275`)
* **Execution Time**: **119.12 seconds** (~476.5 ms / request) via 8-worker concurrent execution.
* **Failure / Fallback Rate**: **0 unhandled exceptions** (100% fault-tolerant).

### 6.2 Decision & Method Distribution
Demonstrates nuanced financial discrimination without degenerate mode collapse:

| Affordability Status | Count | Recommended Method Breakdown |
| :--- | :---: | :--- |
| `affordable_with_plan` | 76 (30.4%) | 59 `installments`, 11 `partial_payment`, 6 `full_payment` (with spending cuts) |
| `not_affordable` | 59 (23.6%) | 59 `not_recommended` |
| `affordable_now` | 58 (23.2%) | 58 `full_payment` |
| `affordable_later` | 57 (22.8%) | 57 `wait` |
| **Total** | **250 (100%)** | |

### 6.3 Audit & Contract Compliance
- **Schema & Types**: 100% compliant (exact 8 columns, 0 nulls in mandatory fields).
- **Arithmetic Precision**: 100% of partial payments sum to exact `requested_amount`.
- **Installment Conformity**: 100% of installment plans match provider payment options.
- **Explanation Quality**: 100% of explanations adhere to golden benchmark dual-clause structure.

### 6.4 Token Usage and Cost Efficiency
*(Source: `evaluation/usage_report.md`)*
* **Total API Calls**: 38 (Multimodal OCR & Semantic Evidence) + live hybrid calls
* **Total Input Tokens**: 46,200
* **Total Output Tokens**: 9,850
* **Total Tokens Consumed**: 56,050
* **Total Run Cost**: **$0.00642** (< $0.01 USD total)
* **Average Cost Per Request**: **$0.000026**

---

## 7. Submission Checklist & Contract Adherence

- [x] **Standalone Terminal Execution**: Runs cleanly via `python code/main.py`.
- [x] **Input Isolation**: Reads strictly from `dataset/`; never touches organizer-only files.
- [x] **No Hardcoded Labels**: All decisions derived from first-principles cash-flow modeling.
- [x] **Zero Secret Commits**: API keys managed solely via `.env` / environment variables.
- [x] **Full Audit Logging**: Continuous per-turn logging in `log.txt` per `AGENTS.md` §5.
- [x] **Mandatory Submission URL**:
  Official submission portal: [https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission](https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission)
