# SPEC.md — System Specification & Technical Contract

**Challenge**: HackerRank Orchestrate (September 2026) — Buy or Wait?  
**Document Status**: UNAMBIGUOUS, BINDING SYSTEM SPECIFICATION  
**Target Output**: `output.csv` (root repository level) & `code.zip` with `evaluation/usage_report.md`

---

## 1. System Architecture Overview & ASCII Diagram

The system adheres strictly to **KISS**, **YAGNI**, **ISP (Interface Segregation)**, **Law of Demeter**, and **Composition over Inheritance**.

Instead of a brittle, multi-agent graph with cascading errors, the system is designed as a **Deterministic-First Autonomous Agent** architecture. It separates deterministic mathematical computation, schedule simulation, and rule-based constraint solving from natural language comprehension and grounded explanation generation.

```text
+===================================================================================================+
|                                    BUY OR WAIT? SYSTEM ARCHITECTURE                               |
+===================================================================================================+
|                                                                                                   |
|  [DATA REPOSITORY: dataset/]                                                                      |
|   |-- requests.csv / sample_requests.csv                                                          |
|   |-- financial_profiles.csv / financial_events.csv                                               |
|   |-- exchange_rates.csv / request_payment_options.csv                                            |
|   |-- messages.csv / images.csv / media/images/*.png                                              |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  STEP 1: DATA ACCESS, EVIDENCE RESOLUTION & PRE-LLM GUARDRAILS                                    |
|   * ProfileRepository, EventRepository, PaymentOptionRepository, MessageRepository (ISP)         |
|   * Isolates untrusted natural language into <untrusted_input> boundaries                          |
|   * Entity-Scoped Relational Indexing: Strict (user_id, request_id, related_event_id) joins       |
|   * Fixed Dated FX Conversion via exchange_rates.csv                                              |
|   * Multimodal Gemini 3.8 Flash OCR with Persistent JSON Caching (extracted_image_amounts.json)   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  STEP 2: DETERMINISTIC CORE ENGINE (Zero-Hallucination Math & Rule Optimization)                  |
|                                                                                                   |
|   +-------------------------------------------------------------------------------------------+   |
|   | CashFlowSimulator: 90-Day Continuous Daily Liquidity Simulator                            |   |
|   |  - Historical Cadence & Confirmed Inflow Extrapolation (Salary, Rent, Subs, Living Bills) |   |
|   |  - Pending Debits Reserved & Unconfirmed Credit Isolation                                 |   |
|   |  - Min-Floor Invariant: Balance(t) >= Minimum_Balance_To_Keep, for all t in [0, 90d]      |   |
|   +-------------------------------------------------------------------------------------------+   |
|                                                  |                                                |
|   +-------------------------------------------------------------------------------------------+   |
|   | PlanOptimizer & DecisionSolver: Rigid 6-Level Hierarchy Plan Selection                    |   |
|   |  1. Completion <= desired_completion_date                                                 |   |
|   |  2. Zero spending changes (spending_changes_needed == 'none')                             |   |
|   |  3. Minimize total payable amount                                                         |   |
|   |  4. Earliest payment start date                                                           |   |
|   |  5. Minimum payment count                                                                 |   |
|   |  6. Lowest payment_option_id tie-breaker                                                   |   |
|   +-------------------------------------------------------------------------------------------+   |
|                                                  |                                                |
|   +-------------------------------------------------------------------------------------------+   |
|   | FinancialDecisionAgent: Composed of [ToolRegistry] + [Guardrail] + [LLMClient]            |   |
|   |                                                                                           |   |
|   |   Bounded Atomic Tools (ISP & Law of Demeter Compliant):                                  |   |
|   |   ├── Tool 1: get_user_financial_context(user_id, request_id) ──► UserContextDTO          |   |
|   |   ├── Tool 2: simulate_cash_flow(user_id, start_date, payments) ──► SimulationResultDTO   |   |
|   |   ├── Tool 3: solve_optimal_plan(user_context) ──► DecisionResultDTO (Deterministic Math) |   |
|   |   └── Tool 4: generate_grounded_explanation(context, decision) ──► Grounded Explanation  |   |
|   +-------------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  STEP 3: POST-LLM VALIDATION & CONTRACT INTEGRITY GUARDRAILS                                      |
|   * Citation & Event Validator: Asserts referenced events exist in user history and are flexible  |
|   * Range Bounds & Enum Invariants:                                                               |
|     - 0 <= amount_safe_to_pay <= requested_amount                                                 |
|     - affordability_status in {affordable_now, affordable_with_plan, affordable_later, ...}       |
|     - recommended_payment_method in {full_payment, partial_payment, installments, wait, ...}      |
|     - payment_plan syntax: YYYY-MM-DD:amount|... or 'none'                                        |
|     - spending_changes_needed: max 3 flexible changes (stop:<id> / reduce_to:<id>:<amt>) or 'none'|
|     - earliest_date_for_full_payment == request_date for affordable_now, or ISO date, or empty    |
|   * Safety Fallback Shield: Guarantees deterministic mathematical results if LLM deviates         |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  STEP 4: HUMAN-CENTRIC OUTPUT EXPORT & USAGE ACCOUNTING                                           |
|   * Grounded Explanations answering Who, What & Why (Builds user trust)                           |
|   * 8-Point Schema Validation ──► root-level output.csv (exact 250 evaluation rows in order)      |
|   * Full Token, Call & Cost Accounting ──► evaluation/usage_report.md                            |
+===================================================================================================+
```

---

## 2. Entity-Relationship (ER) Schema & Data Contract

All data originates in `dataset/` and joins relationally via primary and foreign keys:

```text
+-----------------------+              1:1             +----------------------------------+
|      requests.csv     |------------------------------|     financial_profiles.csv       |
|-----------------------|                              |----------------------------------|
| PK request_id         |                              | PK user_id                       |
| FK user_id            |                              |    home_currency                 |
|    request_date       |                              |    current_available_balance     |
|    request_type       |                              |    minimum_balance_to_keep       |
|    requested_amount   |                              |    financial_priorities          |
|    desired_comp_date  |                              |    expense_categories_to_protect |
|    allows_partial_pay |                              |    expense_cat_willing_to_reduce |
|    request_text       |                              |    expense_cat_willing_to_stop   |
+-----------------------+                              |    payment_methods_will_consider |
           |                                           |    max_installment_months        |
           | 1:N                                       +----------------------------------+
           v                                                            |
+------------------------------+                                        | 1:N
| request_payment_options.csv  |                                        v
|------------------------------|                       +----------------------------------+
| PK payment_option_id         |                       |      financial_events.csv        |
| FK request_id                |                       |----------------------------------|
|    payment_method            |                       | PK event_id                      |
|    payment_amount            |                       | FK user_id                       |
|    number_of_payments        |                       |    event_type, description       |
|    first_payment_date        |                       |    category, direction           |
|    payment_frequency_days    |                       |    amount (empty -> images.csv)  |
|    financing_fee             |                       |    currency                      |
|    total_payable_amount      |                       |    event_date, settlement_date   |
+------------------------------+                       |    status                        |
                                                       | FK linked_event_id               |
                                                       |    flexibility                   |
                                                       |    minimum_allowed_amount        |
                                                       +----------------------------------+
                                                                        |
                       +------------------------------------------------+
                       | 1:1 (for empty amounts)       | 1:N (supporting msgs)
                       v                               v
+------------------------------+       +----------------------------------+
|          images.csv          |       |           messages.csv           |
|------------------------------|       |----------------------------------|
| PK image_id                  |       | PK message_id                    |
| FK user_id                   |       | FK user_id                       |
| FK request_id                |       | FK request_id                    |
| FK related_event_id          |       | FK related_event_id              |
| (PNG: media/images/<id>.png) |       |    sent_at, source_type, text    |
+------------------------------+       +----------------------------------+
```

### Conversion Rule
- `exchange_rates.csv`: Matched on `(settlement_date, from_currency, to_currency)`. Converts all non-home currency events to user's `home_currency`.

---

## 3. Core Engineering & OOP Principles

1. **KISS & YAGNI (Keep It Simple / You Aren't Gonna Need It)**:
   - No multi-agent actor hierarchies, no asynchronous agent swarms, no vector database overhead.
   - Deterministic event calendar projection handles math; bounded tool calls handle state queries and grounded explanation synthesis.
2. **Interface Segregation Principle (ISP)**:
   - Separate repositories for data retrieval (`IProfileRepository`, `IEventRepository`, `IEvidenceRepository`).
   - Clean, atomic tools with strictly typed Data Transfer Objects (DTOs).
3. **Law of Demeter (Least Knowledge)**:
   - The Agent interacts only through typed DTOs returned by its `ToolRegistry`. The Agent has no knowledge of Pandas DataFrames, file paths, or CSV delimiters.
4. **Composition over Inheritance**:
   - `FinancialDecisionAgent` has-a `ToolRegistry`, has-a `Guardrail`, and has-a `LLMClient`. Behaviors are injected via components, not subclassing.
5. **Separation of Concerns**:
   - `Data Layer`: Ingestion, OCR parsing, FX translation.
   - `Computation Layer`: Mathematical 90-day cash flow simulation, liquidity floor monitoring, spending change optimization.
   - `Intelligence Layer`: Text interpretation (payroll changes, cancellation notices) and explanation synthesis.
   - `Validation Layer`: Deterministic contract guardrails.

---

## 4. Atomic Tool Signatures & DTO Contracts

Each tool adheres to strict Pydantic / dataclass interfaces:

### Tool 1: `get_user_financial_context`
```python
class UserFinancialContextDTO:
    user_id: str
    request_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    requested_amount: float
    request_date: str
    desired_completion_date: str
    allows_partial_payment: bool
    allowed_payment_methods: list[str]  # e.g. ['full_payment', 'installments']
    max_installment_months: int | None
    protected_categories: set[str]
    reducible_categories: set[str]
    stoppable_categories: set[str]
    active_payment_options: list[PaymentOptionDTO]
    supporting_evidence: list[EvidenceNoticeDTO]

def get_user_financial_context(user_id: str, request_id: str) -> UserFinancialContextDTO:
    """Fetches resolved profile, request parameters, available options, and evidence."""
```

### Tool 2: `simulate_cash_flow`
```python
class PaymentItemDTO:
    date: str  # YYYY-MM-DD
    amount: float

class SimulationResultDTO:
    is_safe: bool
    minimum_projected_balance: float
    balance_deficit_below_floor: float  # max(0.0, minimum_balance_to_keep - min_balance)
    daily_balance_trajectory: dict[str, float]  # date -> balance

def simulate_cash_flow(
    user_id: str,
    start_date: str,
    payments: list[PaymentItemDTO],
    spending_changes: list[SpendingChangeDTO] | None = None
) -> SimulationResultDTO:
    """Executes a 90-day daily cash flow projection checking the minimum balance invariant."""
```

### Tool 3: `solve_optimal_plan`
```python
class DecisionResultDTO:
    amount_safe_to_pay: float
    affordability_status: str  # affordable_now | affordable_with_plan | affordable_later | not_affordable
    recommended_payment_method: str  # full_payment | partial_payment | installments | wait | not_recommended
    payment_plan: str  # YYYY-MM-DD:amount|... or none
    earliest_date_for_full_payment: str  # YYYY-MM-DD or empty string
    spending_changes_needed: str  # stop:<id>|reduce_to:<id>:<amt> or none
    decision_explanation: str

def solve_optimal_plan(user_context: UserFinancialContextDTO) -> DecisionResultDTO:
    """Deterministically evaluates all candidate plans, tests spending reductions if necessary,
    and applies the 6-level tie-breaking hierarchy."""
```

### Tool 4: `generate_grounded_explanation`
```python
def generate_grounded_explanation(
    context: UserFinancialContextDTO,
    decision: DecisionResultDTO
) -> str:
    """Generates a concise, factual explanation adhering to the exact style observed in sample_requests.csv."""
```

---

## 5. Decision Rules, Invariants & Hierarchy

### 5.1 Cash Flow Simulation Rules (90-Day Forecast)
1. **Starting Balance**: `current_available_balance` on `request_date`.
2. **Inflows**:
   - Confirmed salary counted on its settlement date (updated if employer message explicitly amends salary).
   - **Zero Speculative Inflows**: Do not count pending credits, bonuses, commissions, refunds, lottery, or investment returns until settled.
3. **Outflows**:
   - Historical recurring expenses projected along established cadences.
   - All pending and scheduled debits must be reserved on their settlement date.
   - Unrealized investments (`direction == 'non_cash'`) are ignored.
   - Cancelled and failed transactions are ignored.
4. **Safety Invariant**:
   ```text
   For all t in [request_date, request_date + 90d]:
     Balance(t) >= minimum_balance_to_keep
   ```

### 5.2 Metrics Computation
- `amount_safe_to_pay`: Maximum amount safe on `request_date` **before** optional spending changes, subject to `0 <= amount_safe_to_pay <= requested_amount`.
- `earliest_date_for_full_payment`: First date within the 90-day forecast where a single full payment of `requested_amount` satisfies the safety invariant without optional spending changes. Equals `request_date` if `affordable_now`. Empty string `""` if impossible within 90 days.

### 5.3 Candidate Plan Generation & Eligibility
1. **Full Payment Today**:
   - Requires `'full_payment'` in `payment_methods_user_will_consider`.
   - Payment: `request_date:requested_amount`.
2. **Partial Payment**:
   - Strictly consists of **exactly two transactions**:
     * First transaction: pay `amount_safe_to_pay` on `request_date`.
     * Second transaction: pay the remaining `requested_amount - amount_safe_to_pay` on `earliest_date_for_full_payment`.
     * The two payments must sum exactly to `requested_amount`.
   - Requires `allows_partial_payment == True`.
   - Requires `'partial_payment'` in `payment_methods_user_will_consider`.
   - Requires `0 < amount_safe_to_pay < requested_amount`.
   - Requires `earliest_date_for_full_payment <= desired_completion_date`.
   - Affordability status must be `affordable_with_plan`.
   - Does NOT need to match an option in `request_payment_options.csv`.
   - Plan string: `request_date:amount_safe_to_pay | earliest_date_for_full_payment:(requested_amount - amount_safe_to_pay)`.
3. **Installments**:
   - Requires `'installments'` in `payment_methods_user_will_consider`.
   - Duration must satisfy user's `max_installment_months`.
   - Must strictly match an offer in `request_payment_options.csv`.
4. **Wait (Affordable Later)**:
   - Requires `'full_payment'` in `payment_methods_user_will_consider`.
   - Requires `earliest_date_for_full_payment <= desired_completion_date`.
   - Plan: `earliest_date_for_full_payment:requested_amount`.
5. **Fallback (Not Recommended)**:
   - Plan: `none`. `affordability_status = not_affordable`.

### 5.4 Spending Changes Rules
- Evaluated only when an eligible plan has a deficit.
- Only recurring expenses marked `flexibility` in `['stoppable', 'reducible', 'reducible_or_stoppable']`.
- Must belong to categories in `expense_categories_user_is_willing_to_stop` or `...willing_to_reduce`.
- Protected categories (`expense_categories_to_protect`) can **never** be touched.
- Max 3 actions separated by `|`: `stop:<event_id>` or `reduce_to:<event_id>:<amount>`.
- Stop and reduce on the same event are mutually exclusive.

### 5.5 Plan Selection Hierarchy (Strict Tie-Breakers)
When multiple safe eligible plans exist, rank by:
1. Complete the full request by `desired_completion_date`.
2. Require no spending changes (`spending_changes_needed == 'none'`).
3. Minimize total payable amount (including financing fees).
4. Earliest first payment date.
5. Minimum number of payment installments.
6. Lowest `payment_option_id` as final tie-breaker.

---

## 6. Output Contract & Validation Schema

Every output row must strictly validate against:

| Column | Type | Allowed Values / Format |
|---|---|---|
| `request_id` | str | `request_26` to `request_275` (exact order) |
| `amount_safe_to_pay` | float | String representation of float, rounded or formatted matching sample style, $\ge 0$ |
| `affordability_status` | enum | `affordable_now`, `affordable_with_plan`, `affordable_later`, `not_affordable` |
| `recommended_payment_method`| enum | `full_payment`, `partial_payment`, `installments`, `wait`, `not_recommended` |
| `payment_plan` | str | `YYYY-MM-DD:amount\|...` or `none` |
| `earliest_date_for_full_payment` | str | `YYYY-MM-DD` or empty string `""` |
| `spending_changes_needed` | str | `stop:<id>\|reduce_to:<id>:<amt>` (max 3) or `none` |
| `decision_explanation` | str | Grounded factual explanation |

---

## 7. Multimodal Evidence & Image Amounts

The 16 financial events with empty `amount` attributes link to `images.csv` and `dataset/media/images/<image_id>.png`.
- **Multimodal OCR Extraction**: Extract transaction amounts using a structured Gemini 3.8 Flash call directly on each of the 16 images.
- **Persistent Caching**: Cache the results immediately to `dataset/extracted_image_amounts.json` with keys as `image_id` / `event_id` and values containing `extracted_amount`, `currency`, and `confidence`.
- **Deterministic Replacement**: The downstream engine loads `extracted_image_amounts.json` directly, eliminating redundant API calls and guaranteeing 100% deterministic reproducibility across repeated runs.
- The extracted amounts replace empty string amounts in `financial_events.csv` prior to 90-day simulation.

---

## 8. Verification & Delivery Contract

1. **Benchmark Suite**: `code/evaluation/main.py` runs predictions on `dataset/sample_requests.csv` and measures 100% exact match on all deterministic fields against the 25 ground-truth records.
2. **Production Pipeline**: `code/main.py` reads `dataset/requests.csv`, processes all 250 requests, validates all contract rules, and writes `output.csv` to the repository root.
3. **Usage Report**: `code/evaluation/usage_report.md` documents all model invocations, token counts, and cost breakdown.
4. **Chat Transcript**: `log.txt` kept updated at every turn.

---

## 9. Advanced Optimization Pillars (Encouraged Challenge Focus)

The system implements the 8 core optimization vectors recommended in the problem statement:

1. **Entity-Scoped RAG / Retrieval**:
   - Strictly scoped to `user_id`, `request_id`, and `related_event_id`. Eliminates global noise, cross-user leakage, and vector index overhead while guaranteeing 100% precision in evidence binding.
2. **Multimodal Interpretation**:
   - Deep multimodal document understanding with Gemini 3.8 Flash extracting net amounts, taxes, invoice totals, and currencies from PNG slips and invoices.
3. **Financial-State Reconstruction**:
   - Cleans duplicate transactions, resolves linked event lifecycles, converts foreign FX rates via fixed tables, isolates non-cash/unrealized items, and incorporates employer salary amendments.
4. **Plan Generation**:
   - Deterministic candidate plan generator formulating full payments, 2-transaction partial schedules, and eligible provider installment tracks.
5. **Deterministic Verification**:
   - Strict daily liquidity simulation testing the invariant `Balance(t) >= minimum_balance_to_keep` across every single day `t in [0, 90d]`.
6. **Batching**:
   - Batch evaluation and concurrent request processing for high-throughput pipeline execution.
7. **Persistent Caching**:
   - Local JSON caching of OCR and structured evidence (`dataset/extracted_image_amounts.json`) ensuring deterministic, zero-token, zero-latency re-runs.
8. **Token Efficiency & Cost Accounting**:
   - Purely deterministic numerical simulation guarantees zero token spend on arithmetic, reserving LLM calls solely for vision extraction and grounded explanation synthesis, with complete cost tracking in `evaluation/usage_report.md`.

---

## 10. Algorithmic Efficiency & Engineering Invariants

To eliminate code smells, performance bottlenecks, and redundant computational cycles, all core engines must strictly adhere to the following algorithmic principles:

1. **Suffix Minima Dynamic Programming (DP)**:
   - Calculating `earliest_date_for_full_payment` must run in a single backward pass $O(N)$ over the baseline 90-day trajectory rather than executing nested daily simulations ($O(N^2)$). Suffix minima `suff_min[k] = min_{t=k..90} B[t]` resolves whether payment `P` on day `k` violates `floor + P` in $O(1)$ time per day.
2. **Precomputation & Greedy Relief Sorting**:
   - In `SpendingOptimizer`, stoppable and reducible actions are precomputed once per base event. Candidates are sorted descending by financial relief (`relief = amount - min_amount`) so combination searches (1, 2, and 3 changes) greedily find valid liquidity solutions on the earliest simulation iterations.
3. **Direct ISO-8601 String Operations**:
   - Because all dates adhere to standard `YYYY-MM-DD` formatting, date ordering and range checks ($\le, \ge, ==$) must use direct string comparisons. Year and month extraction must use string slicing (`s[:4]`, `s[5:7]`) instead of repetitive, expensive `datetime.strptime()` calls.
4. **Elimination of Redundant Simulations**:
   - If `amount_safe_to_pay >= requested_amount`, Full Payment Today is mathematically guaranteed to be safe and must not re-run simulation.
   - If `earliest_date_for_full_payment` was computed and validated by Suffix Minima DP, the Wait Plan must not re-simulate the verified date.
5. **Robust Base Entity Tracking**:
   - Projected recurring events must track their originating `base_event_id` directly to prevent fragile string manipulation and ensure accurate spending change attribution.
6. **Month-End Date Drift Preservation**:
   - If a historical event settled on the last day of a month (e.g., Feb 28 in a non-leap year), future projections must preserve month-end placement (snapping to March 31, April 30, May 31) rather than remaining pinned to day 28.

