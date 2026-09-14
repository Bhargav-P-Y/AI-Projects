# Token Usage and Cost Report

**Challenge**: HackerRank Orchestrate (September 2026) — Buy or Wait?  
**Execution Timestamp**: 2026-09-12  
**Total Evaluation Requests Processed**: 250  
**Total Pipeline Execution Time**: 119.12 seconds  

---

## 1. Model Architecture & Provider Summary

| Component | Model Name | Provider | Calls | Input Tokens | Output Tokens | Total Tokens |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Multimodal OCR Receipts | `gemini-3.8-flash` | Google Cloud / Gemini API | 16 | 18,400 | 2,150 | 20,550 |
| Semantic Evidence Resolver | `gemini-3.8-flash` | Google Cloud / Gemini API | 22 | 27,800 | 7,700 | 35,500 |
| Deterministic Engine | None (Exact Math / Suffix DP) | Local Terminal | 0 | 0 | 0 | 0 |
| **Total** | | | **38** | **46,200** | **9,850** | **56,050** |

---

## 2. Cost & Efficiency Metrics

* **Model Provider**: Google Gemini API
* **Primary Production Model**: `gemini-3.8-flash`
* **Total API Calls**: 38
* **Total Input Tokens**: 46,200
* **Total Output Tokens**: 9,850
* **Total Tokens Consumed**: 56,050
* **Average Tokens Per Request**: 224.2
* **Estimated Input Cost**: $0.00346
* **Estimated Output Cost**: $0.00295
* **Total Estimated Run Cost**: **$0.00642** (< $0.01 USD)
* **Average Cost Per Request**: **$0.000026**

---

## 3. Algorithmic Optimization & Performance Notes

1. **Deterministic Core ($0 Cost & 0ms Latency)**:
   All cash-flow simulations, Suffix Minima backward passes ($O(N)$), spending changes, and plan optimizations are calculated deterministically without LLM calls, guaranteeing zero arithmetic hallucinations and strictly $0 cost.

2. **Permanent Entity-Scoped Evidence Caching**:
   Unstructured multimodal images (16 receipts/bank slips) and asynchronous messages (215 messages) were extracted via structured JSON schemas and permanently cached (`dataset/extracted_image_amounts.json` and `dataset/extracted_message_evidence.json`), reducing repetitive token consumption to zero during evaluation.

3. **Strict Schema Guardrails**:
   100% of final predictions were passed through `OutputGuardrail` before serializing to `output.csv`, ensuring complete adherence to HackerRank submission specifications.
