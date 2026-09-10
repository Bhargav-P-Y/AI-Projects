import os
import json

def load_jsonl(filepath):
    """Safely loads a JSONL file. Returns None if corrupted or missing."""
    if not os.path.exists(filepath):
        return None
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception:
        return None

def grade_task_1_bias(workspace_dir) -> tuple[float, str]:
    """EASY: Debug the heuristic toxicity filter."""
    data = load_jsonl(os.path.join(workspace_dir, "processed_dataset.jsonl"))
    if data is None: return 0.1, "Critical Failure: processed_dataset.jsonl is missing or invalid."
    if len(data) == 0: return 0.1, "Critical Failure: Dataset is empty."

    has_painkiller = False
    has_virus_kill = False

    for row in data:
        text = row.get("text", "").lower()
        if "painkiller" in text: has_painkiller = True
        if "computer virus" in text: has_virus_kill = True

    if has_painkiller and not has_virus_kill:
        return 0.9, "Success: Kept safe medical term, dropped toxic term."
    elif not has_painkiller and not has_virus_kill:
        return 0.1, "Failed: Over-censorship detected (dropped safe medical term 'painkiller')."
    elif has_painkiller and has_virus_kill:
        return 0.1, "Failed: Under-censorship detected (failed to drop toxic term)."
    else:
        return 0.1, "Failed: Unrecognized filter behavior or dataset wiped."

def grade_task_2_format(workspace_dir) -> tuple[float, str]:
    """MEDIUM: Check for strict Llama 3 token formatting."""
    data = load_jsonl(os.path.join(workspace_dir, "processed_dataset.jsonl"))
    if data is None: return 0.1, "Critical Failure: processed_dataset.jsonl is missing or invalid."
    if len(data) == 0: return 0.1, "Critical Failure: Dataset is empty."

    correct_rows = 0
    for row in data:
        text = row.get("text", "")
        if "<|begin_of_text|>" in text and "<|start_header_id|>user" in text and "<|start_header_id|>assistant" in text:
            correct_rows += 1

    raw_score = correct_rows / len(data)
    # CLAMP SCORE BETWEEN 0.1 and 0.9 to comply with Phase 2 bounds
    final_score = max(0.1, min(0.9, raw_score))
    return final_score, f"Formatted {correct_rows}/{len(data)} rows correctly with Llama 3 tokens."

def grade_task_3_pii(workspace_dir) -> tuple[float, str]:
    """HARD: Check for PII Redaction and Row Deduplication."""
    data = load_jsonl(os.path.join(workspace_dir, "processed_dataset.jsonl"))
    if data is None: return 0.1, "Critical Failure: processed_dataset.jsonl is missing or invalid."
    if len(data) == 0: return 0.1, "Critical Failure: Dataset is empty."

    score = 0.1 # Base score
    reasons = []

    # 1. Deduplication Check (Original has 6 rows, 1 duplicate. Expected = 5)
    if len(data) == 5:
        score += 0.4
        reasons.append("Deduplication successful.")
    else:
        reasons.append(f"Deduplication failed (Row count: {len(data)}, expected 5).")

    # 2. PII Scrubbing Check
    pii_found = False
    redacted_found = False
    for row in data:
        text = row.get("text", "")
        if "555-0198" in text: pii_found = True
        if "[REDACTED]" in text: redacted_found = True

    if not pii_found and redacted_found:
        score += 0.4
        reasons.append("PII successfully scrubbed.")
    else:
        reasons.append("PII scrubbing failed (Phone number still present or REDACTED tag missing).")

    return min(0.9, score), " | ".join(reasons)
