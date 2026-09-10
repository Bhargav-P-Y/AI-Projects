"""Evaluation Workflow — Message Notification Router.

Evaluates pipeline accuracy and performance against dataset/sample_messages.csv
ground truth benchmark (22 annotated sample messages).

Computes:
1. Action Accuracy (notify / digest / mute)
2. Message Type Accuracy (11 taxonomy categories)
3. Evidence Retrieval Recall & Match Rate
4. Confidence Score Calibration Distribution (mean, min, max, std)
5. Safety Fast-Track Precision
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from data_pipeline.config import load_env_file, get_api_keys
load_env_file(str(code_dir / ".env"))

from data_pipeline.data_loader import DataLoader, DataBundle
from data_pipeline.profile_builder import ProfileBuilder
from media_extractor.media_extractor import MediaExtractor
from safety_filter.safety_filter import SafetyFilter
from safety_filter.semantic_injection import SemanticInjectionDetector
from context_builder.hybrid_retriever import HybridRetriever
from context_builder.context_builder import ContextBuilder
from llm_router.llm_router import LLMRouter, RoutingDecision
from output_writer.confidence_calibrator import ConfidenceCalibrator
from output_writer.output_writer import OutputWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvalWorkflow")


def evaluate_pipeline(
    dataset_dir: str = "dataset",
    sample_file: str = "sample_messages.csv",
    report_output: str = "code/evaluation/eval_report.json",
) -> dict:
    """Runs evaluation workflow against sample_messages.csv benchmark."""
    logger.info("=== Starting Evaluation Workflow ===")
    sample_path = Path(dataset_dir) / sample_file

    if not sample_path.exists():
        logger.error(f"Sample benchmark file '{sample_path}' not found.")
        sys.exit(1)

    df_sample = pd.read_csv(sample_path)
    logger.info(f"Loaded ground truth benchmark: {len(df_sample)} sample messages from '{sample_file}'.")

    # 1. Load data bundle and profiles
    data_bundle = DataLoader(dataset_dir).load_all()
    user_profiles = ProfileBuilder.build_all(data_bundle)

    # 2. Multimodal media extraction
    api_keys = get_api_keys()
    extractor = MediaExtractor(api_keys=api_keys, cache_dir=str(code_dir / "cache"))
    media_cache = extractor.extract_all(data_bundle.images, data_bundle.voice_notes)

    # 3. Run pipeline on sample messages
    semantic_detector = SemanticInjectionDetector()
    safety_filter = SafetyFilter(data_bundle, semantic_detector=semantic_detector)
    retriever = HybridRetriever(data_bundle, semantic_detector=semantic_detector)
    context_builder = ContextBuilder(data_bundle, user_profiles, media_cache, retriever)

    fast_tracked_decisions = {}
    remaining_messages = []
    risk_signals_map = {}

    sample_records = df_sample.to_dict("records")
    for msg in sample_records:
        msg_id = msg["message_id"]
        user_id = msg["user_id"]
        profile = user_profiles.get(user_id)
        hard_threat = safety_filter.check_hard_threats(msg, profile)

        if hard_threat:
            fast_tracked_decisions[msg_id] = LLMRouter.fast_track_to_decision(hard_threat)
        else:
            remaining_messages.append(msg)
            risk_signals_map[msg_id] = safety_filter.compute_risk_signals(msg)

    assembled_contexts = []
    contexts_map = {}
    for msg in remaining_messages:
        msg_id = msg["message_id"]
        risk_sig = risk_signals_map[msg_id]
        ctx = context_builder.build(msg, risk_sig)
        assembled_contexts.append(ctx)
        contexts_map[msg_id] = ctx

    router = LLMRouter(api_keys=api_keys, batch_size=3)
    llm_decisions = router.route_batch(assembled_contexts)
    all_decisions = {**fast_tracked_decisions, **llm_decisions}

    calibrated_decisions = ConfidenceCalibrator.calibrate_all(
        decisions=all_decisions,
        risk_signals_map=risk_signals_map,
        user_profiles=user_profiles,
        data_bundle=data_bundle,
        contexts_map=contexts_map,
    )

    # 4. Calculate Accuracy Metrics
    action_correct = 0
    type_correct = 0
    evidence_hits = 0
    confidences = []

    results_table = []
    for msg in sample_records:
        msg_id = msg["message_id"]
        gt_action = str(msg["action"]).strip().lower()
        gt_type = str(msg["message_type"]).strip().lower()
        gt_ev = str(msg["evidence_message_ids"]).strip().lower()

        pred = calibrated_decisions.get(msg_id)
        if not pred:
            continue

        pred_action = pred.action.strip().lower()
        pred_type = pred.message_type.strip().lower()
        pred_ev = pred.evidence_message_ids.strip().lower()

        a_match = pred_action == gt_action
        t_match = pred_type == gt_type
        e_match = pred_ev == gt_ev or (pred_ev != "none" and gt_ev != "none" and any(e in gt_ev for e in pred_ev.split(";")))

        if a_match:
            action_correct += 1
        if t_match:
            type_correct += 1
        if e_match:
            evidence_hits += 1

        confidences.append(pred.confidence)

        results_table.append({
            "message_id": msg_id,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "action_match": a_match,
            "gt_type": gt_type,
            "pred_type": pred_type,
            "type_match": t_match,
            "confidence": pred.confidence,
        })

    total = len(sample_records)
    action_acc = (action_correct / total) * 100.0 if total > 0 else 0.0
    type_acc = (type_correct / total) * 100.0 if total > 0 else 0.0
    evidence_acc = (evidence_hits / total) * 100.0 if total > 0 else 0.0

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    min_conf = min(confidences) if confidences else 0.0
    max_conf = max(confidences) if confidences else 0.0

    report = {
        "benchmark_file": sample_file,
        "sample_count": total,
        "action_accuracy_pct": round(action_acc, 2),
        "message_type_accuracy_pct": round(type_acc, 2),
        "evidence_retrieval_recall_pct": round(evidence_acc, 2),
        "confidence_statistics": {
            "mean": round(mean_conf, 2),
            "min": round(min_conf, 2),
            "max": round(max_conf, 2),
        },
        "details": results_table,
    }

    # Save evaluation report JSON
    os.makedirs(os.path.dirname(report_output), exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n=== EVALUATION WORKFLOW RESULTS ===")
    logger.info(f"Sample Count: {total}")
    logger.info(f"Action Accuracy: {action_acc:.2f}% ({action_correct}/{total})")
    logger.info(f"Message Type Accuracy: {type_acc:.2f}% ({type_correct}/{total})")
    logger.info(f"Evidence Retrieval Recall: {evidence_acc:.2f}% ({evidence_hits}/{total})")
    logger.info(f"Confidence Stats: mean={mean_conf:.2f}, min={min_conf:.2f}, max={max_conf:.2f}")
    logger.info(f"Saved evaluation report to '{report_output}'.")

    return report


if __name__ == "__main__":
    evaluate_pipeline()
