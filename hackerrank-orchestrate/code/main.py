"""Message Notification Router — Main Pipeline Entrypoint.

Integrates Phases 1-6 end-to-end:
1. Data Loading & User Profile Building
2. Multimodal Media Extraction (OCR / ASR cached)
3. 2-Stage Safety Engine (Hard Threat Fast-Track + Risk Signal Extraction)
4. Hybrid Evidence Retrieval & XML-Sandboxed Context Assembly
5. Gemini 3.6 Flash Batch LLM Routing with 8-Key Round-Robin Rotation
6. Post-Processing Confidence Calibration & Strict 8-Point Output CSV Validation
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure code/ is in Python path regardless of execution directory
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

# Load .env variables from code/.env before importing key dependencies
from data_pipeline.config import load_env_file, get_api_keys
load_env_file(str(code_dir / ".env"))

from data_pipeline.data_loader import DataLoader
from data_pipeline.profile_builder import ProfileBuilder
from media_extractor.media_extractor import MediaExtractor
from safety_filter.safety_filter import SafetyFilter
from safety_filter.semantic_injection import SemanticInjectionDetector
from context_builder.hybrid_retriever import HybridRetriever
from context_builder.context_builder import ContextBuilder
from llm_router.llm_router import LLMRouter, RoutingDecision
from output_writer.confidence_calibrator import ConfidenceCalibrator
from output_writer.output_writer import OutputWriter

# Configure ASCII-safe logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MainPipeline")


def run_pipeline(
    dataset_dir: str = "dataset",
    output_path: str = "dataset/output.csv",
    batch_size: int = 3,
) -> pd.DataFrame:
    """Executes the Message Notification Router pipeline end-to-end."""
    start_time = time.monotonic()
    logger.info("=== Starting Message Notification Router Pipeline ===")

    # -------------------------------------------------------------------------
    # 0. API Keys & Setup
    # -------------------------------------------------------------------------
    api_keys = get_api_keys()
    if not api_keys:
        logger.error("No Gemini API keys found. Please set GEMINI_API_KEYS in code/.env.")
        sys.exit(1)
    logger.info(f"[Setup] Loaded {len(api_keys)} API key(s) for rotation.")

    # -------------------------------------------------------------------------
    # Phase 1: Data Loading & User Profile Building
    # -------------------------------------------------------------------------
    p1_start = time.monotonic()
    logger.info("[Phase 1] Loading dataset CSVs and building User Profiles...")
    data_bundle = DataLoader(dataset_dir).load_all()
    user_profiles = ProfileBuilder.build_all(data_bundle)
    logger.info(
        f"[Phase 1] Loaded {len(data_bundle.messages)} incoming messages, "
        f"{len(data_bundle.message_history)} historical messages, and built {len(user_profiles)} user profiles in {time.monotonic() - p1_start:.2f}s."
    )

    # -------------------------------------------------------------------------
    # Phase 2: Multimodal Media Extraction (Cached)
    # -------------------------------------------------------------------------
    p2_start = time.monotonic()
    logger.info("[Phase 2] Loading/extracting multimodal content (OCR & ASR)...")
    extractor = MediaExtractor(api_keys=api_keys, cache_dir=str(code_dir / "cache"))
    media_cache = extractor.extract_all(data_bundle.images, data_bundle.voice_notes)
    logger.info(f"[Phase 2] Loaded {len(media_cache)} extracted media items in {time.monotonic() - p2_start:.2f}s.")

    # -------------------------------------------------------------------------
    # Phase 3: Safety Filtering & Risk Signal Engine
    # -------------------------------------------------------------------------
    p3_start = time.monotonic()
    logger.info("[Phase 3] Running 2-Stage Safety Engine (Hard Threats + Risk Signals)...")
    semantic_detector = SemanticInjectionDetector()
    safety_filter = SafetyFilter(data_bundle, semantic_detector=semantic_detector)

    fast_tracked_decisions: dict = {}
    remaining_messages: list = []
    risk_signals_map: dict = {}

    messages_list = data_bundle.messages.to_dict("records")
    for msg in messages_list:
        user_id = msg["user_id"]
        profile = user_profiles.get(user_id)
        hard_threat = safety_filter.check_hard_threats(msg, profile)

        if hard_threat:
            fast_tracked_decisions[hard_threat.message_id] = LLMRouter.fast_track_to_decision(hard_threat)
            logger.info(f"  [Safety Fast-Track] {hard_threat.message_id} -> {hard_threat.action}/{hard_threat.message_type} ({hard_threat.filter_layer})")
        else:
            remaining_messages.append(msg)
            risk_signals_map[msg["message_id"]] = safety_filter.compute_risk_signals(msg)

    logger.info(
        f"[Phase 3] Fast-tracked {len(fast_tracked_decisions)} hard security threats. "
        f"{len(remaining_messages)} messages passed to LLM pipeline in {time.monotonic() - p3_start:.2f}s."
    )

    # -------------------------------------------------------------------------
    # Phase 4: Hybrid Evidence Retrieval & XML-Sandboxed Context Assembly
    # -------------------------------------------------------------------------
    p4_start = time.monotonic()
    logger.info("[Phase 4] Retrieving historical evidence and assembling XML-sandboxed contexts...")
    retriever = HybridRetriever(data_bundle, semantic_detector=semantic_detector)
    context_builder = ContextBuilder(data_bundle, user_profiles, media_cache, retriever)

    assembled_contexts: list = []
    contexts_map: dict = {}

    for msg in remaining_messages:
        msg_id = msg["message_id"]
        risk_sig = risk_signals_map[msg_id]
        ctx = context_builder.build(msg, risk_sig)
        assembled_contexts.append(ctx)
        contexts_map[msg_id] = ctx

    logger.info(f"[Phase 4] Assembled {len(assembled_contexts)} structured contexts in {time.monotonic() - p4_start:.2f}s.")

    # -------------------------------------------------------------------------
    # Phase 5: Gemini Batch LLM Routing
    # -------------------------------------------------------------------------
    p5_start = time.monotonic()
    logger.info(f"[Phase 5] Routing {len(assembled_contexts)} contexts via Gemini 3.6 Flash batch API...")
    router = LLMRouter(api_keys=api_keys, batch_size=batch_size)
    llm_decisions = router.route_batch(assembled_contexts)
    logger.info(f"[Phase 5] LLM routing completed in {time.monotonic() - p5_start:.2f}s.")

    # Merge fast-track + LLM decisions
    all_decisions: dict[str, RoutingDecision] = {**fast_tracked_decisions, **llm_decisions}

    # -------------------------------------------------------------------------
    # Phase 6: Confidence Calibration & Output Export
    # -------------------------------------------------------------------------
    p6_start = time.monotonic()
    logger.info("[Phase 6] Calibrating decision confidence scores & validating schema...")
    calibrated_decisions = ConfidenceCalibrator.calibrate_all(
        decisions=all_decisions,
        risk_signals_map=risk_signals_map,
        user_profiles=user_profiles,
        data_bundle=data_bundle,
        contexts_map=contexts_map,
    )

    df_output = OutputWriter.write_csv(
        decisions=calibrated_decisions,
        expected_messages=data_bundle.messages,
        output_path=output_path,
    )
    logger.info(f"[Phase 6] Confidence calibration and CSV export completed in {time.monotonic() - p6_start:.2f}s.")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_elapsed = time.monotonic() - start_time
    logger.info("\n=== END-TO-END PIPELINE EXECUTION SUMMARY ===")
    logger.info(f"Total Messages Processed: {len(df_output)}")
    logger.info(f"Fast-Track Security Mutes: {len(fast_tracked_decisions)}")
    logger.info(f"LLM-Routed Messages: {len(llm_decisions)}")
    logger.info(f"Output File Written: {output_path}")
    logger.info(f"Total Execution Time: {total_elapsed:.2f} seconds")
    print(f"\n[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY! Wrote {len(df_output)} rows to '{output_path}'.")

    return df_output


if __name__ == "__main__":
    import pandas as pd
    run_pipeline()
