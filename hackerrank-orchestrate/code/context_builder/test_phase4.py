import json
import logging
import os
import sys
import time
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from data_pipeline.data_loader import DataLoader
from data_pipeline.profile_builder import ProfileBuilder
from safety_filter.safety_filter import SafetyFilter
from safety_filter.semantic_injection import SemanticInjectionDetector
from .hybrid_retriever import HybridRetriever
from .context_builder import ContextBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestPhase4")


def run_phase4_tests():
    logger.info("=== Starting Phase 4: Context Assembly & Hybrid Evidence Retriever Tests ===")
    start_time = time.time()

    # 1. Load Data
    data_loader = DataLoader("dataset")
    data_bundle = data_loader.load_all()
    logger.info(f"Loaded dataset successfully. Messages: {len(data_bundle.messages)}")

    # 2. Build User Profiles
    profiles = ProfileBuilder.build_all(data_bundle)
    logger.info(f"Built {len(profiles)} user profiles.")

    # 3. Initialize Semantic Detector & Safety Filter
    detector = SemanticInjectionDetector()
    safety_filter = SafetyFilter(data_bundle, semantic_detector=detector)

    # 4. Initialize Hybrid Retriever
    retriever = HybridRetriever(data_bundle, semantic_detector=detector)
    logger.info("HybridRetriever initialized successfully.")

    # 5. Load Media Cache if present
    media_cache_path = code_dir / "cache" / "media_cache.json"
    media_cache = {}
    if media_cache_path.exists():
        with open(media_cache_path, "r", encoding="utf-8") as f:
            media_cache = json.load(f)
        logger.info(f"Loaded {len(media_cache)} media items from cache.")

    # 6. Initialize Context Builder
    context_builder = ContextBuilder(
        data_bundle=data_bundle,
        profiles=profiles,
        media_cache=media_cache,
        retriever=retriever,
    )
    logger.info("ContextBuilder initialized successfully.")

    # 7. Run Test Suite over all 110 incoming messages
    messages_list = data_bundle.messages.to_dict("records")
    valid_history_ids = set(data_bundle.message_history["message_id"])

    evidence_found_count = 0
    none_evidence_count = 0
    xml_sandboxed_count = 0
    media_embedded_count = 0

    for msg in messages_list:
        msg_id = msg["message_id"]

        # Stage 1 & 2 Safety Filter checks
        hard_threat = safety_filter.check_hard_threats(msg, profiles.get(msg["user_id"]))
        risk_signals = safety_filter.compute_risk_signals(msg)

        # Build Context
        ctx = context_builder.build(msg, risk_signals=risk_signals)

        # Assertions & Verification
        assert ctx.message_id == msg_id, f"Mismatch in message_id {ctx.message_id} vs {msg_id}"
        assert "<untrusted_user_message>" in ctx.prompt_text, f"Missing XML sandbox in {msg_id}"
        assert "</untrusted_user_message>" in ctx.prompt_text, f"Missing XML sandbox closing in {msg_id}"
        xml_sandboxed_count += 1

        # Check evidence IDs format and validity
        ev_str = ctx.evidence_message_ids_str
        assert isinstance(ev_str, str), f"Evidence string for {msg_id} must be str"

        if ev_str != "none":
            evidence_found_count += 1
            e_ids = ev_str.split(";")
            for eid in e_ids:
                assert eid in valid_history_ids, f"Invalid evidence ID {eid} for message {msg_id}"
        else:
            none_evidence_count += 1

        if "<untrusted_media_content>" in ctx.prompt_text:
            media_embedded_count += 1

    elapsed = time.time() - start_time
    logger.info("\n=== PHASE 4 TEST RESULTS SUMMARY ===")
    logger.info(f"Total Messages Tested: {len(messages_list)}")
    logger.info(f"XML Sandboxed Contexts: {xml_sandboxed_count} (100.0%)")
    logger.info(f"Evidence Retrieved (IDs): {evidence_found_count} messages")
    logger.info(f"No Evidence ('none'): {none_evidence_count} messages")
    logger.info(f"Multimodal OCR/ASR Embedded: {media_embedded_count} messages")
    logger.info(f"Total Phase 4 Execution Time: {elapsed:.2f} seconds")

    # Sample output inspection
    sample_ctx = context_builder.build(messages_list[0], safety_filter.compute_risk_signals(messages_list[0]))
    logger.info("\n--- SAMPLE ASSEMBLED CONTEXT (msg_001) ---")
    logger.info(sample_ctx.prompt_text[:1000] + "\n... [truncated] ...")

    print("\n[SUCCESS] PHASE 4 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_phase4_tests()
