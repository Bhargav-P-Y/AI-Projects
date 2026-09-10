import json
import logging
import os
import sys
import time
from pathlib import Path

code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

# Load .env from code/ directory before importing anything that reads env vars
from data_pipeline.config import load_env_file
load_env_file(str(code_dir / ".env"))

from data_pipeline.data_loader import DataLoader
from data_pipeline.profile_builder import ProfileBuilder
from safety_filter.safety_filter import SafetyFilter
from safety_filter.semantic_injection import SemanticInjectionDetector
from context_builder.hybrid_retriever import HybridRetriever
from context_builder.context_builder import ContextBuilder
from llm_router import LLMRouter, RoutingDecision, ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestPhase5")

SAMPLE_IDS_TO_TEST = [
    "msg_001",  # business HDFC notify
    "msg_013",  # forward/chain mute
    "msg_023",  # scam fast-tracked
]


def run_phase5_tests():
    logger.info("=== Starting Phase 5: LLM Router Tests ===")
    start_time = time.monotonic()

    # Load API keys via the shared config helper (which already read .env)
    from data_pipeline.config import get_api_keys
    api_keys = get_api_keys()
    if not api_keys:
        logger.error("No API keys found. Ensure .env in code/ has GEMINI_API_KEYS or GEMINI_API_KEY.")
        sys.exit(1)
    logger.info(f"Loaded {len(api_keys)} API key(s).")

    # 1. Load Data
    data = DataLoader("dataset").load_all()
    profiles = ProfileBuilder.build_all(data)
    logger.info(f"Loaded {len(data.messages)} messages, built {len(profiles)} profiles.")

    # 2. Safety Filter
    detector = SemanticInjectionDetector()
    safety = SafetyFilter(data, semantic_detector=detector)

    # 3. Media Cache
    media_cache_path = code_dir / "cache" / "media_cache.json"
    media_cache = {}
    if media_cache_path.exists():
        with open(media_cache_path, "r", encoding="utf-8") as f:
            media_cache = json.load(f)
    logger.info(f"Loaded {len(media_cache)} media items from cache.")

    # 4. Context Assembly
    retriever = HybridRetriever(data, semantic_detector=detector)
    ctx_builder = ContextBuilder(data, profiles, media_cache, retriever)

    # 5. Run safety + build contexts on first 5 messages for smoke test
    messages_list = data.messages.to_dict("records")[:5]
    fast_tracked: dict = {}
    remaining_contexts = []
    risk_signals_map = {}

    for msg in messages_list:
        ft = safety.check_hard_threats(msg, profiles.get(msg["user_id"]))
        if ft:
            fast_tracked[ft.message_id] = ft
            logger.info(f"  Fast-tracked: {ft.message_id} -> {ft.action}/{ft.message_type}")
        else:
            rs = safety.compute_risk_signals(msg)
            risk_signals_map[msg["message_id"]] = rs
            ctx = ctx_builder.build(msg, rs)
            remaining_contexts.append(ctx)

    logger.info(f"Fast-tracked: {len(fast_tracked)}, Remaining for LLM: {len(remaining_contexts)}")

    # 6. Route remaining via LLM
    router = LLMRouter(api_keys=api_keys, batch_size=3)
    llm_decisions = router.route_batch(remaining_contexts)

    # 7. Merge fast-track + LLM decisions
    all_decisions: dict[str, RoutingDecision] = {}
    for ft in fast_tracked.values():
        all_decisions[ft.message_id] = LLMRouter.fast_track_to_decision(ft)
    all_decisions.update(llm_decisions)

    # 8. Validate output schema
    logger.info("\n=== PHASE 5 VALIDATION RESULTS ===")
    schema_errors = 0
    for msg_id, dec in all_decisions.items():
        errors = []
        if dec.action not in ALLOWED_ACTIONS:
            errors.append(f"Invalid action: {dec.action}")
        if dec.message_type not in ALLOWED_MESSAGE_TYPES:
            errors.append(f"Invalid message_type: {dec.message_type}")
        if not (0.0 <= dec.confidence <= 1.0):
            errors.append(f"Confidence out of range: {dec.confidence}")
        if not dec.reason:
            errors.append("Empty reason")
        if errors:
            schema_errors += 1
            logger.error(f"  {msg_id}: {errors}")
        else:
            logger.info(
                f"  {msg_id}: [{dec.source}] action={dec.action}, type={dec.message_type}, "
                f"conf={dec.confidence:.2f}, evidence={dec.evidence_message_ids}"
            )

    elapsed = time.monotonic() - start_time
    logger.info(f"\nTotal Messages Routed: {len(all_decisions)}")
    logger.info(f"Schema Errors: {schema_errors}")
    logger.info(f"Phase 5 Smoke Test Execution Time: {elapsed:.2f}s")

    if schema_errors == 0:
        print("\n[SUCCESS] PHASE 5 SMOKE TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"\n[FAIL] {schema_errors} schema errors detected.")
        sys.exit(1)


if __name__ == "__main__":
    run_phase5_tests()
