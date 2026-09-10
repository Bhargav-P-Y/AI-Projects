import os
import sys

# Add parent directory ('code') to sys.path so imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.data_loader import DataLoader

try:
    from .media_extractor import MediaExtractor, MediaContent
except ImportError:
    from media_extractor import MediaExtractor, MediaContent


def run_phase2_tests():
    print("=== Running Media Extraction Package (Phase 2) Verification Tests ===")

    loader = DataLoader("dataset")
    data = loader.load_all()

    extractor = MediaExtractor(cache_dir="code/cache")

    print(f"[1] Media Extractor initialized.")
    print(f"    - Images in dataset: {len(data.images)}")
    print(f"    - Voice notes in dataset: {len(data.voice_notes)}")

    # 1. Test image extraction / cache lookup on img_008 (used in kurta pickup messages)
    img_008_path = data.image_path_map.get("img_008", "media/images/img_008.jpg")
    content_img008 = extractor.extract_image("img_008", img_008_path, dataset_dir="dataset")

    assert content_img008 is not None, "Failed to extract img_008"
    assert content_img008.media_id == "img_008", "Media ID mismatch for img_008"
    print(f"[2] Image OCR Test (img_008):")
    print(f"    - Extracted text sample: {content_img008.extracted_text[:120]}...")

    # 2. Test voice note transcription / cache lookup on vn_001
    vn_001_path = data.voice_note_path_map.get("vn_001", "media/audio/vn_001.mp3")
    content_vn001 = extractor.extract_voice_note("vn_001", vn_001_path, dataset_dir="dataset")

    assert content_vn001 is not None, "Failed to extract vn_001"
    assert content_vn001.media_id == "vn_001", "Media ID mismatch for vn_001"
    print(f"[3] Voice Note ASR Test (vn_001):")
    print(f"    - Transcribed text sample: {content_vn001.extracted_text[:120]}...")

    # 3. Test caching behavior
    cache_path = os.path.join("code", "cache", "media_cache.json")
    assert os.path.exists(cache_path), "Cache file media_cache.json should be created"
    print(f"[4] Local Caching Verified: {cache_path} exists.")

    print("=== Media Extraction Package (Phase 2) Verification Passed Cleanly! ===")


if __name__ == "__main__":
    run_phase2_tests()
