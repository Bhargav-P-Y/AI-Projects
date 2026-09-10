import os
import json
import base64
import time
import mimetypes
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from data_pipeline.config import get_api_keys


@dataclass
class MediaContent:
    media_id: str
    media_type: str  # "image" or "voice"
    file_path: str
    extracted_text: str
    summary: str
    extracted_at: str


class MediaExtractor:
    """Multimodal media extractor using Gemini 3.6 Flash for image OCR and audio ASR with caching and thread-parallel extraction."""

    IMAGE_PROMPT = (
        "You are an expert document AI and OCR system for WhatsApp notification routing.\n"
        "1. Transcribe ALL visible text in the image verbatim, preserving original layout, headings, dates, phone numbers, UPI/QR details, and prices.\n"
        "2. If the image is a poster, circular, flyer, bill receipt, or screenshot, summarize its core intent, urgency, and action items for the user in 1-2 clear sentences."
    )

    VOICE_PROMPT = (
        "You are an expert speech-to-text transcription system for WhatsApp voice notes.\n"
        "1. Transcribe the audio verbatim in its original language (English, Hindi, or Hinglish/code-mixed).\n"
        "2. Provide an accurate English translation if spoken in Hindi or code-mixed language.\n"
        "3. Note the tone, urgency, and any specific requests or action items mentioned by the speaker."
    )

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        cache_dir: str = "code/cache",
        model_name: str = "gemini-3.6-flash",
    ):
        self.api_keys = api_keys or get_api_keys()
        self.key_index = 0
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "media_cache.json"
        self.cache: Dict[str, dict] = self._load_cache()

    def _load_cache(self) -> Dict[str, dict]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _get_next_key(self) -> str:
        if not self.api_keys:
            raise ValueError("No Gemini API keys provided in environment (.env)")
        key = self.api_keys[self.key_index % len(self.api_keys)]
        self.key_index += 1
        return key

    def _call_gemini_multimodal(self, prompt: str, file_path: str, fallback_mime: str) -> str:
        """Sends an image or audio file to Gemini 3.6 Flash multimodal REST API."""
        if not os.path.exists(file_path):
            return f"[File not found: {file_path}]"

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = fallback_mime

        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
        }

        max_attempts = max(4, len(self.api_keys) * 2) if self.api_keys else 1
        for attempt in range(max_attempts):
            api_key = self._get_next_key()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=35) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    try:
                        text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        return text
                    except (KeyError, IndexError):
                        return "[Empty extraction response]"
            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limit -> rotate to next key
                    time.sleep(0.5)
                    continue
                else:
                    time.sleep(1.0)
                    continue
            except Exception:
                time.sleep(1.0)
                continue

        return f"[Media extraction unavailable for {os.path.basename(file_path)}]"

    def extract_image(self, image_id: str, file_path: str, dataset_dir: str = "dataset") -> MediaContent:
        """Extracts OCR text and scene details from an image file."""
        if image_id in self.cache:
            c = self.cache[image_id]
            return MediaContent(**c)

        full_path = str(Path(dataset_dir) / file_path)
        extracted = self._call_gemini_multimodal(self.IMAGE_PROMPT, full_path, "image/jpeg")

        content = MediaContent(
            media_id=image_id,
            media_type="image",
            file_path=file_path,
            extracted_text=extracted,
            summary=extracted,
            extracted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        self.cache[image_id] = content.__dict__
        self._save_cache()
        return content

    def extract_voice_note(self, voice_note_id: str, file_path: str, dataset_dir: str = "dataset") -> MediaContent:
        """Transcribes audio from a voice note file."""
        if voice_note_id in self.cache:
            c = self.cache[voice_note_id]
            return MediaContent(**c)

        full_path = str(Path(dataset_dir) / file_path)
        extracted = self._call_gemini_multimodal(self.VOICE_PROMPT, full_path, "audio/mp3")

        content = MediaContent(
            media_id=voice_note_id,
            media_type="voice",
            file_path=file_path,
            extracted_text=extracted,
            summary=extracted,
            extracted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        self.cache[voice_note_id] = content.__dict__
        self._save_cache()
        return content

    def extract_all(
        self,
        images_df: Any,
        voice_notes_df: Any,
        dataset_dir: str = "dataset",
        max_workers: int = 4,
    ) -> Dict[str, MediaContent]:
        """Processes and caches all images and voice notes using parallel threads across API keys."""
        results: Dict[str, MediaContent] = {}

        images_records = images_df.to_dict("records")
        voice_records = voice_notes_df.to_dict("records")

        num_workers = min(max_workers, len(self.api_keys)) if self.api_keys else 1


        print(f"Extracting {len(images_records)} images with {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.extract_image, row["image_id"], row["file_path"], dataset_dir): row["image_id"]
                for row in images_records
            }
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    results[img_id] = future.result()
                except Exception as e:
                    print(f"Error extracting image {img_id}: {e}")

        print(f"Extracting {len(voice_records)} voice notes with {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.extract_voice_note, row["voice_note_id"], row["file_path"], dataset_dir): row["voice_note_id"]
                for row in voice_records
            }
            for future in as_completed(futures):
                vn_id = futures[future]
                try:
                    results[vn_id] = future.result()
                except Exception as e:
                    print(f"Error extracting voice note {vn_id}: {e}")

        return results
