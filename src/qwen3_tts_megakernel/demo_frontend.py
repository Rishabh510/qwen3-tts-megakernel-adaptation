"""Shared helpers for browser demos and smoke tests."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env without overriding existing variables."""
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    os.environ.setdefault("HF_HOME", ".cache/huggingface")
    os.environ.setdefault("NLTK_DATA", ".cache/nltk_data")


def generate_gemini_reply(prompt: str, model: str | None = None) -> tuple[str, float]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing in .env")

    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{selected_model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 80, "temperature": 0.3},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc

    text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )
    if not text:
        raise RuntimeError(f"Gemini returned no text: {json.dumps(data)[:500]}")
    return text, (time.perf_counter() - started) * 1000


class WhisperTranscriber:
    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "default"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str | Path) -> tuple[str, float]:
        started = time.perf_counter()
        segments, info = self._load().transcribe(str(audio_path), beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not text:
            raise RuntimeError(f"Whisper returned an empty transcript; language={info.language}")
        return text, elapsed_ms
