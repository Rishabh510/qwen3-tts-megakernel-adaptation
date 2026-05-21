#!/usr/bin/env python
"""Smoke test the non-GPU demo front half: local Whisper STT and Gemini LLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    if not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = ".cache/huggingface"
    if not os.getenv("NLTK_DATA"):
        os.environ["NLTK_DATA"] = ".cache/nltk_data"


def check_gemini(prompt: str, model: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing in .env")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 48, "temperature": 0.2},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
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
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"gemini_ok model={model} latency_ms={elapsed_ms:.1f} reply={text!r}")
    return text


def download_librispeech_sample(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    rows_url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=hf-internal-testing/librispeech_asr_dummy"
        "&config=clean&split=validation&offset=0&length=1"
    )
    with urllib.request.urlopen(rows_url, timeout=30) as response:
        row_data = json.loads(response.read().decode("utf-8"))
    audio_url = row_data["rows"][0]["row"]["audio"][0]["src"]
    with urllib.request.urlopen(audio_url, timeout=60) as response:
        output_path.write_bytes(response.read())
    print(f"downloaded_stt_sample path={output_path}")
    return output_path


def check_whisper(audio_path: Path, model: str, device: str, compute_type: str) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run scripts/setup_frontend_smoke_env.sh first."
        ) from exc

    started = time.perf_counter()
    whisper = WhisperModel(model, device=device, compute_type=compute_type)
    segments, info = whisper.transcribe(str(audio_path), beam_size=1)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not text:
        raise RuntimeError("Whisper returned an empty transcript")
    print(
        "whisper_ok "
        f"model={model} language={info.language} latency_ms={elapsed_ms:.1f} "
        f"transcript={text!r}"
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Reply in one short sentence: hello from the demo.")
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", default="cpu")
    parser.add_argument("--whisper-compute-type", default="int8")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--download-sample", action="store_true", default=True)
    args = parser.parse_args()

    load_dotenv()

    if not args.skip_gemini:
        check_gemini(args.prompt, os.getenv("GEMINI_MODEL", args.gemini_model))

    if not args.skip_whisper:
        audio = args.audio or download_librispeech_sample(Path("outputs/stt_sample.flac"))
        check_whisper(audio, args.whisper_model, args.whisper_device, args.whisper_compute_type)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_failed: {exc}", file=sys.stderr)
        sys.exit(1)
