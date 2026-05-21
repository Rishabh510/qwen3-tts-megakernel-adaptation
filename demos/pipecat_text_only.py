#!/usr/bin/env python
"""Text-only Pipecat-adjacent harness for fast TTS validation.

This is intentionally minimal: it exercises the same streaming service logic
without requiring microphone, browser, STT, or LLM credentials.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen3_tts_megakernel.pipecat_service import float32_to_pcm16
from qwen3_tts_megakernel import StreamingSynthesizer, SynthesizerConfig


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hi there! How can I help you")
    parser.add_argument("--output", default="outputs/pipecat_text_only.wav")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    args = parser.parse_args()

    synth = StreamingSynthesizer(SynthesizerConfig(model_id=args.model_id))
    pcm_chunks = []
    float_chunks = []
    sample_rate = 24000
    async for audio, sample_rate in synth.stream(args.text):
        pcm = float32_to_pcm16(audio)
        pcm_chunks.append(pcm)
        float_chunks.append(audio)
        print(f"pipecat_audio_frame={len(pcm_chunks)} pcm_bytes={len(pcm)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged = np.concatenate(float_chunks) if float_chunks else np.zeros(0, dtype=np.float32)
    sf.write(args.output, merged, sample_rate)
    print(f"wrote={args.output}")
    print(f"frames={len(pcm_chunks)} ttfc_ms={synth.metrics.ttfc_ms:.1f} rtf={synth.metrics.rtf:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

