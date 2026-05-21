#!/usr/bin/env python
"""Generate streamed TTS audio and save it as a WAV file."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen3_tts_megakernel import StreamingSynthesizer, SynthesizerConfig


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hi there! How can I help you")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output", default="outputs/streaming_demo.wav")
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    synth = StreamingSynthesizer(
        SynthesizerConfig(
            model_id=args.model_id,
            chunk_frames=args.chunk_frames,
            notes=args.notes,
        )
    )
    chunks = []
    async for audio, sample_rate in synth.stream(args.text):
        chunks.append(audio)
        print(f"chunk={len(chunks)} samples={len(audio)} sr={sample_rate}")

    merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sf.write(args.output, merged, sample_rate)
    m = synth.metrics
    print(f"output={args.output}")
    print(f"init_ms={m.init_ms:.1f}")
    print(f"ttfc_ms={m.ttfc_ms:.1f}")
    print(f"total_ms={m.total_ms:.1f}")
    print(f"audio_seconds={m.audio_seconds:.3f}")
    print(f"rtf={m.rtf:.3f}")
    print(f"chunks={m.chunks}")


if __name__ == "__main__":
    asyncio.run(main())

