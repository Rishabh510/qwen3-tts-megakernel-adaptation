#!/usr/bin/env python
"""Run a TTS benchmark and append one row to a CSV file."""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen3_tts_megakernel import StreamingSynthesizer, SynthesizerConfig


async def run_once(args) -> None:
    synth = StreamingSynthesizer(
        SynthesizerConfig(
            model_id=args.model_id,
            chunk_frames=args.chunk_frames,
            max_frames=args.max_frames,
            notes=args.notes,
        )
    )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    writer = None

    for run_index in range(1, args.runs + 1):
        run_type = "cold" if run_index == 1 else "warm"
        chunks = []
        sample_rate = 24000
        async for audio, sample_rate in synth.stream(args.text):
            chunks.append(audio)

        output_path = Path(args.output)
        if args.runs > 1:
            output_path = output_path.with_name(
                f"{output_path.stem}_{run_type}{output_path.suffix}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        sf.write(output_path, merged, sample_rate)

        metrics = synth.metrics
        row = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_index": run_index,
            "run_type": run_type,
            "text": args.text,
            "model_id": args.model_id,
            "output": str(output_path),
            "chunk_frames": args.chunk_frames,
            "init_ms": f"{metrics.init_ms:.3f}",
            "ttfc_ms": f"{metrics.ttfc_ms:.3f}",
            "generation_ms": f"{metrics.total_ms:.3f}",
            "end_to_end_ms": f"{metrics.init_ms + metrics.total_ms:.3f}",
            "audio_seconds": f"{metrics.audio_seconds:.6f}",
            "rtf": f"{metrics.rtf:.6f}",
            "chunks": metrics.chunks,
            "frames": metrics.frames,
            "notes": args.notes,
        }
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
                is_new = False
            writer.writerow(row)

        print(
            f"{run_type}: output={output_path} ttfc_ms={metrics.ttfc_ms:.1f} "
            f"rtf={metrics.rtf:.3f} init_ms={metrics.init_ms:.1f}"
        )

    print(f"appended_csv={csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hi there! How can I help you")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--csv", default="outputs/benchmark_results.csv")
    parser.add_argument("--output", default="outputs/benchmark.wav")
    parser.add_argument("--chunk-frames", type=int, default=24)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--notes", default="")
    asyncio.run(run_once(parser.parse_args()))


if __name__ == "__main__":
    main()
