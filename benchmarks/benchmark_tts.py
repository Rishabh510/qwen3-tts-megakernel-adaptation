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


def append_benchmark_row(csv_path: Path, row: dict[str, object]) -> None:
    rows: list[dict[str, str]] = []
    existing_fields: list[str] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            rows = list(reader)

    fields = existing_fields[:]
    for key in row:
        if key not in fields:
            fields.append(key)

    rows.append({key: str(row.get(key, "")) for key in fields})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
            "codebook_ms": f"{metrics.codebook_ms:.3f}",
            "first_codebook_ms": f"{metrics.first_codebook_ms:.3f}",
            "codebook_ms_per_frame": f"{metrics.codebook_ms_per_frame:.3f}",
            "vocoder_ms": f"{metrics.vocoder_ms:.3f}",
            "first_vocoder_ms": f"{metrics.first_vocoder_ms:.3f}",
            "vocoder_ms_per_chunk": f"{metrics.vocoder_ms_per_chunk:.3f}",
            "audio_seconds": f"{metrics.audio_seconds:.6f}",
            "rtf": f"{metrics.rtf:.6f}",
            "chunks": metrics.chunks,
            "frames": metrics.frames,
            "notes": args.notes,
        }
        append_benchmark_row(csv_path, row)

        print(
            f"{run_type}: output={output_path} ttfc_ms={metrics.ttfc_ms:.1f} "
            f"rtf={metrics.rtf:.3f} init_ms={metrics.init_ms:.1f} "
            f"codebook_ms={metrics.codebook_ms:.1f} "
            f"vocoder_ms={metrics.vocoder_ms:.1f}"
        )

    print(f"appended_csv={csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hi there! How can I help you")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--csv", default="outputs/benchmark_results.csv")
    parser.add_argument("--output", default="outputs/benchmark.wav")
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--notes", default="")
    asyncio.run(run_once(parser.parse_args()))


if __name__ == "__main__":
    main()
