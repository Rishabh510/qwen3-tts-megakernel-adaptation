#!/usr/bin/env python
"""Browser demo: microphone audio -> Whisper -> Gemini -> adapted Qwen3-TTS."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from qwen3_tts_megakernel import StreamingSynthesizer, SynthesizerConfig
from qwen3_tts_megakernel.demo_frontend import (
    WhisperTranscriber,
    generate_gemini_reply,
    load_dotenv,
)


class VoiceDemo:
    def __init__(self, args) -> None:
        load_dotenv()
        self.args = args
        self.transcriber = WhisperTranscriber(
            model=args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
        )
        self.synthesizer = StreamingSynthesizer(
            SynthesizerConfig(
                model_id=args.model_id,
                chunk_frames=args.chunk_frames,
                first_chunk_frames=args.first_chunk_frames,
                max_frames=args.max_frames,
                notes="gradio_voice_demo",
            )
        )
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _tts_to_file(self, text: str) -> tuple[str, float, float]:
        chunks = []
        sample_rate = 24000
        started = time.perf_counter()
        async for audio, sample_rate in self.synthesizer.stream(text):
            chunks.append(audio)
        merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        output_path = self.output_dir / "gradio_reply.wav"
        sf.write(output_path, merged, sample_rate)
        total_ms = (time.perf_counter() - started) * 1000
        return str(output_path), self.synthesizer.metrics.ttfc_ms, total_ms

    def handle_audio(self, audio_path: str | None):
        if not audio_path:
            raise gr.Error("Record or upload a short audio clip first.")

        transcript, stt_ms = self.transcriber.transcribe(audio_path)
        prompt = (
            "You are a concise voice assistant. Reply in one short, natural sentence.\n"
            f"User said: {transcript}"
        )
        reply, llm_ms = generate_gemini_reply(prompt)
        output_path, ttfc_ms, tts_ms = asyncio.run(self._tts_to_file(reply))
        timings = (
            f"STT: {stt_ms:.0f} ms\n"
            f"LLM: {llm_ms:.0f} ms\n"
            f"TTS first chunk: {ttfc_ms:.0f} ms\n"
            f"TTS total: {tts_ms:.0f} ms\n"
            f"RTF: {self.synthesizer.metrics.rtf:.3f}"
        )
        return transcript, reply, output_path, timings


def build_app(demo: VoiceDemo) -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS Megakernel Voice Demo") as app:
        gr.Markdown("# Qwen3-TTS Megakernel Voice Demo")
        gr.Markdown("Record a short prompt, then play the generated spoken reply.")
        with gr.Row():
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your voice",
                format="wav",
            )
            audio_out = gr.Audio(type="filepath", label="Generated reply", autoplay=True)
        run = gr.Button("Run voice demo", variant="primary")
        transcript = gr.Textbox(label="Whisper transcript")
        reply = gr.Textbox(label="Gemini reply")
        timings = gr.Textbox(label="Timings", lines=5)
        run.click(
            demo.handle_audio,
            inputs=[audio_in],
            outputs=[transcript, reply, audio_out, timings],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--chunk-frames", type=int, default=20)
    parser.add_argument("--first-chunk-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", default="auto")
    parser.add_argument("--whisper-compute-type", default="default")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    app = build_app(VoiceDemo(args))
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
