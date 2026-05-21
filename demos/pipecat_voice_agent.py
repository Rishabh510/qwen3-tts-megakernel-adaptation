#!/usr/bin/env python
"""Pipecat voice-agent harness with local Whisper STT and pluggable text LLM."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from qwen3_tts_megakernel.pipecat_service import MegakernelQwenTTSService

logger = logging.getLogger(__name__)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env without overriding existing variables."""
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


async def run_websocket_agent(args) -> None:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    stt = WhisperSTTService(
        model=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )

    if args.llm_provider == "gemini":
        from pipecat.services.google.llm import GoogleLLMService

        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            model=os.getenv("GEMINI_MODEL", args.gemini_model),
        )
    else:
        llm_kwargs = {
            "api_key": os.environ["OPENAI_API_KEY"],
            "model": os.getenv("OPENAI_MODEL", args.openai_model),
        }
        if os.getenv("OPENAI_BASE_URL"):
            llm_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        llm = OpenAILLMService(**llm_kwargs)
    tts = MegakernelQwenTTSService(
        model_id=args.model_id,
        chunk_frames=args.chunk_frames,
        notes=args.notes,
    )

    context = LLMContext(
        [
            {
                "role": "system",
                "content": (
                    "You are a concise voice assistant. Keep replies short, plain, "
                    "and easy to speak aloud."
                ),
            }
        ]
    )
    user_ctx, assistant_ctx = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    transport = FastAPIWebsocketTransport(
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
        host=args.host,
        port=args.port,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_ctx,
            llm,
            tts,
            transport.output(),
            assistant_ctx,
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("client disconnected")
        await task.cancel()

    await PipelineRunner().run(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--notes", default="")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", default="auto")
    parser.add_argument("--whisper-compute-type", default="default")
    parser.add_argument("--llm-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    required_key = "GOOGLE_API_KEY" if args.llm_provider == "gemini" else "OPENAI_API_KEY"
    missing = [required_key] if not os.getenv(required_key) else []
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Use demos/pipecat_text_only.py for a no-API-key TTS check.")
        sys.exit(1)

    asyncio.run(run_websocket_agent(args))


if __name__ == "__main__":
    main()
