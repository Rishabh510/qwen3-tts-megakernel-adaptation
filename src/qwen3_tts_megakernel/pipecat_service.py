"""Pipecat TTS service backed by the streaming synthesizer."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

import numpy as np

from .synthesizer import StreamingSynthesizer, SynthesizerConfig

logger = logging.getLogger(__name__)


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


try:
    from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
    from pipecat.services.tts_service import TTSService
except Exception:  # pragma: no cover - lets non-Pipecat demos import this module.
    ErrorFrame = Frame = TTSAudioRawFrame = TTSStartedFrame = TTSStoppedFrame = None
    TTSService = object


class MegakernelQwenTTSService(TTSService):
    """Minimal Pipecat TTSService wrapper.

    The important contract is that audio frames are yielded chunk-by-chunk as the
    synthesizer produces them.
    """

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda",
        chunk_frames: int = 20,
        notes: str = "",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        if TTSService is object:
            raise RuntimeError("pipecat-ai is not installed")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.synthesizer = StreamingSynthesizer(
            SynthesizerConfig(
                model_id=model_id,
                device=device,
                chunk_frames=chunk_frames,
                notes=notes,
            )
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame(context_id=context_id)

            async for audio, sample_rate in self.synthesizer.stream(text):
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=float32_to_pcm16(audio),
                    sample_rate=sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as exc:
            logger.exception("TTS generation failed")
            yield ErrorFrame(error=f"TTS generation failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)
