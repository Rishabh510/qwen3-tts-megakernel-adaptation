"""Streaming text-to-speech orchestration for the adapted megakernel path."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np
import torch

from .adapter import CodebookPredictorKernel, TalkerKernelAdapter, TextProjector
from .constants import (
    CODEC_BOS_ID,
    CODEC_EOS_ID,
    CODEC_NOTHINK_ID,
    CODEC_PAD_ID,
    CODEC_THINK_BOS_ID,
    CODEC_THINK_EOS_ID,
    NUM_CODE_GROUPS,
    TTS_BOS_TOKEN_ID,
    TTS_EOS_TOKEN_ID,
    TTS_PAD_TOKEN_ID,
)
from .weights import collect_talker_weights, resolve_model_dir


@dataclass
class SynthesizerConfig:
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda"
    sample_rate: int = 24000
    chunk_frames: int = 10
    first_chunk_frames: int = 1
    max_frames: int | None = None
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    notes: str = ""


@dataclass
class SynthesisMetrics:
    init_ms: float = 0.0
    ttfc_ms: float = 0.0
    total_ms: float = 0.0
    audio_seconds: float = 0.0
    chunks: int = 0
    frames: int = 0
    notes: str = ""

    @property
    def rtf(self) -> float:
        return self.total_ms / 1000.0 / self.audio_seconds if self.audio_seconds else 0.0


class StreamingSynthesizer:
    """Generate audio chunks from text with adapted talker/codebook kernels."""

    def __init__(self, config: SynthesizerConfig | None = None):
        self.config = config or SynthesizerConfig()
        self.metrics = SynthesisMetrics(notes=self.config.notes)
        self._ready = False

    def initialize(self) -> None:
        if self._ready:
            return
        t0 = time.perf_counter()
        cfg = self.config
        self.weights = collect_talker_weights(cfg.model_id, device=cfg.device)
        self.talker = TalkerKernelAdapter(self.weights, device=cfg.device)
        self.codebook_predictor = CodebookPredictorKernel(self.weights, device=cfg.device)
        self.text_projector = TextProjector(self.weights)
        self.codec_embedding = self.weights["codec_embedding"]
        self.codebook_embeddings = [
            self.weights["code_predictor"][f"codec_embedding.{idx}.weight"]
            for idx in range(NUM_CODE_GROUPS - 1)
        ]

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.speech_tokenizer = self._load_speech_tokenizer()
        self._precompute_constants()
        self._warmup()
        torch.cuda.synchronize()
        self.metrics.init_ms = (time.perf_counter() - t0) * 1000
        self._ready = True

    def _load_speech_tokenizer(self):
        from qwen_tts import Qwen3TTSTokenizer

        model_dir = resolve_model_dir(self.config.model_id)
        tokenizer_dir = model_dir / "speech_tokenizer"
        tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_dir))
        self.config.sample_rate = tokenizer.get_output_sample_rate()
        return tokenizer

    def _precompute_constants(self) -> None:
        device = self.config.device
        special = torch.tensor(
            [TTS_PAD_TOKEN_ID, TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID],
            dtype=torch.long,
            device=device,
        )
        projected = self.text_projector(special)
        self.tts_pad = projected[0]
        self.tts_bos = projected[1]
        self.tts_eos = projected[2]

        role_ids = self.tokenizer.encode("<|im_start|>assistant\n", return_tensors="pt")[0][:3]
        self.role_embeddings = self.text_projector(role_ids.to(device))

        codec_ids = torch.tensor(
            [
                CODEC_NOTHINK_ID,
                CODEC_THINK_BOS_ID,
                CODEC_THINK_EOS_ID,
                CODEC_PAD_ID,
                CODEC_BOS_ID,
            ],
            dtype=torch.long,
            device=device,
        )
        codec = torch.nn.functional.embedding(codec_ids, self.codec_embedding)
        self.prefix_embeddings = torch.cat(
            [
                self.tts_pad.expand(3, -1),
                self.tts_bos.unsqueeze(0),
            ],
            dim=0,
        ) + codec[:4]
        self.codec_bos_embedding = codec[4]

    def _warmup(self) -> None:
        self.talker.reset()
        _, hidden = self.talker.step_token(CODEC_BOS_ID)
        for do_sample in (False, False, True, True, True):
            self.codebook_predictor.predict(
                hidden,
                CODEC_BOS_ID,
                self.codec_embedding,
                do_sample=do_sample,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
        for frame_count in (1, 1, 5):
            dummy_codes = torch.randint(
                0,
                2048,
                (frame_count, NUM_CODE_GROUPS),
                dtype=torch.long,
                device=self.config.device,
            )
            self.speech_tokenizer.decode([{"audio_codes": dummy_codes}])
        torch.cuda.synchronize()
        self.talker.reset()

    def _prompt_embeddings(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer.encode(formatted, return_tensors="pt")[0].to(self.config.device)
        content = ids[3:]
        content_embeddings = self.text_projector(content)
        first_text = content_embeddings[:1] + self.codec_bos_embedding.unsqueeze(0)
        prefill = torch.cat([self.role_embeddings, self.prefix_embeddings, first_text], dim=0)
        trailing = torch.cat([content_embeddings[1:-5], self.tts_eos.unsqueeze(0)], dim=0)
        return prefill, trailing

    def _estimate_frame_limit(self, text: str) -> int:
        if self.config.max_frames:
            return self.config.max_frames
        word_count = max(1, len(text.split()))
        estimated_seconds = word_count / 2.5
        return max(24, min(2048, int(estimated_seconds * 12.5 * 2.0)))

    def iter_codec_frames(self, text: str):
        self.initialize()
        cfg = self.config
        self.talker.reset()
        prefill, trailing = self._prompt_embeddings(text)
        for row in prefill:
            self.talker.step_embedding(row)

        first_token, hidden = self.talker.step_token(CODEC_BOS_ID)
        previous_token = first_token
        trailing_idx = 0
        max_frames = self._estimate_frame_limit(text)

        for _ in range(max_frames):
            if previous_token == CODEC_EOS_ID:
                break
            codes = self.codebook_predictor.predict(
                hidden,
                previous_token,
                self.codec_embedding,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )
            yield codes

            embed_sum = torch.nn.functional.embedding(
                codes[0:1], self.codec_embedding
            ).squeeze(0)
            for group_idx, embedding_table in enumerate(self.codebook_embeddings):
                embed_sum = embed_sum + torch.nn.functional.embedding(
                    codes[group_idx + 1 : group_idx + 2], embedding_table
                ).squeeze(0)
            if trailing_idx < trailing.shape[0]:
                embed_sum = embed_sum + trailing[trailing_idx]
                trailing_idx += 1
            else:
                embed_sum = embed_sum + self.tts_pad
            previous_token, hidden = self.talker.step_embedding(embed_sum)

    def decode_frames(self, frames: list[torch.Tensor]) -> tuple[np.ndarray, int]:
        if not frames:
            return np.zeros(0, dtype=np.float32), self.config.sample_rate
        codes = torch.stack(frames, dim=0)
        wavs, sample_rate = self.speech_tokenizer.decode([{"audio_codes": codes}])
        return wavs[0], sample_rate

    async def stream(self, text: str) -> AsyncGenerator[tuple[np.ndarray, int], None]:
        was_ready = self._ready
        self.initialize()
        init_ms = 0.0 if was_ready else self.metrics.init_ms
        cfg = self.config
        self.metrics = SynthesisMetrics(init_ms=init_ms, notes=cfg.notes)
        buffer: list[torch.Tensor] = []
        first = True
        samples = 0
        start = time.perf_counter()

        for frame in self.iter_codec_frames(text):
            buffer.append(frame)
            target = cfg.first_chunk_frames if first else cfg.chunk_frames
            if len(buffer) >= target:
                audio, sample_rate = self.decode_frames(buffer)
                buffer.clear()
                samples += len(audio)
                self.metrics.chunks += 1
                self.metrics.frames += target
                if first:
                    torch.cuda.synchronize()
                    self.metrics.ttfc_ms = (time.perf_counter() - start) * 1000
                    first = False
                yield audio, sample_rate
                await asyncio.sleep(0)

        if buffer:
            audio, sample_rate = self.decode_frames(buffer)
            samples += len(audio)
            self.metrics.chunks += 1
            self.metrics.frames += len(buffer)
            yield audio, sample_rate

        torch.cuda.synchronize()
        self.metrics.total_ms = (time.perf_counter() - start) * 1000
        self.metrics.audio_seconds = samples / float(cfg.sample_rate)
