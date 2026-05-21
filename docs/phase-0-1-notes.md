# Phase 0/1 Notes

## Goal

Confirm the project setup path and identify the exact model boundary before
spending RTX 5090 rental time.

## GPU Rental Recommendation

Use Vast.ai first, with RunPod as backup.

Reasoning:

- Vast.ai is a practical first choice for short RTX 5090 experiments.
- Vast.ai commonly has low RTX 5090 spot pricing, which matters for short
  budget-limited experiments.
- RunPod is simpler operationally and advertises RTX 5090 on-demand availability,
  but it is usually more expensive.

Instance requirements:

- GPU: single RTX 5090, 32 GB VRAM.
- CUDA: 12.8 or newer.
- Disk: at least 80 GB preferred, because Qwen3-TTS, caches, PyTorch wheels, and
  build artifacts add up quickly.
- Image: PyTorch + CUDA 12.8 if available; otherwise Ubuntu with NVIDIA drivers
  and CUDA 12.8+.
- Access: SSH is enough. A browser UI is optional.

Cost expectation:

- At roughly $0.25-$0.60/hr on Vast.ai spot, $10 buys about 16-40 hours.
- At roughly $0.99/hr on RunPod, $10 buys about 10 hours.
- Prefer preparing scripts locally before starting the paid GPU.

## Service Choices

The project specifies Qwen3-TTS for TTS, but does not specify STT or the
conversation LLM.

Recommended first-version choices:

- STT: browser/manual typed prompt first, then local Whisper if voice input is
  needed.
- Text LLM: Gemini Flash/Lite, OpenAI, or even a fixed canned response.
- TTS: Qwen3-TTS-12Hz-0.6B, because it is closest to the megakernel's Qwen3-0.6B
  target.

The strongest v1 demo can be:

1. Speak or type input.
2. STT/manual transcript creates text.
3. Text LLM or canned responder creates a short response.
4. Custom Qwen3-TTS service streams audio chunks.
5. Pipecat receives audio frames incrementally.

## Shape Audit

`qwen_megakernel` constants:

| Field | Value |
| --- | --- |
| hidden size | 1024 |
| intermediate size | 3072 |
| layers | 28 |
| Q heads | 16 |
| KV heads | 8 |
| head dim | 128 |
| Q size | 2048 |
| KV size | 1024 |
| text vocab / LM head | 151936 |
| max seq len | 2048 |
| dtype | bfloat16 |

`Qwen/Qwen3-TTS-12Hz-0.6B-*` talker config:

| Field | Value |
| --- | --- |
| hidden size | 1024 |
| intermediate size | 3072 |
| layers | 28 |
| Q heads | 16 |
| KV heads | 8 |
| head dim | 128 |
| talker codec vocab | 3072 |
| text vocab | 151936 |
| num code groups | 16 |
| rope theta | 1000000 |
| max position embeddings | 32768 |
| dtype target | bfloat16 |

This means the transformer body is very close to the megakernel's assumptions.
The first adaptation should focus on:

- Loading `talker.model.*` weights instead of `model.*` weights.
- Using `talker.model.codec_embedding.weight` as the input embedding.
- Using `talker.codec_head.weight` as the output projection.
- Reducing the LM/head vocab path from 151936 text tokens to 3072 codec tokens.
- Matching Qwen3-TTS RoPE settings, especially `rope_theta=1000000`.
- Handling `inputs_embeds` prefill, because Qwen3-TTS builds mixed text+codec
  embeddings before calling `self.talker.generate`.

## Important Model Boundary

Qwen3-TTS generation has two token paths:

- Main talker token: first codec codebook token, produced by
  `Qwen3TTSTalkerForConditionalGeneration`.
- Subtalker/code predictor: generates the remaining codebook tokens for each
  audio step.

The target is the talker decoder, not the codebook generator.
So v1 should replace the main talker model forward/decode path while leaving the
subtalker/code predictor and speech tokenizer decode path intact.

## Phase 1 Risk

The megakernel currently takes `input_token_id` and performs an embedding lookup
inside CUDA. Qwen3-TTS often calls the talker with `inputs_embeds`, not raw token
IDs, during prefill. For an honest implementation we likely need one of these:

1. A prefill path that uses PyTorch for prompt setup, then megakernel for
   single-step decode.
2. A kernel entrypoint that accepts a ready hidden vector instead of token id.
3. A wrapper that maps Qwen3-TTS generation into token-id based steps where
   possible.

Fastest v1 path: keep PyTorch prefill and use the megakernel for the iterative
decode step after the cache is initialized.
