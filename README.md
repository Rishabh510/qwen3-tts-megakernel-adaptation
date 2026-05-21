# Qwen3-TTS Megakernel Adaptation

Adapt Qwen3-TTS 12Hz 0.6B talker decoding to an RTX 5090 CUDA megakernel,
stream audio chunks, and record benchmark results.

## Current Scope

First version target:

- Use `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- Use a patched TTS build of the CUDA decode kernel.
- Use text-only benchmarks for TTS measurements.
- Use local Whisper STT and Gemini/OpenAI-compatible LLM only for the demo harness.
- Append benchmark runs to a CSV for iterative comparison.

## Adaptation vs Optimization

Adaptation changes are the minimum needed for Qwen3-TTS to run through the
patched CUDA path: TTS vocab size, talker weight mapping, codec head, RoPE scale,
precomputed embeddings, prefill construction, and vocoder decode.

Optimizations are added after correctness is visible: warmup, codebook predictor
kernel reuse, cached embeddings, first-frame streaming, chunk-size tuning, and
stopping/audio-quality tuning.

See [Optimization Plan](docs/optimization-plan.md) for the sequence to add and
benchmark one change at a time.

## Dependency Boundary

This project does not modify or clone the Qwen3-TTS source repository. Setup
installs `qwen-tts` from PyPI and uses it for tokenizer, vocoder, and checkpoint
compatibility. The only vendored source patched by this repo is the CUDA decode
kernel under `vendor/qwen_megakernel`.

## Setup On RTX 5090

Use a CUDA 12.8+ RTX 5090 machine.

```bash
git clone <your-repo-url>
cd qwen3-tts-megakernel-adaptation
./scripts/gpu_bootstrap.sh
```

The bootstrap script:

- creates `.env` from `.env.template` if it does not exist
- clones the CUDA kernel source into `vendor/`
- applies local CUDA patches
- creates `.venv`
- installs GPU dependencies, `qwen-tts`, and this package
- verifies CUDA visibility

Run the smoke test:

```bash
source .venv/bin/activate
./scripts/run_phase1_smoke.sh
```

Run only the streaming demo:

```bash
python demos/text_to_streaming_audio.py \
  --text "Hi there! How can I help you" \
  --output outputs/streaming_demo.wav
```

Append one benchmark row:

```bash
python benchmarks/benchmark_tts.py \
  --text "Hi there! How can I help you" \
  --notes "baseline_adapted_talker"
```

The benchmark CSV is written to `outputs/benchmark_results.csv` by default. It
runs twice by default: first row is cold, second row is warm. Compare warm rows
when judging latency changes; cold rows help explain model load, JIT, and warmup
cost.

Run the no-API-key Pipecat-style TTS harness:

```bash
python demos/pipecat_text_only.py --text "Hi there! How can I help you"
```

Run the websocket voice-agent harness with local Whisper STT and Gemini:

```bash
# Fill GOOGLE_API_KEY in .env first.
python demos/pipecat_voice_agent.py --port 8765
```

OpenAI-compatible LLM mode is also available:

```bash
# Fill OPENAI_API_KEY in .env first.
python demos/pipecat_voice_agent.py --llm-provider openai --port 8765
```

## Working Notes

- [Phase 0/1 Notes](docs/phase-0-1-notes.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Optimization Plan](docs/optimization-plan.md)
- [Beginner Glossary](docs/beginner-glossary.md)
