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

For Vast.ai, use an RTX 5090 instance with CUDA 12.8 and PyTorch 2.7+ support.
Use `/workspace` for the checkout so model caches and outputs land on the
attached volume.

```bash
cd /workspace
git clone https://github.com/Rishabh510/qwen3-tts-megakernel-adaptation.git
cd qwen3-tts-megakernel-adaptation
cp .env.template .env
nano .env
./scripts/gpu_bootstrap.sh
```

This project assumes the active CUDA toolkit is 12.8. Check before setup:

```bash
nvcc --version
```

If it is not CUDA 12.8, switch to a CUDA 12.8 instance/template before running
the TTS benchmark.

Fill at least these values in `.env`:

```text
GOOGLE_API_KEY=<your-key>
GEMINI_MODEL=gemini-3.1-flash-lite
HF_HOME=/workspace/.cache/huggingface
NLTK_DATA=/workspace/.cache/nltk_data
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
GRADIO_CHUNK_FRAMES=20
GRADIO_FIRST_CHUNK_FRAMES=1
WHISPER_MODEL=tiny
ATTN_IMPLEMENTATION=sdpa
INSTALL_FLASH_ATTN=0
```

The bootstrap script:

- creates `.env` from `.env.template` if it does not exist
- clones the CUDA kernel source into `vendor/`
- applies local CUDA patches
- creates `.venv`
- installs GPU dependencies, `qwen-tts`, and this package
- verifies CUDA visibility

FlashAttention is optional and disabled by default as it takes hours to compile. Leave `INSTALL_FLASH_ATTN=0` for the fastest setup path.

Run the smoke test:

```bash
source .venv/bin/activate
python scripts/smoke_stt_llm.py --whisper-model tiny
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
runs once by default because repeated generation in the same process can hang on
some rented GPU images. For steady comparisons, run the command once per code
state and compare rows with matching notes.

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

Run the browser voice demo:

```bash
cd /workspace/qwen3-tts-megakernel-adaptation
source .venv/bin/activate
./scripts/run_gradio_demo.sh
```

Then open it from your laptop using SSH port forwarding:

```bash
ssh -i <identity-file> -p <ssh-port> -L 7860:localhost:7860 root@<vast-host>
```

Open `http://localhost:7860` locally. If SSH forwarding is inconvenient, set
`GRADIO_SHARE=1` in `.env` before running the demo to request a temporary Gradio
share URL.

For the demo recording, show:

1. `outputs/benchmark_results.csv` with the latest `ttfc_ms`, `rtf`,
   `codebook_ms`, and `vocoder_ms` row.
2. The Gradio page at `http://localhost:7860`.
3. A short microphone prompt and the generated audio reply.
4. The timings textbox in the Gradio page.

Copy generated outputs back to your laptop from a local terminal:

```bash
scp -i <identity-file> -P <ssh-port> -r \
  root@<vast-host>:/workspace/qwen3-tts-megakernel-adaptation/outputs \
  ~/Downloads/qwen3-tts-outputs
```

To pull updates on the rented machine:

```bash
git pull --ff-only
```

Then rerun only the changed step, usually:

```bash
source .venv/bin/activate
./benchmarks/benchmark_current_stage.sh
```

Before renting a GPU, test the constant STT/LLM front half:

```bash
./scripts/setup_frontend_smoke_env.sh
source .venv-front/bin/activate
python scripts/smoke_stt_llm.py --whisper-model tiny
```

## Working Notes

- [Phase 0/1 Notes](docs/phase-0-1-notes.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Optimization Plan](docs/optimization-plan.md)
- [Staging And Benchmarking](docs/staging-and-benchmarking.md)
- [Beginner Glossary](docs/beginner-glossary.md)
