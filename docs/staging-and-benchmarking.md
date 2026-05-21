# Staging And Benchmarking

## Fast GPU Session Order

1. Start the rented RTX 5090 machine and clone this repo.
2. Fill `.env`.
3. Run `./scripts/gpu_bootstrap.sh`.
4. Run `./scripts/run_phase1_smoke.sh`.
5. Record the generated WAV files and CSV rows from `outputs/`.

Use HTTPS for the public repo:

```bash
git clone https://github.com/Rishabh510/qwen3-tts-megakernel-adaptation.git
```

SSH keys are not needed on the rented machine unless you plan to push from it.
For repeated updates, run:

```bash
git pull --ff-only
```

## Browser Demo

Run:

```bash
./scripts/run_gradio_demo.sh
```

Recommended access path from your laptop:

```bash
ssh -L 7860:localhost:7860 root@<vast-host> -p <ssh-port>
```

Then open `http://localhost:7860` locally, record a short prompt, click the run
button, and play the generated reply.

If port forwarding is inconvenient, set `GRADIO_SHARE=1` in `.env` before
running the script. That asks Gradio to create a temporary public share URL.

## Non-GPU Front-Half Smoke Test

Gemini can be tested without GPU:

```bash
python scripts/smoke_stt_llm.py --skip-whisper
```

Gemini plus local Whisper STT can be tested before renting a GPU:

```bash
./scripts/setup_frontend_smoke_env.sh
source .venv-front/bin/activate
python scripts/smoke_stt_llm.py --whisper-model tiny
```

If `--audio` is omitted, the script downloads a tiny public LibriSpeech sample.
For your own sample, create a short voice memo, export it as WAV/M4A/FLAC, and
pass that file to `--audio`.

## Patch Strategy

The CUDA source patch is currently one adaptation patch:

- `patches/qwen-megakernel-00-tts-adaptation.patch`

That patch covers the kernel-side requirements needed before Qwen3-TTS can use
the decode kernel: configurable output vocab size and a precomputed-embedding
sentinel path.

The patch applier supports a sorted patch series named
`patches/qwen-megakernel-*.patch`. If a future optimization edits CUDA, add a
new patch file such as `qwen-megakernel-01-<change>.patch` and rerun
`./scripts/apply_vendor_patches.sh`.

The remaining planned optimizations mostly live in this repo's Python adapter,
synthesizer, benchmark harness, and demo scripts. Stage those as git commits or
branches instead of CUDA patch files unless they actually edit
`vendor/qwen_megakernel`.

Recommended branch sequence after the first GPU validation:

1. `main`: base adaptation and benchmark harness.
2. `opt/01-single-run-benchmark`: benchmark reporting only.
3. `opt/02-codebook-kernel`: codebook predictor kernel path.
4. `opt/03-constant-embedding-cache`: cached prompt/special embeddings.
5. `opt/04-vocoder-sampling-warmup`: warmup timing cleanup.
6. `opt/05-first-frame-streaming`: first audio chunk after one codec frame.
7. `opt/06-chunk-size-sweep`: tune chunk size from CSV evidence.
8. `opt/07-stop-quality-tuning`: stopping and audio quality cleanup.

Benchmark each branch with:

```bash
./benchmarks/benchmark_current_stage.sh
```

Use rows with matching prompt, branch, and notes for latency comparison.
