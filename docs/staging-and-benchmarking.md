# Staging And Benchmarking

## Fast GPU Session Order

1. Start the rented RTX 5090 machine and clone this repo.
2. Fill `.env`.
3. Run `./scripts/gpu_bootstrap.sh`.
4. Run one text-only TTS benchmark.
5. Run the Gradio browser demo and record the screen.
6. Copy the generated WAV files and CSV rows from `outputs/`.

Use HTTPS for the public repo:

```bash
git clone https://github.com/Rishabh510/qwen3-tts-megakernel-adaptation.git
```

SSH keys are not needed on the rented machine unless you plan to push from it.
For repeated updates, run:

```bash
git pull --ff-only
```

## Final Benchmark

The final measured RTX 5090 row for the short prompt was:

```text
ttfc_ms=39.0 rtf=0.161 codebook_ms=606.0 vocoder_ms=241.5 chunk_frames=20
```

Run the benchmark with:

```bash
source .venv/bin/activate
python benchmarks/benchmark_tts.py \
  --text "Hi there! How can I help you" \
  --notes "final_chunk20_direct_vocoder"
```

## Browser Demo

Run:

```bash
cd /workspace/qwen3-tts-megakernel-adaptation
source .venv/bin/activate
./scripts/run_gradio_demo.sh
```

Recommended access path from your laptop:

```bash
ssh -i <identity-file> -p <ssh-port> -L 7860:localhost:7860 root@<vast-host>
```

Then open `http://localhost:7860` locally, record a short prompt, click the run
button, and play the generated reply.

If port forwarding is inconvenient, set `GRADIO_SHARE=1` in `.env` before
running the script. That asks Gradio to create a temporary public share URL.

Demo video: [Google Drive](https://drive.google.com/file/d/1N81Rj-_VG1tV9Lx9xw9Qhgr_RM504wuK/view?usp=sharing)

Copy generated outputs back to a laptop from a local terminal:

```bash
scp -i <identity-file> -P <ssh-port> -r \
  root@<vast-host>:/workspace/qwen3-tts-megakernel-adaptation/outputs \
  ~/Downloads/qwen3-tts-outputs
```

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

Benchmark the current stage with:

```bash
./benchmarks/benchmark_current_stage.sh
```

Use rows with matching prompt and notes for latency comparison.
