# Implementation Plan

## Phase 0 - Local Preparation

- Clone source repos.
- Record current shape audit.
- Prepare setup scripts before renting GPU.
- Decide first-version STT/LLM simplification.

Status: done locally.

## Phase 1 - GPU Baseline

- Rent RTX 5090 instance.
- Verify `nvidia-smi`, CUDA version, and compute capability.
- Install Python dependencies.
- Run original `qwen_megakernel` benchmark unchanged.
- Run official Qwen3-TTS 0.6B inference unchanged.

Exit criteria:

- Original megakernel produces text and benchmark numbers.
- Qwen3-TTS produces a WAV file.

## Phase 2 - TTS Service Baseline

- Build a small Python streaming TTS service around official Qwen3-TTS.
- Start with text input and WAV/chunk output.
- Add timestamps for request start, first token, first audio chunk, final audio.

Exit criteria:

- `text -> audio chunks` works without Pipecat.

## Phase 3 - Megakernel Adaptation

- Add a Qwen3-TTS weight loader for the talker model.
- Adjust output projection to codec vocab size.
- Adjust RoPE theta.
- Validate one-step output against PyTorch for the same hidden/cache state.

Exit criteria:

- Megakernel talker decode returns plausible codec token IDs.
- One-step argmax is close to or matches PyTorch for deterministic settings.

## Phase 4 - Swap Decode Path

- Keep Qwen3-TTS prompt construction, subtalker/code predictor, and speech
  tokenizer decode in PyTorch.
- Replace main talker single-step decode with the adapted megakernel.
- Confirm generated audio is understandable.

Exit criteria:

- `text -> Qwen3-TTS audio` works with megakernel in the main talker path.

## Phase 5 - Pipecat Demo

- Build a simple Pipecat pipeline.
- Use a minimal STT/LLM path: typed/manual input first, cloud STT/LLM if needed.
- Push audio chunks as Pipecat audio frames immediately.

Exit criteria:

- Demo can be recorded end to end.
- Logs prove chunk-by-chunk streaming.

## Phase 6 - Submission

- README with setup/run instructions.
- Architecture notes and kernel modifications.
- Benchmarks: decode tok/s, TTFC, RTF, end-to-end latency.
- Demo recording link or filename.

