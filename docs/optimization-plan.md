# Adaptation And Optimization Plan

## Adaptation Changes

These are required to make Qwen3-TTS run through the CUDA decode path at all.

1. **Install Qwen3-TTS as a package**: use `qwen-tts` from PyPI for model classes,
   tokenizer/vocoder loading, and checkpoint compatibility.
2. **Patch the CUDA vocab constant**: compile the output scan for the Qwen3-TTS
   codec vocabulary size `3072`, not the text vocabulary size.
3. **Load talker weights**: map `talker.model.*` tensors into the same layer
   pointer layout expected by the CUDA kernel.
4. **Use the TTS codec head**: output projection comes from
   `talker.codec_head.weight`, not tied text embeddings.
5. **Use Qwen3-TTS RoPE scale**: build cosine/sine tables with
   `rope_theta=1000000`.
6. **Support precomputed embeddings**: allow the kernel to read an already-summed
   TTS embedding from `hidden_buffer` when token id is negative.
7. **Rebuild Qwen3-TTS prefill**: create the role/text/codec prefix embeddings
   that the talker expects before generation.
8. **Keep vocoder decode intact**: use the published Qwen3-TTS speech tokenizer
   to turn codec frames into waveform audio.

Goal for this stage: audio is intelligible and generated through the adapted
talker path, even if latency is not yet final.

## Sequential Optimizations

Add these one at a time and record a benchmark row after each change.

1. **Single-run benchmark stability**: run one benchmark per process and append
   a CSV row so hangs in repeated same-process generation do not block progress.
2. **TTS LM-head launch shape**: keep the required codec vocab size `3072`, but
   reduce output scan launch geometry because codec vocab is much smaller than
   text vocab.
3. **Shared CUDA op lookup**: resolve the registered CUDA decode op once and
   reuse it across talker and code predictor adapters.
4. **Codebook output buffer reuse**: reuse a CUDA output buffer for the 16
   codebook tokens instead of building a Python list and concatenating every
   frame.
5. **Warm both codebook paths**: run repeated deterministic and sampled
   codebook predictions before request timing so `topk`, `softmax`, and
   `multinomial` do not hit first-call overhead in TTFC.
6. **Warm vocoder decode shapes**: run dummy one-frame and multi-frame vocoder
   decodes during initialization so first streamed audio is not paying lazy
   allocation/JIT cost.
7. **First-frame-first streaming**: emit the first audio chunk after one codec
   frame, then use 10-frame follow-up chunks for playback efficiency.
8. **GPU-resident talker token**: keep the first codebook token on GPU between
   the talker and codebook predictor instead of calling `.item()` once per
   frame.
9. **Stopping heuristic tuning**: tune max-frame limits and EOS handling to avoid
    runaway generation without cutting speech short.
10. **Audio quality pass**: compare short/medium prompts by ear and mark glitches,
    dropped frames, early cutoff, or repeated speech in the CSV notes.

Evidence note: two attempted Python-side buffer/chunk-size changes were removed
from the forward plan because the benchmark CSV showed worse RTF. Keep the
measurement history, but continue from the faster 10-frame streaming path.

## Benchmark Notes

Single-run timing includes model load, extension build/cache lookup, tensor
allocation, and warmup. Use matching single-run rows to compare sequential code
states. If same-process warm generation is fixed later, `--runs 2` can still be
used to collect a cold row and a warm row.

The benchmark writes:

- `init_ms`: initialization and warmup time for the run.
- `ttfc_ms`: time from text request to first emitted audio chunk.
- `generation_ms`: streaming generation time excluding initialization.
- `end_to_end_ms`: `init_ms + generation_ms`.
- `rtf`: generation time divided by produced audio duration.
- `notes`: manual description of the active change.
