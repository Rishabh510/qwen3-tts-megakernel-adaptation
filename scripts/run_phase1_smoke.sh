#!/usr/bin/env bash
set -euo pipefail

# Run after scripts/gpu_bootstrap.sh on the RTX 5090 machine.

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
mkdir -p "${HF_HOME}" "${NLTK_DATA}"

source ./scripts/select_cuda_toolkit.sh

source .venv/bin/activate

echo "== GPU environment =="
python scripts/verify_gpu_env.py

echo "== Original qwen_megakernel benchmark =="
(
  cd vendor/qwen_megakernel
  python -m qwen_megakernel.bench
)

echo "== Qwen3-TTS 0.6B baseline smoke test =="
python scripts/smoke_qwen3_tts.py

echo "== Adapted streaming TTS benchmark =="
python benchmarks/benchmark_tts.py \
  --text "Hi there! How can I help you" \
  --notes "tts_vocab_3072; embedding_sentinel; codebook_predictor_kernel"

echo "== Text-only Pipecat harness =="
python demos/pipecat_text_only.py --text "Hi there! How can I help you"
