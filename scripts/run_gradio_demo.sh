#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
mkdir -p "${HF_HOME}" "${NLTK_DATA}" outputs

source .venv/bin/activate

share_args=()
if [[ "${GRADIO_SHARE:-}" == "1" || "${GRADIO_SHARE:-}" == "true" ]]; then
  share_args=(--share)
fi

python demos/gradio_voice_demo.py \
  --host "${GRADIO_HOST:-0.0.0.0}" \
  --port "${GRADIO_PORT:-7860}" \
  --chunk-frames "${GRADIO_CHUNK_FRAMES:-20}" \
  --first-chunk-frames "${GRADIO_FIRST_CHUNK_FRAMES:-1}" \
  --whisper-model "${WHISPER_MODEL:-tiny}" \
  "${share_args[@]}"
