#!/usr/bin/env bash
set -euo pipefail

# Creates a small local env for testing Gemini and local Whisper without renting a GPU.

export UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-.cache/uv-python}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
mkdir -p "${HF_HOME}" "${NLTK_DATA}"

if [[ ! -d .venv-front ]]; then
  uv venv --python 3.11 .venv-front
fi
source .venv-front/bin/activate
uv pip install faster-whisper "pipecat-ai[google,openai,silero,whisper,websocket]" fastapi uvicorn

echo "Frontend smoke env ready. Activate it with: source .venv-front/bin/activate"
