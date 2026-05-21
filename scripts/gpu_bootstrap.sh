#!/usr/bin/env bash
set -euo pipefail

# It prepares the Python environment, recreates vendor repos, and applies local CUDA patches.

if [[ ! -f .env ]]; then
  cp .env.template .env
  echo "Created .env from .env.template. Fill API keys there before running voice demos."
fi

set -a
source .env
set +a

export HF_HOME="${HF_HOME:-.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
mkdir -p "${HF_HOME}" "${NLTK_DATA}"

./scripts/fetch_vendor_repos.sh
./scripts/apply_vendor_patches.sh

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Prefer a CUDA 12.8-compatible PyTorch image/provider. This project assumes
# the active CUDA toolkit is also 12.8.
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements-gpu.txt
pip install -r vendor/qwen_megakernel/requirements.txt
pip install -e .

# FlashAttention is optional for this project. Keep it opt-in because building
# it can take a long time on rented machines and the baseline smoke test uses
# SDPA by default.
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-4}" pip install -U flash-attn --no-build-isolation
else
  echo "Skipping optional flash-attn install. Set INSTALL_FLASH_ATTN=1 to enable it."
fi

python scripts/verify_gpu_env.py
