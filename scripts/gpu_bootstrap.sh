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

./scripts/fetch_vendor_repos.sh
./scripts/apply_vendor_patches.sh

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Prefer a CUDA 12.8-compatible PyTorch image/provider. If torch is already
# installed, this line is harmless unless the image has a mismatched version.
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements-gpu.txt
pip install -r vendor/qwen_megakernel/requirements.txt
pip install -e .

# FlashAttention is optional for the first smoke test but useful for baseline
# Qwen3-TTS. If it fails, continue with eager/sdpa attention for v1.
MAX_JOBS="${MAX_JOBS:-4}" pip install -U flash-attn --no-build-isolation || true

python scripts/verify_gpu_env.py
