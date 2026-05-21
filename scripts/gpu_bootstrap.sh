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

source ./scripts/select_cuda_toolkit.sh

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

# FlashAttention is optional for this v1. Install it only when nvcc matches
# PyTorch's CUDA version; otherwise PyTorch's extension builder fails noisily.
torch_cuda="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
nvcc_cuda=""
if command -v nvcc >/dev/null 2>&1; then
  nvcc_cuda="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
fi
if [[ "${torch_cuda}" == "12.8" && "${nvcc_cuda}" == "12.8" ]]; then
  MAX_JOBS="${MAX_JOBS:-4}" pip install -U flash-attn --no-build-isolation || true
else
  echo "Skipping optional flash-attn install: torch CUDA=${torch_cuda}, nvcc CUDA=${nvcc_cuda:-missing}."
fi

python scripts/verify_gpu_env.py
