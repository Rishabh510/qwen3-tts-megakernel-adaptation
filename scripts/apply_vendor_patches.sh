#!/usr/bin/env bash
set -euo pipefail

# Apply local CUDA changes after fetching upstream vendor repositories.

if [ ! -d vendor/qwen_megakernel/.git ]; then
  echo "vendor/qwen_megakernel is missing. Run scripts/fetch_vendor_repos.sh first." >&2
  exit 1
fi

if git -C vendor/qwen_megakernel apply --check ../../patches/qwen-megakernel-tts-kernel.patch 2>/dev/null; then
  git -C vendor/qwen_megakernel apply ../../patches/qwen-megakernel-tts-kernel.patch
  echo "Applied qwen_megakernel CUDA patch."
else
  echo "CUDA patch already applied or upstream changed; checking expected markers."
  grep -q "LDG_VOCAB_SIZE" vendor/qwen_megakernel/csrc/kernel.cu
  grep -q "input_token_id >= 0" vendor/qwen_megakernel/csrc/kernel.cu
  echo "Expected CUDA markers found."
fi
