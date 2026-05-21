#!/usr/bin/env bash
set -euo pipefail

# Recreate the upstream CUDA source repo. Run from this repo root.

mkdir -p vendor

if [ ! -d vendor/qwen_megakernel/.git ]; then
  git clone https://github.com/AlpinDale/qwen_megakernel.git vendor/qwen_megakernel
else
  git -C vendor/qwen_megakernel fetch origin
fi

echo "vendor/qwen_megakernel: $(git -C vendor/qwen_megakernel rev-parse --short HEAD)"
