#!/usr/bin/env bash
set -euo pipefail

# Apply local CUDA changes after fetching upstream vendor repositories.

if [ ! -d vendor/qwen_megakernel/.git ]; then
  echo "vendor/qwen_megakernel is missing. Run scripts/fetch_vendor_repos.sh first." >&2
  exit 1
fi

shopt -s nullglob
patches=(patches/qwen-megakernel-*.patch)

if [ ${#patches[@]} -eq 0 ]; then
  echo "No qwen_megakernel patches found."
  exit 0
fi

for patch in "${patches[@]}"; do
  patch_from_vendor="../../${patch}"
  if git -C vendor/qwen_megakernel apply --check "${patch_from_vendor}" 2>/dev/null; then
    git -C vendor/qwen_megakernel apply "${patch_from_vendor}"
    echo "Applied ${patch}."
  elif git -C vendor/qwen_megakernel apply --reverse --check "${patch_from_vendor}" 2>/dev/null; then
    echo "Already applied ${patch}."
  else
    echo "Patch failed or upstream changed: ${patch}" >&2
    exit 1
  fi
done

grep -q "LDG_VOCAB_SIZE" vendor/qwen_megakernel/csrc/kernel.cu
grep -q "input_token_id >= 0" vendor/qwen_megakernel/csrc/kernel.cu
echo "Expected CUDA markers found."
