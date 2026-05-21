#!/usr/bin/env bash

# Source this file before building PyTorch CUDA extensions.
# RTX 5090 needs PyTorch cu128; PyTorch's extension builder rejects mismatched
# nvcc toolkits such as CUDA 13.2.

preferred=(
  "/usr/local/cuda-12.8"
  "/usr/local/cuda-12.8.1"
  "/usr/local/cuda-12.8.0"
  "/usr/local/cuda"
)

for cuda_dir in "${preferred[@]}"; do
  if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
    version="$("${cuda_dir}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [[ "${version}" == "12.8" ]]; then
      export CUDA_HOME="${cuda_dir}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
      echo "Using CUDA toolkit: ${CUDA_HOME} (${version})"
      return 0 2>/dev/null || exit 0
    fi
  fi
done

echo "warning: CUDA 12.8 nvcc not found under /usr/local." >&2
if command -v nvcc >/dev/null 2>&1; then
  echo "active nvcc: $(command -v nvcc)" >&2
  nvcc --version >&2 || true
else
  echo "active nvcc: not found" >&2
fi
return 0 2>/dev/null || exit 0
