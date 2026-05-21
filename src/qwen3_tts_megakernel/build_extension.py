"""Build the CUDA extension with TTS-specific compile-time constants."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_MODULE = None
EXTENSION_NAME = "qwen_megakernel_tts_C"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def get_extension():
    """Compile or return the cached TTS megakernel extension."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    csrc = _root() / "vendor" / "qwen_megakernel" / "csrc"
    if not csrc.exists():
        raise FileNotFoundError(
            f"{csrc} is missing. Run scripts/fetch_vendor_repos.sh first."
        )

    flags = [
        f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
        f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
        f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 16)}",
        f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
        f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
        f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
        f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
        f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
        f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
        f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
        f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
        f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
        f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
        f"-DLDG_VOCAB_SIZE={_env_int('LDG_VOCAB_SIZE', 3072)}",
        "-DLDG_USE_UINT4",
        "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS",
        "-DLDG_MLP_SMEM",
    ]

    _MODULE = load(
        name=EXTENSION_NAME,
        sources=[str(csrc / "torch_bindings.cpp"), str(csrc / "kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-arch=sm_120a",
            f"-I{csrc}",
            *flags,
        ],
        extra_cflags=[f"-I{csrc}"],
        verbose=bool(int(os.getenv("VERBOSE_BUILD", "0"))),
    )
    return _MODULE


def get_decode_op():
    """Build the extension and return the registered decode op."""
    get_extension()
    return getattr(torch.ops, EXTENSION_NAME).decode
