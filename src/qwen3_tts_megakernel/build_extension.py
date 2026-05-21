"""Build the CUDA extension with TTS-specific compile-time constants."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_MODULE = None
_MODULE_NAME = None
_DECODE_OP = None


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _compile_config() -> dict[str, int]:
    return {
        "num_blocks": _env_int("LDG_NUM_BLOCKS", 128),
        "block_size": _env_int("LDG_BLOCK_SIZE", 512),
        "lm_num_blocks": _env_int("LDG_LM_NUM_BLOCKS", 16),
        "lm_block_size": _env_int("LDG_LM_BLOCK_SIZE", 384),
        "lm_rows_per_warp": _env_int("LDG_LM_ROWS_PER_WARP", 2),
        "attn_blocks": _env_int("LDG_ATTN_BLOCKS", 8),
        "vocab_size": _env_int("LDG_VOCAB_SIZE", 3072),
    }


def _extension_name(config: dict[str, int]) -> str:
    return (
        "qwen_megakernel_tts_C"
        f"_v{config['vocab_size']}"
        f"_lm{config['lm_num_blocks']}x{config['lm_block_size']}"
    )


def get_extension():
    """Compile or return the cached TTS megakernel extension."""
    global _MODULE, _MODULE_NAME
    if _MODULE is not None:
        return _MODULE

    csrc = _root() / "vendor" / "qwen_megakernel" / "csrc"
    if not csrc.exists():
        raise FileNotFoundError(
            f"{csrc} is missing. Run scripts/fetch_vendor_repos.sh first."
        )

    config = _compile_config()
    _MODULE_NAME = _extension_name(config)
    print(
        "building megakernel extension "
        f"name={_MODULE_NAME} "
        f"vocab={config['vocab_size']} "
        f"lm_blocks={config['lm_num_blocks']} "
        f"lm_block_size={config['lm_block_size']}"
    )

    flags = [
        f"-DLDG_NUM_BLOCKS={config['num_blocks']}",
        f"-DLDG_BLOCK_SIZE={config['block_size']}",
        f"-DLDG_LM_NUM_BLOCKS={config['lm_num_blocks']}",
        f"-DLDG_LM_BLOCK_SIZE={config['lm_block_size']}",
        f"-DLDG_LM_ROWS_PER_WARP={config['lm_rows_per_warp']}",
        f"-DLDG_ATTN_BLOCKS={config['attn_blocks']}",
        f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
        f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
        f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
        f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
        f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
        f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
        f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
        f"-DLDG_VOCAB_SIZE={config['vocab_size']}",
        "-DLDG_USE_UINT4",
        "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS",
        "-DLDG_MLP_SMEM",
    ]

    _MODULE = load(
        name=_MODULE_NAME,
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
    global _DECODE_OP
    if _DECODE_OP is not None:
        return _DECODE_OP
    get_extension()
    _DECODE_OP = getattr(torch.ops, _MODULE_NAME).decode
    return _DECODE_OP
