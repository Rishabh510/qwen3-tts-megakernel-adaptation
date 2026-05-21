"""Weight loading and shape checks for Qwen3-TTS talker adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .constants import (
    CODE_PREDICTOR_LAYERS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_CODE_GROUPS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    ROPE_THETA,
)


def resolve_model_dir(model_id_or_path: str) -> Path:
    """Return a local directory for a Hugging Face model id or path."""
    path = Path(model_id_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id_or_path))


def load_config(model_id_or_path: str) -> dict[str, Any]:
    model_dir = resolve_model_dir(model_id_or_path)
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict(model_id_or_path: str, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Load all safetensors shards into a single state dict."""
    model_dir = resolve_model_dir(model_id_or_path)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")

    state: dict[str, torch.Tensor] = {}
    for file in files:
        state.update(load_file(str(file), device=device))
    return state


def assert_expected_talker_shape(config: dict[str, Any]) -> None:
    """Fail early if the checkpoint no longer matches the optimized kernel path."""
    talker = config["talker_config"]
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
    }
    mismatches = {
        key: (expected_value, talker.get(key))
        for key, expected_value in expected.items()
        if talker.get(key) != expected_value
    }
    if mismatches:
        pretty = ", ".join(
            f"{key}: expected {exp}, got {got}" for key, (exp, got) in mismatches.items()
        )
        raise ValueError(f"Unsupported talker shape: {pretty}")


def make_rope_tables(max_seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
    return cos, sin


def collect_talker_weights(
    model_id_or_path: str,
    device: str = "cuda",
    max_seq_len: int = 8192,
) -> dict[str, Any]:
    """Map Qwen3-TTS checkpoint tensors into the megakernel adapter layout."""
    config = load_config(model_id_or_path)
    assert_expected_talker_shape(config)
    state = load_state_dict(model_id_or_path, device=device)
    cos_table, sin_table = make_rope_tables(max_seq_len, device)

    layer_weights = []
    for layer_idx in range(NUM_LAYERS):
        prefix = f"talker.model.layers.{layer_idx}."
        layer_weights.extend(
            [
                state[prefix + "input_layernorm.weight"].contiguous(),
                state[prefix + "self_attn.q_proj.weight"].contiguous(),
                state[prefix + "self_attn.k_proj.weight"].contiguous(),
                state[prefix + "self_attn.v_proj.weight"].contiguous(),
                state[prefix + "self_attn.q_norm.weight"].contiguous(),
                state[prefix + "self_attn.k_norm.weight"].contiguous(),
                state[prefix + "self_attn.o_proj.weight"].contiguous(),
                state[prefix + "post_attention_layernorm.weight"].contiguous(),
                state[prefix + "mlp.gate_proj.weight"].contiguous(),
                state[prefix + "mlp.up_proj.weight"].contiguous(),
                state[prefix + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    code_predictor: dict[str, torch.Tensor] = {}
    for layer_idx in range(CODE_PREDICTOR_LAYERS):
        src = f"talker.code_predictor.model.layers.{layer_idx}."
        dst = f"layers.{layer_idx}."
        for key in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            code_predictor[dst + key] = state[src + key].contiguous()

    code_predictor["norm.weight"] = state["talker.code_predictor.model.norm.weight"].contiguous()
    for group_idx in range(NUM_CODE_GROUPS - 1):
        code_predictor[f"codec_embedding.{group_idx}.weight"] = state[
            f"talker.code_predictor.model.codec_embedding.{group_idx}.weight"
        ].contiguous()
        code_predictor[f"lm_head.{group_idx}.weight"] = state[
            f"talker.code_predictor.lm_head.{group_idx}.weight"
        ].contiguous()

    return {
        "config": config,
        "codec_embedding": state["talker.model.codec_embedding.weight"].contiguous(),
        "codec_head": state["talker.codec_head.weight"].contiguous(),
        "final_norm": state["talker.model.norm.weight"].contiguous(),
        "layer_weights": layer_weights,
        "cos_table": cos_table,
        "sin_table": sin_table,
        "text_embedding": state["talker.model.text_embedding.weight"].contiguous(),
        "text_projection_fc1_weight": state["talker.text_projection.linear_fc1.weight"].contiguous(),
        "text_projection_fc1_bias": state["talker.text_projection.linear_fc1.bias"].contiguous(),
        "text_projection_fc2_weight": state["talker.text_projection.linear_fc2.weight"].contiguous(),
        "text_projection_fc2_bias": state["talker.text_projection.linear_fc2.bias"].contiguous(),
        "code_predictor": code_predictor,
    }

