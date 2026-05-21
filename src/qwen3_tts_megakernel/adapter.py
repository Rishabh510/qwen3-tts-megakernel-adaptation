"""Runtime adapters around the patched CUDA decode kernel."""

from __future__ import annotations

import math
import struct
from typing import Any

import torch

from .build_extension import get_decode_op
from .constants import (
    CODE_PREDICTOR_LAYERS,
    CODE_PREDICTOR_MAX_SEQ_LEN,
    CODE_PREDICTOR_VOCAB_SIZE,
    EMBEDDING_SENTINEL_TOKEN_ID,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_SIZE,
    MAX_SEQ_LEN,
    NUM_CODE_GROUPS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_SIZE,
    TALKER_VOCAB_SIZE,
)
from .weights import make_rope_tables


def pack_layer_pointers(layer_weights: list[torch.Tensor], num_layers: int) -> torch.Tensor:
    """Pack layer tensor data pointers into the struct layout expected by CUDA."""
    ptr_size = 8
    pointers_per_layer = 11
    buf = bytearray(num_layers * pointers_per_layer * ptr_size)
    for layer_idx in range(num_layers):
        for ptr_idx in range(pointers_per_layer):
            ptr = layer_weights[layer_idx * pointers_per_layer + ptr_idx].data_ptr()
            struct.pack_into("Q", buf, (layer_idx * pointers_per_layer + ptr_idx) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TextProjector:
    """Text token embedding plus Qwen3-TTS text projection into talker space."""

    def __init__(self, weights: dict[str, Any]):
        self.embedding = weights["text_embedding"]
        self.fc1_weight = weights["text_projection_fc1_weight"]
        self.fc1_bias = weights["text_projection_fc1_bias"]
        self.fc2_weight = weights["text_projection_fc2_weight"]
        self.fc2_bias = weights["text_projection_fc2_bias"]

    @torch.no_grad()
    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.embedding(token_ids, self.embedding)
        x = torch.nn.functional.silu(
            torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias)
        )
        return torch.nn.functional.linear(x, self.fc2_weight, self.fc2_bias).to(torch.bfloat16)


class TalkerKernelAdapter:
    """Stateful Qwen3-TTS talker decoder using the patched megakernel."""

    def __init__(self, weights: dict[str, Any], device: str = "cuda", max_seq_len: int = MAX_SEQ_LEN):
        self._decode = get_decode_op()
        self.device = device
        self.max_seq_len = max_seq_len
        self.position = 0

        self.codec_embedding = weights["codec_embedding"]
        self.codec_head = weights["codec_head"]
        self.final_norm = weights["final_norm"]
        self.cos_table = weights["cos_table"]
        self.sin_table = weights["sin_table"]
        self.layers = pack_layer_pointers(weights["layer_weights"], NUM_LAYERS)
        self.attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        self.k_cache = torch.zeros(
            NUM_LAYERS,
            NUM_KV_HEADS,
            max_seq_len,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self._alloc_scratch()

    def _alloc_scratch(self) -> None:
        f32 = {"dtype": torch.float32, "device": self.device}
        bf16 = {"dtype": torch.bfloat16, "device": self.device}
        self.hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self.activations = torch.empty(HIDDEN_SIZE, **f32)
        self.residual = torch.empty(HIDDEN_SIZE, **f32)
        self.q = torch.empty(Q_SIZE, **f32)
        self.k = torch.empty(KV_SIZE, **f32)
        self.v = torch.empty(KV_SIZE, **f32)
        self.attn_out = torch.empty(Q_SIZE, **f32)
        self.mlp_intermediate = torch.empty(INTERMEDIATE_SIZE, **f32)
        self.norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self.block_max_values = torch.empty(4096, **f32)
        self.block_max_indices = torch.empty(4096, dtype=torch.int32, device=self.device)
        self.output_token = torch.empty(1, dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        self.position = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    @torch.no_grad()
    def step_token(self, token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._step(token_id)

    @torch.no_grad()
    def step_embedding(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden.copy_(embedding.to(torch.bfloat16))
        return self._step(EMBEDDING_SENTINEL_TOKEN_ID)

    @torch.no_grad()
    def step_embedding_into(self, embedding: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self.hidden.copy_(embedding.to(torch.bfloat16))
        token, hidden = self._step(EMBEDDING_SENTINEL_TOKEN_ID)
        out.copy_(hidden)
        return token

    def _step(self, token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._decode(
            self.output_token,
            token_id,
            self.codec_embedding,
            self.layers,
            self.final_norm,
            self.codec_head,
            self.cos_table,
            self.sin_table,
            self.k_cache,
            self.v_cache,
            self.hidden,
            self.activations,
            self.residual,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.mlp_intermediate,
            self.norm_out,
            self.block_max_values,
            self.block_max_indices,
            NUM_LAYERS,
            self.position,
            self.max_seq_len,
            self.attn_scale,
        )
        self.position += 1
        return self.output_token, self.norm_out


class CodebookPredictorKernel:
    """Use the same decode kernel for the 5-layer codebook predictor."""

    def __init__(self, weights: dict[str, Any], device: str = "cuda"):
        self._decode = get_decode_op()
        self.device = device
        cp = weights["code_predictor"]
        layer_weights = []
        for layer_idx in range(CODE_PREDICTOR_LAYERS):
            prefix = f"layers.{layer_idx}."
            layer_weights.extend(
                [
                    cp[prefix + "input_layernorm.weight"],
                    cp[prefix + "self_attn.q_proj.weight"],
                    cp[prefix + "self_attn.k_proj.weight"],
                    cp[prefix + "self_attn.v_proj.weight"],
                    cp[prefix + "self_attn.q_norm.weight"],
                    cp[prefix + "self_attn.k_norm.weight"],
                    cp[prefix + "self_attn.o_proj.weight"],
                    cp[prefix + "post_attention_layernorm.weight"],
                    cp[prefix + "mlp.gate_proj.weight"],
                    cp[prefix + "mlp.up_proj.weight"],
                    cp[prefix + "mlp.down_proj.weight"],
                ]
            )
        self.layers = pack_layer_pointers(layer_weights, CODE_PREDICTOR_LAYERS)
        self.final_norm = cp["norm.weight"]
        self.codec_embeddings = [
            cp[f"codec_embedding.{idx}.weight"] for idx in range(NUM_CODE_GROUPS - 1)
        ]
        self.lm_heads = [cp[f"lm_head.{idx}.weight"] for idx in range(NUM_CODE_GROUPS - 1)]
        self.dummy_embedding = torch.zeros(
            TALKER_VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
        )
        self.dummy_head = torch.zeros_like(self.dummy_embedding)
        self.cos_table, self.sin_table = make_rope_tables(CODE_PREDICTOR_MAX_SEQ_LEN, device)
        self.attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self.token_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.output_codes = torch.empty(NUM_CODE_GROUPS, dtype=torch.long, device=device)
        self._allocate_state()

    def _allocate_state(self) -> None:
        f32 = {"dtype": torch.float32, "device": self.device}
        bf16 = {"dtype": torch.bfloat16, "device": self.device}
        self.position = 0
        self.k_cache = torch.zeros(
            CODE_PREDICTOR_LAYERS,
            NUM_KV_HEADS,
            CODE_PREDICTOR_MAX_SEQ_LEN,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self.activations = torch.empty(HIDDEN_SIZE, **f32)
        self.residual = torch.empty(HIDDEN_SIZE, **f32)
        self.q = torch.empty(Q_SIZE, **f32)
        self.k = torch.empty(KV_SIZE, **f32)
        self.v = torch.empty(KV_SIZE, **f32)
        self.attn_out = torch.empty(Q_SIZE, **f32)
        self.mlp_intermediate = torch.empty(INTERMEDIATE_SIZE, **f32)
        self.norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self.block_max_values = torch.empty(4096, **f32)
        self.block_max_indices = torch.empty(4096, dtype=torch.int32, device=self.device)
        self.output_token = torch.empty(1, dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        self.position = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def _step_embedding(self, embedding: torch.Tensor) -> None:
        self.hidden.copy_(embedding.to(torch.bfloat16))
        self._decode(
            self.output_token,
            EMBEDDING_SENTINEL_TOKEN_ID,
            self.dummy_embedding,
            self.layers,
            self.final_norm,
            self.dummy_head,
            self.cos_table,
            self.sin_table,
            self.k_cache,
            self.v_cache,
            self.hidden,
            self.activations,
            self.residual,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.mlp_intermediate,
            self.norm_out,
            self.block_max_values,
            self.block_max_indices,
            CODE_PREDICTOR_LAYERS,
            self.position,
            CODE_PREDICTOR_MAX_SEQ_LEN,
            self.attn_scale,
        )
        self.position += 1

    @torch.no_grad()
    def predict(
        self,
        talker_hidden: torch.Tensor,
        first_token: int | torch.Tensor,
        talker_codec_embedding: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.reset()
        self._step_embedding(talker_hidden)
        if isinstance(first_token, torch.Tensor):
            self.token_buf.copy_(first_token)
        else:
            self.token_buf[0] = first_token
        first_embed = torch.nn.functional.embedding(self.token_buf, talker_codec_embedding)[0]
        self._step_embedding(first_embed)

        self.output_codes[0] = self.token_buf[0]
        for group_idx in range(NUM_CODE_GROUPS - 1):
            logits = torch.nn.functional.linear(
                self.norm_out.to(torch.bfloat16).unsqueeze(0), self.lm_heads[group_idx]
            )[0]
            if do_sample and temperature > 0:
                logits = logits.float() / temperature
                if top_k:
                    values, _ = torch.topk(logits, min(top_k, CODE_PREDICTOR_VOCAB_SIZE))
                    logits = logits.masked_fill(logits < values[-1], float("-inf"))
                token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            else:
                token = torch.argmax(logits, keepdim=True).long()
            self.output_codes[group_idx + 1] = token[0]
            if group_idx < NUM_CODE_GROUPS - 2:
                embed = torch.nn.functional.embedding(token, self.codec_embeddings[group_idx])[0]
                self._step_embedding(embed)
        return self.output_codes.clone()
