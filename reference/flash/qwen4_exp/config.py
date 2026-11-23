"""Vendored from the community MLX port (nopmobiel/Qwen3.8-Flash-Next-MLX-4bit, mlx_vlm_qwen4_exp/qwen4_exp/config.py).
Changes: absolute mlx_vlm imports; the vision and multimodal ModelConfig classes are dropped (text only)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from mlx_vlm.models.base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    output_gate_type: str = "sigmoid"
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        }
    )

    def __post_init__(self):
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "qwen_sparse_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "qwen_sparse_attention" if t == "full_attention" else t
                for t in self.layer_types
            ]
        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
