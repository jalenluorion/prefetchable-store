"""Vendored from the community MLX port (nopmobiel/Qwen3.8-Flash-Next-MLX-4bit, mlx_vlm_qwen4_exp/qwen4_exp/qwen4_exp.py).
Changes: text only; the three key/norm helpers it imported from mlx-vlm's qwen3_5 are copied in. Model.sanitize becomes the function sanitize(weights, text_config) with the same steps (the vision
tower's keys are dropped instead of mapped); the multimodal Model wrapper is not vendored. Keys come out with the
"language_model." prefix of the wrapper, as in the original."""
import re



# sanitize_key, should_shift_norm_weights, should_offset_norm_weight: copied from mlx-vlm 0.7.12 models/qwen3_5/qwen3_5.py
# (importing that module would pull in the multimodal Qwen3-VL model and transformers)
def sanitize_key(key):
    if key.startswith("model.language_model.visual"):
        key = key.replace("model.language_model.visual", "vision_tower", 1)
    elif key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model", 1)
    elif key.startswith("model.visual"):
        key = key.replace("model.visual", "vision_tower", 1)
    elif key.startswith("lm_head"):
        key = key.replace("lm_head", "language_model.lm_head", 1)
    return key


def should_shift_norm_weights(weights):
    has_mtp_weights = any("mtp." in key for key in weights)
    has_unsanitized_conv1d = any("conv1d.weight" in key and value.shape[-1] != 1 for key, value in weights.items())
    return has_mtp_weights or has_unsanitized_conv1d


def should_offset_norm_weight(original_key, shift_norm_weights):
    return shift_norm_weights or not original_key.startswith("language_model.")


_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
)


def sanitize(weights, text_config):
    shift_norm_weights = should_shift_norm_weights(weights)
    weights = {key: value for key, value in weights.items() if "mtp." not in key and not key.startswith("model.visual.")}
    if text_config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for layer in range(text_config.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer}.mlp"
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        if gate_up_key not in weights:
            continue
        gate_up = weights.pop(gate_up_key)
        middle = gate_up.shape[-2] // 2
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :middle, :]
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., middle:, :]
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(f"{prefix}.experts.down_proj")

    sanitized = {}
    shard_pattern = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
    for key, value in weights.items():
        original_key = key
        key = sanitize_key(key)
        key = shard_pattern.sub(r".ngram_embedding.shards.\1.weight", key)
        if "conv1d.weight" in key and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)
        if value.ndim == 1 and key.endswith(_NORM_SUFFIXES) and should_offset_norm_weight(original_key, shift_norm_weights):
            value = value + 1.0
        sanitized[key] = value
    return sanitized
