"""Vendored from the community MLX port (nopmobiel/Qwen3.8-Flash-Next-MLX-4bit, mlx_vlm_qwen4_exp/qwen4_exp/language.py).
Changes: absolute mlx_vlm imports; ModelConfig is a type alias (text only). The model code is unchanged."""
import math
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel as Qwen3_5LanguageModel,
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RotaryEmbedding,
    _create_qwen3_5_attention_mask,
    _create_qwen3_5_ssm_mask,
    apply_multimodal_rotary_pos_emb,
)
from mlx_vlm.models.switch_layers import SwitchGLU
from .config import TextConfig

ModelConfig = Any  # the multimodal config is not vendored (text only)


class GroupedRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        y = x.astype(mx.float32)
        if self.group_size is not None:
            y = y.reshape(*y.shape[:-1], -1, self.group_size)
        y = mx.fast.rms_norm(y, None, self.eps)
        if self.group_size is not None:
            y = y.flatten(-2)
        return (y * self.weight.astype(mx.float32)).astype(dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, activation: str):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        gate = gate.astype(mx.float32)
        gate = mx.sigmoid(gate) if self.activation == "sigmoid" else nn.silu(gate)
        return (gate * out.astype(mx.float32)).astype(x.dtype)


class GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.norm = GatedRMSNorm(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            activation=config.output_gate_type or "silu",
        )


class QSAKVCache(CacheList):
    """Main K/V, indexer keys, positions, all checkpointed as one cache."""

    def __init__(self):
        super().__init__(KVCache(), KVCache(), ArraysCache(size=1))

    @property
    def offset(self):
        return self.caches[0].offset

    @property
    def _idx(self):
        return self.caches[0].offset

    @property
    def keys(self):
        return self.caches[0].keys

    @property
    def values(self):
        return self.caches[0].values

    @property
    def left_padding(self):
        return getattr(self.caches[0], "left_padding", None)

    def update_positions(self, position_ids: mx.array):
        old = self.caches[2][0]
        if old is not None and old.ndim != position_ids.ndim:
            # Multimodal prefills cache three MRoPE position streams, while
            # subsequent text-only decode steps may provide a single 2-D
            # stream. Text positions are identical on every MRoPE axis.
            if old.ndim == 3 and position_ids.ndim == 2:
                position_ids = mx.broadcast_to(
                    position_ids[None],
                    (old.shape[0], *position_ids.shape),
                )
            elif old.ndim == 2 and position_ids.ndim == 3:
                old = mx.broadcast_to(
                    old[None], (position_ids.shape[0], *old.shape)
                )
            else:
                raise ValueError(
                    "QSA cached and current position IDs have incompatible ranks: "
                    f"{old.ndim} and {position_ids.ndim}"
                )
        self.caches[2][0] = (
            position_ids
            if old is None
            else mx.concatenate([old, position_ids], axis=-1)
        )
        return self.caches[2][0]

    def extract(self, idx: int):
        """Extract one batch row, including on older mlx-vlm KV caches."""
        extracted = []
        for cache in self.caches[:2]:
            if hasattr(cache, "extract"):
                extracted.append(cache.extract(idx))
                continue
            if not isinstance(cache, KVCache):
                raise TypeError(f"Cannot extract {type(cache).__name__} cache")
            item = KVCache()
            if cache.keys is not None:
                batch_size = int(cache.keys.shape[0])
                row = idx + batch_size if idx < 0 else idx
                if row < 0 or row >= batch_size:
                    raise IndexError(
                        f"KV cache row {idx} out of range for batch size {batch_size}"
                    )
                item.keys = mx.contiguous(
                    cache.keys[row : row + 1, :, : cache.offset, :]
                )
                item.values = mx.contiguous(
                    cache.values[row : row + 1, :, : cache.offset, :]
                )
                item.offset = cache.offset
            extracted.append(item)

        positions_cache = ArraysCache(size=1)
        positions = self.positions
        if positions is not None:
            batch_axis = 1 if positions.ndim == 3 else 0
            batch_size = int(positions.shape[batch_axis])
            row = idx + batch_size if idx < 0 else idx
            if row < 0 or row >= batch_size:
                raise IndexError(
                    f"Position cache row {idx} out of range for batch size {batch_size}"
                )
            positions_cache[0] = mx.contiguous(
                positions[:, row : row + 1, :]
                if batch_axis == 1
                else positions[row : row + 1, :]
            )
        result = QSAKVCache()
        result.caches = (*extracted, positions_cache)
        return result

    def filter(self, batch_indices):
        for cache in self.caches[:2]:
            if hasattr(cache, "filter"):
                cache.filter(batch_indices)
            elif isinstance(cache, KVCache) and cache.keys is not None:
                cache.keys = cache.keys[batch_indices]
                cache.values = cache.values[batch_indices]
        positions = self.positions
        if positions is not None:
            self.caches[2][0] = (
                positions[:, batch_indices, :]
                if positions.ndim == 3
                else positions[batch_indices]
            )

    def extend(self, other):
        left_batch = (
            int(self.keys.shape[0])
            if self.keys is not None
            else (int(self.positions.shape[-2]) if self.positions is not None else 1)
        )
        right_batch = (
            int(other.keys.shape[0])
            if other.keys is not None
            else (
                int(other.positions.shape[-2]) if other.positions is not None else 1
            )
        )
        extended = []
        for cache, other_cache in zip(self.caches[:2], other.caches[:2]):
            if hasattr(cache, "extend"):
                cache.extend(other_cache)
                extended.append(cache)
            elif hasattr(cache, "merge"):
                extended.append(cache.merge([cache, other_cache]))
            else:
                raise TypeError(f"Cannot extend {type(cache).__name__} cache")

        left_positions, right_positions = self.positions, other.positions
        positions_cache = ArraysCache(size=1)
        if left_positions is not None or right_positions is not None:
            template = (
                left_positions if left_positions is not None else right_positions
            )
            use_mrope = any(
                p is not None and p.ndim == 3
                for p in (left_positions, right_positions)
            )

            def normalize(positions, batch_size):
                if positions is None:
                    shape = (3, batch_size, 0) if use_mrope else (batch_size, 0)
                    return mx.zeros(shape, dtype=template.dtype)
                if use_mrope and positions.ndim == 2:
                    positions = mx.broadcast_to(
                        positions[None], (3, *positions.shape)
                    )
                return positions

            left_positions = normalize(left_positions, left_batch)
            right_positions = normalize(right_positions, right_batch)
            max_length = max(left_positions.shape[-1], right_positions.shape[-1])

            def left_pad(positions):
                amount = max_length - positions.shape[-1]
                if amount == 0:
                    return positions
                padding = [(0, 0)] * positions.ndim
                padding[-1] = (amount, 0)
                return mx.pad(positions, padding)

            positions_cache[0] = mx.concatenate(
                [left_pad(left_positions), left_pad(right_positions)],
                axis=1 if use_mrope else 0,
            )
        self.caches = (*extended, positions_cache)

    def prepare(self, **kwargs):
        for cache in self.caches[:2]:
            prepare = getattr(cache, "prepare", None)
            if prepare is not None:
                prepare(**kwargs)

    def finalize(self):
        for cache in self.caches[:2]:
            finalize = getattr(cache, "finalize", None)
            if finalize is not None:
                finalize()

    @property
    def positions(self):
        return self.caches[2][0]


class QSAIndexer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.n_heads = config.indexer_n_heads
        self.head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (config.indexer_n_heads + config.indexer_kv_heads)
            * config.indexer_head_dim,
            bias=False,
        )
        self.q_layernorm = GroupedRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )
        self.k_layernorm = GroupedRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            int(config.head_dim * config.rope_parameters["partial_rotary_factor"]),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_parameters["rope_theta"],
            mrope_section=config.rope_parameters["mrope_section"],
        )

    def _rotate(self, x: mx.array, position_ids: mx.array) -> mx.array:
        return self.rotary_emb.apply_rotary(
            x, x, position_ids, unsqueeze_dim=1
        )[0]

    def project(
        self,
        x: mx.array,
        position_ids: mx.array,
        cache: QSAKVCache,
    ):
        B, L, _ = x.shape
        qk = self.index_qk_proj(x)
        q_size = self.n_heads * self.head_dim
        q, raw = mx.split(qk, [q_size], axis=-1)
        q = self.q_layernorm(q.reshape(B, L, self.n_heads, self.head_dim))
        q = q.transpose(0, 2, 1, 3)
        q = self._rotate(q, position_ids)
        raw = raw.reshape(B, L, self.head_dim)
        raw, _ = cache.caches[1].update_and_fetch(
            raw[:, None, :, :], raw[:, None, :, :]
        )
        return q, raw[:, 0]

    def select(
        self,
        q: mx.array,
        raw_keys: mx.array,
        positions: mx.array,
        batch: int,
        query: int,
        visible: int,
    ) -> mx.array:
        ratio = self.compress_ratio
        complete = visible // ratio
        if complete:
            grouped = raw_keys[batch, : complete * ratio].reshape(
                complete, ratio, self.head_dim
            )
            pooled = self.k_layernorm(grouped.mean(axis=1))
            starts = mx.arange(complete, dtype=mx.int32) * ratio
            if positions.ndim == 3:
                key_positions = positions[:, batch : batch + 1, starts]
            else:
                key_positions = positions[batch : batch + 1, starts]
            pooled = self._rotate(pooled[None, None], key_positions)[0, 0]
            scores = mx.matmul(
                q[batch, :, query].astype(mx.float32),
                pooled.astype(mx.float32).T,
            )
            scores = mx.maximum(scores, 0).sum(axis=0) / math.sqrt(self.head_dim)
            k = min(self.block_topk, complete)
            block_ids = mx.argpartition(-scores, kth=k - 1)[:k]
            selected = (
                block_ids[:, None] * ratio
                + mx.arange(ratio, dtype=mx.int32)[None, :]
            ).reshape(-1)
        else:
            selected = mx.array([], dtype=mx.int32)
        tail = mx.arange(complete * ratio, visible, dtype=mx.int32)
        return mx.concatenate([selected.astype(mx.int32), tail])


def _current_positions(position_ids, batch_size, length, offset):
    if position_ids is None:
        return mx.broadcast_to(
            mx.arange(offset, offset + length)[None, :], (batch_size, length)
        )
    return position_ids[..., -length:]


class QSAAttention(Qwen3_5Attention):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.q_norm = GroupedRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GroupedRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = QSAIndexer(config)
        self.token_budget = config.indexer_budget

    def _prepare_projected_qkv(
        self,
        q_proj_output,
        keys,
        values,
        cache,
        position_ids,
        position_embeddings,
        mask,
    ):
        """Q/K/V preparation kept local for older mlx-vlm runtimes."""
        B, L, _ = q_proj_output.shape
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]
        if position_ids is None:
            cache_offset = cache.offset
            if isinstance(cache_offset, mx.array) and cache_offset.ndim > 0:
                offsets = mx.maximum(cache_offset[:B], 0)
                kv_seq_len = kv_seq_len + offsets + 1
                position_ids = offsets[:, None] + mx.arange(L)[None, :]
                position_ids = mx.broadcast_to(
                    position_ids[None], (3, *position_ids.shape)
                )
            else:
                if isinstance(cache_offset, mx.array):
                    cache_offset = int(cache_offset.item())
                kv_seq_len += cache_offset + 1
                position_ids = mx.arange(cache_offset, cache_offset + L)
                position_ids = mx.broadcast_to(
                    position_ids[None, None], (3, 1, L)
                )
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        if position_embeddings is None:
            queries, keys = self.rotary_emb.apply_rotary(
                queries, keys, position_ids, unsqueeze_dim=1
            )
        else:
            cos, sin = position_embeddings
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin
            )

        if mask is not None and isinstance(mask, mx.array):
            if (
                cache is not None
                and hasattr(cache, "_idx")
                and hasattr(cache, "left_padding")
            ):
                kv_seq_len = int(cache._idx) + L
            elif isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        return queries, keys, values, gate, mask

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[QSAKVCache] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        if cache is None:
            cache = QSAKVCache()
        B, L, _ = x.shape
        offset = cache.offset
        current_positions = _current_positions(position_ids, B, L, offset)
        all_positions = cache.update_positions(current_positions)
        index_q, raw_keys = self.indexer.project(x, current_positions, cache)

        # QSA selects every visible token below the budget, so the dense MLX
        # attention path is exactly equivalent and substantially faster.
        if offset + L <= self.token_budget:
            return super().__call__(
                x,
                mask=mask,
                cache=cache.caches[0],
                position_ids=current_positions,
                position_embeddings=position_embeddings,
            )

        q_out = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries, keys, values, gate, _ = self._prepare_projected_qkv(
            q_out,
            keys,
            values,
            cache.caches[0],
            current_positions,
            position_embeddings,
            None,
        )

        if L > 1 and not QSAAttention.loop:
            output = mx.concatenate([self._sparse_batch(queries[b : b + 1], keys[b : b + 1], values[b : b + 1], index_q[b], raw_keys[b], all_positions, b, offset, L) for b in range(B)], axis=0)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output * mx.sigmoid(gate))
        rows = []
        for b in range(B):
            tokens = []
            for i in range(L):
                visible = offset + i + 1
                selected = self.indexer.select(
                    index_q, raw_keys, all_positions, b, i, visible
                )
                k = keys[b : b + 1, :, selected, :]
                v = values[b : b + 1, :, selected, :]
                tokens.append(
                    scaled_dot_product_attention(
                        queries[b : b + 1, :, i : i + 1],
                        k,
                        v,
                        cache=None,
                        scale=self.scale,
                        mask=None,
                    )
                )
            rows.append(mx.concatenate(tokens, axis=2))
        output = mx.concatenate(rows, axis=0)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    # Local change (deep-memory runner, 2026-11-14): the per-query loop above builds ~15 small ops per query and layer in
    # Python (a 4k-token batch past the 2,048-token budget: ~25k selections, most of its prefill time). For a block of
    # queries the same selection is one score matrix and one masked attention: query i sees the top block_topk complete
    # blocks of 4 among those complete at i (indexer scores as in select()), plus its incomplete tail block. Same keys,
    # same softmax; only the summation order (and ties between equal block scores) can differ. Decode (L == 1) keeps the loop.
    loop = False  # True: the community per-query loop for every L (reference for checks)

    def _sparse_batch(self, queries, keys, values, index_q, raw_keys, all_positions, b, offset, L):
        ix = self.indexer
        ratio, hd = ix.compress_ratio, ix.head_dim
        S = offset + L
        nb = S // ratio  # blocks complete for the last query
        grouped = raw_keys[: nb * ratio].reshape(nb, ratio, hd)
        pooled = ix.k_layernorm(grouped.mean(axis=1))
        starts = mx.arange(nb, dtype=mx.int32) * ratio
        key_positions = all_positions[:, b : b + 1, starts] if all_positions.ndim == 3 else all_positions[b : b + 1, starts]
        pooled = ix._rotate(pooled[None, None], key_positions)[0, 0]
        scores = mx.matmul(index_q.astype(mx.float32), pooled.astype(mx.float32).T)  # [heads, L, nb]
        scores = mx.maximum(scores, 0).sum(axis=0) / math.sqrt(hd)  # [L, nb]
        vis = offset + 1 + mx.arange(L, dtype=mx.int32)  # keys visible to each query
        comp = vis // ratio  # its complete blocks
        valid = mx.arange(nb, dtype=mx.int32)[None, :] < comp[:, None]
        k = min(ix.block_topk, nb)
        top = mx.argpartition(-mx.where(valid, scores, -mx.inf), kth=k - 1, axis=-1)[:, :k]
        chosen = mx.put_along_axis(mx.zeros((L, nb), dtype=mx.bool_), top, mx.array(True), axis=-1) & valid
        s = mx.arange(S, dtype=mx.int32)[None, :]
        blocks = mx.repeat(chosen, ratio, axis=1)
        if nb * ratio < S:
            blocks = mx.concatenate([blocks, mx.zeros((L, S - nb * ratio), dtype=mx.bool_)], axis=1)
        allowed = blocks | ((s >= comp[:, None] * ratio) & (s < vis[:, None]))
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=allowed[None, None])


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = Qwen3_5MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        probs = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(probs, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = self.switch_mlp(x, inds)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return routed + shared


class GatedResidual(nn.Module):
    def __init__(self, args: TextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = args.hc_count * args.hidden_size
        self.hc_norm = GroupedRMSNorm(
            hc_hidden, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(
            hc_hidden, args.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden, args.hc_count, bias=False) if use_combine else None
        )

    def __call__(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix)).reshape(
            *normed.shape[:-1], self.hc_count, self.hidden_size
        )
        streams = normed.reshape(
            *normed.shape[:-1], self.hc_count, self.hidden_size
        )
        mixed = (mix * streams).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * mx.sigmoid(
            self.block_inject_weight(normed) / self.hc_count
        )
        return mixed, hyper_input, injection


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _layer_multipliers(vocab_size: int, ngram_size: int, layer: int, seed: int):
    multiplier_max = ((1 << 63) - 1) // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + 10007 * layer
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return mx.array(values, dtype=mx.int64)


class ShardedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, parts: int):
        super().__init__()
        if vocab_size % parts:
            raise ValueError("The padded n-gram vocabulary must divide into shards")
        self.rows_per_shard = vocab_size // parts
        self.embedding_dim = dim
        self.shards = [
            nn.Embedding(self.rows_per_shard, dim) for _ in range(parts)
        ]

    def __call__(self, ids: mx.array) -> mx.array:
        shape = ids.shape
        flat = ids.reshape(-1)
        shard_ids = flat // self.rows_per_shard
        order = mx.argsort(shard_ids)
        sorted_ids = flat[order]
        sorted_shards = shard_ids[order].tolist()
        first_shard = self.shards[int(sorted_shards[0])]
        dtype = (
            first_shard.scales.dtype
            if hasattr(first_shard, "scales")
            else first_shard.weight.dtype
        )
        out = mx.zeros((flat.size, self.embedding_dim), dtype=dtype)
        start = 0
        while start < len(sorted_shards):
            shard_id = int(sorted_shards[start])
            end = start + 1
            while end < len(sorted_shards) and sorted_shards[end] == shard_id:
                end += 1
            local_ids = sorted_ids[start:end] - shard_id * self.rows_per_shard
            out[start:end] = self.shards[shard_id](local_ids)
            start = end
        inverse = mx.argsort(order)
        return out[inverse].reshape(*shape, self.embedding_dim)


class NGramEmbedding(nn.Module):
    def __init__(
        self, args: TextConfig, embedding_dim: int, layer_idx: int, ple_index: int
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * args.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embedding_dim // self.ngram_heads

        sizes = []
        offsets = []
        total = 0
        for head in range(self.ngram_heads):
            global_head = ple_index * self.ngram_heads + head
            size = _find_nth_prime_after(
                args.ngram_vocab_size_base - 1, global_head + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        divisor = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / divisor) * divisor
        self.layer_multipliers = _layer_multipliers(
            args.vocab_size, args.ngram_size, ple_index, args.seed
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = ShardedEmbedding(
            padded, head_dim, args.split_ngram_parts
        )

    def _shift(self, tokens: mx.array, amount: int) -> mx.array:
        if amount == 0:
            return tokens
        pad = mx.full(
            (*tokens.shape[:-1], amount), self.eos_token_id, dtype=tokens.dtype
        )
        shifted = mx.concatenate([pad, tokens[..., :-amount]], axis=-1)
        valid = mx.ones(tokens.shape, dtype=mx.bool_)
        valid[..., :amount] = False
        for delta in range(1, amount + 1):
            previous = mx.concatenate(
                [
                    mx.full(
                        (*tokens.shape[:-1], delta),
                        self.eos_token_id,
                        dtype=tokens.dtype,
                    ),
                    tokens[..., :-delta],
                ],
                axis=-1,
            )
            valid = valid & (previous != self.eos_token_id)
        return mx.where(valid, shifted, self.eos_token_id)

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache]):
        B, L = input_ids.shape
        previous = cache[3] if cache is not None and len(cache.cache) > 3 else None
        if previous is None:
            previous = mx.full(
                (B, self.context_len), self.eos_token_id, dtype=input_ids.dtype
            )
        history = mx.concatenate([previous, input_ids], axis=-1)
        if cache is not None:
            cache[3] = history[:, -self.context_len :]
        shifted = [self._shift(history, i) for i in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed, shifted[position] * self.layer_multipliers[position]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            blocks.append(mixed[..., None] % sizes + offsets)
        ids = mx.concatenate(blocks, axis=-1)[:, -L:]
        return self.ngram_embedding(ids).flatten(-2)


class PLELayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int, ple_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.ple_embedding = NGramEmbedding(
            args, args.ple_embed_dim, layer_idx, ple_index
        )
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(
            hc_hidden, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_query = GroupedRMSNorm(
            hc_hidden, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_conv = GroupedRMSNorm(
            hc_hidden, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.conv1d = nn.Conv1d(
            hc_hidden,
            hc_hidden,
            kernel_size=args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=hc_hidden,
            bias=False,
        )

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache: ArraysCache):
        embeddings = self.ple_embedding(input_ids, cache)
        key = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        query = self.norm_query(hidden).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.flatten(-2)
        normed = self.norm_conv(gated)
        state = cache[2] if cache is not None else None
        if state is None or state.shape[0] != hidden.shape[0]:
            state = mx.zeros(
                (hidden.shape[0], self.state_len, hidden.shape[-1]),
                dtype=hidden.dtype,
            )
        conv_input = mx.concatenate([state, normed], axis=1)
        if cache is not None:
            cache[2] = conv_input[:, -self.state_len :]
        conv = nn.silu(self.conv1d(conv_input))
        return gated + conv[:, -hidden.shape[1] :]


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        self.linear_attn = GatedDeltaNet(args) if self.is_linear else None
        self.self_attn = QSAAttention(args) if not self.is_linear else None
        self.mlp = SparseMoeBlock(args)
        ple_index = (
            args.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in args.ple_layer_ids
            else None
        )
        self.ple = (
            PLELayer(args, layer_idx, ple_index) if ple_index is not None else None
        )
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    @staticmethod
    def _inject(block, hyper_input, weights):
        return hyper_input + (block[..., None, :] * weights[..., :, None]).flatten(-2)

    def __call__(
        self,
        hidden: mx.array,
        input_ids: mx.array,
        mask: Optional[mx.array],
        cache: Any,
        position_ids: Optional[mx.array],
        position_embeddings: Optional[tuple[mx.array, mx.array]],
    ):
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, input_ids, cache)
        block_input, residual, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block = self.linear_attn(block_input, mask=mask, cache=cache)
        else:
            block = self.self_attn(
                block_input,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        hidden = self._inject(block, residual, inject)
        block_input, residual, inject = self.mlp_hyper_connection(hidden)
        block = self.mlp(block_input)
        return self._inject(block, residual, inject)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = 0
        self.fa_idx = next(i for i, layer in enumerate(self.layers) if not layer.is_linear)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        gdn_sink: Optional[list] = None,
    ):
        # mlx-vlm releases with the legacy Qwen3.5 speculative verifier pass
        # this sink through even when no MTP drafter is configured. Qwen4-Exp
        # does not expose compatible GDN states, so the argument is intentionally
        # accepted and ignored for runtime compatibility (notably LM Studio).
        del gdn_sink
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.tile(h, (1, 1, self.args.hc_count))
        if cache is None:
            cache = [None] * len(self.layers)
        fa_cache = cache[self.fa_idx]
        fa_mask = _create_qwen3_5_attention_mask(
            h, fa_cache.caches[0] if isinstance(fa_cache, QSAKVCache) else fa_cache
        )
        ssm_mask = _create_qwen3_5_ssm_mask(h, cache[self.ssm_idx])

        position_embeddings = None
        if position_ids is not None:
            attn = self.layers[self.fa_idx].self_attn
            if not attn.rotary_emb.fused_apply:
                position_embeddings = attn.rotary_emb(h, position_ids)

        capture = set(capture_layer_ids or [])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(
                h,
                inputs,
                ssm_mask if layer.is_linear else fa_mask,
                layer_cache,
                position_ids,
                position_embeddings,
            )
            if hidden_sink is not None and index in capture:
                hidden_sink.append(h)
        return self.hyper_connection_mixer(h)


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        self._rope_deltas = None
        self._position_ids = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if layer.ple is not None else 2))
            else:
                caches.append(QSAKVCache())
        return caches

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if ".ngram_embedding.shards." in path:
                return {"group_size": 32, "bits": 4, "mode": "affine"}
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return predicate
