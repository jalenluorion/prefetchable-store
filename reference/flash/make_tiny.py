"""A synthetic Qwen3.8-Flash-Next release for testing the converter and the runner without the 360 GB download.

  .venv/bin/python make_tiny.py OUT_DIR

Writes a random qwen4_exp model with the real checkpoint's structure at tiny sizes, laid out as a Hugging Face release:
config.json (multimodal wrapper with text_config, the real field names), model-0000K-of-00004.safetensors with
model.safetensors.index.json, generation_config.json, tokenizer.json (a byte-level BPE trained to 512 pieces),
tokenizer_config.json without a chat template and chat_template.jinja (ChatML with tools), LICENSE.

Same tensor names, dtypes (bf16; the n-gram hashing constants int64) and layouts as the release: fused
mlp.experts.gate_up_proj [E, 2I, H] / mlp.experts.down_proj [E, H, I] (in different shards), the n-gram table split into
`split_ngram_parts` tensors ple.ple_embedding.ngram_embedding.shard_K.weight, hyper-connection tensors on every block,
zero-centred RMSNorm weights (the model computes x * (1 + w)), conv1d weights as [C, 1, K], and dummy mtp.* and
model.visual.* tensors that every consumer must skip. Sizes: hidden 128, 4 streams, 8 layers (3 Gated DeltaNet + 1 QSA,
twice), 16 experts top-2 (intermediate 64, shared 64), PLE at layer index 1 (ple_layer_ids [2]) with 2 heads per n-gram
over primes above 1000 (4 heads x 32 dims, 4096 rows in 4 shards), vocab 512, QSA budget 16 (a 40-token prompt crosses
it, so the sparse path runs).
"""
import json, math, os, sys

import torch
from safetensors.torch import save_file

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "tiny_release")
os.makedirs(OUT, exist_ok=True)
g = torch.Generator().manual_seed(int(os.environ.get("SEED", 0)))

# ---------------------------------------------------------------------------------------------------- tokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<think>", "</think>"]
VOCAB = 512
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
corpus = [open(__file__).read()] * 4 + [
    "The history of the city of Pittsburgh begins with the rivers. Hello, how are you today? I am fine, thank you.",
    "def main():\n    print('hello world')\n    return 0\n",
]
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
assert tk.get_vocab_size() <= VOCAB, tk.get_vocab_size()
tk.save(os.path.join(OUT, "tokenizer.json"))
EOS, IM_END = tk.token_to_id("<|endoftext|>"), tk.token_to_id("<|im_end|>")
json.dump({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None,
           "model_max_length": 4096, "add_bos_token": False}, open(os.path.join(OUT, "tokenizer_config.json"), "w"), indent=1)
open(os.path.join(OUT, "chat_template.jinja"), "w").write(
    "{%- if tools %}<|im_start|>system\nYou may call tools:\n{%- for t in tools %}\n{{ t | tojson }}{%- endfor %}<|im_end|>\n{%- endif %}"
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
    "{%- if m.tool_calls %}{%- for tc in m.tool_calls %}\n<tool_call>\n{{ {'name': tc.function.name, 'arguments': tc.function.arguments} | tojson }}\n</tool_call>{%- endfor %}{%- endif %}"
    "<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}")
json.dump({"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "eos_token_id": [IM_END, EOS], "bos_token_id": EOS},
          open(os.path.join(OUT, "generation_config.json"), "w"), indent=1)
open(os.path.join(OUT, "LICENSE"), "w").write("Synthetic test weights (random). No license terms.\n")

# ---------------------------------------------------------------------------------------------------- config
H, HC, LOW, NL, NE, TOPK, I, SH = 128, 4, 64, 8, 16, 2, 64, 64
NKH, NVH, DK, DV, CONVK = 2, 4, 32, 32, 4
NH, NKV, HD = 4, 2, 64
IXH, IXKV, IXD = 2, 1, 32
NG, HPN, NGBASE, DIV, PARTS, SEED = 3, 2, 1000, 128, 4, 1234
layer_types = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(NL)]
text = {
    "model_type": "qwen4_exp_text", "hidden_size": H, "hc_count": HC, "hc_lowrank": LOW, "num_hidden_layers": NL, "layer_types": layer_types,
    "full_attention_interval": 4, "linear_num_key_heads": NKH, "linear_num_value_heads": NVH, "linear_key_head_dim": DK,
    "linear_value_head_dim": DV, "linear_conv_kernel_dim": CONVK, "num_attention_heads": NH, "num_key_value_heads": NKV,
    "head_dim": HD, "attention_bias": False, "attention_dropout": 0.0, "indexer_n_heads": IXH, "indexer_kv_heads": IXKV,
    "indexer_head_dim": IXD, "indexer_budget": 16, "indexer_compress_ratio": 4, "num_experts": NE, "num_experts_per_tok": TOPK,
    "moe_intermediate_size": I, "shared_expert_intermediate_size": SH, "ple_layer_ids": [2], "ple_embed_dim": H,
    "ple_conv_kernel_size": 4, "ngram_size": NG, "heads_per_ngram": HPN, "ngram_vocab_size_base": NGBASE,
    "make_ngram_vocab_size_divisible_by": DIV, "split_ngram_parts": PARTS, "seed": SEED, "vocab_size": VOCAB,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "output_gate_type": "sigmoid", "hidden_act": "silu",
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "partial_rotary_factor": 0.25, "eos_token_id": EOS, "bos_token_id": EOS, "pad_token_id": None, "tie_word_embeddings": False,
    "mtp_num_hidden_layers": 1, "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]}, "dtype": "bfloat16",
    "mamba_ssm_dtype": "float32", "router_aux_loss_coef": 0.001, "use_cache": True,
}
config = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "text_config": text,
          "vision_config": {"model_type": "qwen4_exp", "depth": 1, "hidden_size": 32, "out_hidden_size": H, "num_heads": 2,
                            "intermediate_size": 64, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                            "in_channels": 3, "deepstack_visual_indexes": []},
          "image_token_id": 500, "video_token_id": 501, "vision_start_token_id": 502, "vision_end_token_id": 503,
          "tie_word_embeddings": False, "transformers_version": "5.8.0.dev0"}
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------------- n-gram constants
# exactly the reference's construction (transformers modeling_qwen4_exp: _build_layer_multipliers, _find_nth_prime_after)
def splitmix64(v):
    M = (1 << 64) - 1
    v = (v + 0x9E3779B97F4A7C15) & M
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & M
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & M
    return (v ^ (v >> 31)) & M


def is_prime(v):
    return v >= 2 and (v == 2 or (v % 2 and all(v % d for d in range(3, math.isqrt(v) + 1, 2))))


def nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


ple_index, heads = 0, (NG - 1) * HPN
half = max(1, (((1 << 63) - 1) // VOCAB) // 2)
mult = [2 * (splitmix64((SEED + 10007 * ple_index + 0x9E3779B97F4A7C15 * (i + 1)) & ((1 << 64) - 1)) % half) + 1 for i in range(NG)]
sizes = [nth_prime_after(NGBASE - 1, ple_index * heads + h + 1) for h in range(heads)]
offs = [sum(sizes[:h]) for h in range(heads)]
padded = math.ceil(sum(sizes) / DIV) * DIV
assert padded % PARTS == 0
SHARD_ROWS, NGD = padded // PARTS, H // heads


# ---------------------------------------------------------------------------------------------------- weights
def rnd(*shape, std=1.0, mean=0.0):
    return (torch.randn(*shape, generator=g) * std + mean).to(torch.bfloat16)


def lin(o, i, gain=1.0):
    return rnd(o, i, std=gain / math.sqrt(i))


def gated_residual(p, combine=True):
    d = {f"{p}.hc_norm.weight": rnd(HC * H, std=0.1), f"{p}.input_mix_weight_down.weight": lin(LOW, HC * H),
         f"{p}.input_mix_weight_up.weight": lin(HC * H, LOW)}
    if combine:
        d[f"{p}.block_inject_weight.weight"] = lin(HC, HC * H)
    return d


P = "model.language_model"
shards = [dict() for _ in range(4)]
shards[0][f"{P}.embed_tokens.weight"] = rnd(VOCAB, H)
shards[1]["lm_head.weight"] = lin(VOCAB, H, gain=2.0)
shards[1].update(gated_residual(f"{P}.hyper_connection_mixer", combine=False))
for L in range(NL):
    s = shards[0] if L < NL // 2 else shards[1]
    p = f"{P}.layers.{L}"
    s.update(gated_residual(f"{p}.attn_hyper_connection"))
    s.update(gated_residual(f"{p}.mlp_hyper_connection"))
    if layer_types[L] == "linear_attention":
        kd, vd = NKH * DK, NVH * DV
        s.update({f"{p}.linear_attn.in_proj_qkv.weight": lin(2 * kd + vd, H), f"{p}.linear_attn.in_proj_z.weight": lin(vd, H),
                  f"{p}.linear_attn.in_proj_a.weight": lin(NVH, H), f"{p}.linear_attn.in_proj_b.weight": lin(NVH, H),
                  f"{p}.linear_attn.conv1d.weight": rnd(2 * kd + vd, 1, CONVK, std=0.5), f"{p}.linear_attn.dt_bias": rnd(NVH, std=0.5),
                  f"{p}.linear_attn.A_log": torch.log(torch.rand(NVH, generator=g) * 15 + 1).to(torch.bfloat16),
                  f"{p}.linear_attn.norm.weight": rnd(DV, std=0.1, mean=1.0), f"{p}.linear_attn.out_proj.weight": lin(H, vd)})
    else:
        s.update({f"{p}.self_attn.q_proj.weight": lin(NH * HD * 2, H), f"{p}.self_attn.k_proj.weight": lin(NKV * HD, H),
                  f"{p}.self_attn.v_proj.weight": lin(NKV * HD, H), f"{p}.self_attn.o_proj.weight": lin(H, NH * HD),
                  f"{p}.self_attn.q_norm.weight": rnd(HD, std=0.1), f"{p}.self_attn.k_norm.weight": rnd(HD, std=0.1),
                  f"{p}.self_attn.indexer.index_qk_proj.weight": lin((IXH + IXKV) * IXD, H),
                  f"{p}.self_attn.indexer.q_layernorm.weight": rnd(IXD, std=0.1), f"{p}.self_attn.indexer.k_layernorm.weight": rnd(IXD, std=0.1)})
    s.update({f"{p}.mlp.gate.weight": lin(NE, H, gain=2.0), f"{p}.mlp.shared_expert.gate_proj.weight": lin(SH, H),
              f"{p}.mlp.shared_expert.up_proj.weight": lin(SH, H), f"{p}.mlp.shared_expert.down_proj.weight": lin(H, SH),
              f"{p}.mlp.shared_expert_gate.weight": lin(1, H)})
    shards[0][f"{p}.mlp.experts.gate_up_proj"] = rnd(NE, 2 * I, H, std=1 / math.sqrt(H))  # fused, in a different shard than down_proj
    shards[1][f"{p}.mlp.experts.down_proj"] = rnd(NE, H, I, std=1 / math.sqrt(I))
p = f"{P}.layers.1.ple"
shards[2].update({f"{p}.key_proj.weight": lin(HC * H, H), f"{p}.value_proj.weight": lin(H, H), f"{p}.norm_key.weight": rnd(HC * H, std=0.1),
                  f"{p}.norm_query.weight": rnd(HC * H, std=0.1), f"{p}.norm_conv.weight": rnd(HC * H, std=0.1),
                  f"{p}.conv1d.weight": rnd(HC * H, 1, 4, std=0.3),
                  f"{p}.ple_embedding.layer_multipliers": torch.tensor(mult, dtype=torch.int64),
                  f"{p}.ple_embedding.ngram_heads_vocab_sizes": torch.tensor(sizes, dtype=torch.int64),
                  f"{p}.ple_embedding.ngram_heads_offsets": torch.tensor(offs, dtype=torch.int64)})
for K in range(PARTS):
    shards[2][f"{p}.ple_embedding.ngram_embedding.shard_{K}.weight"] = rnd(SHARD_ROWS, NGD)
# dummies the converter and the runner must skip: an MTP layer with its own experts, a vision block
shards[3].update({"mtp.fc_hidden.weight": lin(H, 2 * H), "mtp.layers.0.mlp.experts.gate_up_proj": rnd(NE, 2 * I, H),
                  "mtp.layers.0.mlp.experts.down_proj": rnd(NE, H, I), "mtp.pre_fc_norm_hidden.weight": rnd(H, std=0.1),
                  "model.visual.blocks.0.attn.qkv.weight": lin(96, 32), "model.visual.blocks.0.attn.qkv.bias": rnd(96),
                  "model.visual.merger.linear_fc2.weight": lin(H, 64)})
index, total = {}, 0
for k, s in enumerate(shards):
    name = f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(s, os.path.join(OUT, name), metadata={"format": "pt"})
    for key, t in s.items():
        index[key] = name
        total += t.numel() * t.element_size()
json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(index.items()))}, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print(f"{OUT}: {len(index)} tensors in {len(shards)} shards, {total / 2**20:.1f} MiB | vocab {tk.get_vocab_size()} eos {EOS} im_end {IM_END} | "
      f"n-gram {PARTS} shards x {SHARD_ROWS} rows x {NGD} (heads {sizes}, offsets {offs}, multipliers {mult})")
print(json.dumps({"NG_SHARD_ROWS": SHARD_ROWS, "NG_DIM": NGD}))
