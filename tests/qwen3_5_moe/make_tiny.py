"""A tiny random Qwen3.6-35B-A3B (qwen3_5_moe) for the engine's checks, built with the repo's own converter.

  python tests/qwen3_5_moe/make_tiny.py OUT      (python: torch, safetensors, numpy)

Writes OUT/release (a Hugging Face-style release: config.json, generation_config.json, model.safetensors.index.json,
two shards, bf16, the release's tensor names and layouts: fused experts gate_up_proj [E, 2I, H] / down_proj [E, H, I],
conv1d [C, 1, K], zero-centred norms, dummy mtp.* tensors the converter skips), then OUT/model: the store
(python/convert/qwen3_5_moe/convert.py of it: store/layer_XX.bin int4, resident.safetensors), a random lag-2 linear
router set as the weight repo publishes it (routers/linear_k2.safetensors: f16 matrices, f32 vectors, the entry's facts as
metadata) and the manifest (tests/tiny_manifest.py). The engine derives its 8-bit in-memory file and router file on
first use (or ms_derive), as for the weight repo.
Geometry: the release's attention (16 query heads x 256, 2 KV heads, partial RoPE 64; Gated DeltaNet 16 key heads x 128,
32 value heads x 128, conv 4) at hidden 256 (a multiple of 256, so the packed i8mm path runs), 8 layers (LLLF twice),
64 experts top 8 (intermediate 128, shared 128), vocabulary 1024. SEED (env, default 0) fixes the weights.
"""
import json, os, subprocess, sys

import numpy as np
import torch
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "q36_tiny")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL, MODEL = os.path.join(OUT, "release"), os.path.join(OUT, "model")
os.makedirs(REL, exist_ok=True)
g = torch.Generator().manual_seed(int(os.environ.get("SEED", 0)))

H, NL, E, K, I, SI, V = 256, 8, 64, 8, 128, 128, 1024
NQ, NKV, HD = 16, 2, 256
NKH, NVH, DK, DV, CK = 16, 32, 128, 128, 4
types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (NL // 4)
tc = {
    "attention_bias": False, "attn_output_gate": True, "bos_token_id": 1, "eos_token_id": 2, "dtype": "bfloat16",
    "full_attention_interval": 4, "head_dim": HD, "hidden_act": "silu", "hidden_size": H, "layer_types": types,
    "linear_conv_kernel_dim": CK, "linear_key_head_dim": DK, "linear_num_key_heads": NKH, "linear_num_value_heads": NVH,
    "linear_value_head_dim": DV, "model_type": "qwen3_5_moe_text", "moe_intermediate_size": I, "num_attention_heads": NQ,
    "num_experts": E, "num_experts_per_tok": K, "num_hidden_layers": NL, "num_key_value_heads": NKV, "norm_topk_prob": True,
    "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
    "partial_rotary_factor": 0.25}, "shared_expert_intermediate_size": SI, "vocab_size": V, "mtp_num_hidden_layers": 1,
}
json.dump({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe", "text_config": tc},
          open(os.path.join(REL, "config.json"), "w"), indent=1)
json.dump({"bos_token_id": 1, "eos_token_id": [2], "temperature": 1.0, "top_p": 0.95, "top_k": 20},
          open(os.path.join(REL, "generation_config.json"), "w"), indent=1)


def rnd(*shape, std=0.05):
    return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)


P = "model.language_model."
shards = [{}, {}]
C = 2 * NKH * DK + NVH * DV
for L in range(NL):
    s = shards[L % 2]
    p = f"{P}layers.{L}."
    s[p + "input_layernorm.weight"] = rnd(H, std=0.1)
    s[p + "post_attention_layernorm.weight"] = rnd(H, std=0.1)
    if types[L] == "linear_attention":
        a = p + "linear_attn."
        s[a + "in_proj_qkv.weight"] = rnd(C, H)
        s[a + "in_proj_z.weight"] = rnd(NVH * DV, H)
        s[a + "in_proj_a.weight"] = rnd(NVH, H)
        s[a + "in_proj_b.weight"] = rnd(NVH, H)
        s[a + "conv1d.weight"] = rnd(C, 1, CK, std=0.3)
        s[a + "A_log"] = (torch.rand(NVH, generator=g) * 2 - 1).to(torch.bfloat16)
        s[a + "dt_bias"] = rnd(NVH, std=0.5)
        s[a + "norm.weight"] = (1 + torch.randn(DV, generator=g) * 0.1).to(torch.bfloat16)
        s[a + "out_proj.weight"] = rnd(H, NVH * DV, std=0.02)
    else:
        a = p + "self_attn."
        s[a + "q_proj.weight"] = rnd(2 * NQ * HD, H)
        s[a + "k_proj.weight"] = rnd(NKV * HD, H)
        s[a + "v_proj.weight"] = rnd(NKV * HD, H)
        s[a + "o_proj.weight"] = rnd(H, NQ * HD, std=0.02)
        s[a + "q_norm.weight"] = rnd(HD, std=0.1)
        s[a + "k_norm.weight"] = rnd(HD, std=0.1)
    m = p + "mlp."
    s[m + "gate.weight"] = rnd(E, H, std=0.2)
    s[m + "experts.gate_up_proj"] = rnd(E, 2 * I, H)
    s[m + "experts.down_proj"] = rnd(E, H, I)
    s[m + "shared_expert.gate_proj.weight"] = rnd(SI, H)
    s[m + "shared_expert.up_proj.weight"] = rnd(SI, H)
    s[m + "shared_expert.down_proj.weight"] = rnd(H, SI)
    s[m + "shared_expert_gate.weight"] = rnd(1, H)
shards[0][P + "embed_tokens.weight"] = rnd(V, H, std=1.0)
shards[1][P + "norm.weight"] = rnd(H, std=0.1)
shards[1]["lm_head.weight"] = rnd(V, H)
shards[1]["mtp.fc.weight"] = rnd(H, 2 * H)  # skipped by the converter, as the release's MTP head is
names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
wmap = {}
for nm, sh in zip(names, shards):
    save_file(sh, os.path.join(REL, nm), metadata={"format": "pt"})
    wmap.update({k: nm for k in sh})
json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(REL, "model.safetensors.index.json"), "w"), indent=1)

env = dict(os.environ, REPO=REL, OUT=MODEL, STATE_DIR=os.path.join(OUT, "_state"), THREADS="4")
env.pop("FINAL_DIR", None)
r = subprocess.run([sys.executable, os.path.join(REPO, "python/convert/qwen3_5_moe/convert.py")], env=env, capture_output=True, text=True)
if r.returncode:
    sys.exit(f"convert.py failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

# a random lag-2 linear router set (the published container: per target L, x = [rmsnorm(emb) * emb_norm, h_prev(, h_lag)])
rng = np.random.default_rng(int(os.environ.get("SEED", 0)) + 1)
LAG = 2
blob = {}
for L in range(NL):
    W = 3 * H if L >= LAG else 2 * H
    blob[f"{L}.emb_norm.weight"] = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    blob[f"{L}.lin.weight"] = (0.05 * rng.standard_normal((E, W))).astype(np.float16)
    blob[f"{L}.lin.bias"] = (0.05 * rng.standard_normal(E)).astype(np.float32)
os.makedirs(os.path.join(MODEL, "routers"), exist_ok=True)
save_np(blob, os.path.join(MODEL, "routers", "linear_k2.safetensors"),
        metadata={"family": "qwen3_5_moe", "lag": str(LAG), "targets": str(NL), "hidden": "0", "run": "tiny", "variant": "linear_k2", "source": "synthetic"})
r = subprocess.run([sys.executable, os.path.join(REPO, "tests/tiny_manifest.py"), MODEL], capture_output=True, text=True)
if r.returncode:
    sys.exit(f"tiny_manifest.py failed:\n{r.stdout}\n{r.stderr}")
print("tiny model:", MODEL)
