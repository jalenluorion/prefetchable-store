"""A tiny random Inkling 2 Mini (inkling2_loop) for the engine's checks, written as a RELEASE (the tensor names, dtypes and
file layout of thinkingmachines/Inkling-2-Mini), then converted with the repo's own converter.

  python tests/inkling2_loop/make_tiny.py OUT      (python: mlx, mlx-lm with the Inkling 2 model (mlx-lm PR 2086), numpy)

Architecture as the real model, scaled down: 12 layers, hidden 128, loop core layers 4-7 run 3 times (gate_rank 8),
attention-residual blocks of 4 layers (the loop starts on a block boundary, as 6 with blocks of 6 in the real model),
global attention on the last layer of each block (3, 7, 11; the real model: 5, 11, 17, 23), sliding window 16, sconv 4,
16 routed experts, top-4, intermediate 64, route_scale 8, the real softcaps; routed experts NVFP4 (group 16, per-expert
global scale) in layers 0-9 and MXFP8 (group 32) in layers 10-11 (the real model: 0-19 and 20-23); a small vision tower.

Built with the PR's own classes: mlx_lm.models.inkling2.Model with random weights, experts quantised by
mlx_lm.utils.quantize_model (nvfp4 with global_scale, mxfp8), exported to release names by the PR's test helper
(tests/models/test_inkling2.py `checkpoint`, MIT, copied below) and written with the release's dtypes: bf16 everywhere
except the F32 tensors of the release (attn_res queries, cross-pass queries, router bias, amax, denominators); nvfp4
codes U8 with F8_E4M3 scales, mxfp8 codes F8_E4M3 with U8 (UE8M0) scales.
OUT/release is the release; OUT/model is python/convert/inkling2_loop/convert.py of it, plus a random lag-2 linear router
set as the weight repo publishes it (routers/linear_k2.safetensors) and the manifest (tests/tiny_manifest.py). The engine
derives its 8-bit in-memory file and router file on first use (or ms_derive), as for the weight repo. SEED (env, default 0).
"""
import argparse
import shutil
import subprocess
import json
import os
import re
import sys

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python", "convert", "inkling2_loop"))
import store as S  # noqa: E402
from mlx_lm.models import inkling2  # noqa: E402
from mlx_lm.utils import quantize_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
ap.add_argument("--denominator", type=float, default=2688.0, help="outer_scale_denominator written for every nvfp4 matrix (the release: 2688 = 6 * 448 everywhere seen)")
a = ap.parse_args()
OUT = os.path.abspath(a.out)
REL, MOD = os.path.join(OUT, "release"), os.path.join(OUT, "model")
if os.path.exists(OUT):
    shutil.rmtree(OUT)

NL, MX8 = 12, (10, 11)
TEXT = {
    "model_type": "inkling2_loop_text", "hidden_size": 128, "num_hidden_layers": NL, "vocab_size": 256,
    "unpadded_vocab_size": 250, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 32, "sliding_window_size": 16,
    "local_layer_ids": [L for L in range(NL) if L not in (3, 7, 11)], "d_rel": 8, "rel_extent": 16,
    "log_scaling_n_floor": 8, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 4, "n_routed_experts": 16,
    "num_experts_per_tok": 4, "n_shared_experts": 0, "intermediate_size": 64, "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": True, "final_logit_softcapping": 30.0,
    "loop_config": {"core_start": 4, "core_end": 8, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
}
VISION = {"model_type": "inkling2_vision", "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
          "native_patch_size": 2, "adapter_merge_size": 2, "posemb_grid_size": 4, "projection_hidden_dim": 48}
CONFIG = {"dtype": "bfloat16", "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop",
          "eos_token_id": 249, "image_token_id": 248, "text_config": TEXT, "vision_config": VISION,
          "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}

# the PR's export helper (release names): mlx-lm PR 2086 tests/models/test_inkling2.py, MIT licence (c) 2023 Apple Inc.
_CHECKPOINT_NAMES = [
    (r"^lm_head\.", "model.llm.unembed."),
    (r"^model\.language_model\.embed_tokens\.", "model.llm.embed."),
    (r"^model\.language_model\.", "model.llm."),
    (r"\.final_input\.query$", ".attn_res.output_query"),
    (r"\.layers\.(\d+)\.mlp_input\.query$", r".attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.", ".attn.wv_dv."),
    (r"\.self_attn\.r_proj\.", ".attn.wr_du."),
    (r"\.self_attn\.o_proj\.", ".attn.wo_ud."),
    (r"\.self_attn\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.", ".attn."),
    (r"\.q_message_gate\.", ".controllers.attn."),
    (r"\.(k|v|mlp)_message_gate\.", r".controllers.\1."),
    (r"(\.controllers\.\w+)\.down_proj\.weight$", r"\1.down_weight"),
    (r"(\.controllers\.\w+)\.forget_gate_proj\.weight$", r"\1.forget_weight"),
    (r"(\.controllers\.\w+)\.message_gate_proj\.weight$", r"\1.gain_weight"),
    (r"\.post_attention_convolution\.", ".attn_sconv."),
    (r"\.post_mlp_convolution\.", ".mlp_sconv."),
    (r"\.conv1d\.weight$", ".weight"),
    (r"sconv\.static_weight$", "sconv.static_w"),
    (r"sconv\.weight_proj\.weight$", "sconv.weight_proj"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
    (r"^model\.vision_tower\.pre_layernorm\.", "model.visual.backbone.ln_pre."),
    (r"^model\.vision_tower\.merger\.linear_1\.", "model.visual.proj_in."),
    (r"^model\.vision_tower\.merger\.linear_2\.", "model.visual.proj_out."),
    (r"^model\.vision_tower\.", "model.visual.backbone."),
    (r"position_embedding\.weight$", "position_embedding"),
    (r"\.layernorm_before\.", ".ln_1."),
    (r"\.layernorm_after\.", ".ln_2."),
    (r"\.mlp\.fc1\.", ".mlp.c_fc."),
    (r"\.mlp\.fc2\.", ".mlp.c_proj."),
    (r"\.lambda_(\d)$", r".ls_\1.gamma"),
]


def checkpoint(model: inkling2.Model) -> dict:
    weights = dict(tree_flatten(model.parameters()))
    ckpt = {}
    denominator = mx.array([7.5])
    for k, v in weights.items():
        if k.endswith(".message_input.query"):
            q, kv = mx.concatenate([v[:, :1], v[:, 3:]], axis=1), v[:, 1:3]
            ckpt["model.llm.middle_cycle_cross_pass_queries"] = q.flatten(0, 1)
            ckpt["model.llm.middle_cycle_cross_pass_kv_queries"] = kv.flatten(0, 1)
            continue
        if m := re.match(r"(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$", k):
            prefix, name, param = m.groups()
            prefix = prefix.replace("model.language_model.", "model.llm.")
            w = f"{prefix}.{'w13' if name == 'gate_up' else 'w2'}_weight"
            if param == "weight":
                ckpt[w] = v.view(mx.uint8)
            elif param == "scales":
                ckpt[f"{w}_scale"] = v
            else:
                ckpt[f"{w}_amax"] = v * denominator / 2688.0
                ckpt[f"{w}_outer_scale_denominator"] = denominator
            continue
        if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
            ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
            continue
        for pattern, replacement in _CHECKPOINT_NAMES:
            k = re.sub(pattern, replacement, k)
        if k.endswith("sconv.weight"):
            v = v.swapaxes(1, 2)
        ckpt[k] = v
    return ckpt



mx.random.seed(a.seed)
model = inkling2.Model(inkling2.ModelArgs.from_dict(CONFIG))
weights = []
for k, v in tree_flatten(model.parameters()):  # as the PR's create_model, with a larger spread for the router and queries
    if k.endswith(("norm.weight", "l2_clip.scale", "lambda_1", "lambda_2")):
        v = 1 + 0.1 * mx.random.normal(v.shape)
    elif k.endswith(".mlp.gate.e_score_correction_bias"):
        v = 0.05 * mx.random.normal(v.shape)
    else:
        fan_in = v.shape[-1] if v.ndim > 1 else 16
        v = mx.random.normal(v.shape) * fan_in ** -0.5
    weights.append((k, v))
model.update(tree_unflatten(weights))


def predicate(path, module):
    if ".mlp.experts." not in path:
        return False
    if any(f".layers.{L}." in path for L in MX8):
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}


quantize_model(model, {}, None, None, quant_predicate=predicate)
mx.eval(model.parameters())

ckpt = checkpoint(model)  # release names; experts: codes as uint8, scales uint8, amax = gs * denominator / 2688
# The helper also renames the vision tower's self_attn.{q,k,v}_proj / self_attn.out_proj to the text names; the release
# keeps them (model.visual.backbone.layers.N.self_attn.q_proj.weight), so undo that for model.visual.*.
VIS_BACK = [(".attn.wq_du.", ".self_attn.q_proj."), (".attn.wk_dv.", ".self_attn.k_proj."),
            (".attn.wv_dv.", ".self_attn.v_proj."), (".attn.out_proj.", ".self_attn.out_proj.")]
for k in [k for k in ckpt if k.startswith("model.visual.")]:
    k2 = k
    for x, y in VIS_BACK:
        k2 = k2.replace(x, y)
    ckpt[k2] = ckpt.pop(k)
den = mx.array([a.denominator], dtype=mx.float32)
for k in list(ckpt):  # the PR helper uses 7.5; use --denominator (amax rescaled so the global scale is unchanged)
    if k.endswith("_outer_scale_denominator"):
        amax_k = k.replace("_outer_scale_denominator", "_amax")
        ckpt[amax_k] = ckpt[amax_k] / ckpt[k] * den
        ckpt[k] = den
F32 = re.compile(r"(attn_res\.|cross_pass_|\.mlp\.gate\.bias$|_amax$|_outer_scale_denominator$)")
EXPW = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight(_scale)?$")


def to_np(v, dt):
    if dt == "BF16":
        return np.array(v.astype(mx.bfloat16).view(mx.uint16))
    if dt == "F32":
        return np.array(v.astype(mx.float32))
    assert v.dtype == mx.uint8, v.dtype
    return np.array(v)


entries = []
for k in sorted(ckpt):
    v = ckpt[k]
    m = EXPW.fullmatch(k)
    if m:
        nv = int(m.group(1)) not in MX8
        dt = ("F8_E4M3" if m.group(3) else "U8") if nv else ("U8" if m.group(3) else "F8_E4M3")
    else:
        dt = "F32" if F32.search(k) else "BF16"
    arr = to_np(v, dt)
    entries.append((k, dt, list(arr.shape), arr.nbytes, arr.tobytes()))
os.makedirs(REL, exist_ok=True)
S.write_safetensors(os.path.join(REL, "model-00001-of-00001.safetensors"), entries, metadata={"format": "pt"})
json.dump(CONFIG, open(os.path.join(REL, "config.json"), "w"), indent=2)
groups = [{"compute_dtype": "nvfp4", "tensors": [f"model.llm.layers.{L}.mlp.experts.{w}_weight" for L in range(NL) if L not in MX8 for w in ("w13", "w2")]},
          {"compute_dtype": "mxfp8", "tensors": [f"model.llm.layers.{L}.mlp.experts.{w}_weight" for L in MX8 for w in ("w13", "w2")]}]
json.dump({"model": {"parameter_groups": groups}, "quant_method": "inkling2"}, open(os.path.join(REL, "hf_quant_config.json"), "w"), indent=1)
json.dump({"eos_token_id": 249}, open(os.path.join(REL, "generation_config.json"), "w"))
n_exp = sum(e[3] for e in entries if EXPW.fullmatch(e[0]))
print(f"tiny release -> {REL}: {len(entries)} tensors, {sum(e[3] for e in entries)} B ({n_exp} B routed experts)")

r = subprocess.run([sys.executable, os.path.join(REPO, "python/convert/inkling2_loop/convert.py"), "--release", REL, "--out", MOD],
                   capture_output=True, text=True)
if r.returncode:
    sys.exit(f"convert.py failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
# a random lag-2 linear router set (the published container), one target per step (application): 4 + 4 x 3 + 4 = 20
LC = TEXT["loop_config"]
APPS = LC["core_start"] + (LC["core_end"] - LC["core_start"]) * LC["num_passes"] + NL - LC["core_end"]
H, E, LAG = TEXT["hidden_size"], TEXT["n_routed_experts"], 2
rng = np.random.default_rng(a.seed + 1)
blob = {}
for t in range(APPS):
    W = 3 * H if t >= LAG else 2 * H
    blob[f"{t}.emb_norm.weight"] = mx.array((1 + 0.1 * rng.standard_normal(H)).astype(np.float32))
    blob[f"{t}.lin.weight"] = mx.array((0.05 * rng.standard_normal((E, W))).astype(np.float16))
    blob[f"{t}.lin.bias"] = mx.array((0.05 * rng.standard_normal(E)).astype(np.float32))
os.makedirs(os.path.join(MOD, "routers"), exist_ok=True)
mx.save_safetensors(os.path.join(MOD, "routers", "linear_k2.safetensors"), blob,
                    metadata={"family": "inkling2_loop", "lag": str(LAG), "targets": str(APPS), "hidden": "0", "run": "tiny", "variant": "linear_k2",
                              "source": "synthetic"})
r = subprocess.run([sys.executable, os.path.join(REPO, "tests/tiny_manifest.py"), MOD], capture_output=True, text=True)
if r.returncode:
    sys.exit(f"tiny_manifest.py failed:\n{r.stdout}\n{r.stderr}")
print("tiny model:", MOD)
