"""Inkling 2 Mini resident file for dmeng: DMENGRES1 format.

  python python/derive/inkling2_loop.py MODEL_DIR [--bits 8|0] [--out F]   (default F: MODEL_DIR/dmeng_resident.bin)

--bits 8 (default): the six attention projections of every layer (q, k, v, r, o, gate) and the output head as MLX affine
  8-bit group 64, taken BYTE FOR BYTE from mlx_run.py --resident-bits 8's cache (model/resident_q8.safetensors: nn.quantize
  of the bf16 weights, bf16 scales and biases). The scales/biases are written as fp16 (the kernels' format); the
  conversion is checked exact for every value (bf16 -> fp16 loses nothing above 2^-17; the smallest scale is 4.6e-5).
--bits 0: the same matrices as released (bf16), for the exact reference path (--prec f32) and the tiny-model check.
Everything else as mlx_run.py holds it: every depth-attention query in float32; the router and the message-gate
forget/gain projections (float32 in MLX, cast from the release's bf16) as bf16, exact (checked), computed in fp32, the message-gate down projections bf16 (computed in fp32), small tensors (norms, convolutions,
clip scales, relative-logit projections, sink projections, dynamic-conv weight projections) widened to float32 (exact).
The embedding is not copied: the engine reads its rows from resident.safetensors (cfg emb_file / emb_off).
"""
import argparse, json, os, struct, sys

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--bits", type=int, default=8, choices=[0, 8])
ap.add_argument("--out", default=None)
a = ap.parse_args()
M = os.path.abspath(a.model)
cfg_full = json.load(open(os.path.join(M, "config.json")))
tc = cfg_full["text_config"]
meta = json.load(open(os.path.join(M, "store", "meta.json")))
ALIGN = 16384
P = "model.language_model."


def up(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


class Writer:
    RESERVE = 1 << 20

    def __init__(self, path):
        self.path, self.tmp = path, path + ".tmp"
        self.f = open(self.tmp, "wb")
        self.f.write(b"\0" * self.RESERVE)
        self.cfg, self.tab, self.off = [], [], 0

    def add(self, name, arr, dtype):
        a_ = np.ascontiguousarray(arr)
        assert a_.dtype.itemsize == {"u8": 1, "f16": 2, "bf16": 2, "f32": 4}[dtype], (name, a_.dtype, dtype)
        self.tab.append((name, dtype, self.off, a_.nbytes, a_.shape))
        self.f.write(a_.tobytes())
        end = up(self.off + a_.nbytes)
        self.f.write(b"\0" * (end - self.off - a_.nbytes))
        self.off = end

    def close(self):
        lines = ["cfg %s %s" % (k, v) for k, v in self.cfg]
        lines += ["t %s %s %d %d %d %s" % (n, d, o, nb, len(sh), " ".join(str(x) for x in sh)) for n, d, o, nb, sh in self.tab]
        lines.append("end")
        head = ("DMENGRES1 %d\n" % self.RESERVE + "\n".join(lines) + "\n").encode()
        assert len(head) <= self.RESERVE
        self.f.seek(0)
        self.f.write(head)
        self.f.close()
        os.replace(self.tmp, self.path)


# ---------------------------------------------------------------- the sanitized weights (MLX names)
if a.bits:
    W = mx.load(os.path.join(M, f"resident_q{a.bits}.safetensors"))
    src_note = f"resident_q{a.bits}.safetensors (mlx_run.py --resident-bits {a.bits} cache)"
else:
    from mlx_lm.models import inkling2

    model = inkling2.Model(inkling2.ModelArgs.from_dict({k: v for k, v in cfg_full.items() if k != "vision_config"}))
    raw = mx.load(os.path.join(M, "resident.safetensors"))
    raw.pop("model.llm.embed.weight")
    W = {k: v for k, v in model.sanitize(raw).items() if not k.endswith(".global_scale") and not k.startswith("model.vision_tower.")}
    src_note = "resident.safetensors via the PR's sanitize (bf16 as released)"

TAG = "tiny" if "_tiny" in M else "real"
out = Writer(a.out or os.path.join(M, "dmeng_resident.bin"))
bad = [0]


def f32(name, key):
    out.add(name, np.array(W[key].astype(mx.float32)), "f32")


def bf16(name, key):
    v = W[key]
    assert v.dtype == mx.bfloat16, (key, v.dtype)
    out.add(name, np.array(v.view(mx.uint16)), "bf16")


def exact_bf16(name, key):
    """float32 in MLX but bf16 in the release (the PR casts them): stored as bf16, which loses nothing (checked), read at
    half the bytes and computed in fp32 exactly as before"""
    v = W[key].astype(mx.float32)
    b = v.astype(mx.bfloat16)
    assert mx.array_equal(b.astype(mx.float32), v).item(), f"{key} is not exactly bf16"
    out.add(name, np.array(b.view(mx.uint16)), "bf16")


def lin(name, key):
    """a big projection: AFF8 (q/s/b) when quantised, else bf16 as released"""
    if key + ".scales" in W:
        w, s, b = W[key + ".weight"], W[key + ".scales"], W[key + ".biases"]
        R = w.shape[0]
        C = w.shape[1] * 4
        assert w.dtype == mx.uint32 and s.shape == (R, C // 64), (key, w.shape, s.shape)
        out.add(name + ".q", np.array(w).view(np.uint8).reshape(R, C), "u8")
        for t, x in (("s", s), ("b", b)):
            x32 = np.array(x.astype(mx.float32))
            h = x32.astype(np.float16)
            bad[0] += int((h.astype(np.float32) != x32).sum())
            out.add(name + "." + t, h.view(np.uint16), "f16")
    else:
        bf16(name, key + ".weight")


H, NL = tc["hidden_size"], tc["num_hidden_layers"]
loop = tc["loop_config"]
CS, CE = loop["core_start"], loop["core_end"]
glob = "".join("L" if L in tc["local_layer_ids"] else "G" for L in range(NL))
fmts = "".join("N" if meta["layer_format"][L] == "nvfp4" else "M" for L in range(NL))
F = meta["formats"]
rhdr_len = struct.unpack("<Q", open(os.path.join(M, "resident.safetensors"), "rb").read(8))[0]
rhdr = json.loads(open(os.path.join(M, "resident.safetensors"), "rb").read(8 + rhdr_len)[8:])
emb = rhdr["model.llm.embed.weight"]
assert emb["dtype"] == "BF16" and emb["shape"] == [tc["vocab_size"], H]
assert tc["swa_num_attention_heads"] == tc["num_attention_heads"] and tc["swa_head_dim"] == tc["head_dim"]
assert tc["swa_num_key_value_heads"] == tc["num_key_value_heads"]
out.cfg += [("model", "inkling2_loop"), ("bits", a.bits), ("source", src_note.replace(" ", "_")), ("hidden", H), ("layers", NL),
            ("experts", tc["n_routed_experts"]), ("topk", tc["num_experts_per_tok"]), ("inter", tc["intermediate_size"]),
            ("heads", tc["num_attention_heads"]), ("kv_heads", tc["num_key_value_heads"]), ("head_dim", tc["head_dim"]),
            ("d_rel", tc["d_rel"]), ("rel_extent", tc["rel_extent"]), ("window", tc["sliding_window_size"]),
            ("gate_rank", loop["gate_rank"]), ("core_start", CS), ("core_end", CE), ("passes", loop["num_passes"]),
            ("block", tc["n_layers_per_attn_res_block"]), ("vocab", tc["vocab_size"]), ("unpadded_vocab", tc["unpadded_vocab_size"]),
            ("softcap", tc["final_logit_softcapping"]), ("gate_cap", tc["mlp_gate_softcap"]), ("up_cap", tc["mlp_up_softcap"]),
            ("route_scale", tc["route_scale"]), ("eps", tc["rms_norm_eps"]), ("qk_clip", repr(tc["qk_l2_clip_threshold"])),
            ("rel_clip", tc["rel_l2_clip_threshold"]), ("log_floor", tc["log_scaling_n_floor"]), ("log_alpha", tc["log_scaling_alpha"]),
            ("sconv", tc["sconv_kernel_size"]), ("layer_kind", glob), ("layer_format", fmts),
            ("emb_file", "resident.safetensors"), ("emb_off", 8 + rhdr_len + emb["data_offsets"][0]),
            ("layer_file", meta["layer_file"].replace("{L:02d}", "%02d")), ("eos", cfg_full.get("eos_token_id", -1))]
for k, f in F.items():
    tag = "nv" if k == "nvfp4" else "mx"
    fl = f["fields"]
    out.cfg += [(f"{tag}_rec", f["record_bytes"]), (f"{tag}_gu_c", fl["gate_up.weight"]["offset"]), (f"{tag}_gu_s", fl["gate_up.scales"]["offset"]),
                (f"{tag}_dn_c", fl["down.weight"]["offset"]), (f"{tag}_dn_s", fl["down.scales"]["offset"])]
    if "trailer" in fl:
        out.cfg.append((f"{tag}_trailer", fl["trailer"]["offset"]))

f32("embed_norm", P + "embed_norm.weight")
f32("final_q", P + "final_input.query")
f32("norm", P + "norm.weight")
f32("msgq", P + "message_input.query")
lin("head", "lm_head")
for L in range(NL):
    p, s = f"{P}layers.{L}.", f"{L}."
    g = glob[L] == "G"
    if L > 0:
        f32(s + "aq", p + "attention_input.query")
    f32(s + "mq", p + "mlp_input.query")
    if g:
        for x in "qkv":
            f32(s + "ln" + x, p + f"{x}_input_layernorm.weight")
    else:
        f32(s + "ln_in", p + "input_layernorm.weight")
    for nm, key in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("r", "r_proj"), ("o", "o_proj"), ("g", "gate_proj")):
        lin(s + nm, p + "self_attn." + key)
    f32(s + "sink", p + "self_attn.sink_proj.weight")
    f32(s + "qclip", p + "self_attn.q_l2_clip.scale")
    f32(s + "rclip", p + "self_attn.rel_l2_clip.scale")
    f32(s + "rel", p + "self_attn.rel_logits_proj.proj")
    for x in "kv":
        if g:
            f32(s + x + "stat", p + f"self_attn.{x}_sconv.static_weight")
            f32(s + x + "wp", p + f"self_attn.{x}_sconv.weight_proj.weight")
        else:
            out.add(s + x + "conv", np.array(W[p + f"self_attn.{x}_sconv.conv1d.weight"].astype(mx.float32)).reshape(-1, tc["sconv_kernel_size"]), "f32")
    out.add(s + "aconv", np.array(W[p + "post_attention_convolution.conv1d.weight"].astype(mx.float32)).reshape(H, -1), "f32")
    out.add(s + "mconv", np.array(W[p + "post_mlp_convolution.conv1d.weight"].astype(mx.float32)).reshape(H, -1), "f32")
    f32(s + "mlnorm", p + "post_attention_layernorm.weight")
    exact_bf16(s + "router", p + "mlp.gate.weight")
    f32(s + "rbias", p + "mlp.gate.e_score_correction_bias")
    if CS <= L < CE:
        for x in "qkvm":
            q = p + ("mlp" if x == "m" else x) + "_message_gate."
            bf16(s + "mg" + x + ".down", q + "down_proj.weight")
            exact_bf16(s + "mg" + x + ".forget", q + "forget_gate_proj.weight")
            exact_bf16(s + "mg" + x + ".gain", q + "message_gate_proj.weight")
    mx.clear_cache()
out.cfg.append(("fp16_inexact", bad[0]))
out.close()
assert bad[0] == 0, f"{bad[0]} scale/bias values not exact in fp16"
print(f"wrote {out.path}: {len(out.tab)} tensors, {out.off / 1e9:.3f} GB; {src_note}; fp16 conversion exact")
