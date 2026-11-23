"""dmeng's own resident file (and router file), produced once from the q36 model folder.

  .venv/bin/python python/convert.py resident MODEL_DIR            # -> MODEL_DIR/dmeng_resident.bin
  .venv/bin/python python/convert.py router   MODEL_DIR ROUTER.npz # -> ROUTER.dmr (8-bit, same quantiser)

The resident file (dmeng_resident.bin):
  bytes 0..: a text header in a reserved region of the stated size (1 MiB); tensor offsets count from its end:
      DMENGRES1 <header_bytes>
      cfg <key> <value>                     model shape (from config.json) and the expert record layout (store/meta.json)
      t <name> <dtype> <offset> <nbytes> <ndim> <d0> <d1> ...
      end
  then every tensor at a 16 KB-aligned offset. dtypes: u8, f16, bf16, f32.
  An 8-bit matrix W [R, C] is three tensors NAME.q (u8 codes [R, C]), NAME.s, NAME.b (f16 scale / bias [R, C/64]):
  w = q * s + b per group of 64 along C (MLX affine 8-bit, group 64).

Which weights, and how (the Python engine's rule, local/q36/mlx_run.py `qpred`, so the codes are the same bytes):
  * every nn.Linear with >= 64 outputs and inputs % 64 == 0 except the router (mlp.gate) -> 8-bit, quantised with
    mx.quantize(bf16 -> fp16 weight, group_size=64, bits=8): exactly what nn.quantize does in mlx_run.py's
    resident_int8.safetensors (that file stores the same codes as uint32 words, 4 codes per word, little end first).
  * kept exact, as bf16 (the release's bytes): the router mlp.gate [E, H], shared_expert_gate [1, H], the Gated DeltaNet's
    in_proj_a / in_proj_b [Hv, H]. (mlx_run.py keeps these as fp16 casts of the bf16.)
  * f32: the norms (zero-centred ones + 1, computed in f32 from the bf16), conv1d [C, K] (kernel tap k of channel c),
    A_log, dt_bias, the Gated DeltaNet's one-centred gated-norm weight.
  * the embedding table is NOT copied: the engine reads one bf16 row per token from resident.safetensors (cfg emb_off).

The router file (.dmr) holds a lag-k router set (mlx_run.py's format: per layer emb_norm, lin, mlp.0, mlp.2) with the
three matrices 8-bit by the same quantiser, biases and emb_norm f32, and cfg lag / feats / per-layer input widths.
"""
import json, os, struct, sys

import mlx.core as mx
import numpy as np

ALIGN = 16384


def up(n, a=ALIGN):
    return (n + a - 1) // a * a


class Writer:
    """Streams tensors to disk as they are added (nothing is held in memory); the header goes into a reserved region at
    the front when the file is closed."""

    RESERVE = 1 << 20

    def __init__(self, path):
        self.path, self.tmp = path, path + ".tmp"
        self.f = open(self.tmp, "wb")
        self.f.write(b"\0" * self.RESERVE)
        self.cfg, self.tab, self.off = [], [], 0

    def add(self, name, arr, dtype):
        a = np.ascontiguousarray(arr)
        assert a.dtype.itemsize == {"u8": 1, "f16": 2, "bf16": 2, "f32": 4}[dtype], (name, a.dtype, dtype)
        assert self.f.tell() == self.RESERVE + self.off
        self.tab.append((name, dtype, self.off, a.nbytes, a.shape))
        self.f.write(a.tobytes())
        end = up(self.off + a.nbytes)
        self.f.write(b"\0" * (end - self.off - a.nbytes))
        self.off = end

    def q8(self, name, w):
        """w: an mx array [R, C] -> NAME.q/.s/.b with MLX's affine 8-bit quantiser on the fp16 cast (bf16 weights as
        mlx_run.py casts them; fp32 router weights straight to fp16, as mlx_run.py loads them)."""
        wq, s, b = mx.quantize(w.astype(mx.float16), group_size=64, bits=8)
        mx.eval(wq, s, b)
        R, C = w.shape
        self.add(name + ".q", np.array(wq).view(np.uint8).reshape(R, C), "u8")
        self.add(name + ".s", np.array(s).view(np.uint16).reshape(R, C // 64), "f16")
        self.add(name + ".b", np.array(b).view(np.uint16).reshape(R, C // 64), "f16")
        del wq, s, b
        mx.clear_cache()

    def q4(self, name, w):
        """w: an mx array [R, C] -> NAME.q (codes [R, C/2], MLX's 4-bit packing: element 2i in the low nibble of byte i)
        /.s/.b with MLX's affine 4-bit quantiser, group 64, on the fp16 cast (Stage 3c: 4-bit router weights)."""
        wq, s, b = mx.quantize(w.astype(mx.float16), group_size=64, bits=4)
        mx.eval(wq, s, b)
        R, C = w.shape
        self.add(name + ".q", np.array(wq).view(np.uint8).reshape(R, C // 2), "u8")
        self.add(name + ".s", np.array(s).view(np.uint16).reshape(R, C // 64), "f16")
        self.add(name + ".b", np.array(b).view(np.uint16).reshape(R, C // 64), "f16")
        del wq, s, b
        mx.clear_cache()

    def bf16(self, name, w):
        self.add(name, np.array(w.astype(mx.bfloat16).view(mx.uint16)), "bf16")

    def f32(self, name, w):
        self.add(name, np.array(w.astype(mx.float32)), "f32")

    def close(self):
        lines = ["cfg %s %s" % (k, v) for k, v in self.cfg]
        lines += ["t %s %s %d %d %d %s" % (n, d, o, nb, len(sh), " ".join(str(x) for x in sh)) for n, d, o, nb, sh in self.tab]
        lines.append("end")
        head = ("DMENGRES1 %d\n" % self.RESERVE + "\n".join(lines) + "\n").encode()
        assert len(head) <= self.RESERVE, "header larger than its reserved region"
        self.f.seek(0)
        self.f.write(head)
        self.f.close()
        os.replace(self.tmp, self.path)


def resident(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    meta = json.load(open(os.path.join(model_dir, "store", "meta.json")))
    src = os.path.join(model_dir, "resident.safetensors")
    with open(src, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hl))
    P = "model.language_model."
    emb = hdr[P + "embed_tokens.weight"]
    assert emb["dtype"] == "BF16"
    mx.set_cache_limit(256 << 20)
    W = mx.load(src)  # lazy
    out = Writer(os.path.join(model_dir, "dmeng_resident.bin"))
    H, NL = tc["hidden_size"], tc["num_hidden_layers"]
    types = "".join("F" if t == "full_attention" else "L" for t in tc["layer_types"])
    rp = tc.get("rope_parameters", {})
    hd = tc["head_dim"]
    out.cfg += [("hidden", H), ("layers", NL), ("layer_types", types), ("experts", tc["num_experts"]), ("topk", tc["num_experts_per_tok"]),
                ("moe_inter", tc["moe_intermediate_size"]), ("shared_inter", tc["shared_expert_intermediate_size"]),
                ("norm_topk", int(tc.get("norm_topk_prob", True))),
                ("lin_nk", tc["linear_num_key_heads"]), ("lin_nv", tc["linear_num_value_heads"]), ("lin_dk", tc["linear_key_head_dim"]),
                ("lin_dv", tc["linear_value_head_dim"]), ("conv_k", tc["linear_conv_kernel_dim"]), ("n_heads", tc["num_attention_heads"]),
                ("n_kv", tc["num_key_value_heads"]), ("head_dim", hd), ("rot_dim", int(hd * rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))),
                ("rope_theta", rp.get("rope_theta", 10000000)), ("eps", tc["rms_norm_eps"]), ("vocab", tc["vocab_size"]),
                ("emb_file", "resident.safetensors"), ("emb_off", 8 + hl + emb["data_offsets"][0]),
                ("rec_bytes", meta["record_bytes"]), ("store_layer_file", "store/layer_%02d.bin")]
    for kind in ("gate_proj", "up_proj", "down_proj"):
        out.cfg += [(f"off_{kind}", meta["offsets"][kind]), (f"soff_{kind}", meta["scale_offsets"][kind]), (f"boff_{kind}", meta["bias_offsets"][kind])]
    assert meta["dtype"] == "int4" and meta["group"] == 64 and meta["record_order"] == ["gate_proj", "up_proj", "down_proj"]

    def g(k):  # popped: an evaluated array left in the dict would keep its buffer
        return W.pop(P + k)

    for L in range(NL):
        p = f"layers.{L}."
        out.f32(f"{L}.ln_in", g(p + "input_layernorm.weight").astype(mx.float32) + 1.0)
        out.f32(f"{L}.ln_post", g(p + "post_attention_layernorm.weight").astype(mx.float32) + 1.0)
        if types[L] == "L":
            a = p + "linear_attn."
            out.q8(f"{L}.qkv", g(a + "in_proj_qkv.weight"))
            out.q8(f"{L}.z", g(a + "in_proj_z.weight"))
            out.bf16(f"{L}.a", g(a + "in_proj_a.weight"))
            out.bf16(f"{L}.b", g(a + "in_proj_b.weight"))
            cw = g(a + "conv1d.weight")  # [C, 1, K]
            out.f32(f"{L}.conv", cw.reshape(cw.shape[0], cw.shape[2]))
            out.f32(f"{L}.A_log", g(a + "A_log"))
            out.f32(f"{L}.dt_bias", g(a + "dt_bias"))
            out.f32(f"{L}.gnorm", g(a + "norm.weight"))
            out.q8(f"{L}.o", g(a + "out_proj.weight"))
        else:
            a = p + "self_attn."
            out.q8(f"{L}.q", g(a + "q_proj.weight"))
            out.q8(f"{L}.k", g(a + "k_proj.weight"))
            out.q8(f"{L}.v", g(a + "v_proj.weight"))
            out.q8(f"{L}.o", g(a + "o_proj.weight"))
            out.f32(f"{L}.qn", g(a + "q_norm.weight").astype(mx.float32) + 1.0)
            out.f32(f"{L}.kn", g(a + "k_norm.weight").astype(mx.float32) + 1.0)
        m = p + "mlp."
        out.bf16(f"{L}.router", g(m + "gate.weight"))
        out.q8(f"{L}.sg", g(m + "shared_expert.gate_proj.weight"))
        out.q8(f"{L}.su", g(m + "shared_expert.up_proj.weight"))
        out.q8(f"{L}.sd", g(m + "shared_expert.down_proj.weight"))
        out.bf16(f"{L}.shgate", g(m + "shared_expert_gate.weight"))
        mx.clear_cache()
    out.f32("norm", g("norm.weight").astype(mx.float32) + 1.0)
    out.q8("head", W.pop("lm_head.weight"))
    out.close()
    print(f"wrote {out.path}: {len(out.tab)} tensors, {os.path.getsize(out.path) / 2**30:.3f} GiB")


def router(model_dir, npz, bits=8, split=False):
    """bits 4: every router matrix 4-bit (Stage 3c); split: "pre" written as "pre_e" (embedding columns) and "pre_p"
    (previous-token columns), so the engine can keep the embedding part's output per token id. The split is at a multiple
    of 64, so the 8-bit codes equal those of the whole matrix."""
    blob = np.load(npz)
    lag = int(blob["__lag__"])
    feats = str(blob["__feats__"]) if "__feats__" in blob.files else "full"
    assert feats == "full", "only the 'full' feature set is wired into the engine"
    NL = 1 + max(int(k.split(".")[0]) for k in blob.files if not k.startswith("__"))
    path = npz[: -len(".npz")].replace("_fp16", "") + ("_b4" if bits == 4 else "") + ("_split" if split else "") + ".dmr"
    out = Writer(path)
    qq = out.q4 if bits == 4 else out.q8
    H = blob["0.emb_norm.weight"].shape[0]
    hid = blob["0.mlp.0.weight"].shape[0] if "0.mlp.0.weight" in blob.files else 0  # 0: a linear router (lin_k*)
    out.cfg += [("lag", lag), ("feats", feats), ("layers", NL), ("hidden", H), ("hidden_mlp", hid)]
    if bits == 4:
        out.cfg.append(("bits", 4))
    if split:
        out.cfg.append(("split_pre", 1))
    # z = lin(x) + mlp.2(gelu(mlp.0(x))), x = [emb_norm(emb), prev, lag]. lin and mlp.0 share x, so their rows are stacked
    # into one matrix, and split by columns: "pre" (the emb and prev columns, known at the token start, computed for every
    # layer in one pass) and "lagm" (the lag columns, computed at layer L - lag). A split at a multiple of 64 leaves every
    # quantisation group as it was, so the codes equal those of the whole matrix.
    widths = []
    for L in range(NL):
        lw = blob[f"{L}.lin.weight"]
        m0 = blob[f"{L}.mlp.0.weight"] if hid else np.zeros((0, lw.shape[1]), lw.dtype)
        widths.append(lw.shape[1])
        assert lw.shape[1] == m0.shape[1] and lw.shape[1] in (2 * H, 3 * H)
        stack = np.concatenate([lw, m0], axis=0).astype(np.float32)
        out.f32(f"{L}.emb_norm", mx.array(blob[f"{L}.emb_norm.weight"].astype(np.float32)))
        if split:
            qq(f"{L}.pre_e", mx.array(stack[:, :H]))
            qq(f"{L}.pre_p", mx.array(stack[:, H : 2 * H]))
        else:
            qq(f"{L}.pre", mx.array(stack[:, : 2 * H]))
        if lw.shape[1] == 3 * H:
            qq(f"{L}.lagm", mx.array(stack[:, 2 * H :]))
        out.f32(f"{L}.b_pre", mx.array(np.concatenate([blob[f"{L}.lin.bias"]] + ([blob[f"{L}.mlp.0.bias"]] if hid else [])).astype(np.float32)))
        if hid:
            qq(f"{L}.m2", mx.array(blob[f"{L}.mlp.2.weight"].astype(np.float32)))
            out.f32(f"{L}.m2_b", mx.array(blob[f"{L}.mlp.2.bias"].astype(np.float32)))
        mx.clear_cache()
    out.cfg.append(("widths", ",".join(map(str, widths))))
    out.close()
    print(f"wrote {path}: lag {lag}, {NL} layers, widths {sorted(set(widths))}, {os.path.getsize(path) / 2**20:.0f} MiB")


if __name__ == "__main__":
    if sys.argv[1] == "resident":
        resident(sys.argv[2])
    elif sys.argv[1] == "router":
        router(sys.argv[2], sys.argv[3], bits=4 if "--bits4" in sys.argv else 8, split="--split" in sys.argv)
    else:
        raise SystemExit(__doc__)
