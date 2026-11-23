"""Trained Inkling early routers (jobs/ink_router.py "full" variants, fetched by jobs/ink_xfer.sh as fp16 npz)
-> the engine's router file (DMENGRES1, like q36's .dmr): one per lag, 8-bit (default) or 4-bit affine group 64.

  python python/derive/inkling2_loop_routers.py RUN --dir MODEL_DIR/routers [--bits 8|4]
  -> local/inkling/model/routers/<run>/ink_router_full_k<k>_h<H>_b<bits>.dmr for every ink_router_full_*_fp16.npz there
     (and ink_router_lin_k<k>_b<bits>.dmr for every ink_router_lin_*_fp16.npz: the linear path only, hidden_mlp 0, no m2)

Router a (one per application, 48) on x = [rmsnorm_w(emb), h_prev@a, h@(a-k)] (emb = the model's embed_tokens output, which in
transformers PR 51050 is embed_norm(embedding); h = MoE-block inputs; a < k: no third block): z = lin(x) + mlp.2(gelu(mlp.0(x))).
Stored per router a:  a.emb_norm f32 [H];  a.pre [NE + hid, 2H] = rows [lin; mlp.0], columns of the first two blocks;
a.lagm [NE + hid, H] = their columns of the third block (a >= k);  a.b_pre f32 [NE + hid] = [lin.bias; mlp.0.bias];
a.m2 [NE, hid] = mlp.2.weight;  a.m2_b f32 [NE].  Matrices as NAME.q (8-bit codes) or NAME.q4 (4-bit, 2 per byte, low nibble
= even element), NAME.s / NAME.b fp16 scales / biases: mx.quantize of the fp16 weights.
"""
import argparse, glob, json, os, sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = 16384


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



ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
ap.add_argument("--dir", required=True)
a = ap.parse_args()
D = os.path.join(a.dir, a.run)
files = sorted(glob.glob(os.path.join(D, "ink_router_full_*_fp16.npz")) + glob.glob(os.path.join(D, "ink_router_lin_*_fp16.npz")))
assert files, f"no ink_router_full_*_fp16.npz or ink_router_lin_*_fp16.npz in {D}"
for f in files:
    z = np.load(f)
    lag, hid, na = int(z["__lag__"]), int(z["__hidden__"]), int(z["__n_apps__"])
    assert str(z["__feats__"]) == "full" and not bool(z["__shared__"]) and not bool(z["__pp__"])
    rid = z["__router_of_app__"].tolist()
    assert rid == list(range(na)), "per-application routers expected"
    H = z["0.emb_norm.weight"].shape[0]
    NE = z["0.lin.bias"].shape[0]
    out_path = f.replace("_fp16.npz", f"_b{a.bits}.dmr")
    W = Writer(out_path)
    stats = {"maxrel": 0.0}

    def q(name, w):
        w = mx.array(w.astype(np.float16))
        wq, s, b = mx.quantize(w, group_size=64, bits=a.bits)
        deq = mx.dequantize(wq, s, b, group_size=64, bits=a.bits)
        err = float(mx.abs(deq.astype(mx.float32) - w.astype(mx.float32)).max().item() / max(1e-30, mx.abs(w).max().item()))
        stats["maxrel"] = max(stats["maxrel"], err)
        R, C = w.shape
        codes = np.array(wq).view(np.uint8).reshape(R, C * a.bits // 8)
        W.add(name + (".q" if a.bits == 8 else ".q4"), codes, "u8")
        W.add(name + ".s", np.array(s).view(np.uint16), "f16")
        W.add(name + ".b", np.array(b).view(np.uint16), "f16")

    for r in range(na):
        p = f"{r}."
        lw = z[p + "lin.weight"].astype(np.float32)
        l0 = z[p + "mlp.0.weight"].astype(np.float32) if hid > 0 else np.zeros((0, lw.shape[1]), np.float32)  # linear: no MLP rows
        n_in = lw.shape[1]
        assert n_in in (2 * H, 3 * H) and (n_in == 3 * H) == (r >= lag), (r, n_in)
        stacked = np.concatenate([lw, l0], 0)  # [NE + hid, n_in]
        W.add(p + "emb_norm", z[p + "emb_norm.weight"].astype(np.float32), "f32")
        q(p + "pre", stacked[:, : 2 * H])
        if n_in == 3 * H:
            q(p + "lagm", stacked[:, 2 * H :])
        W.add(p + "b_pre", np.concatenate([z[p + "lin.bias"]] + ([z[p + "mlp.0.bias"]] if hid > 0 else [])).astype(np.float32), "f32")
        if hid > 0:  # a linear router (lin_k<k>, hidden 0) has no m2: the engine reads it only when hidden_mlp > 0
            q(p + "m2", z[p + "mlp.2.weight"].astype(np.float32))
            W.add(p + "m2_b", z[p + "mlp.2.bias"].astype(np.float32), "f32")
        mx.clear_cache()
    W.cfg += [("kind", "ink_router"), ("run", a.run), ("variant", str(z["__variant__"])), ("lag", lag), ("apps", na), ("hidden", H),
              ("experts", NE), ("hidden_mlp", hid), ("bits", a.bits), ("emb", "embed_norm_output"), ("rms_eps", "1.1920929e-07"),
              ("source", os.path.basename(f))]
    W.close()
    print(json.dumps({"out": out_path, "bytes": os.path.getsize(out_path), "lag": lag, "bits": a.bits, "quant_max_rel_err": round(stats["maxrel"], 5)}))
