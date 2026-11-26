"""Cross-token routers (the router trainer's "xtok" variants, exported as fp16 npz) -> the engine's router file (DMENGRES1),
kind ink_xtok, read by Xtok::load in src/core/address.cpp (dmeng --xtok FILE).

  python python/derive/inkling2_loop_xtok.py RUN VARIANT --dir MODEL_DIR/routers [--targets 0-5] [--bits 8|4]
  -> MODEL_DIR/routers/RUN/ink_router_VARIANT_t<t0>-<t1>_b<bits>.dmr from ink_router_VARIANT_fp16.npz there
  python python/derive/inkling2_loop_xtok.py --from MODEL_DIR/routers/xtok_s47_h512.safetensors --out FILE [--targets 0-5]
  -> the same file from the weight repo's router container (matrices f16, vectors f32; run, variant and source name
     from its metadata), byte-identical to the one derived from the npz

Router for target a on x = [h_t@S, h_t@a] (S = the source application, both MoE-block inputs of token t) names token t+1's
released routing at a: z = lin(x) + mlp.2(gelu(mlp.0(x))). Stored per kept target a (--targets, every a < S):
a.pre [NE + hid, 2H] = rows [lin; mlp.0];  a.b_pre f32 [NE + hid] = [lin.bias; mlp.0.bias];  a.m2 [NE, hid] = mlp.2.weight;
a.m2_b f32 [NE].  Matrices as NAME.q (8-bit codes) or NAME.q4 (4-bit, 2 per byte, low nibble = even element), NAME.s /
NAME.b fp16 scales / biases: mx.quantize of the fp16 weights, group 64 (as python/derive/inkling2_loop_routers.py).
Header keys: kind=ink_xtok, source_app, targets (comma list), apps, hidden, experts, hidden_mlp, bits.
Needs mlx and numpy.
"""
import argparse, glob, json, os, sys

import mlx.core as mx
import numpy as np

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



class SafeTensorsSource:
    """The weight repo's router container (routers/xtok_s<S>_h<hidden>.safetensors) read like the npz: z[name] gives a
    tensor, z["__key__"] a value of the npz's own metadata (kept as JSON in source_meta)."""

    DT = {"F16": np.float16, "F32": np.float32}

    def __init__(self, path):
        import struct
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.h = json.loads(fh.read(n))
        self.path, self.base = path, 8 + n
        self.meta = self.h.pop("__metadata__", {})
        assert self.meta.get("format") == "moved-store-router" and self.meta.get("form") == "xtok", path
        self.src_meta = json.loads(self.meta["source_meta"])

    def __getitem__(self, k):
        if k.startswith("__"):
            return np.array(self.src_meta[k])
        e = self.h[k]
        a_, b_ = e["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.base + a_)
            return np.frombuffer(fh.read(b_ - a_), dtype=self.DT[e["dtype"]]).reshape(e["shape"])


ap = argparse.ArgumentParser()
ap.add_argument("run", nargs="?")
ap.add_argument("variant", nargs="?")
ap.add_argument("--targets", default="0-5")
ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
ap.add_argument("--dir", help="the routers directory holding RUN/ink_router_VARIANT_fp16.npz")
ap.add_argument("--from", dest="src", help="a router container (.safetensors) instead of RUN VARIANT --dir")
ap.add_argument("--out", help="with --from: the engine file to write")
a = ap.parse_args()
t0, t1 = map(int, a.targets.split("-"))
if a.src:
    assert a.out, "--from needs --out"
    z = SafeTensorsSource(a.src)
    a.run, f = z.meta["run"], z.meta["source"]
    out_path = a.out
else:
    assert a.run and a.variant and a.dir, "RUN VARIANT --dir DIR, or --from FILE --out FILE"
    f = os.path.join(a.dir, a.run, f"ink_router_{a.variant}_fp16.npz")
    z = np.load(f)
    out_path = f.replace("_fp16.npz", f"_t{t0}-{t1}_b{a.bits}.dmr")
assert int(z["__lag__"]) == -1 and str(z["__feats__"]) == "xtok", (z["__lag__"], z["__feats__"])
src, hid = int(z["__source_app__"]), int(z["__hidden__"])
tg = z["__targets__"].tolist()
NE = z["0.lin.bias"].shape[0]
H = z["0.lin.weight"].shape[1] // 2
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


kept = []
for i, t in enumerate(tg):
    if not (t0 <= t <= t1):
        continue
    p = f"{i}."
    lw, l0 = z[p + "lin.weight"].astype(np.float32), z[p + "mlp.0.weight"].astype(np.float32)
    assert lw.shape[1] == 2 * H
    q(f"{t}.pre", np.concatenate([lw, l0], 0))
    W.add(f"{t}.b_pre", np.concatenate([z[p + "lin.bias"], z[p + "mlp.0.bias"]]).astype(np.float32), "f32")
    q(f"{t}.m2", z[p + "mlp.2.weight"].astype(np.float32))
    W.add(f"{t}.m2_b", z[p + "mlp.2.bias"].astype(np.float32), "f32")
    kept.append(t)
    mx.clear_cache()
W.cfg += [("kind", "ink_xtok"), ("run", a.run), ("variant", str(z["__variant__"])), ("source_app", src), ("targets", ",".join(map(str, kept))),
          ("apps", int(z["__n_apps__"])), ("hidden", H), ("experts", NE), ("hidden_mlp", hid), ("bits", a.bits), ("source", os.path.basename(f))]
W.close()
print(json.dumps({"out": out_path, "bytes": os.path.getsize(out_path), "source_app": src, "targets": kept, "bits": a.bits, "quant_max_rel_err": round(stats["maxrel"], 5)}))
