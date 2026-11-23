"""Gate for the Inkling 2 Mini store: records are the released bytes, and they dequantise to what the PR computes.

  .venv/bin/python check.py [--release release] [--model model] [--layers 0,6,19,20,23] [--experts 0,37,127] [--sha]

For each chosen (layer, expert), nvfp4 and mxfp8 layers both:
  1. bytes: every field of the store record equals the release's bytes for that expert (gate_up / down codes and scales;
     the nvfp4 trailer equals the release's amax and outer_scale_denominator); the padding is zero.
  2. values: the record dequantised by this file's own arithmetic (store.dequant_np: E2M1 / E4M3 / UE8M0 tables in numpy)
     equals, value for value (float32, bit-exact), the same expert taken STRAIGHT FROM THE RELEASE through the PR's code
     path: mlx_lm.models.inkling2.checkpoint.sanitize (the PR's global scale and mode selection) then
       (a) mx.dequantize(mode, group, bits, global_scale), and
       (b) mx.gather_qmm with x = identity (float32): the exact matmul kernel the model runs, returning W^T.
  3. a decode-shaped product (one random float32 row through gather_qmm, the M = 1 qmv kernel the runner will use)
     against float64 numpy on the dequantised weights: relative error reported, must be < 1e-5 (accumulation order only).
--sha: also re-hash every store file against model/sha256.txt.
Memory: a few experts at a time (at most ~100 MB). Exit 1 on any failure.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import store as S  # noqa: E402
from mlx_lm.layers.moe import SwitchLinear  # noqa: E402
from mlx_lm.models.inkling2 import checkpoint as pr_checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--release", default=os.path.join(HERE, "release"))
ap.add_argument("--model", default=os.path.join(HERE, "model"))
ap.add_argument("--layers", default=None, help="default: first nvfp4, first looped, last nvfp4, first and last mxfp8")
ap.add_argument("--experts", default=None, help="default: 0, a middle one, the last")
ap.add_argument("--sha", action="store_true")
a = ap.parse_args()

meta = json.load(open(os.path.join(a.model, "store", "meta.json")))
NL, NE, H, I = meta["layers"], meta["experts"], meta["hidden"], meta["intermediate"]
REL = os.path.join(a.release, "model-00001-of-00001.safetensors")
hdr, _, DATA = S.read_header(REL)
kinds = meta["layer_format"]
if a.layers:
    layers = [int(x) for x in a.layers.split(",")]
else:
    nv = [L for L in range(NL) if kinds[L] == "nvfp4"]
    mx8 = [L for L in range(NL) if kinds[L] == "mxfp8"]
    loop = meta.get("loop_config") or {}
    layers = sorted({nv[0], nv[-1], mx8[0], mx8[-1]} | ({loop["core_start"]} if loop else set()))
experts = [int(x) for x in a.experts.split(",")] if a.experts else sorted({0, (NE * 37) // 128, NE - 1})
fails = []


def fail(msg):
    fails.append(msg)
    print("FAIL", msg, flush=True)


# ------------------------------------------------------------------------------------------------ 0. sizes, sha
for L in range(NL):
    p = os.path.join(a.model, meta["layer_file"].format(L=L))
    if os.path.getsize(p) != NE * meta["record_bytes"][L]:
        fail(f"layer {L}: file size {os.path.getsize(p)} != {NE} x {meta['record_bytes'][L]}")
if a.sha:
    want = dict(reversed(line.split()) for line in open(os.path.join(a.model, "sha256.txt")))
    for L in range(NL):
        rel = meta["layer_file"].format(L=L)
        h = hashlib.sha256()
        with open(os.path.join(a.model, rel), "rb") as fh:
            S.nocache(fh.fileno())
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != want.get(rel):
            fail(f"{rel}: sha256 mismatch")
    print(f"sha256 of {NL} store files checked", flush=True)

src = os.open(REL, os.O_RDONLY)


def release_slice(name, e, nb):
    return os.pread(src, nb, DATA + hdr[name]["data_offsets"][0] + e * nb)


class FakeModel:
    """Just enough of the PR's Model for checkpoint.sanitize: args.vision, named_modules, update_modules."""

    def __init__(self, L, n):
        self.args = SimpleNamespace(vision=None)
        p = f"model.language_model.layers.{L}.mlp.experts."
        self.mods = [(p + "gate_up_proj", SwitchLinear(H, 2 * I, n, bias=False)), (p + "down_proj", SwitchLinear(I, H, n, bias=False))]
        self.quantized = {}

    def named_modules(self):
        return self.mods

    def update_modules(self, tree):
        self.quantized.update(dict(tree_flatten(tree, is_leaf=lambda x: isinstance(x, nn.Module))))


def pr_path(L, ex):
    """The experts `ex` of layer L straight from the release through the PR's sanitize: sanitized dict + modules."""
    names = S.release_names(L)
    fmt = meta["formats"][kinds[L]]
    w = {}
    for f in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales"):
        fl = fmt["fields"][f]
        arr = np.stack([np.frombuffer(release_slice(names[f], e, fl["bytes"]), dtype=np.uint8).reshape(fl["shape"]) for e in ex])
        w[names[f]] = mx.array(arr)  # mx.load reads F8_E4M3 and U8 both as uint8, as here
    if fmt["global_scale"]:
        for f in ("gate_up.amax", "down.amax"):
            w[names[f]] = mx.array(np.array([np.frombuffer(release_slice(names[f], e, 4), dtype="<f4")[0] for e in ex], dtype=np.float32))
        for f in ("gate_up.denominator", "down.denominator"):
            w[names[f]] = mx.array(np.frombuffer(os.pread(src, 4, DATA + hdr[names[f]]["data_offsets"][0]), dtype="<f4").copy())
    fake = FakeModel(L, len(ex))
    san = pr_checkpoint.sanitize(fake, w)
    return san, fake.quantized


fd_cache = {}
rng = np.random.default_rng(0)
summary = []
for L in layers:
    kind, fmt = kinds[L], meta["formats"][kinds[L]]
    fd = fd_cache.setdefault(L, os.open(os.path.join(a.model, meta["layer_file"].format(L=L)), os.O_RDONLY))
    names = S.release_names(L)
    san, mods = pr_path(L, experts)
    p = f"model.language_model.layers.{L}.mlp.experts."
    for j, e in enumerate(experts):
        rec = S.read_record(fd, meta, L, e)
        v = S.record_views(rec, fmt)
        # 1. bytes
        for f in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales"):
            fl = fmt["fields"][f]
            if v[f].tobytes() != release_slice(names[f], e, fl["bytes"]):
                fail(f"L{L} e{e} {f}: record bytes differ from the release")
        if fmt["global_scale"]:
            want = [np.frombuffer(release_slice(names["gate_up.amax"], e, 4), "<f4")[0], np.frombuffer(release_slice(names["down.amax"], e, 4), "<f4")[0],
                    np.frombuffer(os.pread(src, 4, DATA + hdr[names["gate_up.denominator"]]["data_offsets"][0]), "<f4")[0],
                    np.frombuffer(os.pread(src, 4, DATA + hdr[names["down.denominator"]]["data_offsets"][0]), "<f4")[0]]
            if v["trailer"].tobytes() != np.array(want, dtype="<f4").tobytes():
                fail(f"L{L} e{e}: trailer differs from the release amax/denominator")
        used = fmt["record_bytes_used"]
        if any(np.frombuffer(rec, np.uint8)[used:]):
            fail(f"L{L} e{e}: padding not zero")
        # 2. values
        for m, name in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            mod = mods[p + name]
            gs = None
            if fmt["global_scale"]:
                t = v["trailer"]
                gs_mine = S.global_scale_f32(t[0 if m == "gate_up" else 1], t[2 if m == "gate_up" else 3])
                gs_pr = np.array(san[p + name + ".global_scale"])[j]
                if gs_mine.tobytes() != np.float32(gs_pr).tobytes():
                    fail(f"L{L} e{e} {m}: global scale {gs_mine} != PR {gs_pr}")
                gs = gs_mine
            mine = S.dequant_np(v[f"{m}.weight"], v[f"{m}.scales"], kind, gs)  # [out, in] float32
            W, Sc = san[p + name + ".weight"], san[p + name + ".scales"]
            if (mod.mode, mod.group_size, mod.bits) != (kind, fmt["group"], fmt["bits"]):
                fail(f"L{L} {m}: PR chose {(mod.mode, mod.group_size, mod.bits)}")
            GS = san.get(p + name + ".global_scale")
            deq = mx.dequantize(W[j], Sc[j], group_size=mod.group_size, bits=mod.bits, mode=mod.mode,
                                global_scale=None if GS is None else GS[j:j + 1], dtype=mx.float32)
            K = W.shape[-1] * 32 // mod.bits
            ident = mx.eye(K, dtype=mx.float32)[None]
            qmm = mx.gather_qmm(ident, W, Sc, None, rhs_indices=mx.array([j], dtype=mx.uint32), transpose=True,
                                group_size=mod.group_size, bits=mod.bits, mode=mod.mode, global_scale=GS)
            x = rng.standard_normal(K).astype(np.float32)
            qmv = mx.gather_qmm(mx.array(x)[None, None], W, Sc, None, rhs_indices=mx.array([j], dtype=mx.uint32), transpose=True,
                                group_size=mod.group_size, bits=mod.bits, mode=mod.mode, global_scale=GS)
            mx.eval(deq, qmm, qmv)
            deq, qmm, qmv = np.array(deq), np.array(qmm)[0].T, np.array(qmv).reshape(-1)
            ref = mine.astype(np.float64) @ x.astype(np.float64)
            rel_err = float(np.abs(qmv - ref).max() / (np.abs(ref).max() + 1e-30))
            n_deq = int((deq.view(np.uint32) != mine.view(np.uint32)).sum())
            n_qmm = int((qmm != mine).sum())  # values: a matmul turns a -0.0 weight into +0.0 (sum of +0 products)
            nan = int(np.isnan(mine).sum())
            summary.append((L, kind, e, m, mine.shape, n_deq, n_qmm, rel_err, float(np.abs(mine).max()), nan))
            if n_deq or n_qmm:
                fail(f"L{L} e{e} {m}: {n_deq} values differ from mx.dequantize, {n_qmm} from gather_qmm(identity)")
            if rel_err > 1e-5 or nan:
                fail(f"L{L} e{e} {m}: qmv rel err {rel_err:.2e}, NaN {nan}")
        mx.clear_cache()

print(f"{'layer':>5} {'kind':>6} {'expert':>6} {'matrix':>8} {'shape':>12} {'!=deq':>6} {'!=qmm':>6} {'qmv rel err':>12} {'max |w|':>9}")
for L, kind, e, m, shp, nd, nq, re_, mxw, nan in summary:
    print(f"{L:5d} {kind:>6} {e:6d} {m:>8} {str(shp):>12} {nd:6d} {nq:6d} {re_:12.2e} {mxw:9.4f}")
print(f"checked {len(summary)} matrices: layers {layers}, experts {experts}")
if fails:
    print(f"CHECK FAILED: {len(fails)} failures")
    sys.exit(1)
print("CHECK PASSED: records are the released bytes; own dequantisation == PR dequantize (bit for bit) == PR gather_qmm (value for value)")
