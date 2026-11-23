"""Gate G1 check for the Qwen3.8-Flash-Next store (run by convert.py once layer 0 and n-gram part 0 are written).

  python check.py --out <OUT> --ref <STATE_DIR>/_ref [--expert 7]

1. Expert record: reads expert E of store/layer_00.bin, unpacks the int4 codes with its own bit arithmetic (uint32, element
   i at bits 4i), dequantises w = code * scale + bias per group of 64, and compares with the bf16 originals (gate = rows
   0:I of experts.gate_up_proj[E], up = rows I:2I, down = experts.down_proj[E]) saved during conversion: max abs error and
   relative RMS error. It also re-quantises the originals independently (numpy) and requires bit-identical codes, scales
   and biases, so the numbers measure 4-bit group-64 rounding and nothing else. Record padding must be zero, and the
   first and last records of the layer must be populated.
2. N-gram part 0: 1,000 random rows (fixed seed in convert.py) read back from ngram/part_00.bin (fp16 scale | int8 codes),
   dequantised and compared with the bf16 rows; independent re-quantisation must match exactly.
Exit 0 only if every check passes.
"""
import argparse, glob, json, os, sys

import numpy as np
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--expert", type=int, default=7)
ap.add_argument("--layer", type=int, default=0)
a = ap.parse_args()
meta = json.load(open(os.path.join(a.out, "store", "meta.json")))
ngm = json.load(open(os.path.join(a.out, "ngram", "meta.json")))
G, REC, I, H, NE = meta["group"], meta["record_bytes"], meta["intermediate"], meta["hidden"], meta["experts"]
ok = True


def fail(msg):
    global ok
    ok = False
    print("FAIL", msg)


def stats(name, w, w_hat):
    err = w_hat - w
    rel = float(np.sqrt((err.astype(np.float64) ** 2).sum() / (w.astype(np.float64) ** 2).sum()))
    print(f"  {name:10s} shape {list(w.shape)}  max|w| {np.abs(w).max():.4g}  max abs err {np.abs(err).max():.4g}  "
          f"rel RMS err {rel * 100:.3f}%  mean abs err {np.abs(err).mean():.3g}")
    return rel


# ---------- 1. expert record ----------
lay = os.path.join(a.out, "store", f"layer_{a.layer:02d}.bin")
assert os.path.getsize(lay) == NE * REC, (lay, os.path.getsize(lay), NE * REC)
with open(lay, "rb") as fh:
    fh.seek(a.expert * REC)
    rec = fh.read(REC)
    first, last = (fh.seek(0), fh.read(REC))[1], (fh.seek((NE - 1) * REC), fh.read(REC))[1]
gu = load_file(os.path.join(a.ref, f"l{a.layer}_e{a.expert}_gate_up_proj.safetensors"))["gate_up_proj"].float().numpy()
dn = load_file(os.path.join(a.ref, f"l{a.layer}_e{a.expert}_down_proj.safetensors"))["down_proj"].float().numpy()
orig = {"gate_proj": gu[:I], "up_proj": gu[I:], "down_proj": dn}
print(f"== expert record: layer {a.layer} expert {a.expert}, record {REC} B, offsets {meta['offsets']}, "
      f"scale_offsets {meta['scale_offsets']}, bias_offsets {meta['bias_offsets']}")
used = 0
for kind, (r, c) in meta["shapes"].items():
    w = orig[kind]
    assert w.shape == (r, c), (kind, w.shape, (r, c))
    words = np.frombuffer(rec, dtype="<u4", count=r * c // 8, offset=meta["offsets"][kind]).reshape(r, c // 8)
    codes = ((words[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(r, c)
    sc = np.frombuffer(rec, dtype="<f2", count=r * c // G, offset=meta["scale_offsets"][kind]).reshape(r, c // G)
    bi = np.frombuffer(rec, dtype="<f2", count=r * c // G, offset=meta["bias_offsets"][kind]).reshape(r, c // G)
    used = max(used, meta["bias_offsets"][kind] + r * c // G * 2)
    w_hat = codes.astype(np.float32) * np.repeat(sc.astype(np.float32), G, axis=1) + np.repeat(bi.astype(np.float32), G, axis=1)
    rel = stats(kind, w, w_hat)
    # independent re-quantisation (numpy): min/max per group of 64, fp16 scale=(max-min)/15 and bias=min, round half-even
    wg = w.reshape(r, c // G, G)
    lo, hi = wg.min(-1), wg.max(-1)
    s16, b16 = ((hi - lo) / np.float32(15.0)).astype(np.float16), lo.astype(np.float16)
    s, b = s16.astype(np.float32), b16.astype(np.float32)
    q = np.clip(np.round((wg - b[..., None]) / np.where(s > 0, s, 1)[..., None]), 0, 15).reshape(r, c)
    n_code = int((q != codes).sum())
    n_s, n_b = int((s16.view(np.uint16) != sc.view(np.uint16)).sum()), int((b16.view(np.uint16) != bi.view(np.uint16)).sum())
    print(f"             vs independent re-quantisation: code mismatches {n_code}/{codes.size}, scale mismatches {n_s}, bias mismatches {n_b}")
    # the rounding bound: every element within half a step (+ fp16 slack) of the original
    bound = np.repeat(s, G, axis=1) * 0.5 + np.abs(np.repeat(b, G, axis=1)) * 2 ** -10 + 1e-7
    over = int((np.abs(w_hat - w) > bound + np.repeat(s, G, axis=1) * 0.02).sum())
    print(f"             elements beyond half a quantisation step: {over}")
    if n_code or n_s or n_b:
        fail(f"{kind}: record differs from the independent quantisation")
    if rel > 0.15:
        fail(f"{kind}: rel RMS {rel:.3f} too large for 4-bit group-64")
if any(rec[used:]):
    fail("record padding is not zero")
print(f"  record bytes used {used}, padding {REC - used} B zero: {not any(rec[used:])}; "
      f"record 0 populated: {any(first)}; record {NE - 1} populated: {any(last)}")
if not any(first) or not any(last):
    fail("first/last record empty")

# ---------- 2. n-gram part 0 ----------
refs = sorted(glob.glob(os.path.join(a.ref, "ng_*.safetensors")))
idx = np.concatenate([load_file(p)["idx"].numpy() for p in refs])
rows = np.concatenate([load_file(p)["rows"].float().numpy() for p in refs])
order = np.argsort(idx)
idx, rows = idx[order], rows[order]
BPR, D, NR0 = ngm["bytes_per_row"], ngm["dim"], ngm["part_rows"][0]
part = np.memmap(os.path.join(a.out, "ngram", "part_00.bin"), dtype=np.uint8, mode="r", shape=(NR0, BPR))
print(f"== n-gram part 0: {len(idx)} random rows of {NR0} (from {len(refs)} shard refs), {BPR} B/row")
recs = np.asarray(part[idx])
scale = recs[:, :2].copy().view("<f2")[:, 0]
codes = recs[:, 2:].copy().view(np.int8)
x_hat = codes.astype(np.float32) * scale.astype(np.float32)[:, None]
rel = stats("ngram", rows, x_hat)
per_row = np.sqrt(((x_hat - rows) ** 2).sum(1) / np.maximum((rows ** 2).sum(1), 1e-30))
print(f"             per-row rel RMS: median {np.median(per_row) * 100:.3f}%  p99 {np.percentile(per_row, 99) * 100:.3f}%  "
      f"max {per_row.max() * 100:.3f}%; scale range [{scale.min():.3g}, {scale.max():.3g}], zero rows {int((scale == 0).sum())}")
s16 = (np.abs(rows).max(1) / np.float32(127.0)).astype(np.float16)
s = s16.astype(np.float32)
q = np.clip(np.round(rows / np.where(s > 0, s, 1)[:, None]), -127, 127).astype(np.int8)
n_code, n_s = int((q != codes).sum()), int((s16.view(np.uint16) != scale.view(np.uint16)).sum())
print(f"             vs independent re-quantisation: code mismatches {n_code}/{codes.size}, scale mismatches {n_s}")
if n_code or n_s:
    fail("n-gram rows differ from the independent quantisation")
if rel > 0.01:
    fail(f"n-gram rel RMS {rel:.4f} >= 1%")
if len(idx) != min(1000, NR0):
    fail(f"expected {min(1000, NR0)} reference rows, found {len(idx)}")
print("CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
