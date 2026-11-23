"""Gate check for the Qwen3.6-35B-A3B int4 store (run by convert.py once layer 0 is written; exit 0 only if all pass).

  python check.py --out <OUT> --ref <STATE_DIR>/_ref [--expert 7] [--layer 0]

Reads expert E of store/layer_00.bin, unpacks the int4 codes with its own bit arithmetic (little-endian uint32, element i
at bits 4i), dequantises w = code * scale + bias per group of 64 along the input dim, and compares with the bf16
originals saved during conversion (gate = rows 0:I of experts.gate_up_proj[E], up = rows I:2I, down = experts.down_proj[E]
[H, I]; layout from transformers Qwen3_5MoeExperts.forward): max abs error and relative RMS error. It also re-quantises
the originals independently (numpy, not the converter's torch code) and requires bit-identical codes, scales and biases,
so the error numbers measure 4-bit group-64 rounding and nothing else. As a layout control it reports the relative RMS
the same record would have against the wrong matrix (gate vs up swapped, the neighbouring expert): those must be ~100%+.
Record padding must be zero, and the first and last records of the layer must be populated.
"""
import argparse, json, os, sys

import numpy as np  # no torch: the originals are read from the safetensors bytes directly

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--expert", type=int, default=7)
ap.add_argument("--layer", type=int, default=0)
a = ap.parse_args()
meta = json.load(open(os.path.join(a.out, "store", "meta.json")))
G, REC, I, H, NE = meta["group"], meta["record_bytes"], meta["intermediate"], meta["hidden"], meta["experts"]
assert meta["dtype"] == "int4" and meta["record_order"] == ["gate_proj", "up_proj", "down_proj"], meta
ok = True


def fail(msg):
    global ok
    ok = False
    print("FAIL", msg)


def bf16_to_f32(path, name):
    """A bf16 tensor from a safetensors file as float32, by bit arithmetic (bf16 = the top 16 bits of an fp32)."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
        info = hdr[name]
        assert info["dtype"] == "BF16", info
        s, e = info["data_offsets"]
        fh.seek(8 + n + s)
        raw = np.frombuffer(fh.read(e - s), dtype="<u2")
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])


def relrms(w, w_hat):
    err = (w_hat - w).astype(np.float64)
    return float(np.sqrt((err ** 2).sum() / (w.astype(np.float64) ** 2).sum()))


def decode(rec, kind):
    r, c = meta["shapes"][kind]
    words = np.frombuffer(rec, dtype="<u4", count=r * c // 8, offset=meta["offsets"][kind]).reshape(r, c // 8)
    codes = ((words[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(r, c)
    sc = np.frombuffer(rec, dtype="<f2", count=r * c // G, offset=meta["scale_offsets"][kind]).reshape(r, c // G)
    bi = np.frombuffer(rec, dtype="<f2", count=r * c // G, offset=meta["bias_offsets"][kind]).reshape(r, c // G)
    w_hat = codes.astype(np.float32) * np.repeat(sc.astype(np.float32), G, axis=1) + np.repeat(bi.astype(np.float32), G, axis=1)
    return codes, sc, bi, w_hat


lay = os.path.join(a.out, "store", f"layer_{a.layer:02d}.bin")
assert os.path.getsize(lay) == NE * REC, (lay, os.path.getsize(lay), NE * REC)
with open(lay, "rb") as fh:
    fh.seek(a.expert * REC)
    rec = fh.read(REC)
    first, last = (fh.seek(0), fh.read(REC))[1], (fh.seek((NE - 1) * REC), fh.read(REC))[1]
    nb = (a.expert + 1) % NE
    neigh = (fh.seek(nb * REC), fh.read(REC))[1]
gu = bf16_to_f32(os.path.join(a.ref, f"l{a.layer}_e{a.expert}_gate_up_proj.safetensors"), "gate_up_proj")
dn = bf16_to_f32(os.path.join(a.ref, f"l{a.layer}_e{a.expert}_down_proj.safetensors"), "down_proj")
assert gu.shape == (2 * I, H) and dn.shape == (H, I), (gu.shape, dn.shape)
orig = {"gate_proj": gu[:I], "up_proj": gu[I:], "down_proj": dn}
print(f"== expert record: layer {a.layer} expert {a.expert}, record {REC} B, offsets {meta['offsets']}, "
      f"scale_offsets {meta['scale_offsets']}, bias_offsets {meta['bias_offsets']}")
used = 0
for kind, (r, c) in meta["shapes"].items():
    w = orig[kind]
    assert w.shape == (r, c), (kind, w.shape, (r, c))
    codes, sc, bi, w_hat = decode(rec, kind)
    used = max(used, meta["bias_offsets"][kind] + r * c // G * 2)
    err = w_hat - w
    rel = relrms(w, w_hat)
    print(f"  {kind:10s} shape {[r, c]}  max|w| {np.abs(w).max():.4g}  max abs err {np.abs(err).max():.4g}  "
          f"rel RMS err {rel * 100:.3f}%  mean abs err {np.abs(err).mean():.3g}")
    # independent re-quantisation (numpy): min/max per group of 64, fp16 scale=(max-min)/15 and bias=min, round half-even
    wg = w.reshape(r, c // G, G)
    lo, hi = wg.min(-1), wg.max(-1)
    s16, b16 = ((hi - lo) / np.float32(15.0)).astype(np.float16), lo.astype(np.float16)
    s, b = s16.astype(np.float32), b16.astype(np.float32)
    q = np.clip(np.round((wg - b[..., None]) / np.where(s > 0, s, 1)[..., None]), 0, 15).reshape(r, c)
    n_code = int((q != codes).sum())
    n_s, n_b = int((s16.view(np.uint16) != sc.view(np.uint16)).sum()), int((b16.view(np.uint16) != bi.view(np.uint16)).sum())
    print(f"             vs independent re-quantisation: code mismatches {n_code}/{codes.size}, scale mismatches {n_s}, bias mismatches {n_b}")
    bound = np.repeat(s, G, axis=1) * 0.5 + np.abs(np.repeat(b, G, axis=1)) * 2 ** -10 + 1e-7
    over = int((np.abs(err) > bound + np.repeat(s, G, axis=1) * 0.02).sum())
    print(f"             elements beyond half a quantisation step: {over}")
    # layout controls: the same decoded matrix against the wrong original
    wrong = {"gate_proj": orig["up_proj"], "up_proj": orig["gate_proj"]}.get(kind)
    ctl = f"vs the {'up' if kind == 'gate_proj' else 'gate'} half {relrms(wrong, w_hat) * 100:.1f}%, " if wrong is not None else ""
    nb_hat = decode(neigh, kind)[3]
    print(f"             controls (must be large): {ctl}record of expert {nb} vs this expert's original {relrms(w, nb_hat) * 100:.1f}%")
    if n_code or n_s or n_b:
        fail(f"{kind}: record differs from the independent quantisation")
    if over:
        fail(f"{kind}: {over} elements beyond half a quantisation step")
    if rel > 0.15:
        fail(f"{kind}: rel RMS {rel:.3f} too large for 4-bit group-64")
    if (wrong is not None and relrms(wrong, w_hat) < 0.5) or relrms(w, nb_hat) < 0.5:
        fail(f"{kind}: a wrong matrix matches the record as well as the right one")
if any(rec[used:]):
    fail("record padding is not zero")
print(f"  record bytes used {used}, padding {REC - used} B zero: {not any(rec[used:])}; "
      f"record 0 populated: {any(first)}; record {NE - 1} populated: {any(last)}")
if not any(first) or not any(last):
    fail("first/last record empty")
if used != meta.get("record_bytes_used", used):
    fail(f"meta record_bytes_used {meta.get('record_bytes_used')} != {used}")
print("CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
