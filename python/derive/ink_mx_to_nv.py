"""Inkling 2 Mini, profile phone-small only: expert layers 20-23 rewritten from the release's MXFP8 records
(8,224,768 B) into NVFP4 records (4,489,216 B, the format of layers 0-19), so one kernel serves all 24 layers and every
record is 4.49 MB. The originals are never touched; the new layer files go to OUT_DIR.

  python ink_mx_to_nv.py MODEL_DIR OUT_DIR [--layers 20,21,22,23] [--workers 2] [--check-only]

Per record and matrix (gate_up [2304, 2304] interleaved rows, down [2304, 1152]):
  1. decode the MXFP8 record to float32 weights (E4M3 codes x 2^(E8M0 - 127) per 32 consecutive elements of a row);
  2. global scale as the NVFP4 format in the manifest specifies: trailer [gate_up.amax, down.amax, 2688, 2688] with
     amax = max |w| of the matrix, so the engine's multiplier is g = amax * (2688 / 2688) / 2688 (store.cpp nv_gmul);
  3. per block of 16 consecutive elements of a row, the E4M3 block scale that MINIMISES the block's squared error:
     candidates are the 7 E4M3 codes around the max-based one (round(max|block| / (6 g))), each with every element
     rounded to the nearest E2M1 value (0, .5, 1, 1.5, 2, 3, 4, 6, signed); the lowest error wins (ties: the max-based);
  4. codes two per byte (low nibble = even element), scales row-major, trailer, zero padding to the record size.
--check-only: dequantise 8 records of each written layer and of the original, and report the
relative squared error sum((new - orig)^2) / sum(orig^2) per layer (experts 0, 16, ..., 112), with the max-based scale alone for comparison.
"""
import json, os, struct, sys
from multiprocessing import Pool

import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
MID = (E2M1[1:] + E2M1[:-1]) / 2  # rounding thresholds


def e4m3_table():
    v = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            x = np.nan
        elif e == 0:
            x = m / 8 * 2.0 ** -6
        else:
            x = (1 + m / 8) * 2.0 ** (e - 7)
        v[c] = -x if s else x
    return v


E4M3 = e4m3_table()
E4M3_POS = E4M3[:127]  # codes 0..126: 0 and the positive finite values, increasing


def mx_decode(rec, off_c, off_s, R, C):
    codes = np.frombuffer(rec, np.uint8, R * C, off_c).reshape(R, C)
    sc = np.frombuffer(rec, np.uint8, R * (C // 32), off_s).reshape(R, C // 32)
    w = E4M3[codes].reshape(R, C // 32, 32) * np.ldexp(np.float32(1), sc.astype(np.int32) - 127)[:, :, None].astype(np.float32)
    return w.reshape(R, C).astype(np.float32)


def nv_quant(w, search=True):
    """w [R, C] float32 -> (codes [R, C/2] uint8, scales [R, C/16] uint8, amax float32)"""
    R, C = w.shape
    amax = np.float32(np.abs(w).max())
    g = np.float32(np.float32(amax * np.float32(6.0 * 448.0 / 2688.0)) / np.float32(6.0 * 448.0))  # = nv_gmul
    b = w.reshape(R, C // 16, 16)
    a = np.abs(b)
    bmax = a.max(axis=2)
    t = bmax / (np.float32(6) * g) if g > 0 else np.zeros_like(bmax)
    c0 = np.clip(np.searchsorted(E4M3_POS, t), 1, 126)  # nearest E4M3 code to t
    lower = np.clip(c0 - 1, 0, 126)
    c0 = np.where(np.abs(E4M3_POS[lower] - t) <= np.abs(E4M3_POS[c0] - t), lower, c0)
    c0 = np.where(bmax == 0, 0, c0)
    cands = [0] if not search else [0, -1, 1, -2, 2, -3, 3]
    best_c, best_e = None, None
    for d in cands:
        c = np.clip(c0 + d, 0, 126)
        S = E4M3_POS[c] * g  # [R, NB]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(S[:, :, None] > 0, a / S[:, :, None], 0)
        qi = np.searchsorted(MID, x)  # nearest E2M1 index (ties up)
        err = ((a - E2M1[qi] * S[:, :, None]) ** 2).sum(axis=2)
        if best_e is None:
            best_c, best_e, best_q = c, err, qi
        else:
            better = err < best_e
            best_c = np.where(better, c, best_c)
            best_q = np.where(better[:, :, None], qi, best_q)
            best_e = np.where(better, err, best_e)
    nib = (best_q | np.where((b < 0) & (best_q > 0), 8, 0)).astype(np.uint8).reshape(R, C)
    codes = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return codes, best_c.astype(np.uint8), amax


def nv_decode(rec, off_c, off_s, R, C, gmul):
    by = np.frombuffer(rec, np.uint8, R * C // 2, off_c).reshape(R, C // 2)
    nib = np.empty((R, C), np.uint8)
    nib[:, 0::2] = by & 15
    nib[:, 1::2] = by >> 4
    v = E2M1[nib & 7] * np.where(nib & 8, -1, 1).astype(np.float32)
    sc = np.frombuffer(rec, np.uint8, R * (C // 16), off_s).reshape(R, C // 16)
    return (v.reshape(R, C // 16, 16) * E4M3[sc][:, :, None] * gmul).reshape(R, C)


def fmt(m, kind):
    f = next(v for v in m["experts"]["formats"].values() if v["kind"] == kind)
    mats = {x["name"]: x for x in f["matrices"]}
    return f, mats


def gmul_of(rec, trailer, which):
    t = np.frombuffer(rec, np.float32, 4, trailer)
    gs = np.float32(t[which] * np.float32(np.float32(6.0 * 448.0) / t[2 + which]))
    return np.float32(gs / np.float32(6.0 * 448.0))


def convert_record(args):
    rec, MX, NV, nv_bytes, trailer, search = args
    out = bytearray(nv_bytes)
    amaxes = []
    for name in ("gate_up", "down"):
        R, C = MX[name]["shape"]
        w = mx_decode(rec, MX[name]["codes"], MX[name]["scales"], R, C)
        codes, scales, amax = nv_quant(w, search)
        out[NV[name]["codes"]:NV[name]["codes"] + codes.size] = codes.tobytes()
        out[NV[name]["scales"]:NV[name]["scales"] + scales.size] = scales.tobytes()
        amaxes.append(amax)
    out[trailer:trailer + 16] = struct.pack("<4f", amaxes[0], amaxes[1], 2688.0, 2688.0)
    return bytes(out)


def main():
    a = sys.argv[1:]
    model, out_dir = a[0], a[1]
    layers = [int(x) for x in a[a.index("--layers") + 1].split(",")] if "--layers" in a else [20, 21, 22, 23]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 2
    m = json.load(open(os.path.join(model, "moved-store.json")))
    fmx, MX = fmt(m, "mxfp8")
    fnv, NV = fmt(m, "nvfp4")
    mx_bytes, nv_bytes, trailer = fmx["record_bytes"], fnv["record_bytes"], fnv["trailer"]["offset"]
    E = m["experts"]["experts_per_layer"]
    lf = m["experts"]["layer_file"]
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for L in layers:
        src = os.path.join(model, lf.format(L=L))
        dst = os.path.join(out_dir, "layer_%02d.bin" % L)
        if "--check-only" not in a:
            with open(src, "rb") as f, open(dst + ".tmp", "wb") as g, Pool(workers) as p:
                recs = (f.read(mx_bytes) for _ in range(E))
                for i, nv in enumerate(p.imap(convert_record, ((r, MX, NV, nv_bytes, trailer, True) for r in recs), chunksize=1)):
                    g.write(nv)
            os.replace(dst + ".tmp", dst)
        # the check: 8 records, new vs original, and the max-based scale alone
        num = den = num_max = 0.0
        with open(src, "rb") as f, open(dst, "rb") as g:
            for e in range(0, E, E // 8)[:8]:
                f.seek(e * mx_bytes)
                g.seek(e * nv_bytes)
                r, n = f.read(mx_bytes), g.read(nv_bytes)
                plain = convert_record((r, MX, NV, nv_bytes, trailer, False))
                for which, name in enumerate(("gate_up", "down")):
                    R, C = MX[name]["shape"]
                    w = mx_decode(r, MX[name]["codes"], MX[name]["scales"], R, C).astype(np.float64)
                    v = nv_decode(n, NV[name]["codes"], NV[name]["scales"], R, C, gmul_of(n, trailer, which)).astype(np.float64)
                    v0 = nv_decode(plain, NV[name]["codes"], NV[name]["scales"], R, C, gmul_of(plain, trailer, which)).astype(np.float64)
                    num += ((v - w) ** 2).sum()
                    num_max += ((v0 - w) ** 2).sum()
                    den += (w ** 2).sum()
        report[L] = {"file": dst, "bytes": os.path.getsize(dst), "rel_sq_err": num / den, "rel_sq_err_max_based": num_max / den}
        print(json.dumps({"layer": L} | report[L]), flush=True)
    json.dump(report, open(os.path.join(out_dir, "convert_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
