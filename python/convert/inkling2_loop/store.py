"""The Inkling 2 Mini expert store: record layout, safetensors helpers, and record -> MLX arrays.

Shared by convert.py, check.py, make_tiny.py and test_tiny.py, and meant for the runner.

A record holds ONE routed expert exactly as released (bytes copied, nothing re-quantised), its fields in this order, each
starting on a 64-byte boundary, the record padded with zeros to a whole number of 16 KB pages:

  nvfp4 (layers whose w13_weight is U8):
    gate_up.weight   uint8 [2I, H/2]   two E2M1 codes per byte, element 2j in the LOW nibble (MLX packs the same way)
    gate_up.scales   e4m3  [2I, H/16]  one FP8 E4M3 scale per 16 inputs
    down.weight      uint8 [H, I/2]
    down.scales      e4m3  [H, I/16]
    trailer          4 x float32: gate_up amax, down amax, gate_up outer_scale_denominator, down outer_scale_denominator
  mxfp8 (layers whose w13_weight is F8_E4M3):
    gate_up.weight   e4m3  [2I, H]     one FP8 E4M3 code per byte
    gate_up.scales   e8m0  [2I, H/32]  one UE8M0 (power of two, bias 127) scale per 32 inputs
    down.weight      e4m3  [H, I]
    down.scales      e8m0  [H, I/32]

gate_up rows alternate gate / up (row 2j = gate j, row 2j+1 = up j), as in the release and in the PR's Experts
(gate_up[..., 0::2], gate_up[..., 1::2]). Matrices are [out, in]; groups run along `in`.
MLX views (mlx-lm PR 2086, checkpoint.sanitize): weight = bytes viewed as uint32 [out, in*bits/32], scales = uint8
[out, in/group], and for nvfp4 a per-expert float32 global_scale = amax * (6 * 448 / denominator); gather_qmm then uses
group scale * global_scale / (6 * 448), i.e. w = code * scale * amax / denominator.
"""
import json
import os
import struct

import numpy as np

PAGE = 16384
ALIGN = 64
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "U32": 4, "I32": 4, "I64": 8}


# ----------------------------------------------------------------------------------------------- safetensors, raw
def read_header(path):
    """(tensors {name: {dtype, shape, data_offsets}}, metadata, data_start) without reading any tensor data."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    meta = h.pop("__metadata__", None)
    return h, meta, 8 + n


def nbytes(info):
    a, b = info["data_offsets"]
    return b - a


def write_safetensors(path, entries, metadata=None, chunk=1 << 26, no_cache=False):
    """Stream tensors into a safetensors file without holding them in memory.
    entries: list of (name, dtype, shape, nbytes, source); source is bytes, or (fd, offset) to copy nbytes from a file.
    Tensors are written in the order given. Returns the sha256 of the whole file."""
    import hashlib
    hdr, pos = {}, 0
    for name, dt, shape, nb, _ in entries:
        assert nb == int(np.prod(shape, dtype=np.int64)) * DTYPE_BYTES[dt], (name, dt, shape, nb)
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [pos, pos + nb]}
        pos += nb
    if metadata:
        hdr["__metadata__"] = metadata
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    h = hashlib.sha256()
    tmp = path + ".part"
    with open(tmp, "wb") as fo:
        if no_cache:
            nocache(fo.fileno())
        for b in (struct.pack("<Q", len(hb)), hb):
            fo.write(b)
            h.update(b)
        for name, dt, shape, nb, src in entries:
            if isinstance(src, (bytes, bytearray, memoryview)):
                assert len(src) == nb, name
                fo.write(src)
                h.update(src)
                continue
            fd, off = src
            done = 0
            while done < nb:
                b = os.pread(fd, min(chunk, nb - done), off + done)
                assert b, f"short read for {name}"
                fo.write(b)
                h.update(b)
                done += len(b)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, path)
    return h.hexdigest()


def nocache(fd):
    """macOS: do not keep this file's pages in the unified buffer cache (the laptop has 16 GB and apps open)."""
    try:
        import fcntl
        fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------- record layout
def layer_format(header, L):
    """'nvfp4' or 'mxfp8' for layer L, from the release header dtypes."""
    dt = header[f"model.llm.layers.{L}.mlp.experts.w13_weight"]["dtype"]
    return {"U8": "nvfp4", "F8_E4M3": "mxfp8"}[dt]


def release_names(L):
    p = f"model.llm.layers.{L}.mlp.experts."
    return {"gate_up.weight": p + "w13_weight", "gate_up.scales": p + "w13_weight_scale",
            "down.weight": p + "w2_weight", "down.scales": p + "w2_weight_scale",
            "gate_up.amax": p + "w13_weight_amax", "down.amax": p + "w2_weight_amax",
            "gate_up.denominator": p + "w13_weight_outer_scale_denominator",
            "down.denominator": p + "w2_weight_outer_scale_denominator"}


def make_format(kind, H, I):
    """The record layout of one format: fields with offset, bytes, release dtype, per-expert shape; and MLX params."""
    if kind == "nvfp4":
        bits, group, wdt, sdt = 4, 16, "U8", "F8_E4M3"
    elif kind == "mxfp8":
        bits, group, wdt, sdt = 8, 32, "F8_E4M3", "U8"
    else:
        raise ValueError(kind)
    mats = {"gate_up": (2 * I, H), "down": (H, I)}  # [out, in]
    fields, pos = {}, 0
    for m, (o, i) in mats.items():
        assert i % group == 0 and (i * bits) % 32 == 0, (kind, m, i)
        for part, shape, dt in (("weight", (o, i * bits // 8), wdt), ("scales", (o, i // group), sdt)):
            pos = -(-pos // ALIGN) * ALIGN
            nb = shape[0] * shape[1]
            fields[f"{m}.{part}"] = {"offset": pos, "bytes": nb, "shape": list(shape), "release_dtype": dt}
            pos += nb
    if kind == "nvfp4":
        pos = -(-pos // ALIGN) * ALIGN
        fields["trailer"] = {"offset": pos, "bytes": 16, "shape": [4], "release_dtype": "F32",
                             "order": ["gate_up.amax", "down.amax", "gate_up.denominator", "down.denominator"]}
        pos += 16
    return {"kind": kind, "bits": bits, "group": group, "mode": kind, "global_scale": kind == "nvfp4",
            "matrices": {m: list(s) for m, s in mats.items()},
            "packed_u32_shapes": {m: [o, i * bits // 32] for m, (o, i) in mats.items()},
            "fields": fields, "record_bytes_used": pos, "record_bytes": -(-pos // PAGE) * PAGE,
            "code_dtype": "fp4_e2m1" if bits == 4 else "fp8_e4m3fn", "scale_dtype": "fp8_e4m3fn" if bits == 4 else "fp8_e8m0"}


def record_views(rec, fmt):
    """numpy uint8 views of one record's fields (rec: bytes-like or uint8 array of record_bytes)."""
    a = np.frombuffer(rec, dtype=np.uint8) if not isinstance(rec, np.ndarray) else rec
    out = {}
    for k, f in fmt["fields"].items():
        v = a[f["offset"]:f["offset"] + f["bytes"]]
        out[k] = v.view("<f4") if k == "trailer" else v.reshape(f["shape"])
    return out


def read_record(fd, meta, L, e):
    fmt = meta["formats"][meta["layer_format"][L]]
    R = fmt["record_bytes"]
    b = os.pread(fd, R, e * R)
    assert len(b) == R
    return b


def global_scale_f32(amax, denominator):
    """The PR's per-expert nvfp4 global scale (checkpoint.sanitize): amax.f32 * (6 * 448 / denominator.f32), in float32."""
    s = np.float32(6.0 * 448.0) / np.asarray(denominator, dtype=np.float32)
    return (np.asarray(amax, dtype=np.float32) * s).astype(np.float32)


def experts_to_mlx(recs, fmt):
    """A list of records (one layer, any experts) -> the PR's sanitized MLX arrays for the stacked experts:
    {'gate_up_proj.weight': uint32 [n, 2I, H*bits/32], 'gate_up_proj.scales': uint8 [n, 2I, H/group],
     'down_proj.weight', 'down_proj.scales', and for nvfp4 '*.global_scale' float32 [n]}."""
    import mlx.core as mx
    out = {}
    views = [record_views(r, fmt) for r in recs]
    for m, name in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
        w = np.stack([v[f"{m}.weight"] for v in views])
        out[f"{name}.weight"] = mx.array(np.ascontiguousarray(w).view("<u4"))
        out[f"{name}.scales"] = mx.array(np.stack([v[f"{m}.scales"] for v in views]))
    if fmt["global_scale"]:
        t = np.stack([v["trailer"] for v in views])  # [n, 4]
        out["gate_up_proj.global_scale"] = mx.array(global_scale_f32(t[:, 0], t[:, 2]))
        out["down_proj.global_scale"] = mx.array(global_scale_f32(t[:, 1], t[:, 3]))
    return out


def release_expert_tensors(store_dir, meta, L):
    """Layer L's expert tensors rebuilt in the RELEASE naming and dtypes (as numpy uint8 / float32), from the store.
    Used by test_tiny.py to rebuild a release from store + resident; holds a whole layer, so tiny models only."""
    fmt = meta["formats"][meta["layer_format"][L]]
    n = meta["experts"]
    with open(os.path.join(store_dir, meta["layer_file"].format(L=L)), "rb") as fh:
        recs = [fh.read(fmt["record_bytes"]) for _ in range(n)]
    views = [record_views(r, fmt) for r in recs]
    names = release_names(L)
    out = {}
    for k in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales"):
        out[names[k]] = (fmt["fields"][k]["release_dtype"], np.stack([v[k] for v in views]))
    if fmt["global_scale"]:
        t = np.stack([v["trailer"] for v in views])
        out[names["gate_up.amax"]] = ("F32", np.ascontiguousarray(t[:, 0]))
        out[names["down.amax"]] = ("F32", np.ascontiguousarray(t[:, 1]))
    return out


# ----------------------------------------------------------------------------------------------- independent decoders
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0xF and m == 0x7:
            t[b] = np.nan  # e4m3fn: only S.1111.111 is NaN, no infinities
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()
E8M0 = np.array([2.0 ** (b - 127) if b != 255 else np.nan for b in range(256)], dtype=np.float32)


def dequant_np(weight_u8, scales_u8, kind, global_scale=None):
    """Independent dequantisation (numpy, float32): weight_u8 [out, in*bits/8], scales_u8 [out, in/group] -> [out, in].
    Same operation order as MLX's fp_dequantize: scale = float(group scale) * (global_scale / 2688); w = scale * code."""
    if kind == "nvfp4":
        lo, hi = weight_u8 & 0x0F, weight_u8 >> 4
        codes = np.empty((weight_u8.shape[0], weight_u8.shape[1] * 2), dtype=np.uint8)
        codes[:, 0::2], codes[:, 1::2] = lo, hi
        vals, sc, g = E2M1[codes], E4M3[scales_u8], 16
        if global_scale is not None:
            sc = (sc * np.float32(np.float32(global_scale) / np.float32(448.0 * 6.0))).astype(np.float32)
    else:
        vals, sc, g = E4M3[weight_u8], E8M0[scales_u8], 32
    return (np.repeat(sc, g, axis=1) * vals).astype(np.float32)
