"""Check a calibrated 4-bit resident (ms_derive resident DIR --bits 4 --from-calibrated CAL --out OUT) without a model run.

usage: python q4c_verify.py OUT CAL PLAIN
  OUT    the calibrated container (DMENGRES1, resident_bits 4)
  CAL    the calibrated export (the trainer's q36_q4c_export.py: BASE.codes / .scales / .biases / .shape, metadata r2_q4c)
  PLAIN  today's resident_q4 container (ms_derive resident DIR --bits 4)
Checks: (1) the name mapping: the export's matrices[].engine_name, the name derived from its release key, and the
container's NAME.q4 stems are the same 251 names, shapes [R, C/2] / [R, C/64] equal; (2) every NAME.q4 / .s / .b
payload equals the export's codes / scales / biases bytes; (3) dequantised f32 (code * s + b, codes unpacked low nibble
first, the same numpy expression on both sides) bitwise equal for a spread of matrices; (4) every other tensor bitwise
equal to PLAIN's; (5) header (cfg lines, tensor table) identical to PLAIN's, file size equal.
"""
import json, re, struct, sys

import numpy as np

SHORT = {"linear_attn.in_proj_qkv": "qkv", "linear_attn.in_proj_z": "z", "linear_attn.out_proj": "o", "self_attn.q_proj": "q",
         "self_attn.k_proj": "k", "self_attn.v_proj": "v", "self_attn.o_proj": "o", "mlp.shared_expert.gate_proj": "sg",
         "mlp.shared_expert.up_proj": "su", "mlp.shared_expert.down_proj": "sd"}
DT = {"u8": np.uint8, "f16": np.float16, "f32": np.float32, "bf16": np.uint16}


def dm_open(path):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    first = bytes(mm[:64]).split(b"\n")[0].decode()
    assert first.startswith("DMENGRES1 "), path
    hdr = int(first.split()[1])
    text = bytes(mm[:hdr]).rstrip(b"\0").decode()
    cfg, ts, done = [], {}, False
    for line in text.split("\n")[1:]:
        f = line.split()
        if not f:
            continue
        if f[0] == "cfg":
            cfg.append((f[1], " ".join(f[2:])))
        elif f[0] == "t":
            nd = int(f[5])
            ts[f[1]] = (f[2], int(f[3]), int(f[4]), [int(x) for x in f[6:6 + nd]])
        elif f[0] == "end":
            done = True
            break
    assert done, path + ": no end line"
    return mm, hdr, text, cfg, ts


def dm_get(dm, name):
    mm, hdr, _, _, ts = dm
    dt, off, nb, shp = ts[name]
    return mm[hdr + off: hdr + off + nb].view(DT[dt]).reshape(shp)


def st_open(path):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = struct.unpack("<Q", bytes(mm[:8]))[0]
    h = json.loads(bytes(mm[8:8 + n]))
    meta = h.pop("__metadata__", {})
    return mm, 8 + n, h, meta


def st_get(st, name):
    mm, base, h, _ = st
    t = h[name]
    dt = {"U8": np.uint8, "F16": np.float16, "I64": np.int64}[t["dtype"]]
    b, e = t["data_offsets"]
    return mm[base + b: base + e].view(dt).reshape(t["shape"])


def engine_name(base):
    if base == "lm_head":
        return "head"
    m = re.fullmatch(r"model\.(?:language_model\.)?layers\.(\d+)\.(.+)", base)
    return f"{m.group(1)}.{SHORT[m.group(2)]}"


def deq(codes_packed, s, b, r0, r1):
    p = np.asarray(codes_packed[r0:r1])
    q = np.empty((p.shape[0], p.shape[1] * 2), np.uint8)
    q[:, 0::2], q[:, 1::2] = p & 0xF, p >> 4
    R, C = q.shape
    w = q.reshape(R, C // 64, 64).astype(np.float32) * np.asarray(s[r0:r1], np.float32)[..., None] + \
        np.asarray(b[r0:r1], np.float32)[..., None]
    return w.reshape(R, C)


def main(out, cal, plain):
    O, P, S = dm_open(out), dm_open(plain), st_open(cal)
    res = {}
    # (1) mapping
    mats = json.loads(S[3]["r2_q4c"])["matrices"]
    meta_names = {x["base"]: x["engine_name"] for x in mats}
    derived = {b: engine_name(b) for b in meta_names}
    stems = sorted(n[:-3] for n in O[4] if n.endswith(".q4"))
    bases = sorted({k.rsplit(".", 1)[0] for k in S[2]})
    assert derived == meta_names, "metadata engine_name differs from the name derived from the release key"
    assert sorted(bases) == sorted(meta_names), "export tensors vs metadata matrix list"
    assert sorted(meta_names.values()) == stems, "export names vs container stems"
    assert len(stems) == 251 and len(set(meta_names.values())) == 251
    assert all(k.rsplit(".", 1)[1] in ("codes", "scales", "biases", "shape") for k in S[2]) and len(S[2]) == 251 * 4
    res["mapping"] = f"251 of 251, none missing, none extra"
    # (2) payloads + shapes
    by_name = {v: k for k, v in meta_names.items()}
    for n in stems:
        base = by_name[n]
        R, C = st_get(S, base + ".shape").tolist()
        for sfx, key, shp, dt in ((".q4", ".codes", [R, C // 2], "u8"), (".s", ".scales", [R, C // 64], "f16"),
                                  (".b", ".biases", [R, C // 64], "f16")):
            t = O[4][n + sfx]
            assert t[0] == dt and t[3] == shp and list(st_get(S, base + key).shape) == shp, (n, sfx, t, shp)
            assert np.array_equal(dm_get(O, n + sfx).view(np.uint8), st_get(S, base + key).view(np.uint8)), (n, sfx)
    res["payloads"] = "753 of 753 tensors (251 x .q4/.s/.b) byte-equal to the export, shapes match"
    # (3) dequantised f32 bitwise, from the container and from the export
    pick = ["0.qkv", "0.z", "0.o", "1.qkv", "3.q", "3.k", "3.v", "3.o", "20.qkv", "21.o", "23.q", "23.o", "39.q", "39.o",
            "0.sg", "0.su", "0.sd", "17.sg", "38.su", "39.sd", "head"]
    nel = 0
    for n in pick:
        base = by_name[n]
        R = O[4][n + ".q4"][3][0]
        for r0 in range(0, R, 8192):
            r1 = min(R, r0 + 8192)
            a = deq(dm_get(O, n + ".q4"), dm_get(O, n + ".s"), dm_get(O, n + ".b"), r0, r1)
            b = deq(st_get(S, base + ".codes"), st_get(S, base + ".scales"), st_get(S, base + ".biases"), r0, r1)
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (n, r0)
            nel += a.size
    res["dequant"] = f"{len(pick)} matrices ({', '.join(pick)}), {nel:,} weights bitwise equal as f32"
    # (4) every other tensor vs PLAIN
    four = {n + x for n in stems for x in (".q4", ".s", ".b")}  # (L.b = in_proj_b is bf16, not a 4-bit part)
    others = [n for n in P[4] if n not in four]
    for n in others:
        assert O[4][n] == P[4][n], n
        assert np.array_equal(dm_get(O, n).view(np.uint8), dm_get(P, n).view(np.uint8)), n
    diff4 = sum(not np.array_equal(dm_get(O, n).view(np.uint8), dm_get(P, n).view(np.uint8)) for n in four)
    res["others"] = f"{len(others)} of {len(others)} non-4-bit tensors bitwise equal to the plain file ({diff4} of 753 4-bit tensors differ from it)"
    # (5) header + size
    assert O[1] == P[1] and O[2] == P[2], "header text differs from the plain file's"
    assert len(O[0]) == len(P[0]), (len(O[0]), len(P[0]))
    res["header"] = f"header identical to the plain file's ({len(O[3])} cfg lines incl. resident_bits {dict(O[3]).get('resident_bits')}, {len(O[4])} tensors); size {len(O[0]):,} B = plain"
    for k, v in res.items():
        print(f"{k}: {v}")
    print("PASS")


if __name__ == "__main__":
    main(*sys.argv[1:4])
