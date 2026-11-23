#!/usr/bin/env python3
"""The light draft-head source (include/dmeng/draft.h LightHeadSource) on the tiny Qwen3.6 model with random head weights
in the trainer's format (jobs/babel/q36_draft_head.py export: bf16 safetensors, mlp.{j}.*, proj / vocab), seconds, no real
model. Exit 0 only if all hold:
  1. with the head held as bf16 (--draft-light-q8 0) every run's drafts equal a numpy reference of the trainer's convention
     on the same h_t and the same tokens (embedding rows from the release), confidences within 2e-3;
  2. ids with --draft lookup --draft-light F equal plain decoding (low-rank and full variants, q8 on and off, the head
     every step; and gated as decoding runs it);
  3. the 8-bit head (q8 1) agrees with the bf16 one on most first drafts (reported).
  tests/draft_light_check.py TINY_MODEL_DIR TINY_RELEASE_DIR [INK_RESIDENT]   (Python with numpy; DMENG_BIN overrides
  build/dmeng). With INK_RESIDENT: the tiny Inkling (its resident file), the same checks through Inkling's hooks."""
import json, os, struct, subprocess, sys, tempfile

import numpy as np

R_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.environ.get("DMENG_BIN", os.path.join(R_, "build", "dmeng"))
MODEL, RELEASE = sys.argv[1], sys.argv[2]
INK_RES = sys.argv[3] if len(sys.argv) > 3 else None


def bf16(a):  # fp32 -> bf16 bits (round to nearest even)
    u = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def write_st(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k, v in tensors.items():
        b = v.tobytes()
        hdr[k] = {"dtype": "BF16", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in h.items():
            if k == "__metadata__":
                continue
            f.seek(base + v["data_offsets"][0])
            raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
            out[k] = (v["dtype"], v["shape"], raw)
        return out


emb = None
for fn in sorted(os.listdir(RELEASE)):
    if fn.endswith(".safetensors"):
        t = read_st(os.path.join(RELEASE, fn))
        for name in ("model.language_model.embed_tokens.weight", "model.llm.embed.weight"):
            if name in t:
                dt, sh, raw = t[name]
                emb = f32(np.frombuffer(raw, np.uint16).reshape(sh))
assert emb is not None, "no embedding in the release"
V, D = emb.shape
_c = json.load(open(os.path.join(MODEL, "config.json")))
_c = _c.get("text_config", _c)
VOUT = _c.get("unpadded_vocab_size", V)  # rows that count for the argmax (Inkling: the unpadded vocabulary)
rng = np.random.default_rng(0)
Hh, RK, ND = 128, 64, 3


def make(kind):
    t = {}
    for j in range(1, ND + 1):
        t[f"mlp.{j}.norm_z.weight"] = bf16(1 + 0.1 * rng.standard_normal(D))
        t[f"mlp.{j}.norm_e.weight"] = bf16(1 + 0.1 * rng.standard_normal(D))
        t[f"mlp.{j}.up.weight"] = bf16(rng.standard_normal((Hh, 2 * D)) / np.sqrt(2 * D))
        t[f"mlp.{j}.up.bias"] = bf16(0.1 * rng.standard_normal(Hh))
        t[f"mlp.{j}.down.weight"] = bf16(rng.standard_normal((D, Hh)) / np.sqrt(Hh))
        t[f"mlp.{j}.down.bias"] = bf16(0.1 * rng.standard_normal(D))
    if kind == "lr":
        t["proj.weight"] = bf16(rng.standard_normal((RK, D)) / np.sqrt(D))
        t["vocab.weight"] = bf16(3 * rng.standard_normal((V, RK)) / np.sqrt(RK))
    return t


def rms(x, w):
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6).astype(np.float32) * w


def ref(t, h, tok):
    z, out, conf = f32(bf16(h)), [], []  # h_t in bf16 (the trainer's convention)
    for j in range(1, ND + 1):
        g = lambda n: f32(t[f"mlp.{j}.{n}"])
        u = np.concatenate([rms(z, g("norm_z.weight")), rms(emb[tok], g("norm_e.weight"))])
        a = g("up.weight") @ u + g("up.bias")
        a = a / (1 + np.exp(-a))
        z = z + g("down.weight") @ a + g("down.bias")
        lg = (f32(t["vocab.weight"]) @ (f32(t["proj.weight"]) @ z))[:VOUT]
        d = int(np.argmax(lg))
        p = 1 / np.sum(np.exp((lg - lg[d]).astype(np.float64)))
        out.append(d)
        conf.append(p)
        tok = d
    return out, conf


tmp = tempfile.mkdtemp(prefix="draft_light")
P = ",".join(str(x) for x in np.random.default_rng(3).integers(0, 500, 60))
P = ",".join(str(x) for x in np.random.default_rng(3).integers(0, min(500, V), 60))
Q = ["--threads", "2", "--readers", "2", "--mem", "plain", "--res-mem", "plain", "--cache-gb", "0.05" if INK_RES else "0.01", "--model", MODEL,
     "--prompt-ids", P, "--n", "80"] + (["--resident", INK_RES, "--ctx", "1024"] if INK_RES else [])
fail = 0
subprocess.run([B] + Q + ["--out-ids", os.path.join(tmp, "plain.ids")], capture_output=True, check=True)
plain = open(os.path.join(tmp, "plain.ids")).read()
heads = {k: make(k) for k in ("lr", "full")}
firsts = {}
for kind, t in heads.items():
    path = os.path.join(tmp, f"heads_{kind}.safetensors")
    write_st(path, t)
    for q8 in (0, 1):
        dump = os.path.join(tmp, f"{kind}_q8{q8}.dump")
        js = os.path.join(tmp, f"{kind}_q8{q8}.json")
        r = subprocess.run([B] + Q + ["--draft", "lookup", "--draft-light", path, "--draft-light-q8", str(q8), "--draft-light-dump", dump, "--draft-light-backoff", "0",
                                     "--out-ids", os.path.join(tmp, "d.ids"), "--json", js], capture_output=True, text=True)
        if r.returncode:
            print("FAIL run", kind, q8, r.stderr[-400:])
            fail = 1
            continue
        same = open(os.path.join(tmp, "d.ids")).read() == plain
        lt = json.load(open(js))["draft"].get("light", {})
        print(("PASS" if same else "FAIL"), f"ids {kind} q8={q8} = plain", f"(head runs {lt.get('runs')}, offers {lt.get('offers')}, drafts kept {lt.get('kept')} / {lt.get('drafts')})")
        fail |= not same
        raw = open(dump, "rb").read()
        rec = 4 * (1 + D + 2 + 2 * ND)
        runs = []
        for k in range(len(raw) // rec):
            o = k * rec + 4  # (the leading position)
            h = np.frombuffer(raw, np.float32, D, o)
            tok, n = struct.unpack_from("<ii", raw, o + 4 * D)
            d = list(struct.unpack_from(f"<{ND}i", raw, o + 4 * D + 8))
            c = list(struct.unpack_from(f"<{ND}f", raw, o + 4 * D + 8 + 4 * ND))
            runs.append((h, tok, n, d, c))
        firsts[(kind, q8)] = [x[3][0] for x in runs]
        if kind == "lr" and q8 == 0:
            bad, cmax = 0, 0.0
            for h, tok, n, d, c in runs:
                rd, rc = ref(t, h, tok)
                # (n < ND: the run was asked for fewer drafts, near the token budget's end)
                if n < 1 or d[:n] != rd[:n]:
                    bad += 1
                    print("  differ: tok", tok, "n", n, "engine", d, c, "reference", rd, [round(x, 6) for x in rc])
                else:
                    cmax = max(cmax, max(abs(a - b) for a, b in zip(c[:n], rc[:n])))
            ok = bad == 0 and cmax < 2e-3 and len(runs) > 0
            print(("PASS" if ok else "FAIL"), f"reference: {len(runs)} runs, {bad} differ in drafts, max confidence difference {cmax:.2e}")
            fail |= not ok
# the gated head (back-off and confidence-gated width on, as decoding runs it): ids
r = subprocess.run([B] + Q + ["--draft", "lookup", "--draft-light", os.path.join(tmp, "heads_lr.safetensors"), "--out-ids", os.path.join(tmp, "g.ids")],
                   capture_output=True, text=True)
same = r.returncode == 0 and open(os.path.join(tmp, "g.ids")).read() == plain
print(("PASS" if same else "FAIL"), "ids lr gated (back-off, gated width) = plain")
fail |= not same
for kind in heads:
    a, b = firsts.get((kind, 0), []), firsts.get((kind, 1), [])
    m = min(len(a), len(b))
    if m:
        print(f"INFO {kind}: 8-bit head's first draft = bf16 head's on {sum(x == y for x, y in zip(a[:m], b[:m]))} / {m} runs (same inputs only while the runs agree)")
print("DRAFT LIGHT CHECK", "PASSED" if not fail else "FAILED")
sys.exit(fail)
