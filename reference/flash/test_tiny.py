"""Synthetic validation of the flash runner (mlx_run.py) against the vendored reference model, on make_tiny.py's release.

  .venv/bin/python make_tiny.py WORK/tiny_release
  REPO=WORK/tiny_release OUT=WORK/tiny_store STATE_DIR=WORK/tiny_state NG_SHARD_ROWS=1024 NG_DIM=32 NG_PART_ROWS=1000 .venv/bin/python convert.py
  .venv/bin/python test_tiny.py --release WORK/tiny_release --store WORK/tiny_store --work WORK/refs

References: the vendored community model (qwen4_exp/, the released SparseMoeBlock with SwitchGLU and the in-memory
n-gram table), loaded from the release's safetensors through the port's own sanitize (in float32, so the zero-centred
norm weights get their + 1 exactly), one forward pass over the prompt:
  R0  released weights, float32
  R1  the store's values: experts = mx.dequantize of the int4 records, n-gram rows = int8 code * scale; float16
  R2  R1 with the resident linears quantised to int8 (group 64) by the runner's rule
  R1f R1 in float32 (the runner's --dtype float32: everything but the expert step in float32)
  R8f R1f with the experts of an int8 one-scale-per-neuron store instead (the earlier record format; written here from
      the release with the Qwen3-Next-80B store's quantisation), to check that the runner still serves it
Each also token by token through the cache (_inc), the order the runner computes in when it decodes.
And, independent of the community port: the official transformers implementation (Qwen4ExpForCausalLM, float32, CPU,
its reference PyTorch paths for the Gated DeltaNet and the conv) on the same release, against R0 (--hf, default on).
The runner (exact mode, released routing) runs through its --check front-end against them; every comparison is next-token
agreement and the max |logit diff| on the reference's top-5 at every position, plus whole-vocabulary diffs where two
runner outputs are compared directly (--check-save).
"""
import argparse, glob, json, os, re, subprocess, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--release", required=True)
ap.add_argument("--store", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--no-hf", action="store_true", help="skip the transformers reference")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(args.work, exist_ok=True)
PY = sys.executable

# the runner's import path for the vendored model (same stubs: mlx_vlm's package __init__ files are not run)
import importlib.util, types  # noqa: E402

root = importlib.util.find_spec("mlx_vlm").submodule_search_locations[0]
for name in ("mlx_vlm", "mlx_vlm.speculative", "mlx_vlm.models.qwen3_5", "mlx_vlm.models.qwen3_vl"):
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(root, *name.split(".")[1:])]
    sys.modules[name] = m
sys.path.insert(0, HERE)
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import torch  # noqa: E402
from qwen4_exp.config import TextConfig  # noqa: E402
from qwen4_exp.language import LanguageModel  # noqa: E402
from qwen4_exp.qwen4_exp import sanitize  # noqa: E402

cfg = json.load(open(os.path.join(args.release, "config.json")))["text_config"]
targs = TextConfig.from_dict(cfg)
meta = json.load(open(os.path.join(args.store, "store", "meta.json")))
ngm = json.load(open(os.path.join(args.store, "ngram", "meta.json")))
H, I, NL, NE, REC, G = meta["hidden"], meta["intermediate"], meta["layers"], meta["experts"], meta["record_bytes"], meta["group"]
V = cfg["vocab_size"]

release = {}
for f in sorted(glob.glob(os.path.join(args.release, "model-*.safetensors"))):
    release.update(mx.load(f))
release = {k: (v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in release.items()}


STORE8 = os.path.join(args.work, "store_int8")


def make_int8_store():
    """The release's experts as int8 records with one fp16 scale per neuron (the Qwen3-Next-80B store's converter: gate/up per output
    row, down per input column; scale = max|w| / 127), next to symlinks of the int4 store's other files."""
    os.makedirs(os.path.join(STORE8, "store"), exist_ok=True)
    for f in os.listdir(args.store):
        if f not in ("store", "resident_int8.safetensors") and not os.path.exists(os.path.join(STORE8, f)):
            os.symlink(os.path.abspath(os.path.join(args.store, f)), os.path.join(STORE8, f))
    MAT = I * H
    rec8 = -(-(3 * MAT + 3 * I * 2) // 16384) * 16384
    OFF = {"gate_proj": 0, "up_proj": MAT, "down_proj": 2 * MAT}
    SOFF = {"gate_proj": 3 * MAT, "up_proj": 3 * MAT + 2 * I, "down_proj": 3 * MAT + 4 * I}
    deq = {}
    for L in range(NL):
        buf = np.zeros((NE, rec8), dtype=np.uint8)
        gu = torch.from_numpy(np.array(release[f"model.language_model.layers.{L}.mlp.experts.gate_up_proj"]))
        dn = torch.from_numpy(np.array(release[f"model.language_model.layers.{L}.mlp.experts.down_proj"]))
        for e in range(NE):
            for kind, w in (("gate_proj", gu[e, :I]), ("up_proj", gu[e, I:]), ("down_proj", dn[e])):
                dim = 0 if kind == "down_proj" else 1
                sc = (w.abs().amax(dim=dim).clamp_min(1e-8) / 127.0).to(torch.float16)
                sf = sc.float()
                q = torch.clamp((w / (sf[None, :] if kind == "down_proj" else sf[:, None])).round(), -127, 127).to(torch.int8)
                buf[e, OFF[kind] : OFF[kind] + MAT] = q.numpy().reshape(-1).view(np.uint8)
                buf[e, SOFF[kind] : SOFF[kind] + 2 * I] = sc.numpy().view(np.uint8)
                deq[(L, e, kind)] = (q.float() * (sf[None, :] if kind == "down_proj" else sf[:, None])).numpy()
        buf.tofile(os.path.join(STORE8, "store", f"layer_{L:02d}.bin"))
    m8 = {k: v for k, v in meta.items() if k not in ("bias_offsets", "group", "record_order", "shapes")}
    m8.update(record_bytes=rec8, offsets=OFF, scale_offsets=SOFF, dtype="int8")
    json.dump(m8, open(os.path.join(STORE8, "store", "meta.json"), "w"), indent=1)
    return deq


DEQ8 = make_int8_store()


def store_experts(L):
    """Layer L's experts from the store, dequantised: gate/up [NE, I, H], down [NE, H, I] (float16, as mx.dequantize gives)."""
    raw = np.fromfile(os.path.join(args.store, "store", f"layer_{L:02d}.bin"), dtype=np.uint8).reshape(NE, REC)
    out = {}
    for kind, (r, c) in meta["shapes"].items():
        o, so, bo = meta["offsets"][kind], meta["scale_offsets"][kind], meta["bias_offsets"][kind]
        w = mx.array(np.ascontiguousarray(raw[:, o : o + r * c // 2]).view(np.uint32).reshape(NE, r, c // 8))
        s = mx.array(np.ascontiguousarray(raw[:, so : so + r * c // G * 2]).view(np.float16).reshape(NE, r, c // G))
        b = mx.array(np.ascontiguousarray(raw[:, bo : bo + r * c // G * 2]).view(np.float16).reshape(NE, r, c // G))
        out[kind] = mx.dequantize(w, s, b, group_size=G, bits=4)
    return out


def store_ngram():
    parts = [np.fromfile(os.path.join(args.store, f), dtype=np.uint8).reshape(-1, ngm["bytes_per_row"]) for f in ngm["files"]]
    recs = np.concatenate(parts)
    assert len(recs) == ngm["rows"]
    return recs[:, 2:].view(np.int8).astype(np.float32) * recs[:, :2].copy().view(np.float16).astype(np.float32)


def qpred(path, m):  # the runner's rule
    return isinstance(m, nn.Linear) and m.weight.shape[0] >= 64 and m.weight.shape[1] % 64 == 0 and not path.endswith("mlp.gate")


def build(variant, targs=targs):
    w = sanitize(dict(release), targs)
    w = {k.replace("language_model.", "", 1): v for k, v in w.items()}
    dt = mx.float32 if variant in ("R0", "R1f", "R8f") else mx.float16
    if variant in ("R1", "R2", "R1f", "R8f"):
        for L in range(NL):
            e = store_experts(L) if variant != "R8f" else {k: mx.array(np.stack([DEQ8[(L, x, k)] for x in range(NE)])) for k in ("gate_proj", "up_proj", "down_proj")}
            for kind in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{L}.mlp.switch_mlp.{kind}.weight"
                assert w[key].shape == e[kind].shape, (key, w[key].shape, e[kind].shape)
                w[key] = e[kind]
        table = store_ngram()
        shard_keys = sorted((k for k in w if ".ngram_embedding.shards." in k), key=lambda k: int(k.split(".shards.")[1].split(".")[0]))
        rows = ngm["shard_rows"]
        for i, k in enumerate(shard_keys):
            assert w[k].shape == (rows, ngm["dim"])
            w[k] = mx.array(table[i * rows : (i + 1) * rows])
    w = [(k, v.astype(dt) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in w.items()]
    lm = LanguageModel(targs)
    lm.load_weights(w, strict=True)  # strict: every parameter of the port comes from the release, nothing left over
    if variant == "R2":
        nn.quantize(lm, group_size=64, bits=8, class_predicate=qpred)
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def run_ref(lm, ids, incremental=False):
    cache = lm.make_cache()
    if not incremental:
        return np.array(lm.lm_head(lm.model(mx.array([ids]), cache=cache))[0].astype(mx.float32))
    out = []
    for t in ids:
        out.append(np.array(lm.lm_head(lm.model(mx.array([[t]]), cache=cache))[0, -1].astype(mx.float32)))
    return np.stack(out)


# the prompt: 40 tokens from a fixed generator, with the n-gram EOS at two places (the hashing's EOS-aware shifts)
rng = np.random.default_rng(1)
eos = cfg["eos_token_id"]
ids = [int(t) for t in rng.integers(10, V, args.n)]
ids[13] = ids[27] = eos
refs = {}
for variant in ("R0", "R1", "R2", "R1f", "R8f"):
    lm = build(variant)
    refs[variant] = run_ref(lm, ids)
    refs[variant + "_inc"] = run_ref(lm, ids, incremental=True)
    del lm


def save_ref(path, lg, n_prompt=1, **extra):
    top = np.argsort(-lg, axis=1)[:, :5]
    torch.save({"ids": torch.tensor(ids), "n_prompt": n_prompt, "rule": "orig", "all_argmax": torch.tensor(lg.argmax(1)),
                "all_top_ids": torch.tensor(top), "all_top_logits": torch.tensor(np.take_along_axis(lg, top, 1)),
                "final_logits": torch.tensor(lg[-1]), **extra}, path)
    return path


def cmp(x, y):
    top = np.argsort(-y, axis=1)[:, :5]
    return {"agree": int((x.argmax(1) == y.argmax(1)).sum()), "n": len(x), "max_top5": float(np.abs(np.take_along_axis(x, top, 1) - np.take_along_axis(y, top, 1)).max()),
            "max_vocab": float(np.abs(x - y).max())}


HF_CODE = r"""
import json, sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import Qwen4ExpForCausalLM
m, info = Qwen4ExpForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32, output_loading_info=True)
bad = {k: [x for x in v if not x.startswith(("mtp.", "model.visual."))] for k, v in info.items() if v}
bad = {k: v for k, v in bad.items() if v}
assert not bad, bad
m.eval()
with torch.no_grad():
    np.save(sys.argv[3], m(torch.tensor([json.loads(sys.argv[2])]), use_cache=False).logits[0].float().numpy())
"""
if not args.no_hf:
    # also with the QSA budget above the prompt length (dense attention in both), to separate the sparse selection
    dense = os.path.join(args.work, "release_dense")
    os.makedirs(dense, exist_ok=True)
    for f in os.listdir(args.release):
        if f != "config.json" and not os.path.exists(os.path.join(dense, f)):
            os.symlink(os.path.abspath(os.path.join(args.release, f)), os.path.join(dense, f))
    cd = json.load(open(os.path.join(args.release, "config.json")))
    cd["text_config"]["indexer_budget"] = 4096
    json.dump(cd, open(os.path.join(dense, "config.json"), "w"))
    for tag, rel in (("HF", args.release), ("HF_dense", dense)):
        out = os.path.join(args.work, f"{tag}_logits.npy")
        r = subprocess.run([PY, "-c", HF_CODE, rel, json.dumps(ids), out], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        refs[tag] = np.load(out)
    refs["R0_dense"] = run_ref(build("R0", TextConfig.from_dict(cd["text_config"])), ids)
print("== references (one forward pass over the 40-token prompt unless noted)")
if "HF" in refs:
    print("transformers Qwen4ExpForCausalLM (fp32, torch) vs R0 (the vendored MLX port, fp32), QSA budget 4096 (dense):", cmp(refs["HF_dense"], refs["R0_dense"]))
    print("the same, QSA budget 16 (sparse selection from position 19):", cmp(refs["HF"], refs["R0"]),
          "| per position max |diff|:", " ".join(f"{p}:{np.abs(refs['HF'][p] - refs['R0'][p]).max():.0e}" for p in range(len(ids))))
print("R1 vs R0 (what the int4 experts + int8 n-gram rows + fp16 change):", cmp(refs["R1"], refs["R0"]))
print("R2 vs R1 (what the int8 resident changes):", cmp(refs["R2"], refs["R1"]))
for v in ("R0", "R1f", "R1", "R2"):
    print(f"{v} token by token vs {v} one pass (the reference's own noise in its dtype):", cmp(refs[v + "_inc"], refs[v]))
P = {v: save_ref(os.path.join(args.work, f"ref_{v}.pt"), refs[v]) for v in refs}


def runner(name, ref, *extra):
    out = os.path.join(args.work, f"run_{name}.npz")
    cmd = ["/usr/bin/time", "-l", PY, os.path.join(HERE, "mlx_run.py"), "--dir", args.store, "--mode", "exact", "--check", ref, "--check-save", out, "-v", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = r.stdout + r.stderr
    line = next((l for l in text.splitlines() if l.startswith("CHECK")), None)
    if r.returncode != 0 or line is None:
        print(text)
        raise SystemExit(f"{name}: runner failed")
    m = re.search(r"agreement (\d+)/(\d+) \| max \|logit diff\| on the reference top-5 ([\d.]+) \| max \|logit diff\| over the whole vocabulary at the last position ([\d.]+)", line)
    rss = int(re.search(r"(\d+)\s+maximum resident set size", text).group(1)) / 2**20
    feeds = re.search(r"feeds: (.*?); --chunk", line).group(1)
    res = {"agree": f"{m.group(1)}/{m.group(2)}", "max_top5": float(m.group(3)), "last_vocab": float(m.group(4)), "rss_mb": round(rss), "feeds": feeds}
    batch_lines = [l for l in text.splitlines() if l.startswith("[batch]")]
    print(f"{name:34s} {json.dumps(res)}")
    for l in batch_lines:
        print("    ", l[:400])
    z = np.load(out)
    return dict(zip(z["positions"].tolist(), z["logits"]))


print("\n== runner, exact mode, token by token (n_prompt 1: all 40 positions)")
tbt16 = runner("fp16 resident vs R1_inc", P["R1_inc"], "--dtype", "float16", "--chunk", "0")
tbt8 = runner("int8 resident vs R2_inc", P["R2_inc"], "--chunk", "0")
tbt32 = runner("fp32 resident vs R1f_inc", P["R1f_inc"], "--dtype", "float32", "--chunk", "0")
runner("fp32 resident vs R0_inc (released)", P["R0_inc"], "--dtype", "float32", "--chunk", "0")
runner("fp16 resident vs R1 (one pass)", P["R1"], "--dtype", "float16", "--chunk", "0")
runner("int8 resident vs R2 (one pass)", P["R2"], "--chunk", "0")
runner("int8 resident vs R0 (released, fp32)", P["R0"], "--chunk", "0")
runner("fp16 resident vs R0 (released, fp32)", P["R0"], "--dtype", "float16", "--chunk", "0")



def direct(x, y, label):
    common = sorted(set(x) & set(y))
    X, Y = np.stack([x[p] for p in common]), np.stack([y[p] for p in common])
    print(f"  {label:66s} positions {common[0]}..{common[-1]} ({len(common)}): same argmax {int((X.argmax(1) == Y.argmax(1)).sum())}/{len(common)}, max |diff| over the vocabulary {np.abs(X - Y).max():.5f}")


alt = ids[:12] + [int(t) for t in rng.integers(10, V, 12)]
snap = os.path.join(args.work, "snapshots")
for dtype, R, tbt in (("float32", "R1f", tbt32), ("int8", "R2", tbt8)):
    D = ["--dtype", dtype]
    print(f"\n== {dtype} resident: batched prefill (--chunk 8), turns, snapshots; against {R}_inc and against the runner's own token-by-token run")
    P2 = save_ref(os.path.join(args.work, f"ref_{R}_turns.pt"), refs[R + "_inc"], n_prompt=24, cuts=[12])
    P3 = save_ref(os.path.join(args.work, f"ref_{R}_onebatch.pt"), refs[R + "_inc"], n_prompt=40)
    b1 = runner(f"{dtype}: turns 12+12, each a batch", P2, *D, "--chunk", "8")
    b2 = runner(f"{dtype}: chunk-max 8, keep 0, attn-block 4", P2, *D, "--chunk", "8", "--chunk-max", "8", "--keep", "0", "--attn-block", "4")
    t1 = runner(f"{dtype}: turns 12+12, token by token", P2, *D, "--chunk", "0")
    b3 = runner(f"{dtype}: one batch of 40", P3, *D, "--chunk", "8")
    b4 = runner(f"{dtype}: one batch of 40, every expert streamed", P3, *D, "--chunk", "8", "--stream-from", "8")
    P4 = save_ref(os.path.join(args.work, f"ref_{R}_hot.pt"), refs[R + "_inc"], n_prompt=24, feeds=[ids[:24], alt, ids[:24]], marks=[12])
    P5 = save_ref(os.path.join(args.work, f"ref_{R}_disk.pt"), refs[R + "_inc"], n_prompt=24, feeds=[ids[:24]], marks=[12])
    for f in glob.glob(os.path.join(snap, "*")):
        os.remove(f)
    h1 = runner(f"{dtype}: 24 (snapshot at 12), diverge, back: hot", P4, *D, "--chunk", "0", "--prompt-cache", snap)
    h2 = runner(f"{dtype}: new process, snapshot at 12 from disk", P5, *D, "--chunk", "0", "--prompt-cache", snap)
    for f in glob.glob(os.path.join(snap, "*")):
        os.remove(f)
    h3 = runner(f"{dtype}: same hot test, batched", P4, *D, "--chunk", "8", "--prompt-cache", snap)
    h4 = runner(f"{dtype}: same disk test, batched", P5, *D, "--chunk", "8", "--prompt-cache", snap)
    print(f"  -- the runner against its own token-by-token run from one cold feed (whole vocabulary):")
    direct(b1, tbt, "turns 12+12, each a batch")
    direct(b2, tbt, "chunk-max 8 / keep 0 / attn-block 4")
    direct(b3, tbt, "one batch of 40 (position 39)")
    direct(b4, tbt, "one batch of 40, streamed (position 39)")
    direct(t1, tbt, "turns 12+12 token by token (live prefix, n-gram context carried)")
    direct(h1, tbt, "hot snapshot restore after a divergent turn")
    direct(h2, tbt, "disk snapshot restore in a new process")
    direct(h3, b1, "hot snapshot restore, batched, vs turns 12+12 batched")
    direct(h4, b1, "disk snapshot restore, batched, vs turns 12+12 batched")

print("\n== the int8 one-scale-per-neuron store (store/meta.json dtype int8), fp32 resident, against R8f_inc")
r8a = runner("int8 records: token by token", P["R8f_inc"], "--dtype", "float32", "--chunk", "0", "--dir", STORE8)
P8 = save_ref(os.path.join(args.work, "ref_R8f_turns.pt"), refs["R8f_inc"], n_prompt=24, cuts=[12])
r8b = runner("int8 records: turns 12+12, each a batch", P8, "--dtype", "float32", "--chunk", "8", "--dir", STORE8)
r8c = runner("int8 records: batched, keep 0, streamed", P8, "--dtype", "float32", "--chunk", "8", "--keep", "0", "--stream-from", "8", "--dir", STORE8)
direct(r8b, r8a, "int8 records: batched vs token by token")
direct(r8c, r8a, "int8 records: batched, keep 0, streamed vs token by token")
