"""Inkling 2 Mini release -> expert store + resident file (lossless: every byte is copied from the release).

  .venv/bin/python convert.py [--release release] [--out model]

Reads release/model-00001-of-00001.safetensors through its header and os.pread (never loads a tensor whole; at most a
few records in memory, F_NOCACHE on the big files so the conversion does not fill the page cache) and writes, in --out:

  store/layer_XX.bin     24 files x 128 records, record = one expert as released (layout in store.py and meta.json):
                         nvfp4 (layers 0-19): 4,489,216 B (274 pages, 4,478,992 used); mxfp8 (20-23): 8,224,768 B (502 pages).
  store/meta.json        the layout (per-layer format, offsets, shapes), model facts the runner needs, sizes.
  resident.safetensors   every text tensor that is not a routed-expert matrix or scale, in the release's names, dtypes and
                         bytes (bf16 / f32), the per-expert amax and outer_scale_denominator included (they are also in
                         each nvfp4 record's trailer). Load it through the PR's Model.sanitize like the release.
  vision.safetensors     the vision tower (model.visual.*), as released, kept out of the resident.
  mtp.safetensors, config.json, generation_config.json, hf_quant_config.json, tokenizer files, chat_template.jinja,
  processor_config.json, tiktoken/tokenizer.model, README.md (copied), convert.py, store.py, check.py, sha256.txt.

Resumable: model/_convert_state.json records each finished file with its sha256; a layer is written to .part and renamed
when complete, so a killed run redoes at most the layer in progress. Runs check.py at the end (exit 3 if it fails).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

import store as S

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--release", default=os.path.join(HERE, "release"))
ap.add_argument("--out", default=os.path.join(HERE, "model"))
ap.add_argument("--batch", type=int, default=4, help="records per read/write batch")
ap.add_argument("--no-check", action="store_true")
a = ap.parse_args()
t0 = time.time()


def log(*x):
    print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)


REL = os.path.join(a.release, "model-00001-of-00001.safetensors")
OUT, STORE = a.out, os.path.join(a.out, "store")
os.makedirs(STORE, exist_ok=True)
hdr, hmeta, DATA = S.read_header(REL)
assert os.path.getsize(REL) == DATA + max(v["data_offsets"][1] for v in hdr.values()), "release file size does not match its header"
full_cfg = json.load(open(os.path.join(a.release, "config.json")))
assert full_cfg["model_type"] == "inkling2_loop", full_cfg["model_type"]
cfg = full_cfg["text_config"]
H, I, NL, NE, K = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["num_experts_per_tok"]
assert cfg.get("n_shared_experts", 0) == 0

# --- the expert tensors: names, dtypes and shapes as expected; everything else is resident (or vision)
EXP = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight(_scale)?$")
kinds = [S.layer_format(hdr, L) for L in range(NL)]
FMT = {k: S.make_format(k, H, I) for k in sorted(set(kinds))}
for L in range(NL):
    fmt, names = FMT[kinds[L]], S.release_names(L)
    for f in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales"):
        info, want = hdr[names[f]], fmt["fields"][f]
        assert info["dtype"] == want["release_dtype"] and info["shape"] == [NE] + want["shape"], (L, f, info, want)
    if fmt["global_scale"]:
        for f in ("gate_up.amax", "down.amax"):
            assert hdr[names[f]]["dtype"] == "F32" and hdr[names[f]]["shape"] == [NE], (L, f)
        for f in ("gate_up.denominator", "down.denominator"):
            assert hdr[names[f]]["dtype"] == "F32" and hdr[names[f]]["shape"] == [1], (L, f)
    else:
        assert names["gate_up.amax"] not in hdr and names["gate_up.denominator"] not in hdr, L
qcfg = json.load(open(os.path.join(a.release, "hf_quant_config.json")))
for g in qcfg["model"]["parameter_groups"]:  # cross-check the per-layer kind against the vendor's quant config
    if g["compute_dtype"] in ("nvfp4", "mxfp8"):
        for t in g["tensors"]:
            m = EXP.fullmatch(t)
            assert m and kinds[int(m.group(1))] == g["compute_dtype"], (t, g["compute_dtype"])

expert_keys = {S.release_names(L)[f] for L in range(NL) for f in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales")}
vision_keys = sorted(k for k in hdr if k.startswith("model.visual."))
resident_keys = sorted(k for k in hdr if k not in expert_keys and not k.startswith("model.visual."))
assert not [k for k in resident_keys if ".experts." in k and not k.endswith(("_amax", "_outer_scale_denominator"))]

REC = [FMT[k]["record_bytes"] for k in kinds]
meta = {
    "repo": "thinkingmachines/Inkling-2-Mini", "release_file": os.path.basename(REL), "release_bytes": os.path.getsize(REL),
    "model_type": full_cfg["model_type"], "hidden": H, "intermediate": I, "layers": NL, "experts": NE, "top_k": K,
    "norm_topk_prob": True,
    "router": {"logits": "x.float32 @ mlp.gate.weight.float32.T", "scores": "sigmoid(logits)",
               "select": "top_k of scores + mlp.gate.bias (e_score_correction_bias)",
               "weights": "unbiased scores of the selected / (sum + 1e-20) * route_scale", "route_scale": cfg["route_scale"]},
    "expert_act": {"form": "softcap(gate, g) * sigmoid(gate) * softcap(up, u), softcap(x, c) = c * tanh(x / c), in float32",
                   "gate_softcap": cfg["mlp_gate_softcap"], "up_softcap": cfg["mlp_up_softcap"]},
    "gate_up_rows": "interleaved: row 2j = gate j, row 2j + 1 = up j",
    "loop_config": cfg.get("loop_config"), "local_layer_ids": cfg["local_layer_ids"],
    "layer_types": ["local" if L in cfg["local_layer_ids"] else "global" for L in range(NL)],
    "layer_format": kinds, "formats": FMT, "record_bytes": REC, "page": S.PAGE, "align": S.ALIGN,
    "layer_file": "store/layer_{L:02d}.bin", "record_order": ["gate_up_proj", "down_proj"],
    "mlx_names": {"gate_up.weight": "model.language_model.layers.{L}.mlp.experts.gate_up_proj.weight",
                  "gate_up.scales": "model.language_model.layers.{L}.mlp.experts.gate_up_proj.scales",
                  "gate_up.global_scale": "model.language_model.layers.{L}.mlp.experts.gate_up_proj.global_scale",
                  "down.weight": "model.language_model.layers.{L}.mlp.experts.down_proj.weight",
                  "down.scales": "model.language_model.layers.{L}.mlp.experts.down_proj.scales",
                  "down.global_scale": "model.language_model.layers.{L}.mlp.experts.down_proj.global_scale"},
    "global_scale_rule": "nvfp4: global_scale = amax * (6 * 448 / outer_scale_denominator) (float32, per expert); "
                         "dequant w = e2m1 * e4m3 * global_scale / (6 * 448)",
    "store_bytes": int(sum(REC) * NE),
    "expert_bytes_used": {k: FMT[k]["record_bytes_used"] for k in FMT},
    "expert_params": NL * NE * 3 * H * I,
    "resident_file": "resident.safetensors", "resident_tensors": len(resident_keys),
    "resident_bytes": int(sum(S.nbytes(hdr[k]) for k in resident_keys)),
    "resident_params": int(sum(np.prod(hdr[k]["shape"]) for k in resident_keys)),
    "vision_file": "vision.safetensors", "vision_bytes": int(sum(S.nbytes(hdr[k]) for k in vision_keys)),
}

state_path = os.path.join(OUT, "_convert_state.json")
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
state.setdefault("sha", {})


def save_state():
    tmp = state_path + ".tmp"
    json.dump(state, open(tmp, "w"), indent=0)
    os.replace(tmp, state_path)


def done(rel, size=None):
    p = os.path.join(OUT, rel)
    return rel in state["sha"] and os.path.exists(p) and (size is None or os.path.getsize(p) == size)


log(f"release {REL}: {len(hdr)} tensors, data at {DATA} | formats " +
    ", ".join(f"{k}: record {f['record_bytes']} B ({f['record_bytes_used']} used)" for k, f in FMT.items()) +
    f" | store {meta['store_bytes'] / 2**30:.2f} GiB | resident {len(resident_keys)} tensors {meta['resident_bytes'] / 2**30:.3f} GiB")

src = os.open(REL, os.O_RDONLY)
S.nocache(src)

# --- 1. the store, layer by layer
for L in range(NL):
    rel = f"store/layer_{L:02d}.bin"
    fmt, names, R = FMT[kinds[L]], S.release_names(L), REC[L]
    if done(rel, NE * R):
        continue
    t1 = time.time()
    part = os.path.join(OUT, rel + ".part")
    fo = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    S.nocache(fo)
    h = hashlib.sha256()
    base = {f: DATA + hdr[names[f]]["data_offsets"][0] for f in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales")}
    if fmt["global_scale"]:
        tr = np.stack([np.frombuffer(os.pread(src, nb, DATA + hdr[names[f]]["data_offsets"][0]), dtype="<f4")
                       for f, nb in (("gate_up.amax", 4 * NE), ("down.amax", 4 * NE))])  # [2, NE]
        den = [np.frombuffer(os.pread(src, 4, DATA + hdr[names[f]]["data_offsets"][0]), dtype="<f4")[0]
               for f in ("gate_up.denominator", "down.denominator")]
    for e0 in range(0, NE, a.batch):
        n = min(a.batch, NE - e0)
        buf = np.zeros((n, R), dtype=np.uint8)
        for f in ("gate_up.weight", "gate_up.scales", "down.weight", "down.scales"):
            fl = fmt["fields"][f]
            b = os.pread(src, n * fl["bytes"], base[f] + e0 * fl["bytes"])  # experts are contiguous in the release
            assert len(b) == n * fl["bytes"], (L, f)
            buf[:, fl["offset"]:fl["offset"] + fl["bytes"]] = np.frombuffer(b, dtype=np.uint8).reshape(n, fl["bytes"])
        if fmt["global_scale"]:
            fl = fmt["fields"]["trailer"]
            t = np.zeros((n, 4), dtype="<f4")
            t[:, 0], t[:, 1], t[:, 2], t[:, 3] = tr[0, e0:e0 + n], tr[1, e0:e0 + n], den[0], den[1]
            buf[:, fl["offset"]:fl["offset"] + 16] = t.view(np.uint8)
        mv = memoryview(buf).cast("B")
        w = 0
        while w < len(mv):
            w += os.pwrite(fo, mv[w:], e0 * R + w)
        h.update(mv)
    os.fsync(fo)
    os.close(fo)
    assert os.path.getsize(part) == NE * R
    os.replace(part, os.path.join(OUT, rel))
    state["sha"][rel] = h.hexdigest()
    save_state()
    log(f"{rel} {kinds[L]} {NE} x {R} B in {time.time() - t1:.1f}s")

# --- 2. resident and vision, streamed in the release's order of offsets
for rel, keys, md in (("resident.safetensors", resident_keys, {"format": "pt", "source": "Inkling-2-Mini release, non-expert text tensors"}),
                      ("vision.safetensors", vision_keys, {"format": "pt", "source": "Inkling-2-Mini release, vision tower"})):
    if done(rel):
        continue
    t1 = time.time()
    keys = sorted(keys, key=lambda k: hdr[k]["data_offsets"][0])
    ent = [(k, hdr[k]["dtype"], hdr[k]["shape"], S.nbytes(hdr[k]), (src, DATA + hdr[k]["data_offsets"][0])) for k in keys]
    state["sha"][rel] = S.write_safetensors(os.path.join(OUT, rel), ent, metadata=md, no_cache=True)
    save_state()
    log(f"{rel}: {len(keys)} tensors, {os.path.getsize(os.path.join(OUT, rel)) / 2**30:.3f} GiB in {time.time() - t1:.1f}s")
os.close(src)

# --- 3. small files and the draft head, copied
COPY = ["mtp.safetensors", "config.json", "generation_config.json", "hf_quant_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "processor_config.json",
        "tiktoken/tokenizer.model", "README.md"]
for f in COPY:
    s = os.path.join(a.release, f)
    if not os.path.exists(s):
        log("not in release:", f)
        continue
    if done(f, os.path.getsize(s)):
        continue
    os.makedirs(os.path.dirname(os.path.join(OUT, f)), exist_ok=True)
    shutil.copyfile(s, os.path.join(OUT, f))
    h = hashlib.sha256()
    with open(os.path.join(OUT, f), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    state["sha"][f] = h.hexdigest()
    save_state()
for f in ("convert.py", "store.py", "check.py"):
    shutil.copyfile(os.path.join(HERE, f), os.path.join(OUT, f))
    state["sha"][f] = hashlib.sha256(open(os.path.join(OUT, f), "rb").read()).hexdigest()

meta_s = json.dumps(meta, indent=1)
open(os.path.join(STORE, "meta.json"), "w").write(meta_s)
state["sha"]["store/meta.json"] = hashlib.sha256(meta_s.encode()).hexdigest()
save_state()

# --- 4. gate: check.py on the finished store
if not a.no_check:
    r = subprocess.run([sys.executable, os.path.join(HERE, "check.py"), "--release", a.release, "--model", OUT],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    open(os.path.join(OUT, "check_output.txt"), "w").write(text)
    print(text, flush=True)
    if r.returncode != 0:
        log("CHECK_FAILED")
        sys.exit(3)
    state["sha"]["check_output.txt"] = hashlib.sha256(text.encode()).hexdigest()
with open(os.path.join(OUT, "sha256.txt"), "w") as fh:
    for k in sorted(state["sha"]):
        fh.write(f"{state['sha'][k]}  {k}\n")
save_state()
log(f"CONVERT_DONE store {meta['store_bytes']} B ({meta['store_bytes'] / 2**30:.2f} GiB), resident {meta['resident_bytes']} B "
    f"({meta['resident_params'] / 1e9:.3f}B params), vision {meta['vision_bytes']} B -> {OUT}")
