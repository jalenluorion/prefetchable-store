"""moved-store.json for a tiny random model folder (tests/*/make_tiny.py), from its family's published manifest.

  python tests/tiny_manifest.py MODEL_DIR [--derived-sha FILE]      (python: the standard library only)

The family's manifest (python/convert/<family>/moved-store.json, the weight repo's as published) gives the declared
properties a run resolves from (`modes`, `profiles`, `defaults`, the in-memory rule `resident.q8_rule`, `layout`); this
script replaces what depends on the folder: the parts (every file, its size and sha256), the geometry (`model`,
`experts` from config.json and store/meta.json), the routers the folder holds, `extras` and `derived`. A profile's own
record files (Inkling's phone-small NVFP4 layers) and its part lists are left out: a tiny folder has neither.

--derived-sha FILE: a `sha256sum` list of the engine-ready files (`ms_derive` output in MODEL_DIR/.dmeng/), written into
`derived[].sha256` so the engine checks each file it uses against it. Without it the hashes are left null (the
engine then derives without a check), as they are before the first derivation.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("dir")
ap.add_argument("--derived-sha", default=None)
a = ap.parse_args()
D = os.path.abspath(a.dir)
cfg_full = json.load(open(os.path.join(D, "config.json")))
fam = cfg_full["model_type"]
tc = cfg_full.get("text_config", cfg_full)
meta = json.load(open(os.path.join(D, "store", "meta.json")))
m = json.load(open(os.path.join(REPO, "python", "convert", fam, "moved-store.json")))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


ROLE = {"config.json": "config", "generation_config.json": "config", "hf_quant_config.json": "config", "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer", "vocab.json": "tokenizer", "merges.txt": "tokenizer", "chat_template.jinja": "chat_template",
        "store/meta.json": "meta", "ngram/meta.json": "meta", "resident.safetensors": "resident", "mtp.safetensors": "extra",
        "vision.safetensors": "legacy", "sha256.txt": "checksums"}
parts = []
for root, dirs, files in os.walk(D):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "derived"]
    for f in sorted(files):
        rel = os.path.relpath(os.path.join(root, f), D)
        if rel == "moved-store.json" or f.startswith(("_", ".")):
            continue
        role = ROLE.get(rel) or ("experts" if rel.startswith("store/layer_") else "extra" if rel.startswith("ngram/part_")
                                 else "router" if rel.startswith("routers") and rel.endswith((".safetensors", ".npz")) else "legacy")
        prec = {"experts": None, "resident": "bf16", "router": "f16+f32"}.get(role, "text")
        parts.append({"path": rel, "bytes": os.path.getsize(os.path.join(D, rel)), "sha256": sha(os.path.join(D, rel)), "role": role, "precision": prec})
parts.sort(key=lambda p: p["path"])
NL = tc["num_hidden_layers"]
known = {}
if a.derived_sha:
    for line in open(a.derived_sha):
        p = line.split()
        if len(p) == 2:
            known[os.path.basename(p[1])] = p[0]

# ---------------------------------------------------------------- geometry: config.json and store/meta.json
if fam in ("qwen3_5_moe", "qwen4_exp"):
    letter = "F" if fam == "qwen3_5_moe" else "Q"
    types = "".join("L" if t == "linear_attention" else letter for t in tc["layer_types"])
    m["model"].update({"hidden": tc["hidden_size"], "vocab": tc["vocab_size"], "physical_layers": NL, "steps": NL, "step_layer": None,
                       "step_pass": None, "layer_types": types})
    off, soff, boff, shp = meta["offsets"], meta["scale_offsets"], meta["bias_offsets"], meta["shapes"]
    f = m["experts"]["formats"]["aff4_g64"]
    f.update({"record_bytes": meta["record_bytes"], "record_bytes_used": meta.get("record_bytes_used", meta["record_bytes"]),
              "matrices": [{"name": n, "shape": shp[k], "codes": off[k], "scales": soff[k], "biases": boff[k]}
                           for n, k in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj"))]})
    m["experts"].update({"experts_per_layer": tc["num_experts"], "top_k": tc["num_experts_per_tok"], "page": meta.get("page", 16384)})
    for p in parts:
        if p["role"] == "experts":
            p["precision"] = "aff4_g64"
elif fam == "inkling2_loop":
    lc = tc["loop_config"]
    cs, ce, npass = lc["core_start"], lc["core_end"], lc["num_passes"]
    step_layer = list(range(cs)) + list(range(cs, ce)) * npass + list(range(ce, NL))
    step_pass = [-1] * cs + [p for p in range(npass) for _ in range(cs, ce)] + [-1] * (NL - ce)
    glob = [L for L in range(NL) if L not in tc["local_layer_ids"]]
    m["model"].update({"hidden": tc["hidden_size"], "vocab": tc["vocab_size"], "vocab_unpadded": tc["unpadded_vocab_size"],
                       "physical_layers": NL, "steps": len(step_layer), "step_layer": step_layer, "step_pass": step_pass,
                       "layer_types": "".join("G" if L in glob else "S" for L in range(NL))})
    ids = {"nvfp4": "nvfp4_g16", "mxfp8": "mxfp8_g32"}
    fmts = {}
    for kind, fm in meta["formats"].items():
        fl = fm["fields"]
        e = dict(m["experts"]["formats"][ids[kind]])
        e.update({"record_bytes": fm["record_bytes"], "record_bytes_used": fm["record_bytes_used"],
                  "matrices": [{"name": "gate_up", "shape": fm["matrices"]["gate_up"], "interleave": "gate_up_rows",
                                "codes": fl["gate_up.weight"]["offset"], "scales": fl["gate_up.scales"]["offset"]},
                               {"name": "down", "shape": fm["matrices"]["down"], "codes": fl["down.weight"]["offset"], "scales": fl["down.scales"]["offset"]}]})
        if "trailer" in fl:
            e["trailer"] = dict(e["trailer"], offset=fl["trailer"]["offset"])
        fmts[ids[kind]] = e
    m["experts"].update({"formats": fmts, "layer_format": [ids[k] for k in meta["layer_format"]], "experts_per_layer": tc["n_routed_experts"],
                         "top_k": tc["num_experts_per_tok"], "route_scale": tc["route_scale"], "page": meta.get("page", 16384),
                         "activation_caps": {"gate": tc["mlp_gate_softcap"], "up": tc["mlp_up_softcap"]}})
    m["experts"]["router_rule"] = dict(m["experts"].get("router_rule", {}), route_scale=tc["route_scale"])
    for p in parts:
        if p["role"] == "experts":
            p["precision"] = ids[meta["layer_format"][int(p["path"][12:14])]]
else:
    sys.exit(f"{fam}: no tiny manifest rule")
emb_tensor = m["model"]["embedding"]["tensor"]
m["model"]["embedding"] = {"file": "resident.safetensors", "tensor": emb_tensor, "dtype": "bf16"}

# ---------------------------------------------------------------- routers the folder holds (routers/<id>.safetensors|.npz)
rt = []
for p in parts:
    if p["role"] != "router":
        continue
    old = next((r for r in m["routers"] if r["path"] == p["path"]), None)  # the published entry of this file, else a new id
    rid = old["id"] if old else os.path.splitext(os.path.basename(p["path"]))[0]
    old = old or next((r for r in m["routers"] if r["id"] == rid), None) or (m["routers"][0] if m["routers"] else {})
    e = dict(old, id=rid, path=p["path"], default=rid in (m["defaults"].get("routers") or []), targets=m["model"]["steps"])
    e.pop("training", None)
    rt.append(e)
m["routers"] = rt
m.pop("cross_token_routers", None)  # a tiny folder has none (the engine notes "no file found, left out")
for p in m["profiles"].values():
    for k in ("parts", "bytes", "bytes_rule", "optional", "bytes_with_optional", "optional_note", "experts", "status"):
        p.pop(k, None)

# ---------------------------------------------------------------- extras
if fam == "qwen4_exp":
    ngm = json.load(open(os.path.join(D, "ngram", "meta.json")))
    ple = [int(x) - 1 for x in tc.get("ple_layer_ids", [])]
    m["extras"]["ngram"].update({"files": "ngram/part_{P:02d}.bin", "parts": ngm["parts"], "rows": ngm["rows"], "rows_per_part": ngm["rows_per_part"],
                                 "dim": ngm["dim"], "bytes_per_row": ngm["bytes_per_row"],
                                 "rows_per_token": (tc.get("ngram_size", 3) - 1) * tc.get("heads_per_ngram", 8), "layer": ple[0]})
    for p in parts:
        if p["role"] == "extra":
            p["precision"] = "int8_row_ngram"
if fam == "inkling2_loop" and not os.path.exists(os.path.join(D, "mtp.safetensors")):
    m["extras"]["draft_head"] = None

# ---------------------------------------------------------------- engine-ready files: what the engine derives, and from what
src = {p["path"]: p["sha256"] for p in parts}
derived = []
for d in m["derived"]:
    if d["id"] == "resident_q8":
        derived.append({k: d[k] for k in ("id", "from", "engine_format") if k in d})
for r in rt:
    derived.append({"id": f"router.{r['id']}.q8", "from": [r["path"]], "engine_format": "dmr1"})
for d in derived:
    name = f"{d['id']}.{d['engine_format']}.{src[d['from'][0]][:12]}.bin"
    d["sha256"] = known.get(name)
    d["made_by"] = "ms_derive (the engine's own derivation) from " + d["from"][0]
m["derived"] = derived
m.update({"name": f"tiny-{fam}", "source": {"repo": "synthetic", "revision": None, "license": m["source"].get("license"), "license_file": None},
          "converter": {"code": f"tests/{fam}/make_tiny.py", "commit": None, "sha256": None, "date": None}, "complete": True, "parts": parts,
          "legacy": [p["path"] for p in parts if p["role"] == "legacy"], "pending": [], "unknown": [], "notes": [],
          "generated": {"by": "tests/tiny_manifest.py", "date": None}})
json.dump(m, open(os.path.join(D, "moved-store.json"), "w"), indent=1)
print(f"{D}/moved-store.json: {fam}, {len(parts)} parts, {len(rt)} router(s), derived sha256 {'set' if known else 'unset'}")
