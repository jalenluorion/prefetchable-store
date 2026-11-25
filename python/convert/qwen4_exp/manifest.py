"""moved-store.json for a Qwen3.8-Flash-Next folder in the store layout (reference/flash/convert.py's output), in the schema of
the other two weight repos (docs/MANIFEST-PRESETS.md), with one profile: mac.

  python python/convert/qwen4_exp/manifest.py DIR [--out FILE] [--sha256 FILE] [--size PATH=BYTES ...] [--name NAME]

DIR holds at least config.json, store/meta.json and ngram/meta.json. A part present in DIR is listed with its size and
sha256 (computed, or taken from --sha256, a `sha256sum` list such as the repo's sha256.txt); a part that is not present
(a folder holding only the small files) takes its sha256 from --sha256 and its size from the layout (store layers:
experts x record bytes; n-gram parts: rows x bytes per row) or from --size. Engine-ready files in DIR/derived/ and an
embeddings.safetensors are listed when present, and the mac profile then lists them instead of the sources they come
from (resident.safetensors, routers_k4.npz), so a Mac never holds the 9.9 GB bf16 resident.

Nothing is read but the small JSON files and the parts' bytes for their hashes (the trainer's npz is not opened: the
router's hidden width is read from it when the engine derives its file).
"""
import argparse, hashlib, json, os, sys

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("dir")
ap.add_argument("--out", default=None, help="default: DIR/moved-store.json")
ap.add_argument("--sha256", default=None, help="a sha256sum list (sha256.txt) for parts not hashed here")
ap.add_argument("--size", action="append", default=[], help="PATH=BYTES for a part that is not in DIR")
ap.add_argument("--name", default="Qwen3.8-Flash-Next-prefetchable-store")
ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
a = ap.parse_args()
D = os.path.abspath(a.dir)
cfg_full = json.load(open(os.path.join(D, "config.json")))
tc = cfg_full.get("text_config", cfg_full)
meta = json.load(open(os.path.join(D, "store", "meta.json")))
ngm = json.load(open(os.path.join(D, "ngram", "meta.json")))
known = {}
if a.sha256:
    for line in open(a.sha256):
        p = line.split()
        if len(p) == 2:
            known[p[1]] = p[0]
sizes = {}
for s in a.size:
    k, v = s.split("=")
    sizes[k] = int(v)

H, NL, NE, K = tc["hidden_size"], tc["num_hidden_layers"], tc["num_experts"], tc["num_experts_per_tok"]
assert meta["hidden"] == H and meta["layers"] == NL and meta["experts"] == NE and meta["top_k"] == K, "store/meta.json is not this config's"
assert meta["dtype"] == "int4" and meta["group"] == 64 and meta["record_order"] == ["gate_proj", "up_proj", "down_proj"]
REC = meta["record_bytes"]
for f in ngm["files"]:
    sizes.setdefault(f, ngm["part_rows"][ngm["files"].index(f)] * ngm["bytes_per_row"])
for L in range(NL):
    sizes.setdefault(f"store/layer_{L:02d}.bin", NE * REC)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


ROLE = {"config.json": "config", "generation_config.json": "config", "tokenizer.json": "tokenizer", "tokenizer_config.json": "tokenizer",
        "vocab.json": "tokenizer", "merges.txt": "tokenizer", "chat_template.jinja": "chat_template", "LICENSE": "license", "README.md": "card",
        "store/meta.json": "meta", "ngram/meta.json": "meta", "resident.safetensors": "resident", "routers_k4.npz": "router",
        "embeddings.safetensors": "embedding", "sha256.txt": "checksums", "convert.py": "legacy", "check.py": "legacy",
        "check_output.txt": "legacy", "ref_released.json": "legacy"}


def part(path, role=None, precision="text", optional=False, **extra):
    full = os.path.join(D, path)
    if os.path.exists(full):
        n, h = os.path.getsize(full), known.get(path) or sha(full)
    else:
        n, h = sizes.get(path), known.get(path)
        if (n is None and h is None) or (optional and (n is None or h is None)):
            return None  # not here and nothing known of it (an optional part is listed only when both are known)
    p = {"path": path, "bytes": n, "sha256": h, "role": role or ROLE.get(path, "other"), "precision": precision}
    if optional:
        p["optional"] = True
    p.update(extra)
    return p


parts = []
for f in ["LICENSE", "README.md", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
          "tokenizer_config.json", "vocab.json", "store/meta.json", "ngram/meta.json"]:
    q = part(f)
    if q:
        parts.append(q)
parts.append(part("resident.safetensors", precision="bf16"))
q = part("routers_k4.npz", precision="f32", note="the lag-4 router set as the trainer exported it (flash_router3; keys L.emb_norm.weight, "
         "L.lin.weight, L.lin.bias, L.mlp.0.weight, L.mlp.0.bias, L.mlp.2.weight, L.mlp.2.bias, __lag__); the engine derives its "
         "8-bit file from it in place")
if q:
    parts.append(q)
for L in range(NL):
    parts.append(part(f"store/layer_{L:02d}.bin", role="experts", precision="aff4_g64"))
for f in ngm["files"]:
    parts.append(part(f, role="extra", precision="int8_row_ngram"))
for f in ["convert.py", "check.py", "check_output.txt", "ref_released.json"]:
    q = part(f, optional=True)
    if q:
        parts.append(q)
missing = [p["path"] for p in parts if p["bytes"] is None or p["sha256"] is None]
if missing:
    sys.exit(f"no size or sha256 for {missing}: pass --size / --sha256")


def src_sha(path):
    return next(p["sha256"] for p in parts if p["path"] == path)


# engine-ready files the folder already holds (made elsewhere, e.g. by ms_derive on another machine)
derived = [{"id": "resident_q8", "from": ["resident.safetensors"], "engine_format": "dmres1", "sha256": None,
            "note": "made by the engine (or ms_derive resident DIR) from resident.safetensors on first use; no full-size reference file exists yet"}]
if any(p["path"] == "routers_k4.npz" for p in parts):
    derived.append({"id": "router.mlp_k4.q8", "from": ["routers_k4.npz"], "engine_format": "dmr1", "sha256": None,
                    "note": "made by the engine (or ms_derive router DIR mlp_k4) from routers_k4.npz on first use"})
shipped = []
for d in derived:
    name = f"derived/{d['id']}.{d['engine_format']}.{src_sha(d['from'][0])[:12]}.bin"
    q = part(name, role="derived", precision=d["engine_format"], optional=True, derived_id=d["id"])
    if q:
        parts.append(q)
        d["path"], d["bytes"], d["sha256"] = name, q["bytes"], q["sha256"]
        d.pop("note")
        shipped.append(d["from"][0])
emb = part("embeddings.safetensors", precision="bf16", optional=True, tensor="model.language_model.embed_tokens.weight",
           check="its data region equals the bytes of model.language_model.embed_tokens.weight in resident.safetensors")
if emb:
    parts.append(emb)
    # an engine-ready file too: listed under derived with its sha256, so the engine checks it when it reads the rows from it
    derived.append({"id": "embeddings", "from": ["resident.safetensors"], "engine_format": "safetensors", "path": "embeddings.safetensors",
                    "bytes": emb["bytes"], "sha256": emb["sha256"],
                    "made_by": "the embed_tokens table of resident.safetensors as its own safetensors file (the same bytes)"})

types = "".join("L" if t == "linear_attention" else "Q" for t in tc["layer_types"])
off, soff, boff, shp = meta["offsets"], meta["scale_offsets"], meta["bias_offsets"], meta["shapes"]
ple = [int(x) - 1 for x in tc.get("ple_layer_ids", [])]
assert len(ple) == 1, "one PLE layer expected"
# fast mode and the names, measured on the base M4 (docs/MANIFEST-PRESETS.md): with 10 picks the routing weights are
# about 0.1 each, so a per-pick threshold of 0.1 left out a third of the weight; instead the heaviest missing picks are read
# until what is left out weighs <= 0.13 (mean 0.087 left out); the routers name a top 4 (fewer wasted reads than 10, same
# speed); one slot pool (per-layer caps refuse names below 2 x 10 slots per layer)
FAST_DEF = ("The model's own router still picks. The picked experts that are not in memory are read, heaviest first, until "
            "those left weigh at most 0.13 of the step's routing weight; those are left out (no substitute, no renormalisation). "
            "A light pick whose read is in flight is not waited for. Router names are on.")
NAMES_M = 4
m = {
    "format": "moved-store", "format_version": "1.0", "family": "qwen4_exp", "name": a.name,
    "source": {"repo": a.repo, "revision": None, "license": "qwen-community-1.0", "license_file": "LICENSE"},
    "converter": {"code": "reference/flash/convert.py (the store, the n-gram parts and resident.safetensors as published)", "commit": None,
                  "sha256": known.get("convert.py"), "date": None},
    "complete": True,
    "parts": parts,
    "model": {"hidden": H, "vocab": tc["vocab_size"], "physical_layers": NL, "steps": NL, "step_layer": None, "step_pass": None,
              "layer_types": types, "layer_type_letters": {"L": "Gated DeltaNet (linear attention)", "Q": "sparse attention (QSA: dense up to the indexer budget, then the indexer's blocks)"},
              "hyper_connections": tc.get("hc_count", 4),
              "embedding": {"file": "resident.safetensors", "tensor": "model.language_model.embed_tokens.weight", "dtype": "bf16"}},
    "experts": {"layer_file": "store/layer_{L:02d}.bin", "experts_per_layer": NE, "top_k": K, "router": "softmax_topk_renorm", "route_scale": 1.0,
                "shared_expert": True, "activation": "swiglu", "page": meta.get("page", 16384), "layer_format": "aff4_g64",
                "formats": {"aff4_g64": {"kind": "affine", "bits": 4, "group": 64, "record_bytes": REC, "record_bytes_used": REC,
                                         "matrices": [{"name": n, "shape": shp[k], "codes": off[k], "scales": soff[k], "biases": boff[k]}
                                                      for n, k in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj"))],
                                         "code_packing": "two per byte, element 2j in the low nibble of byte j", "scale_dtype": "f16", "bias_dtype": "f16",
                                         "dequant": "w = code * scale + bias per group of 64 along the input"}}},
    "resident": {"file": "resident.safetensors", "precision": "bf16",
                 "q8_rule": {"matrices": "every 2-D linear with >= 64 outputs and inputs % 64 == 0, except the released router (reference/flash/mlx_run.py qpred)",
                             "quantiser": "mlx_affine", "bits": 8, "group": 64, "input_cast": "f16", "scale_dtype": "f16",
                             "keep_exact": ["mlp.gate", "mlp.shared_expert_gate", "block_inject_weight", "linear_attn.in_proj_a", "linear_attn.in_proj_b"],
                             "embedding": "not copied: one bf16 row per token is read from resident.safetensors (or embeddings.safetensors)"}},
    "routers": [],
    "extras": {"ngram": {"required": True, "files": "ngram/part_{P:02d}.bin", "parts": ngm["parts"], "rows": ngm["rows"], "rows_per_part": ngm["rows_per_part"],
                         "dim": ngm["dim"], "bytes_per_row": ngm["bytes_per_row"], "precision": "int8_row_ngram",
                         "layout": "per row: f16 scale (2 B) | int8 codes [dim]; x = code * scale",
                         "row_of": "qwen4_exp n-gram hash of the token and its two predecessors: shard-concatenated row id (ngram/meta.json source)",
                         "rows_per_token": (tc.get("ngram_size", 3) - 1) * tc.get("heads_per_ngram", 8), "layer": ple[0],
                         "hash_constants": "in resident.safetensors (.ple.ple_embedding. tensors); the engine recomputes them and compares",
                         "meta": "ngram/meta.json"},
               "draft_head": None, "vision": None},
    "derived": derived,
    "legacy": [p["path"] for p in parts if p["role"] == "legacy"],
    "prefill": {"chunk": 256, "span": 2048, "whole_from": 256, "join": 0.25, "gpu_min": 64,
                "notes": "the batched prompt path (dmeng/prefill.h): layer-major spans of 2048 in sub-chunks of 256, the next layer named whole in spans of 256 or more, the last span joined under a quarter span, the GPU's share from sub-chunks of 64; the experts all at once once landed (no waves)"},
    "drafting": {"cost_prior": None, "note": "no measured pass costs declared: the draft controller starts from the neutral prior (a pass of n positions costs n steps) until a session's measured costs (DIR/.dmeng/draft_cost.txt) replace it"},
    "modes": {"exact": {"definition": f"Generated tokens never depend on the cache, the routers or timing: the model's own router picks {K} experts per layer and the engine waits for every one.",
                        "engine": "lossless"},
              "fast": {"definition": FAST_DEF, "engine": "approx:drop:w=0.13:mass=0.13:late=-1",
                       "read_if": {"missing_routing_weight_over": 0.13, "order": "heaviest missing pick first"}, "otherwise": "left out", "substitute": False, "renormalise": False}},
    "profiles": {},
    "defaults": {"routers": [], "names": {"rule": "top", "m": NAMES_M}, "cross_token": None, "eviction": "auto", "layer_caps": "none", "drafts": False},
    "layout": {"derived_dir": "derived/", "engine_cache_dir": ".dmeng/",
               "rule": "engine-ready files are in derived/ under the engine's own names (<id>.<engine_format>.<first 12 hex of the source part's sha256>.bin); the engine looks for them in DIR/derived/ as well as DIR/.dmeng/ and derives what is missing into DIR/.dmeng/"},
}
if any(p["path"] == "routers_k4.npz" for p in parts):
    m["routers"].append({"id": "mlp_k4", "path": "routers_k4.npz", "default": True, "form": "mlp", "lag": 4, "hidden": None, "index": "layer", "targets": NL,
                         "features": {"inputs": ["emb", "h_prev", "h_lag"], "layout": "x = [rmsnorm(emb) * emb_norm.weight, h_prev, h_lag]; h_lag absent for targets t < lag",
                                      "emb": "the embed_tokens row of the token (bf16 widened to f32)",
                                      "h": "the MLP hyper-connection's mixed output (the MoE block input); h_prev at target t for the previous token (zero at position 0), h_lag at target t - lag for this token",
                                      "rms_eps": 1.1920929e-07, "activation": "gelu_erf",
                                      "output": "z = lin.weight @ x + lin.bias + mlp.2.weight @ gelu(mlp.0.weight @ x + mlp.0.bias) + mlp.2.bias; the top m of z name the experts of target t"},
                         "dtypes": "f32 (the trainer's npz); the hidden width is read from the npz when the engine derives its file",
                         "training": {"run": "flash_router3"}})
    m["defaults"]["routers"] = ["mlp_k4"]
# the mac profile: everything a Mac run reads; the engine-ready forms where the folder holds them
mac = [p for p in parts if p["role"] in ("config", "tokenizer", "chat_template", "license", "meta", "experts", "extra")
       and p["path"] not in ("vocab.json", "merges.txt")]
for p in parts:
    if p["role"] in ("resident", "router") and p["path"] not in shipped:
        mac.append(p)
    if p["role"] == "derived":
        mac.append(p)
if emb and "resident.safetensors" in shipped:
    mac = [p for p in mac if p["path"] != "resident.safetensors"] + [emb]
m["profiles"]["mac"] = {"when": "macOS", "mode": "exact", "resident": "resident_q8", "precision": "8-bit", "precision_detail": "8-bit affine, groups of 64",
                        "names": {"rule": "top", "m": NAMES_M}, "cache": "auto",
                        "cache_rule": "locked; the start keeps the kernel's free level at 14% once everything is in (cache_start), the ceiling physical memory - 5.5 GB - the process's other memory, never more than the system will lock; shrinks 1 GB per check under memory pressure and grows back",
                        "cache_start": {"rule": "free_level", "floor_pct": 14, "measured": "base M4 16 GB, this model only: from a 42-44% free level at start, a 3.5 GB cache (9.8 GB in all) ran with no swap growth; 4.5 GB (10.8 GB) let the level fall to ~9% and compute slowed 13-55%; 5.75 GB could not be locked. The built-in headroom rule (what the 35B and Inkling use) is not measured on this model"},
                        "prefill_gpu": {"projections": 0.4, "experts": 0.4}, "run_ahead": {"picks": K, "depth": 1},
                        "parts": [{"path": p["path"], "bytes": p["bytes"], "sha256": p["sha256"]} for p in mac],
                        "bytes": sum(p["bytes"] for p in mac), "bytes_rule": "sum of the listed parts' bytes; moved-store.json is fetched first and is not listed"}
out = a.out or os.path.join(D, "moved-store.json")
json.dump(m, open(out, "w"), indent=1)
print(f"{out}: {len(parts)} parts, mac profile {len(mac)} parts {m['profiles']['mac']['bytes'] / 1e9:.2f} GB")
