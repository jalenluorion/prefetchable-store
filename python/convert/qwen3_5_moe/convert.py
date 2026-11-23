"""One-time conversion of Qwen/Qwen3.6-35B-A3B (model_type qwen3_5_moe: fused experts) into the store layout, int4 experts.

  REPO=Qwen/Qwen3.6-35B-A3B OUT=/scratch/... STATE_DIR=<data>/q36/pack_state \
      FINAL_DIR=<data>/q36/store-int4 python convert.py

Derived from local/flash/convert.py (resumable, one shard at a time, two shards prefetched; same int4 record and the
same store/meta.json fields the Flash runner reads), without the n-gram table and without any Hugging Face upload:
finished files are copied to FINAL_DIR (shared disk) instead. Writes, in OUT (node-local) and then FINAL_DIR:

  store/layer_XX.bin   40 files x 256 records. Record (REC bytes, a whole number of 16 KB pages), per matrix in the order
                       gate_proj [I, H] | up_proj [I, H] | down_proj [H, I]: codes (8 per little-endian uint32, element i
                       at bits 4i, rows*cols/2 bytes), then fp16 scales [rows, cols/64], then fp16 biases [rows, cols/64];
                       groups of 64 run along each matrix's input dim (cols); w ~= code * scale + bias
                       (mlx.core.gather_qmm(bits=4, group_size=64)). Offsets in store/meta.json.
                       The release stores experts fused (transformers Qwen3_5MoeExperts: gate_up_proj [E, 2I, H],
                       `gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)` so rows 0:I are gate and I:2I up;
                       down_proj [E, H, I] applied as linear(h, down_proj[e]), i.e. [out=H, in=I]), and for some layers
                       the two tensors are in different shards; a layer is complete when 3*256 matrices have been written.
  resident.safetensors every other tensor in bf16 as released, minus SKIP_KEYS (default ^mtp\\. = the MTP
                       head, ^model\\.visual\\. = the vision tower). The router mlp.gate.weight, the shared expert and
                       mlp.shared_expert_gate stay here.
  config.json, generation_config.json, tokenizer files, chat_template.jinja, LICENSE (copied), convert.py, check.py,
  check_output.txt, sha256.txt (every file above, layers included).

Env: REPO (a hub id, or a local directory holding a release, e.g. make_tiny.py's: files are copied from it), REVISION
(hub revision, pinned by default; ignored for a local directory), OUT (large files; node-local disk), STATE_DIR (small
state, resident parts and check references; shared disk so a requeued job resumes), FINAL_DIR (optional: where finished
files are copied), SKIP_KEYS, CHECK_EXPERT (default 7), THREADS, STOP_AFTER (tests only: exit after N shards).
Gate: once layer 0 is written, check.py dequantises one of its experts against the bf16 copies saved during conversion and
requires bit-identical records against an independent re-quantisation; nothing is copied to FINAL_DIR before it passes,
and a failure stops the conversion (exit 3).
"""
import hashlib, json, os, re, shutil, subprocess, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO = os.environ.get("REPO", "Qwen/Qwen3.6-35B-A3B")
LOCAL = os.path.isdir(REPO)
REVISION = None if LOCAL else os.environ.get("REVISION", "345d467b117d1fd5eda9b879a5988de9ae5f47a7")  # main on 2026-11-16
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", HERE)
WORK, STORE = os.path.join(OUT, "_download"), os.path.join(OUT, "store")
for d in (WORK, STORE):
    os.makedirs(d, exist_ok=True)
STATE_DIR = os.environ.get("STATE_DIR", OUT)
REF_DIR = os.path.join(STATE_DIR, "_ref")
os.makedirs(REF_DIR, exist_ok=True)
FINAL = os.environ.get("FINAL_DIR") or None
if FINAL:
    os.makedirs(os.path.join(FINAL, "store"), exist_ok=True)
SKIP = re.compile(os.environ.get("SKIP_KEYS", r"^(mtp\.|model\.visual\.)"))
CHECK_E = int(os.environ.get("CHECK_EXPERT", "7"))
PAGE, G = 16384, 64
torch.set_num_threads(int(os.environ.get("THREADS", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 8)))
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)


def retry(fn, what, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network: HF hub hiccups (a missing file is not retried)
            if i == tries - 1 or isinstance(e, FileNotFoundError) or "NotFound" in type(e).__name__:
                raise
            log(f"retry {what} after {type(e).__name__}: {e}")
            time.sleep(30 * (i + 1))


def fetch(f):
    """One file of the release into WORK: from the hub (public; no token is used or needed), or copied from a local
    release directory (the copy is removed after conversion like a download; the source is never touched)."""
    if LOCAL:
        src = os.path.join(REPO, f)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(WORK, f)
        shutil.copy(src, dst)
        return dst
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f, revision=REVISION, local_dir=WORK, token=False)


SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "chat_template.jinja", "LICENSE", "model.safetensors.index.json")
for f in SMALL:
    try:
        shutil.copy(retry(lambda: fetch(f), f, tries=3), os.path.join(OUT, f))
    except Exception as e:
        print("no", f, type(e).__name__, flush=True)
full_cfg = json.load(open(os.path.join(OUT, "config.json")))
cfg = full_cfg.get("text_config", full_cfg)
assert full_cfg.get("model_type") == "qwen3_5_moe", full_cfg.get("model_type")
H, I, NL, NE = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_hidden_layers"], cfg["num_experts"]
TOPK = cfg["num_experts_per_tok"]
assert H % G == 0 and I % G == 0, (H, I)
MAT = I * H
SHAPES = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}  # [rows = out, cols = in]; groups along cols
OFF, SOFF, BOFF, pos = {}, {}, {}, 0
for kind, (r, c) in SHAPES.items():
    OFF[kind] = pos
    pos += r * c // 2
    SOFF[kind] = pos
    pos += r * (c // G) * 2
    BOFF[kind] = pos
    pos += r * (c // G) * 2
REC_USED = pos
REC = -(-pos // PAGE) * PAGE
meta = {"repo": REPO, "revision": REVISION, "model_type": full_cfg.get("model_type"), "hidden": H, "intermediate": I,
        "layers": NL, "experts": NE, "top_k": TOPK, "norm_topk_prob": True,  # Qwen3_5MoeTopKRouter: softmax, top-k, renormalise
        "shared_expert_intermediate": cfg.get("shared_expert_intermediate_size"), "layer_types": cfg.get("layer_types"),
        "record_bytes": REC, "record_bytes_used": REC_USED, "page": PAGE, "matrix_bytes": MAT, "offsets": OFF,
        "scale_offsets": SOFF, "bias_offsets": BOFF, "dtype": "int4", "group": G, "scale_dtype": "float16",
        "record_order": ["gate_proj", "up_proj", "down_proj"], "shapes": {k: list(v) for k, v in SHAPES.items()},
        "source_layout": {"gate_proj": "experts.gate_up_proj[e, 0:I, :]", "up_proj": "experts.gate_up_proj[e, I:2I, :]",
                          "down_proj": "experts.down_proj[e] ([H, I])"},
        "layer_file": "store/layer_{L:02d}.bin", "resident_params": None, "expert_params": NL * NE * 3 * MAT}

index = json.load(open(os.path.join(OUT, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(index.values()))
EXP = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")
want_exp = {f"model.language_model.layers.{L}.mlp.experts.{n}" for L in range(NL) for n in ("gate_up_proj", "down_proj")}
assert want_exp <= set(index), sorted(want_exp - set(index))[:4]
json.dump(meta, open(os.path.join(STORE, "meta.json"), "w"), indent=1)

state_path = os.path.join(STATE_DIR, "_convert_state.json")
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
for k, v in {"done": [], "mats": {}, "copied": [], "shard_layers": {}, "res_parts": [], "res_keys": {}, "skipped": {},
             "sha": {}, "check_ok": False, "marker": None}.items():
    state.setdefault(k, v)


def save_state():
    tmp = state_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, state_path)


# OUT is node-local and may be gone after a requeue: a marker file ties the local store files to the state. If it is
# missing, layers that were not copied to FINAL_DIR are rebuilt (layer 0 too while the check has not passed, since the
# check reads the local file), by redoing every shard that wrote into them.
marker_path = os.path.join(OUT, "_marker")
local_marker = open(marker_path).read().strip() if os.path.exists(marker_path) else None
if state["marker"] is not None and local_marker != state["marker"]:
    keep_L = {L for L in state["copied"] if state["check_ok"] or L != 0}
    redo = {sh for sh in state["done"] if any(L not in keep_L for L in state["shard_layers"].get(sh, []))}
    state["copied"] = sorted(keep_L)
    state["mats"] = {L: n for L, n in state["mats"].items() if int(L) in keep_L}
    state["done"] = [sh for sh in state["done"] if sh not in redo]
    log(f"local files gone (node {os.uname().nodename}): {len(keep_L)} layers already in FINAL_DIR, {len(redo)} shards to redo")
if local_marker is None or local_marker != state["marker"]:
    local_marker = uuid.uuid4().hex
    open(marker_path, "w").write(local_marker)
    state["marker"] = local_marker
save_state()

fds = []
for L in range(NL):
    p = os.path.join(STORE, f"layer_{L:02d}.bin")
    if not os.path.exists(p) or os.path.getsize(p) != NE * REC:
        with open(p, "wb") as fh:
            fh.truncate(NE * REC)
    fds.append(os.open(p, os.O_RDWR))


def q4(w):
    """w [n, rows, cols] float32 -> codes uint32 [n, rows, cols/8], scales fp16 [n, rows, cols/G], biases fp16 (same).
    MLX affine: per group of G along cols, scale = (max - min) / 15, bias = min (both rounded to fp16 first),
    code = round_half_even((w - bias) / scale) clamped to 0..15."""
    n, r, c = w.shape
    wg = w.reshape(n, r, c // G, G)
    lo, hi = wg.amin(dim=-1), wg.amax(dim=-1)
    scale16, bias16 = ((hi - lo) / 15.0).to(torch.float16), lo.to(torch.float16)
    s, b = scale16.float(), bias16.float()
    s_safe = torch.where(s > 0, s, torch.ones_like(s))  # constant group: every code 0, w = bias
    q = ((wg - b[..., None]) / s_safe[..., None]).round().clamp_(0, 15).to(torch.int64).reshape(n, r, c // 8, 8)
    packed = q[..., 0].clone()
    for i in range(1, 8):
        packed |= q[..., i] << (4 * i)
    return packed.numpy().astype("<u4"), scale16.numpy(), bias16.numpy()


def write_experts(L, name, t):
    """t: experts.gate_up_proj [NE, 2I, H] or experts.down_proj [NE, H, I] (bf16), written record by record."""
    want = (NE, 2 * I, H) if name == "gate_up_proj" else (NE, H, I)
    assert tuple(t.shape) == want, (L, name, tuple(t.shape), want)
    assert t.dtype == torch.bfloat16, (L, name, t.dtype)
    if L == 0:
        save_file({name: t[CHECK_E].clone()}, os.path.join(REF_DIR, f"l0_e{CHECK_E}_{name}.safetensors"))
    parts = [("gate_proj", slice(0, I)), ("up_proj", slice(I, 2 * I))] if name == "gate_up_proj" else [("down_proj", slice(None))]
    CH = 16
    for e0 in range(0, NE, CH):
        w = t[e0:e0 + CH].to(torch.float32)
        for kind, sl in parts:
            codes, sc, bi = q4(w[:, sl].contiguous())
            for j in range(w.shape[0]):
                base = (e0 + j) * REC
                os.pwrite(fds[L], codes[j].tobytes(), base + OFF[kind])
                os.pwrite(fds[L], sc[j].tobytes(), base + SOFF[kind])
                os.pwrite(fds[L], bi[j].tobytes(), base + BOFF[kind])
    state["mats"][str(L)] = state["mats"].get(str(L), 0) + len(parts) * NE


def copy_hashed(src, rel):
    """src -> FINAL_DIR/rel through a temporary name, hashing the bytes read; the copy is re-read and must hash the same."""
    h = hashlib.sha256()
    dst = os.path.join(FINAL, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for b in iter(lambda: fi.read(1 << 24), b""):
            h.update(b)
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())
    digest = h.hexdigest()
    assert sha256(tmp) == digest, f"copy of {rel} does not read back identically"
    os.replace(tmp, dst)
    return digest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def layer_done(L):
    return state["mats"].get(str(L), 0) >= 3 * NE


def maybe_check():
    """Gate: once layer 0 is on disk, dequantise one expert against the bf16 references; stop on failure."""
    if state["check_ok"] or not layer_done(0):
        return
    os.fsync(fds[0])
    r = subprocess.run([sys.executable, os.path.join(HERE, "check.py"), "--out", OUT, "--ref", REF_DIR, "--expert", str(CHECK_E)],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    open(os.path.join(STATE_DIR, "check_output.txt"), "w").write(text)
    shutil.copy(os.path.join(STATE_DIR, "check_output.txt"), os.path.join(OUT, "check_output.txt"))
    print(text, flush=True)
    if r.returncode != 0:
        log("CHECK_FAILED: stopping; nothing copied to FINAL_DIR")
        sys.exit(3)
    state["check_ok"] = True
    save_state()
    log("CHECK_PASSED")


def copy_complete():
    """Every finished layer not yet in FINAL_DIR (only after the check has passed). Without FINAL_DIR only the sha is kept."""
    if not state["check_ok"]:
        return
    for L in range(NL):
        if layer_done(L) and L not in state["copied"]:
            os.fsync(fds[L])
            path, rel = os.path.join(STORE, f"layer_{L:02d}.bin"), f"store/layer_{L:02d}.bin"
            assert os.path.getsize(path) == NE * REC, (path, os.path.getsize(path))
            state["sha"][rel] = copy_hashed(path, rel) if FINAL else sha256(path)
            state["copied"].append(L)
            save_state()
            log("copied" if FINAL else "hashed", rel)


part_path = os.path.join(STATE_DIR, "_resident_part_{}")  # {} = shard file name
todo = [s for s in shards if s not in state["done"]]
log(f"{REPO}@{REVISION} | {len(shards)} shards, {len(todo)} to convert | record {REC} B ({REC_USED} used, {REC // PAGE} pages) | "
    f"layer {NE * REC} B | store {NL * NE * REC} B ({NL * NE * REC / 2**30:.2f} GiB) | OUT {OUT} | STATE_DIR {STATE_DIR} | FINAL_DIR {FINAL}")
pool = ThreadPoolExecutor(max_workers=2)
futs = {}


def prefetch(i):
    for s in todo[i:i + 2]:
        if s not in futs:
            futs[s] = pool.submit(retry, lambda s=s: fetch(s), s)


maybe_check()
copy_complete()
for i, shard in enumerate(todo):
    prefetch(i)
    t_wait = time.time()
    path = futs.pop(shard).result()
    prefetch(i + 1)
    log("converting", shard, f"{os.path.getsize(path) / 2**30:.2f} GiB (waited {time.time() - t_wait:.0f}s for the download)")
    res_part, touched_L, skipped = {}, set(), []
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        assert all(index.get(k) == shard for k in keys), f"{shard} holds keys the index puts elsewhere"
        for k in keys:
            if SKIP.match(k):
                skipped.append(k)
                continue
            m = EXP.fullmatch(k)
            if m:
                L = int(m.group(1))
                write_experts(L, m.group(2), f.get_tensor(k))
                touched_L.add(L)
                continue
            assert ".experts." not in k, f"unexpected expert tensor {k}"
            res_part[k] = f.get_tensor(k)
    if res_part:
        save_file(res_part, part_path.format(shard))
        if shard not in state["res_parts"]:
            state["res_parts"].append(shard)
    for fd in fds:
        os.fsync(fd)
    os.remove(path)
    state["done"].append(shard)
    state["shard_layers"][shard] = sorted(touched_L)
    state["res_keys"][shard] = sorted(res_part)
    state["skipped"][shard] = skipped
    save_state()
    maybe_check()
    copy_complete()
    log("done", shard, f"| {len(state['done'])}/{len(shards)} shards | layers {sum(layer_done(L) for L in range(NL))}/{NL} "
        f"(copied {len(state['copied'])}) | resident {len(res_part)} skipped {len(skipped)} | free GB {shutil.disk_usage(OUT).free / 2**30:.0f}")
    if i + 1 == int(os.environ.get("STOP_AFTER", 0)):  # resume test only: stop as if preempted
        log("STOP_AFTER reached")
        os._exit(4)
pool.shutdown()
for fd in fds:
    os.close(fd)
bad_L = [L for L in range(NL) if state["mats"].get(str(L), 0) != 3 * NE and L not in state["copied"]]
assert not bad_L and all(layer_done(L) for L in range(NL)), f"incomplete layers {bad_L}"
assert state["check_ok"], "check never ran"
assert len(state["copied"]) == NL, "not every layer copied/hashed"

# every key of the index accounted for exactly once: expert (store), skipped, or resident
res_keys = {k for v in state["res_keys"].values() for k in v}
skip_keys = {k for v in state["skipped"].values() for k in v}
assert res_keys | skip_keys | want_exp == set(index) and not (res_keys & skip_keys), "index keys not accounted for"
resident = {}
for sh in state["res_parts"]:
    resident.update(load_file(part_path.format(sh)))
assert set(resident) == res_keys, (len(resident), len(res_keys))
bad = [k for k in resident if SKIP.match(k) or ".experts." in k]
assert not bad, bad[:5]
assert all(t.dtype == torch.bfloat16 for t in resident.values()), "non-bf16 resident tensor"
res_path = os.path.join(OUT, "resident.safetensors")
save_file(resident, res_path, metadata={"format": "pt"})
meta["resident_params"] = int(sum(t.numel() for t in resident.values()))
meta["resident_tensors"] = len(resident)
meta["skipped"] = {"pattern": SKIP.pattern, "tensors": len(skip_keys),
                   "by_prefix": {p: sum(k.startswith(p) for k in skip_keys) for p in ("mtp.", "model.visual.")}}
json.dump(meta, open(os.path.join(STORE, "meta.json"), "w"), indent=1)
log(f"resident: {len(resident)} tensors, {meta['resident_params'] / 1e9:.3f}B params, {os.path.getsize(res_path) / 2**30:.2f} GiB | "
    f"skipped {meta['skipped']}")
del resident
for f in ("convert.py", "check.py"):
    shutil.copy(os.path.join(HERE, f), os.path.join(OUT, f))
FILES = ["resident.safetensors", "store/meta.json", "config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "LICENSE", "convert.py", "check.py",
         "check_output.txt"]
FILES = [f for f in FILES if os.path.exists(os.path.join(OUT, f))]
for f in FILES:
    state["sha"][f] = copy_hashed(os.path.join(OUT, f), f) if FINAL else sha256(os.path.join(OUT, f))
with open(os.path.join(OUT, "sha256.txt"), "w") as fh:
    for k in sorted(state["sha"]):
        fh.write(f"{state['sha'][k]}  {k}\n")
if FINAL:
    copy_hashed(os.path.join(OUT, "sha256.txt"), "sha256.txt")
save_state()
for sh in state["res_parts"]:
    os.remove(part_path.format(sh))
shutil.rmtree(WORK, ignore_errors=True)
os.replace(state_path, os.path.join(STATE_DIR, "_convert_state.done.json"))
log("CONVERT_DONE resident params", meta["resident_params"], "expert params", meta["expert_params"],
    "store bytes", NL * NE * REC, f"({NL * NE * REC / 2**30:.2f} GiB)", "| final dir", FINAL or OUT)
