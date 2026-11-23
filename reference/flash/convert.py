"""One-time conversion of Qwen/Qwen3.8-Flash-Next (qwen4_exp: fused experts, a 51B n-gram table) into the store layout.

  OUT=/fast/disk/flash_pack STATE_DIR=/shared/disk/flash_pack_state [HF_REPO=you/your-repo] python convert.py

Derived from the Qwen3-Next-80B store's converter (resumable, shard at a time, each finished store layer uploaded at
once), with a 4-bit record (MLX affine 4-bit, group 64). Downloads the release one shard at a time
(two shards prefetched), writes

  store/layer_XX.bin   48 files x 512 records. Record (REC bytes, padded to 16 KB pages), per matrix in the order
                       gate_proj [I, H] | up_proj [I, H] | down_proj [H, I]: codes (8 per little-endian uint32, element i
                       at bits 4i, rows*cols/2 bytes), then fp16 scales [rows, cols/64], then fp16 biases [rows, cols/64];
                       w ~= code * scale + bias (mlx.core.gather_qmm(bits=4, group_size=64)). Offsets in store/meta.json.
                       The release stores experts fused: experts.gate_up_proj [512, 2I, H] (rows 0:I gate, I:2I up) and
                       experts.down_proj [512, H, I], usually in different shards; a layer is complete when 3*512 matrices
                       have been written.
  ngram/part_PP.bin    the PLE n-gram table (128 tensors ngram_embedding.shard_K.weight [2500012, 160], one flat table in
                       K order) as int8 rows: fp16 scale = max|row|/127, then 160 int8 codes (162 B/row), rows contiguous
                       in global row order, 20,000,096 rows per part. ngram/meta.json describes it.
  resident.safetensors everything else in bf16 as released, minus SKIP_KEYS (mtp.*, model.visual.*).

Env: REPO (a hub id, or a local directory holding a release: files are copied from it, e.g. make_tiny.py's), OUT (large files; node-local disk), STATE_DIR (small state, resident parts and check references; shared disk so
a requeued job resumes), HF_REPO (upload each finished layer / n-gram part), SKIP_KEYS, CHECK_EXPERT (default 7).
Gate: once layer 0 and n-gram part 0 are written, check.py dequantises them against bf16 copies saved during conversion;
nothing but layer 0 is uploaded before it passes, and a failure stops the job.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO = os.environ.get("REPO", "Qwen/Qwen3.8-Flash-Next")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", HERE)
WORK, STORE, NGDIR = os.path.join(OUT, "_download"), os.path.join(OUT, "store"), os.path.join(OUT, "ngram")
for d in (WORK, STORE, NGDIR):
    os.makedirs(d, exist_ok=True)
STATE_DIR = os.environ.get("STATE_DIR", OUT)
REF_DIR = os.path.join(STATE_DIR, "_ref")
os.makedirs(REF_DIR, exist_ok=True)
HF_REPO_UP = os.environ.get("HF_REPO")
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
    """One file of the release into WORK: from the hub, or copied from a local release directory (the copy is removed
    after conversion like a download; the source is never touched)."""
    if os.path.isdir(REPO):
        src = os.path.join(REPO, f)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(WORK, f)
        shutil.copy(src, dst)
        return dst
    return hf_hub_download(REPO, f, local_dir=WORK)


SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "chat_template.jinja", "LICENSE", "model.safetensors.index.json")
for f in SMALL:
    try:
        shutil.copy(retry(lambda: fetch(f), f, tries=3), os.path.join(OUT, f))
    except Exception as e:
        print("no", f, type(e).__name__, flush=True)
full_cfg = json.load(open(os.path.join(OUT, "config.json")))
cfg = full_cfg.get("text_config", full_cfg)
H, I, NL, NE = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_hidden_layers"], cfg["num_experts"]
TOPK = cfg.get("num_experts_per_tok", 10)
MAT = I * H
SHAPES = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
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
meta = {"repo": REPO, "model_type": full_cfg.get("model_type"), "hidden": H, "intermediate": I, "layers": NL, "experts": NE,
        "top_k": TOPK, "norm_topk_prob": True, "record_bytes": REC, "page": PAGE, "matrix_bytes": MAT, "offsets": OFF,
        "scale_offsets": SOFF, "bias_offsets": BOFF, "dtype": "int4", "group": G, "scale_dtype": "float16",
        "record_order": ["gate_proj", "up_proj", "down_proj"], "shapes": {k: list(v) for k, v in SHAPES.items()},
        "layer_file": "store/layer_{L:02d}.bin", "resident_params": None, "expert_params": NL * NE * 3 * MAT}

index = json.load(open(os.path.join(OUT, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(index.values()))
EXP = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")
NG = re.compile(r"model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight")
NG_K = sorted(int(NG.fullmatch(k).group(1)) for k in index if NG.fullmatch(k))
assert NG_K == list(range(len(NG_K))), "n-gram shards are not 0..K-1"
# shapes asserted when each tensor is read; the env overrides exist only for the synthetic test
NG_SHARDS, NG_SHARD_ROWS, NG_DIM = len(NG_K), int(os.environ.get("NG_SHARD_ROWS", 2500012)), int(os.environ.get("NG_DIM", 160))
NG_ROWS, NG_BPR, NG_PART_ROWS = NG_SHARDS * NG_SHARD_ROWS, 2 + NG_DIM, int(os.environ.get("NG_PART_ROWS", 20000096))
NG_PARTS = -(-NG_ROWS // NG_PART_ROWS)
part_rows = [min(NG_PART_ROWS, NG_ROWS - p * NG_PART_ROWS) for p in range(NG_PARTS)]
ng_meta = {"repo": REPO, "rows": NG_ROWS, "dim": NG_DIM, "bytes_per_row": NG_BPR, "rows_per_part": NG_PART_ROWS,
           "parts": NG_PARTS, "part_rows": part_rows, "files": [f"ngram/part_{p:02d}.bin" for p in range(NG_PARTS)],
           "dtype": "int8", "scale_dtype": "float16", "layout": "per row: fp16 scale (2 B) | int8 codes [dim]; x = code * scale",
           "source": "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_K.weight, K = 0.."
                     f"{NG_SHARDS - 1}, [{NG_SHARD_ROWS}, {NG_DIM}] each, concatenated in K order (row r in shard r // {NG_SHARD_ROWS})",
           "shard_rows": NG_SHARD_ROWS}
json.dump(meta, open(os.path.join(STORE, "meta.json"), "w"), indent=1)
json.dump(ng_meta, open(os.path.join(NGDIR, "meta.json"), "w"), indent=1)

state_path = os.path.join(STATE_DIR, "_convert_state.json")
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
for k, v in {"done": [], "mats": {}, "uploaded": [], "shard_layers": {}, "shard_parts": {}, "ng_rows": {}, "ng_uploaded": [],
             "res_parts": [], "sha": {}, "check_ok": False, "marker": None, "meta_uploaded": False}.items():
    state.setdefault(k, v)


def save_state():
    tmp = state_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, state_path)


# The local disk is node-local and wiped between jobs: a marker file ties the local store files to the state. If it is
# missing, layers/parts that were not uploaded are rebuilt (layer 0 too while the check has not passed, since the check
# reads the local files), by redoing every shard that wrote into them.
marker_path = os.path.join(OUT, "_marker")
local_marker = open(marker_path).read().strip() if os.path.exists(marker_path) else None
if state["marker"] is not None and local_marker != state["marker"]:
    keep_L = {L for L in state["uploaded"] if state["check_ok"] or L != 0}
    keep_P = set(state["ng_uploaded"])
    redo = {sh for sh in state["done"] if any(L not in keep_L for L in state["shard_layers"].get(sh, []))
            or any(p not in keep_P for p in state["shard_parts"].get(sh, []))}
    state["uploaded"] = sorted(keep_L)
    state["mats"] = {L: n for L, n in state["mats"].items() if int(L) in keep_L}
    state["ng_rows"] = {p: n for p, n in state["ng_rows"].items() if int(p) in keep_P}
    state["done"] = [sh for sh in state["done"] if sh not in redo]
    log(f"local files gone (new node {os.uname().nodename}): {len(keep_L)} layers + {len(keep_P)} parts uploaded, {len(redo)} shards to redo")
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
ng_fds = []
for p in range(NG_PARTS):
    path = os.path.join(NGDIR, f"part_{p:02d}.bin")
    if not os.path.exists(path) or os.path.getsize(path) != part_rows[p] * NG_BPR:
        with open(path, "wb") as fh:
            fh.truncate(part_rows[p] * NG_BPR)
    ng_fds.append(os.open(path, os.O_RDWR))
# 1,000 random rows of n-gram part 0 (fixed seed): their bf16 originals are saved for the check.
NG_REF = np.sort(np.random.default_rng(0).choice(part_rows[0], min(1000, part_rows[0]), replace=False)).astype(np.int64)


def q4(w):
    """w [n, rows, cols] float32 -> codes uint32 [n, rows, cols/8], scales fp16 [n, rows, cols/G], biases fp16 (same)."""
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


def write_ngram(K, t):
    """t: ngram_embedding.shard_K.weight [2500012, 160] bf16 -> int8 rows with one fp16 scale, at global row K*2500012."""
    assert tuple(t.shape) == (NG_SHARD_ROWS, NG_DIM), (K, tuple(t.shape))
    g0 = K * NG_SHARD_ROWS
    sel = NG_REF[(NG_REF >= g0) & (NG_REF < g0 + NG_SHARD_ROWS)]
    if len(sel):
        save_file({"idx": torch.from_numpy(sel), "rows": t[torch.from_numpy(sel - g0)].clone()}, os.path.join(REF_DIR, f"ng_{K:03d}.safetensors"))
    touched = set()
    CH = 500000
    for r0 in range(0, NG_SHARD_ROWS, CH):
        w = t[r0:r0 + CH].to(torch.float32)
        s16 = (w.abs().amax(dim=1) / 127.0).to(torch.float16)
        s = s16.float()
        s_safe = torch.where(s > 0, s, torch.ones_like(s))
        q = (w / s_safe[:, None]).round().clamp_(-127, 127).to(torch.int8)
        rec = np.empty((w.shape[0], NG_BPR), dtype=np.uint8)
        rec[:, :2] = s16.numpy().view(np.uint8).reshape(-1, 2)
        rec[:, 2:] = q.numpy().view(np.uint8)
        g, i = g0 + r0, 0
        while i < len(rec):  # split at part boundaries
            p = g // NG_PART_ROWS
            n = min(len(rec) - i, (p + 1) * NG_PART_ROWS - g)
            os.pwrite(ng_fds[p], rec[i:i + n].tobytes(), (g - p * NG_PART_ROWS) * NG_BPR)
            state["ng_rows"][str(p)] = state["ng_rows"].get(str(p), 0) + n
            touched.add(p)
            g, i = g + n, i + n
    return touched


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def upload(path, in_repo, msg):
    state["sha"][in_repo] = sha256(path)
    retry(lambda: HfApi().upload_file(path_or_fileobj=path, path_in_repo=in_repo, repo_id=HF_REPO_UP, commit_message=msg), in_repo)


def layer_done(L):
    return state["mats"].get(str(L), 0) >= 3 * NE


def part_done(p):
    return state["ng_rows"].get(str(p), 0) >= part_rows[p]


def maybe_check():
    """Gate G1: once layer 0 and n-gram part 0 are on disk, dequantise them against the bf16 references."""
    if state["check_ok"] or not (layer_done(0) and part_done(0)):
        return
    os.fsync(fds[0])
    os.fsync(ng_fds[0])
    r = subprocess.run([sys.executable, os.path.join(HERE, "check.py"), "--out", OUT, "--ref", REF_DIR, "--expert", str(CHECK_E)],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    open(os.path.join(STATE_DIR, "check_output.txt"), "w").write(text)
    shutil.copy(os.path.join(STATE_DIR, "check_output.txt"), os.path.join(OUT, "check_output.txt"))
    print(text, flush=True)
    if r.returncode != 0:
        log("CHECK_FAILED: stopping before any further upload")
        sys.exit(3)
    state["check_ok"] = True
    save_state()
    log("CHECK_PASSED")


def upload_complete():
    if not HF_REPO_UP:
        return
    for L in range(NL):
        if layer_done(L) and L not in state["uploaded"] and (L == 0 or state["check_ok"]):
            os.fsync(fds[L])
            path = os.path.join(STORE, f"layer_{L:02d}.bin")
            assert os.path.getsize(path) == NE * REC, (path, os.path.getsize(path))
            upload(path, f"store/layer_{L:02d}.bin", f"store layer {L}")
            state["uploaded"].append(L)
            save_state()
            log("uploaded", f"store/layer_{L:02d}.bin")
    if not state["check_ok"]:
        return
    if not state["meta_uploaded"]:  # early copies so the runner track can start; resident_params is filled in at the end
        for rel in ("store/meta.json", "ngram/meta.json"):
            upload(os.path.join(OUT, rel), rel, f"{rel} (preliminary)")
        state["meta_uploaded"] = True
        save_state()
    for p in range(NG_PARTS):
        if part_done(p) and p not in state["ng_uploaded"]:
            os.fsync(ng_fds[p])
            upload(os.path.join(NGDIR, f"part_{p:02d}.bin"), f"ngram/part_{p:02d}.bin", f"ngram part {p}")
            state["ng_uploaded"].append(p)
            save_state()
            log("uploaded", f"ngram/part_{p:02d}.bin")


part_path = os.path.join(STATE_DIR, "_resident_part_{}")  # {} = shard file name
todo = [s for s in shards if s not in state["done"]]
log(f"{len(shards)} shards, {len(todo)} to convert | record {REC} B ({REC_USED} used) | layer {NE * REC / 2**30:.2f} GiB | "
    f"ngram {NG_ROWS} rows x {NG_BPR} B in {NG_PARTS} parts | OUT {OUT} | STATE_DIR {STATE_DIR}")
pool = ThreadPoolExecutor(max_workers=2)
futs = {}


def prefetch(i):
    for s in todo[i:i + 2]:
        if s not in futs:
            futs[s] = pool.submit(retry, lambda s=s: fetch(s), s)


upload_complete()
maybe_check()
for i, shard in enumerate(todo):
    prefetch(i)
    t_wait = time.time()
    path = futs.pop(shard).result()
    prefetch(i + 1)
    log("converting", shard, f"{os.path.getsize(path) / 2**30:.1f} GB (waited {time.time() - t_wait:.0f}s for the download)")
    res_part, touched_L, touched_P = {}, set(), set()
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if SKIP.match(k):
                continue
            m = EXP.fullmatch(k)
            if m:
                L = int(m.group(1))
                write_experts(L, m.group(2), f.get_tensor(k))
                touched_L.add(L)
                continue
            m = NG.fullmatch(k)
            if m:
                touched_P |= write_ngram(int(m.group(1)), f.get_tensor(k))
                continue
            assert ".experts." not in k and "ngram_embedding" not in k, f"unexpected tensor {k}"
            res_part[k] = f.get_tensor(k)
    if res_part:
        save_file(res_part, part_path.format(shard))
        if shard not in state["res_parts"]:
            state["res_parts"].append(shard)
    for fd in fds + ng_fds:
        os.fsync(fd)
    os.remove(path)
    state["done"].append(shard)
    state["shard_layers"][shard] = sorted(touched_L)
    state["shard_parts"][shard] = sorted(touched_P)
    save_state()
    maybe_check()
    upload_complete()
    log("done", shard, f"| {len(state['done'])}/{len(shards)} shards | layers {sum(layer_done(L) for L in range(NL))}/{NL} "
        f"(up {len(state['uploaded'])}) | parts {sum(part_done(p) for p in range(NG_PARTS))}/{NG_PARTS} (up {len(state['ng_uploaded'])}) "
        f"| free GB {shutil.disk_usage(OUT).free / 2**30:.0f}")
pool.shutdown()
for fd in fds + ng_fds:
    os.close(fd)
bad_L = [L for L in range(NL) if not layer_done(L)]
bad_P = [p for p in range(NG_PARTS) if not part_done(p)]
assert not bad_L and not bad_P, f"incomplete layers {bad_L} parts {bad_P}"
assert state["check_ok"], "check never ran"
if HF_REPO_UP:
    assert len(state["uploaded"]) == NL and len(state["ng_uploaded"]) == NG_PARTS, "not every layer/part uploaded"

resident = {}
for sh in state["res_parts"]:
    resident.update(load_file(part_path.format(sh)))
bad = [k for k in resident if SKIP.match(k) or ".experts." in k or "ngram_embedding.shard_" in k]
assert not bad, bad[:5]
res_path = os.path.join(OUT, "resident.safetensors")
save_file(resident, res_path, metadata={"format": "pt"})
meta["resident_params"] = int(sum(t.numel() for t in resident.values()))
json.dump(meta, open(os.path.join(STORE, "meta.json"), "w"), indent=1)
log(f"resident: {len(resident)} tensors, {meta['resident_params'] / 1e9:.3f}B params, {os.path.getsize(res_path) / 2**30:.2f} GiB")
del resident
for f in ("convert.py", "check.py"):
    shutil.copy(os.path.join(HERE, f), os.path.join(OUT, f))
FINAL = ["resident.safetensors", "store/meta.json", "ngram/meta.json", "config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "convert.py", "check.py", "check_output.txt"]
FINAL = [f for f in FINAL if os.path.exists(os.path.join(OUT, f))]
for f in FINAL:
    if f != "check_output.txt":
        state["sha"][f] = sha256(os.path.join(OUT, f))
with open(os.path.join(OUT, "sha256.txt"), "w") as fh:
    for k in sorted(state["sha"]):
        fh.write(f"{state['sha'][k]}  {k}\n")
FINAL.append("sha256.txt")
if HF_REPO_UP:
    retry(lambda: HfApi().upload_folder(repo_id=HF_REPO_UP, folder_path=OUT, allow_patterns=FINAL,
                                        commit_message="resident, meta, tokenizer, sha256"), "final files")
    log("uploaded", FINAL)
state["final_uploaded"] = True
save_state()
for sh in state["res_parts"]:
    os.remove(part_path.format(sh))
shutil.rmtree(WORK, ignore_errors=True)
os.replace(state_path, os.path.join(STATE_DIR, "_convert_state.done.json"))
log("CONVERT_DONE resident params", meta["resident_params"], "expert params", meta["expert_params"],
    "store GiB", round(NL * NE * REC / 2**30, 2), "ngram GiB", round(NG_ROWS * NG_BPR / 2**30, 2))
