"""The moved store on Apple Silicon, in MLX, for Qwen3.8-Flash-Next (qwen4_exp). One engine, two ways in:

  python mlx_run.py --mode exact                      # research mode: a chat loop over the store (quiet; -v for statistics)
  python mlx_run.py --serve 8080 --prompt-cache demo/cache --warm   # demo mode: the same engine behind an OpenAI-compatible API (OpenCode)
  python mlx_run.py --bench --max-new 80              # the timing JSON (token-by-token prefill, argmax)
  python mlx_run.py --mode exact --check ref.pt       # the released model's outputs, against a reference
  python mlx_run.py --prompt "..." --max-new 64       # one turn

The engine of the Qwen3-Next-80B moved store (huggingface.co/jalenluorion/Qwen3-Next-80B-A3B-moved-store), sections and
front-ends unchanged; what this model changes:
- The skeleton is the community MLX port of qwen4_exp, vendored in qwen4_exp/ (on mlx-vlm's qwen3_5 pieces): a residual of
  4 hyper-connection streams with a GatedResidual before every block, 36 Gated DeltaNet + 12 Qwen Sparse Attention layers,
  a softmax top-10 MoE with a sigmoid-gated shared expert on every layer, the PLE n-gram memory at layer index 1. The
  expert block's input is the MLP hyper-connection's `mixed` tensor; StoreMoE takes the place of the released block.
- The experts are MLX affine 4-bit records (group 64: packed codes, fp16 scales, fp16 biases) and go to gather_qmm as they
  are. The int8 one-scale-per-neuron records of the earlier stores still work: store/meta.json says which.
- The n-gram memory (51B params; 16 rows of 160 per token) is a second store, ngram/part_PP.bin (int8 rows with one fp16
  scale). A row's address follows from the token ids alone, so the rows of the next token are read the moment it is
  sampled, while the embedding and layer 0 run; a batch's rows are all read at the start of the batch.
- The resident weights (4.2B, bf16 on disk) are quantised to int8 (MLX affine, group 64) at the first load and cached in
  resident_int8.safetensors; --dtype float16 keeps them in fp16 on bigger Macs.
- The lag-4 routers read the mixed input (embedding, previous token's mixed at L, this token's mixed at L-4). Until they
  are trained, --mode exact (the released router, misses read on the spot); prefetch mode needs a routers file.

The research contribution (sections 3-5): experts live on disk; lag-4 routers name each layer's experts four layers early
so the reads overlap the layers in between; a fetched expert stays in a slot while it is named again within --keep tokens;
the released router picks its top-k among the experts in memory (prefetch mode: commit, nothing read on the spot) or reads
a missing one synchronously (exact mode).

The engineering around it, shared by both modes (sections 1, 5, 6): the model's memory wired and kept hot for the macOS
pager; prompts of 48+ tokens prefilled as one batch with every needed expert streamed once; a session that reuses the
decoded prefix and snapshots prefixes in memory and on disk. Nothing in it changes what the model computes.
"""
import argparse, json, os, queue, re, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ============================================================================================================ 0. flags
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
g = ap.add_argument_group("model and store")
g.add_argument("--dir", default=HERE, help="folder with resident.safetensors, store/, ngram/, routers, config and tokenizer")
g.add_argument("--store", default="store")
g.add_argument("--ngram", default="ngram", help="the PLE n-gram row store (folder under --dir)")
g.add_argument("--routers", default=None, help="lag-4 router weights (.npz or .pt; default: the first of routers_k4.npz, routers_k4.pt, mlp_k4.npz, mlp_k4.pt, routers.npz, routers.pt in --dir)")
g.add_argument("--mode", default="prefetch", choices=["prefetch", "exact"], help="prefetch: released top-k among the experts in memory; exact: the released choice, missing experts read on the spot")
g.add_argument("--m", type=int, default=0, help="experts named per layer (0: the model's top-k)")
g.add_argument("--keep", type=int, default=3, help="tokens a fetched expert stays in memory while it keeps being named")
g.add_argument("--max-slots", type=int, default=-1, help="expert slots per layer (-1: 2k+4, the evaluated cells: 24 for the top-10 models; 0: unbounded)")
g.add_argument("--dtype", default="int8", choices=["int8", "float16", "bfloat16", "float32"], help="resident weights: int8 (MLX affine, group 64, quantised at the first load and cached; fp16 activations) for 16 GB Macs, or float16/bfloat16 weights and activations (the expert step is always fp16; float32 is for validation)")
g.add_argument("--router-dtype", default="float16", choices=["float16", "float32"])
g.add_argument("--embed", default="disk", choices=["disk", "memory"], help="input embeddings: one 5 KB row read per token, or the whole table in memory")
g = ap.add_argument_group("generation")
g.add_argument("--prompt", default=None)
g.add_argument("--chat", action="store_true", default=None)
g.add_argument("--no-chat", action="store_false", dest="chat")
g.add_argument("--reasoning-effort", default=None, choices=["xhigh", "medium", "low"], help="thinking effort when a request names none (the chat template's own default is xhigh); the template states it at the start of the system prompt, so changing it re-prefills a cached prompt")
g.add_argument("--max-new", type=int, default=None)
g.add_argument("--greedy", action="store_true", help="argmax decoding (the default for --bench and --check); otherwise the release's generation_config.json")
g.add_argument("--temp", type=float, default=None)
g.add_argument("--top-p", type=float, default=None)
g.add_argument("--top-k", type=int, default=None)
g.add_argument("--seed", type=int, default=0)
g = ap.add_argument_group("prefill and prompt cache (both modes)")
g.add_argument("--chunk", type=int, default=None, help="prompt segments of at least this many tokens are prefilled as one batch; 0: token by token (default 48; 0 for --bench and --check)")
g.add_argument("--chunk-max", type=int, default=4096, help="longest batch; longer segments are split (a batch that runs out of GPU memory is retried at half the size)")
g.add_argument("--stream-from", type=int, default=2048, help="batches of at least this many tokens read every expert of every layer in file order, two chunks ahead across layer boundaries (a 4k batch touches >80% of them anyway); shorter batches read only the experts they need")
g.add_argument("--chunk-copy", action="store_true", help="batched prefill, int4: copy each chunk's fields out of the staging ring (the pre-2026-11-14 path) instead of strided views")
g.add_argument("--attn-block", type=int, default=512, help="inside a batch, the attention layers run this many tokens at a time against the cache (bounds the peak; the experts still stream once per batch)")
g.add_argument("--prompt-cache", default=None, help="directory where prefixes (an agent's system prompt) are snapshotted, so they are prefilled once per machine")
g.add_argument("--snapshot-every", type=int, default=256, help="with --prompt-cache: also snapshot every N tokens of a token-by-token prefill")
g.add_argument("--disk-snapshots", default="first", choices=["first", "all"], help="with --prompt-cache: which message boundaries are written to disk: the first (an agent's system prompt), or all of them; the rest stay in memory")
g.add_argument("--cache-gb", type=float, default=3.0, help="with --prompt-cache: the snapshot folder's size limit; least recently used snapshots are deleted beyond it")
g = ap.add_argument_group("demo mode")
g.add_argument("--serve", type=int, default=0, help="serve an OpenAI-compatible API on this port (/v1/chat/completions with streaming and tools, /v1/models)")
g.add_argument("--warm", action="store_true", help="with --serve and --prompt-cache: replay the client's last request at start-up and when the date changes, so a thread's first message finds its prefix in memory")
g = ap.add_argument_group("machine")
g.add_argument("--threads", type=int, default=6, help="reader threads (measured on an M4 SSD: 4/6/8/10 = 5.35/5.45/5.38/5.06 tok/s; after zero-copy landing and the async layer pipeline, 2026-11-14: 4/6/8/12 = 5.80/5.78/5.59/5.62, two runs each)")
g.add_argument("--qos", type=int, default=1, help="1: reader and model threads ask for user-interactive QoS (performance cores)")
g.add_argument("--switch-ms", type=float, default=0, help="Python's GIL switch interval in ms (0: Python's default, 5 ms): how long a reader whose read has finished can wait for the GIL")
g.add_argument("--cache-mb", type=int, default=64, help="MLX buffer cache limit")
g.add_argument("--no-keep-hot", action="store_true", help="do not touch the model's pages from the CPU (see PageKeeper)")
g.add_argument("--cache", action="store_true", help="allow the OS page cache on the store (default: F_NOCACHE, every read hits the drive)")
g = ap.add_argument_group("research and diagnostics")
g.add_argument("--bench", action="store_true", help="timing JSON: token-by-token prefill, argmax")
g.add_argument("--check", default=None, help="a reference .pt from the cluster (exact mode: ref_released.pt), or from test_tiny.py")
g.add_argument("--check-save", default=None, help="with --check: save the logits of every compared position (.npz)")
g.add_argument("--sim-reads", action="store_true", help="timing only: records are not read from disk")
g.add_argument("--sim-lzfse", default="off", choices=["off", "scratch", "slab", "ioonly", "fields"], help="timing only (outputs are WRONG): what an LZFSE-compressed store would cost in decode. Each decode-path record read "
               "reads only ceil(--sim-lzfse-ratio x REC / 16 KB) x 16 KB from disk into its slab row; scratch / slab: then one real LZFSE decode of a pre-encoded full record into a per-thread buffer / into the slab row; ioonly: no decode; "
               "fields: only the six scale/bias blocks compressed (pass their ratio, e.g. 0.958), each decoded into the slab row")
g.add_argument("--sim-record", default=None, help="timing experiments: save every name (token clock, layer -> experts) and every generated token of this run to a JSON file")
g.add_argument("--sim-replay", default=None, help="timing experiments: replay a --sim-record file (names and generated tokens), so a run whose outputs are wrong (--sim-lzfse) issues exactly the reads of the recorded run")
g.add_argument("--sim-lzfse-ratio", type=float, default=0.9196, help="compressed size / record size for --sim-lzfse")
g.add_argument("--no-experts", action="store_true", help="timing only: skip the routed expert math")
g.add_argument("--sync-name", action="store_true", help="read each layer's prediction back immediately (default: one layer later)")
g.add_argument("--profile", action="store_true", help="per-stage ms per token (forces a sync at stage boundaries)")
g.add_argument("--trace", default=None, help="with --bench: write a timeline (every expert read with its layer and token, host events at the token boundary and per layer) to this JSON file; adds no syncs")
g.add_argument("--trace-sync", action="store_true", help="with --trace: also sync after layers 40-47 and the head, to time their GPU work (changes the overlap; diagnostic only)")
g.add_argument("-v", "--verbose", action="store_true", help="chat mode: per-turn statistics and the full load line (always on for --serve, --bench, --check)")
a = ap.parse_args()
VERBOSE = a.verbose or bool(a.serve) or a.bench or bool(a.check)
HERE = os.path.abspath(a.dir)
DT, RDT = (mx.float16 if a.dtype == "int8" else getattr(mx, a.dtype)), getattr(mx, a.router_dtype)
if a.chunk is None:
    a.chunk = 0 if a.bench or a.check else 48
t00 = time.time()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {msg}" if a.serve else msg, flush=True)


# ===================================================================================== 1. the machine: memory and threads
mx.set_cache_limit(a.cache_mb << 20)
# The model lives in GPU (unified) memory. macOS's pager ages pages by CPU references, which GPU reads never produce, so
# under any pressure it compresses the model's working set while it is being read every token (measured on a 16 GB M4:
# 13 GB of a 9 GB server in the compressor within minutes, first steps 5-15 s) and it revokes wiring requests. Two
# defences, both free: wire what we can (mlx-lm does the same for generation) and keep the pages hot from the CPU.
try:
    WIRED = int(mx.device_info().get("max_recommended_working_set_size", 0))
    if WIRED:
        mx.set_wired_limit(WIRED)
except Exception:
    WIRED = 0


def kernel_wired_gb():
    """Wired memory as the kernel reports it; the wiring request is best-effort and can be refused or revoked."""
    try:
        import subprocess

        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        return int(re.search(r"Pages wired down:\s+(\d+)", out).group(1)) * 16384 / 2**30
    except Exception:
        return -1.0


def ensure_wired():
    """Re-request residency when the kernel shows less wired memory than the model's size."""
    if not WIRED:
        return
    need = mx.get_active_memory() / 2**30
    w = kernel_wired_gb()
    if 0 <= w < need:
        try:
            mx.set_wired_limit(0)
            mx.set_wired_limit(WIRED)
        except Exception:
            pass


def set_qos_user_interactive():
    """Performance cores for the latency-critical threads (macOS puts default-QoS threads on the efficiency cores when it likes)."""
    if not a.qos:
        return
    try:
        import ctypes

        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(ctypes.c_int(0x21), ctypes.c_int(0))  # QOS_CLASS_USER_INTERACTIVE
    except Exception:
        pass


set_qos_user_interactive()
if a.switch_ms > 0:
    import sys

    sys.setswitchinterval(a.switch_ms / 1000)
pool =ThreadPoolExecutor(a.threads, initializer=set_qos_user_interactive)

# ======================================================================================= 2. model skeleton and tokenizer
cfg_full = json.load(open(os.path.join(HERE, "config.json")))
cfg = cfg_full.get("text_config", cfg_full)  # the release's config is the multimodal wrapper; the text model is all we run
meta = json.load(open(os.path.join(HERE, a.store, "meta.json")))
FMT = meta.get("dtype") or "int8"  # int8: one scale per neuron (the earlier stores); int4: MLX affine 4-bit, group 64, with biases
assert FMT in ("int8", "int4"), f"unknown expert record format {FMT!r} in {a.store}/meta.json"
H, I, NL, NE, REC, MAT = meta["hidden"], meta["intermediate"], meta["layers"], meta["experts"], meta["record_bytes"], meta["matrix_bytes"]  # MAT: elements per matrix
K, NORM, G = meta.get("top_k", 8), meta.get("norm_topk_prob", False), meta.get("group", 64)
assert G == 64 and H % G == 0 and I % G == 0 and H % 8 == 0 and I % 8 == 0
assert (H, I, NL, NE, K) == (cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_hidden_layers"], cfg["num_experts"], cfg["num_experts_per_tok"]), "store/meta.json does not match config.json"
if FMT == "int4":  # per matrix (gate | up | down): packed codes (8 per uint32, element i at bits 4i), fp16 scales, fp16 biases
    KINDS = meta["record_order"]
    assert KINDS == ["gate_proj", "up_proj", "down_proj"] and meta["shapes"] == {"gate_proj": [I, H], "up_proj": [I, H], "down_proj": [H, I]}
    OFF, SOFF, BOFF = meta["offsets"], meta["scale_offsets"], meta["bias_offsets"]
    assert all(OFF[k] % 4 == 0 and SOFF[k] % 2 == 0 and BOFF[k] % 2 == 0 for k in KINDS) and max(BOFF.values()) + 2 * MAT // G <= REC
a.m = a.m or K
if a.max_slots < 0:
    a.max_slots = 2 * K + 4
CAP = min(NE, max(8 + 4 * a.keep, a.max_slots))  # slabs start at the cap: every layer reaches it within the first tokens, and a growth copies the slab



def _vendored_imports():
    """The vendored qwen4_exp text model (qwen4_exp/, next to this file) on mlx-vlm's qwen3_5 pieces. The package
    __init__ files of mlx_vlm, mlx_vlm.speculative and the qwen3_5/qwen3_vl model packages import transformers and torch
    (~320 MB of RSS, seconds at start-up) for the processor and speculative-decoding registries; the modules the text model
    uses need neither, so those packages are entered without running their __init__ (measured: 66 MB, 0.3 s)."""
    import importlib.util, sys, types

    root = importlib.util.find_spec("mlx_vlm").submodule_search_locations[0]
    for name in ("mlx_vlm", "mlx_vlm.speculative", "mlx_vlm.models.qwen3_5", "mlx_vlm.models.qwen3_vl"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(root, *name.split(".")[1:])]
            sys.modules[name] = m
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


_vendored_imports()
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402
from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_attention_mask, _create_qwen3_5_ssm_mask  # noqa: E402
from qwen4_exp.config import TextConfig  # noqa: E402
from qwen4_exp.language import LanguageModel  # noqa: E402

assert cfg.get("model_type") in ("qwen4_exp_text", "qwen4_exp"), f"this runner serves qwen4_exp, not {cfg.get('model_type')}"
model = LanguageModel(TextConfig.from_dict(cfg))  # lazily initialised: nothing is allocated until the resident weights replace it
HC = model.args.hc_count
make_prompt_cache = lambda _model: _model.make_cache()  # noqa: E731
from tokenizers import Tokenizer  # noqa: E402  (transformers would cost ~290 MB of RSS for the same tokenizer and template)


class Tok:
    """The release's tokenizer and chat template with the small libraries only: `tokenizers` for the pieces, jinja2 for the
    template (renders byte-for-byte what transformers renders; checked on plain, tool-call and thinking conversations)."""

    def __init__(self, path):
        import jinja2
        import jinja2.ext

        self.tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        tc = json.load(open(os.path.join(path, "tokenizer_config.json")))
        self.chat_template, self.bos_token, self.eos_token = tc.get("chat_template"), tc.get("bos_token"), tc.get("eos_token")
        if not self.chat_template and os.path.exists(os.path.join(path, "chat_template.jinja")):  # newer releases ship the template as its own file
            self.chat_template = open(os.path.join(path, "chat_template.jinja")).read()
        self.bos_token, self.eos_token = [t.get("content") if isinstance(t, dict) else t for t in (self.bos_token, self.eos_token)]
        self.eos_token_id = self.tk.token_to_id(self.eos_token) if self.eos_token else None
        class Generation(jinja2.ext.Extension):  # transformers' {% generation %} ... {% endgeneration %} (it marks the assistant's tokens): the body renders unchanged
            tags = {"generation"}

            def parse(self, parser):
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

            def _body(self, caller):
                return caller()

        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", Generation])

        def raise_exception(m):
            raise jinja2.TemplateError(m)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
        self.tpl = env.from_string(self.chat_template) if self.chat_template else None

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        if a.reasoning_effort and "reasoning_effort" not in kw:  # every front-end (chat, --prompt, server, warm) renders here
            kw["reasoning_effort"] = a.reasoning_effort
        return self.tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, bos_token=self.bos_token, eos_token=self.eos_token, **kw)

    def prompt_ids(self, messages, tools=None, **kw):
        """Token ids of a conversation, and the positions where each message ends (the prefixes worth snapshotting). The
        boundaries are found in the one rendering, after each "<|im_end|>\\n" (ChatML; this template refuses to render a
        conversation without a user message, so the system prompt cannot be rendered alone), at a token boundary."""
        text = self.render(messages, tools, **kw)
        enc = self.tk.encode(text, add_special_tokens=False)
        ids, starts = enc.ids, {s: i for i, (s, e) in enumerate(enc.offsets)}
        marks, sep = [], "<|im_end|>\n"
        p = text.find(sep)
        while p >= 0:
            q = p + len(sep)
            if q in starts:
                marks.append(starts[q])
            p = text.find(sep, q)
        return ids, marks


tok = Tok(HERE)
if a.chat is None:
    a.chat = bool(tok.chat_template) and not a.check and not a.bench and not a.serve
a.max_new = a.max_new or (4096 if a.chat or a.serve else 40)  # a turn runs to the stop token; 4096 matches the demo config (OpenCode asks for 4096); --bench and --prompt --no-chat keep 40

# ============================================================================ 3. the store: records, slots, routers, experts
fds = []
for L in range(NL):
    fd = os.open(os.path.join(HERE, a.store, f"layer_{L:02d}.bin"), os.O_RDONLY)
    if not a.cache:
        try:
            import fcntl

            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except Exception:
            pass
    fds.append(fd)

# One packed slab per layer, its fields depending on the record format. int8: slot = the record's three int8 matrices as
# uint32 code words (gate, up, down) plus the three per-neuron scale vectors; the per-group scale tensors MLX's kernel wants
# are broadcast from the per-neuron scales at use time (int8 with one scale per neuron == affine 8-bit with code q+128,
# scale s, bias -128 s for every group of that neuron; down_proj's per-input-column scales are applied to the activation).
# int4: the records are already MLX affine 4-bit, so a slot is the three matrices' packed codes, scales and biases as they
# are on disk. Landing an expert is one copy either way. A slab grows by GROW slots whenever the keep window needs more
# than the cap allows, so the window is never cut short by the allocation.
FIELDS = ({"w": ((3, MAT // 4), mx.uint32), "sc": ((3, I), mx.float16)} if FMT == "int8" else
          {"w": ((3, MAT // 8), mx.uint32), "s": ((3, MAT // G), mx.float16), "b": ((3, MAT // G), mx.float16)})
# Zero-copy landing: a layer's slab is one [cap, REC/4] uint32 array holding the records as they are on disk, row = slot.
# A reader thread preadv's a record straight into its row through a numpy view of the evaluated MLX buffer (unified
# memory: the GPU sees the bytes), and the fields are strided views of the rows (mx.as_strided, no copy) with the record's
# own layout. MLX never writes a slab (no slice updates, so the buffer is never reallocated under a reader); only the
# reader threads (decode), the batched prefill's window copy and grow() write it, from the host, while no GPU work that
# reads those rows is pending. int8 records are relabelled (q ^ 0x80) in place right after the read.
assert REC % 4 == 0
if FMT == "int4":
    PER = OFF["up_proj"] - OFF["gate_proj"]
    assert OFF["down_proj"] - OFF["up_proj"] == PER and len({SOFF[k] - OFF[k] for k in KINDS}) == 1 and len({BOFF[k] - OFF[k] for k in KINDS}) == 1
    FVIEW = {"w": (mx.uint32, MAT // 8, PER // 4, OFF["gate_proj"] // 4), "s": (mx.float16, MAT // G, PER // 2, SOFF["gate_proj"] // 2), "b": (mx.float16, MAT // G, PER // 2, BOFF["gate_proj"] // 2)}
else:
    FVIEW = {"w": (mx.uint32, MAT // 4, MAT // 4, 0), "sc": (mx.float16, I, I, 3 * MAT // 2)}
SLAB_BASE = [mx.zeros((CAP, REC // 4), dtype=mx.uint32) for _ in range(NL)]
mx.eval(*SLAB_BASE)
SLAB_NP = [np.asarray(b).view(np.uint8) for b in SLAB_BASE]  # [cap, REC] zero-copy host views (the reader threads write rows)
assert all(v.flags.writeable and not v.flags.owndata for v in SLAB_NP)


def slab_fields(L):
    """Layer L's slab as the expert kernel's fields: strided views of the record rows (built per use, so they always read
    the current bytes)."""
    base, n = SLAB_BASE[L], SLAB_BASE[L].shape[0]
    out = {}
    for k, (dt, length, stride, off) in FVIEW.items():
        src = base if dt == mx.uint32 else base.view(dt)
        q = 4 if dt == mx.uint32 else 2
        out[k] = mx.as_strided(src, (n, 3, length), (REC // q, stride, 1), off)
    return out


CAPS = [CAP] * NL
GROW = 4
SDB = {}  # cap -> (down_proj group scales = 1, biases = -128), shared by every layer of that size
SLOT_OF = [dict() for _ in range(NL)]  # expert -> slot
SLOT_BY = [dict() for _ in range(NL)]  # slot -> expert
TAB = [np.full(NE, -1, dtype=np.int32) for _ in range(NL)]  # expert -> slot once LANDED (-1 otherwise); copied to the device per layer
PENDING = []  # (layer, prediction array) issued asynchronously, read back one layer later
FREE = [list(range(CAP)) for _ in range(NL)]
LAST = [dict() for _ in range(NL)]  # expert -> token at which it was last named (prefetch) or used (exact)
INFL = [dict() for _ in range(NL)]  # expert -> future of its read
st = {"t": -1, "bytes": 0, "bytes_batch": 0, "wait": 0.0, "hits_dev": mx.array(0), "mass_dev": mx.zeros(2), "asked": 0, "grown": 0, "early": 0, "held": [0, 0], "sync": 0, "emb": None, "h_prev": [None] * NL}
mx.eval(st["hits_dev"], st["mass_dev"])  # materialised: the server evaluates them on its model thread
PROF = {}
RLOG = []  # (submitted, started, finished, layer, token) per read, under --profile or --trace
TR = []  # (event, time, token) under --trace


def tr(k):
    if a.trace:
        TR.append((k, time.perf_counter(), st["t"]))


def sdb(cap):
    if cap not in SDB:
        SDB[cap] = (mx.ones((cap, H, I // G), dtype=mx.float16), mx.full((cap, H, I // G), -128.0, dtype=mx.float16))
        mx.eval(*SDB[cap])
    return SDB[cap]


if FMT == "int8":
    sdb(CAP)


class stage:
    """with stage("x"): ... accumulates wall time; under --profile the enclosed graph is evaluated at exit."""

    def __init__(self, k, *arrs):
        self.k, self.arrs = k, arrs

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *_):
        if a.profile and self.arrs:
            mx.eval(*self.arrs)
        PROF[self.k] = PROF.get(self.k, 0.0) + time.perf_counter() - self.t0


def grow(L):
    """GROW more slots: a new slab with the old rows copied on the host. The reads in flight write the old rows, so they
    are waited for first (they are landed by flush as usual); the old buffer stays alive while queued GPU work holds it."""
    for slot, f in INFL[L].values():
        f.result()
    n = CAPS[L]
    new = mx.zeros((n + GROW, REC // 4), dtype=mx.uint32)
    mx.eval(new)
    new_np = np.asarray(new).view(np.uint8)
    new_np[:n] = SLAB_NP[L][:n]
    SLAB_BASE[L], SLAB_NP[L] = new, new_np
    FREE[L].extend(range(n, n + GROW))
    CAPS[L] = n + GROW
    st["grown"] += 1


def relabel_int8(row):
    """int8 records in MLX's affine 8-bit layout, in place: code = q + 128 == q ^ 0x80 (four codes per little-endian
    uint32); the per-neuron scales stay as they are (bias = -128 * scale is formed at use time)."""
    np.bitwise_xor(row[: 3 * MAT], np.uint8(0x80), out=row[: 3 * MAT])


LZ = {"dec_s": 0.0, "n": 0, "io_s": 0.0}
if a.sim_lzfse != "off":  # timing experiment: a handful of real records, LZFSE-encoded once, decoded in turn by the reader threads
    import ctypes, itertools

    _LZLIB = ctypes.CDLL("/usr/lib/libcompression.dylib")  # ctypes releases the GIL for the duration of the call
    _LZLIB.compression_encode_buffer.argtypes = _LZLIB.compression_decode_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    _LZLIB.compression_encode_buffer.restype = _LZLIB.compression_decode_buffer.restype = ctypes.c_size_t
    _LZLIB.compression_decode_scratch_buffer_size.argtypes, _LZLIB.compression_decode_scratch_buffer_size.restype = [ctypes.c_int], ctypes.c_size_t
    LZ_ALG = 0x801  # COMPRESSION_LZFSE
    LZ_RED = -(-int(a.sim_lzfse_ratio * REC) // 16384) * 16384
    LZ_ENC, LZ_RAW_CRC, LZ_FIELDS = [], [], []
    _raw = ctypes.create_string_buffer(REC)
    for k_ in range(16):
        L_, e_ = (k_ * 7) % NL, (k_ * 53) % NE
        os.preadv(fds[L_], [memoryview(_raw).cast("B")], e_ * REC)
        out_ = ctypes.create_string_buffer(REC + 65536)
        n_ = _LZLIB.compression_encode_buffer(out_, REC + 65536, _raw, REC, None, LZ_ALG)
        assert n_ > 0
        LZ_ENC.append((out_, n_))
        LZ_RAW_CRC.append(hash(_raw.raw))
        if a.sim_lzfse == "fields":  # the six fp16 scale/bias blocks of the record, each LZFSE-encoded on its own
            blk = []
            for o_ in [SOFF[k] for k in KINDS] + [BOFF[k] for k in KINDS]:
                src_ = ctypes.create_string_buffer(_raw.raw[o_ : o_ + 2 * MAT // G], 2 * MAT // G)
                ob_ = ctypes.create_string_buffer(2 * MAT // G + 65536)
                nb_ = _LZLIB.compression_encode_buffer(ob_, len(ob_), src_, 2 * MAT // G, None, LZ_ALG)
                blk.append((o_, ob_, nb_))
            LZ_FIELDS.append(blk)
    _chk = ctypes.create_string_buffer(REC)  # every sample decodes back to its record byte for byte
    for (b_, n_), h_ in zip(LZ_ENC, LZ_RAW_CRC):
        assert _LZLIB.compression_decode_buffer(_chk, REC, b_, n_, None, LZ_ALG) == REC and hash(_chk.raw) == h_
    LZ["sample_ratio"] = sum(n_ for _, n_ in LZ_ENC) / (len(LZ_ENC) * REC)
    LZ["sample_padded_ratio"] = sum(-(-n_ // 16384) * 16384 for _, n_ in LZ_ENC) / (len(LZ_ENC) * REC)
    del _raw, _chk
    LZ_NEXT = itertools.count()
    _LZ_TL = threading.local()


def lz_decode(row):
    """One real LZFSE decode of a full record (a pre-encoded sample) into this thread's scratch or into the slab row."""
    tl = _LZ_TL
    if not hasattr(tl, "scr"):
        tl.scr = ctypes.create_string_buffer(_LZLIB.compression_decode_scratch_buffer_size(LZ_ALG))
        tl.out = ctypes.create_string_buffer(REC)
    if a.sim_lzfse == "fields":
        base_ = row.ctypes.data
        for o_, ob_, nb_ in LZ_FIELDS[next(LZ_NEXT) % len(LZ_FIELDS)]:
            got = _LZLIB.compression_decode_buffer(base_ + o_, 2 * MAT // G, ob_, nb_, tl.scr, LZ_ALG)
            assert got == 2 * MAT // G
        return
    b_, n_ = LZ_ENC[next(LZ_NEXT) % len(LZ_ENC)]
    dst = row.ctypes.data if a.sim_lzfse == "slab" else ctypes.addressof(tl.out)
    got = _LZLIB.compression_decode_buffer(dst, REC, b_, n_, tl.scr, LZ_ALG)
    assert got == REC, f"lzfse decode gave {got} of {REC}"


def read_record(L, e, row, t_sub=None, tok=None):
    """One expert from disk, straight into its slab row (a [REC] uint8 view; one aligned read, no host copy)."""
    t_start = time.perf_counter()
    if a.sim_lzfse != "off" and not a.sim_reads:  # timing experiment (see --sim-lzfse): the reduced read, then the decode
        n = os.preadv(fds[L], [row[:LZ_RED]], e * REC)
        assert n == LZ_RED, f"short read: {n} of {LZ_RED}"
        t_io = time.perf_counter()
        if a.sim_lzfse != "ioonly":
            lz_decode(row)
        LZ["io_s"] += t_io - t_start
        LZ["dec_s"] += time.perf_counter() - t_io
        LZ["n"] += 1
        st["bytes"] += LZ_RED
    elif not a.sim_reads:
        n = os.preadv(fds[L], [row], e * REC)
        assert n == REC, f"short read: {n} of {REC}"
        if FMT == "int8":
            relabel_int8(row)
    if a.sim_lzfse == "off" or a.sim_reads:
        st["bytes"] += REC
    if a.profile or a.trace:
        RLOG.append((t_sub or t_start, t_start, time.perf_counter(), L, tok))


def land(L, slots):
    """Mark read records as in memory (their bytes are already in their rows)."""
    for slot in slots:
        TAB[L][SLOT_BY[L][slot]] = slot


def take_slot(L, t):
    """A free slot: evict the least recently named expert outside the keep window; else grow the slab; at --max-slots,
    evict the least recently named expert inside the window but outside this token's set (the evaluated capped cell)."""
    if FREE[L]:
        return FREE[L].pop()
    victim = min((e for e in SLOT_OF[L] if LAST[L].get(e, -1) < t - a.keep and e not in INFL[L]), key=lambda e: LAST[L][e], default=None)
    if victim is None and (not a.max_slots or CAPS[L] < a.max_slots):
        grow(L)
        return FREE[L].pop()
    if victim is None:
        cands = [e for e in SLOT_OF[L] if LAST[L].get(e, -1) < t and e not in INFL[L]]
        if not cands:  # everything held is this token's own set or still in flight: grow regardless of the cap
            grow(L)
            return FREE[L].pop()
        victim = min(cands, key=lambda e: LAST[L][e])
        st["early"] += 1
    del LAST[L][victim]
    TAB[L][victim] = -1
    return SLOT_OF[L].pop(victim)


def reset_slots():
    """Forget every held expert (a new session or a restored prefix: the token clock restarts, so the window state is void)."""
    for L in range(NL):
        for e, (slot, f) in list(INFL[L].items()):
            f.result()
        INFL[L].clear()
        SLOT_OF[L].clear()
        SLOT_BY[L].clear()
        LAST[L].clear()
        TAB[L][:] = -1
        FREE[L] = list(range(CAPS[L]))
    PENDING.clear()
    ng_drop()


def slot_state():
    """What a snapshot needs of the expert slots for the restored prefix to continue exactly as the live one would: per
    layer the held experts in slot-table order (the order breaks ties between eviction candidates) and the token clock at
    which each was last named. Taken between steps, when nothing is in flight."""
    out = []
    for L in range(NL):
        flush(L, must=set(INFL[L]))
        held = [e for e in SLOT_OF[L] if TAB[L][e] >= 0]
        out.append((np.array(held, dtype=np.int32), np.array([LAST[L].get(e, -1) for e in held], dtype=np.int64)))
    return out


def restore_slots(state):
    """Re-read a snapshot's held experts into the slabs (~1,100 records, ~3 GB, on the reader pool) and rebuild the table."""
    reset_slots()
    futs = []
    for L, (held, last) in enumerate(state):
        for slot, (e, t_) in enumerate(zip(held.tolist(), last.tolist())):
            if slot >= CAPS[L]:
                break
            SLOT_OF[L][e], SLOT_BY[L][slot], LAST[L][e] = slot, e, t_
            futs.append((L, e, slot, pool.submit(read_record, L, e, SLAB_NP[L][slot])))
        FREE[L] = list(range(min(len(held), CAPS[L]), CAPS[L]))
    for L, e, slot, f in futs:
        f.result()
        TAB[L][e] = slot


SIMREC = json.load(open(a.sim_replay)) if a.sim_replay else ({"names": {}, "tokens": []} if a.sim_record else None)


def name(L, pred):
    """Request layer L's named experts that are not in memory."""
    t = st["t"]
    if SIMREC is not None:  # timing experiments only
        if a.sim_replay:
            pred = SIMREC["names"][f"{t}:{L}"]
        else:
            SIMREC["names"][f"{t}:{L}"] = [int(c) for c in pred]
    for c in pred:  # first mark what is already here, so it cannot be chosen as a victim for the rest
        if c in SLOT_OF[L]:
            LAST[L][c] = t
    for c in pred:
        if c in SLOT_OF[L]:
            continue
        slot = take_slot(L, t)
        SLOT_OF[L][c], SLOT_BY[L][slot], LAST[L][c] = slot, c, t
        INFL[L][c] = (slot, pool.submit(read_record, L, c, SLAB_NP[L][slot], time.perf_counter(), t))


def flush(L, must=()):
    """Land the reads for layer L that have finished, plus the experts in `must` (waited for)."""
    done = []
    for e in list(INFL[L]):
        slot, f = INFL[L][e]
        if e in must or f.done():
            f.result()
            done.append(slot)
            del INFL[L][e]
    if done:
        land(L, done)


# -- routers (lag-4 MLPs trained on GPUs with the model frozen), in MLX. A .pt is converted once, in a subprocess, to a .npz next to it.
# Their inputs here: the token embedding, the previous token's mixed input at L and this token's at L - lag (the mixed input
# = the MLP hyper-connection's output that feeds the expert block). Without a routers file only exact mode runs.
a.routers = a.routers or next((os.path.join(HERE, f) for f in ("routers_k4.npz", "routers_k4.pt", "mlp_k4.npz", "mlp_k4.pt", "routers.npz", "routers.pt") if os.path.exists(os.path.join(HERE, f))), None)
ROUT, LAG = [], 0
if a.routers is None or not os.path.exists(a.routers):
    if a.mode == "prefetch":
        raise SystemExit(f"prefetch mode needs the lag-4 routers, and there is no routers file ({a.routers or 'none of routers_k4.npz/.pt, mlp_k4.npz/.pt, routers.npz/.pt in ' + HERE}). "
                         "They are routers_k4.npz in the store's Hugging Face repository; pass --routers PATH, or run --mode exact "
                         "(the released router, every missing expert read on the spot).")
else:
    npz = a.routers if a.routers.endswith(".npz") else a.routers[: -len(".pt")] + ".npz"
    if not os.path.exists(npz):
        import subprocess, sys

        log(f"converting {os.path.basename(a.routers)} -> {os.path.basename(npz)} (once)")
        subprocess.run([sys.executable, "-c", "import sys,torch,numpy as np; b=torch.load(sys.argv[1],map_location='cpu'); sd=b['state'] if 'state' in b else b; "
                        "np.savez(sys.argv[2], __lag__=np.array(b.get('lag',0) if 'state' in b else 0), **{k:v.float().numpy() for k,v in sd.items()})", a.routers, npz], check=True)
    blob = np.load(npz)
    LAG = int(blob["__lag__"])
    for L in range(NL):
        r = {k[len(f"{L}.") :]: mx.array(blob[k]).astype(RDT) for k in blob.files if k.startswith(f"{L}.")}
        assert r["lin.weight"].shape[0] == NE and r["lin.weight"].shape[1] in (2 * H, 3 * H), f"router {L}: {r['lin.weight'].shape} is not for this model"
        r["lagged"] = r["lin.weight"].shape[1] == 3 * H
        ROUT.append(r)
    mx.eval(*[v for r in ROUT for v in r.values() if isinstance(v, mx.array)])
    del blob
EPS = 1.1920929e-07  # torch.nn.RMSNorm default for float32


def route(L, emb, h_prev, h_lag):
    r = ROUT[L]
    e = emb.astype(mx.float32)
    e = (e * mx.rsqrt(mx.mean(e * e, axis=-1, keepdims=True) + EPS)).astype(RDT) * r["emb_norm.weight"]
    parts = [e, h_prev.astype(RDT)] + ([h_lag.astype(RDT)] if r["lagged"] else [])
    x = mx.concatenate(parts, axis=-1)
    z = x @ r["lin.weight"].T + r["lin.bias"]
    if "mlp.0.weight" in r:
        h = nn.gelu(x @ r["mlp.0.weight"].T + r["mlp.0.bias"])
        z = z + h @ r["mlp.2.weight"].T + r["mlp.2.bias"]
    return mx.argpartition(-z, kth=a.m - 1, axis=-1)[..., : a.m]


# -- the expert math: one token from the slab, or a whole batch streamed through it
def moe_qmm(x, F, idx, sorted_indices=False):
    """The experts' MLP from a packed buffer of n records (the slab's fields F: a slab, or a chunk of a batch) as three
    fused quantised gathers. x [..., 1, H] and idx [...] (which record each row uses) -> [..., H].
    int4: F["w"] [n, 3, MAT/8] uint32, F["s"], F["b"] [n, 3, MAT/64] fp16, straight into MLX's affine 4-bit kernel.
    int8: F["w"] [n, 3, MAT/4] uint32, F["sc"] [n, 3, I] fp16; 8-bit gathers with the scales broadcast from the per-neuron
    vectors, down_proj's folded into the activation."""
    n = F["w"].shape[0]
    if FMT == "int4":
        def q4(x_, j, rows, cols):
            return mx.gather_qmm(x_, F["w"][:, j].reshape(n, rows, cols // 8), F["s"][:, j].reshape(n, rows, cols // G), F["b"][:, j].reshape(n, rows, cols // G),
                                 rhs_indices=idx, transpose=True, group_size=G, bits=4, sorted_indices=sorted_indices)

        return q4(nn.silu(q4(x, 0, I, H)) * q4(x, 1, I, H), 2, H, I).squeeze(-2)
    w, sc = F["w"], F["sc"]
    sg = mx.broadcast_to(sc[:, 0, :, None], (n, I, H // G))
    su = mx.broadcast_to(sc[:, 1, :, None], (n, I, H // G))
    sd1, bd128 = sdb(n)

    def qmm(x_, m, scales, biases):
        return mx.gather_qmm(x_, m, scales, biases, rhs_indices=idx, transpose=True, group_size=G, bits=8, sorted_indices=sorted_indices)

    gt = qmm(x, w[:, 0].reshape(n, I, H // 4), sg, sg * -128)
    up = qmm(x, w[:, 1].reshape(n, I, H // 4), su, su * -128)
    act = nn.silu(gt) * up * mx.expand_dims(sc[idx, 2], -2)
    return qmm(act, w[:, 2].reshape(n, H, I // 4), sd1, bd128).squeeze(-2)


def expert_step(xf, F, slots, scores):
    """One token's chosen experts from the slab ([1, K] slots), their outputs summed with the gate's scores."""
    y = moe_qmm(mx.expand_dims(xf.astype(mx.float16), (-2, -3)), F, slots)  # [1, K, H]
    return (y * scores[..., None]).sum(axis=-2)


STAGE = []  # three staging buffers for batched prefill: records are read straight into GPU-visible memory, two batches ahead
STAGE_VIEWS = []


def stage_buffers():
    if not STAGE:
        for _ in range(3):
            STAGE.append(mx.zeros((CAP, REC), dtype=mx.uint8))
        mx.eval(*STAGE)
        STAGE_VIEWS.extend(np.asarray(b) for b in STAGE)  # zero-copy host views the reader threads write into
    return STAGE, STAGE_VIEWS


def stage_free():
    """After a batch: give the staging ring (3 x CAP records, ~200 MB) back to the system; the next batch allocates it again."""
    STAGE.clear()
    STAGE_VIEWS.clear()
    mx.clear_cache()


RT = {"lag": 0.0, "read": 0.0, "n": 0}  # read pipeline timing: submission -> start lag, read duration (diagnostic, printed under -v)
readers = ThreadPoolExecutor(8, initializer=set_qos_user_interactive)  # chunk reads of batched prefill: their own pool, so two chunks in flight never queue behind each other's records


def read_into(L, experts, view, t_sub=None):
    """Records `experts` of layer L read straight into rows of a staging buffer (no host copy): one read for a contiguous
    run of experts (the streamed schedule), else one read per record on the reader pool."""
    t_start = time.perf_counter()
    n_exp = len(experts)
    if not a.sim_reads:
        if experts[-1] - experts[0] == n_exp - 1:  # contiguous in the file: a few large parallel reads (one big read on one thread gets 1.2 GB/s under load; six in flight get the drive's rate)
            k = min(6, n_exp)
            bounds = [(i * n_exp // k, (i + 1) * n_exp // k) for i in range(k)]

            def rd_part(b):
                lo, hi = b
                n = os.preadv(fds[L], [view[lo:hi]], (experts[0] + lo) * REC)
                assert n == (hi - lo) * REC, f"short read: {n} of {(hi - lo) * REC}"

            list(readers.map(rd_part, bounds))
        else:
            def rd(ie):
                i, e = ie
                n = os.preadv(fds[L], [view[i]], e * REC)
                assert n == REC, f"short read: {n} of {REC}"

            list(readers.map(rd, enumerate(experts)))
    st["bytes"] += n_exp * REC
    RT["lag"] += t_start - (t_sub or t_start)
    RT["read"] += time.perf_counter() - t_start
    RT["n"] += 1


X80 = mx.array(0x80, dtype=mx.uint8)


def chunk_fields(buf, n):
    """The first n records of a staging buffer ([CAP, REC] uint8, on the GPU) as the slab's fields."""
    if FMT == "int8":  # relabelled (q ^ 0x80) on the GPU
        return {"w": (buf[:n, : 3 * MAT] ^ X80).view(mx.uint32).reshape(n, 3, MAT // 4), "sc": (buf[:n, 3 * MAT : 3 * MAT + 6 * I] + 0).view(mx.float16).reshape(n, 3, I)}

    if not a.chunk_copy:  # int4: strided views of the ring rows, as slab_fields does for the slab (no GPU copy of the chunk)
        b32 = buf.view(mx.uint32)
        out = {}
        for k, (dt, length, stride, off) in FVIEW.items():
            src = b32 if dt == mx.uint32 else b32.view(dt)
            q = 4 if dt == mx.uint32 else 2
            out[k] = mx.as_strided(src, (n, 3, length), (REC // q, stride, 1), off)
        return out

    def part(off, nbytes, dt):
        return mx.stack([(buf[:n, off[k] : off[k] + nbytes] + 0).view(dt) for k in KINDS], axis=1)

    return {"w": part(OFF, MAT // 2, mx.uint32), "s": part(SOFF, 2 * (MAT // G), mx.float16), "b": part(BOFF, 2 * (MAT // G), mx.float16)}
STREAM = {"on": False, "chunks": [], "next": 0, "taken": -1, "futs": {}}  # long batches: the fixed schedule of (layer, experts) chunks and the reads in flight


def stream_begin():
    """Plan the whole prefill's reads: every expert of every layer in file order (a long batch touches >80% of them anyway),
    in chunks of CAP; start two chunks ahead. The reader runs through layer boundaries, so the drive never waits on routing."""
    chunks = [(L, list(range(e0, min(e0 + CAP, NE)))) for L in range(NL) for e0 in range(0, NE, CAP)]
    STREAM.update(on=True, chunks=chunks, next=0, taken=-1, futs={})
    _, views = stage_buffers()
    for i in range(min(2, len(chunks))):
        STREAM["futs"][i] = readers.submit(read_into, chunks[i][0], chunks[i][1], views[i % 3], time.perf_counter())
    STREAM["next"] = min(2, len(chunks))


def stream_take():
    """The next chunk, read and in its buffer; queue the chunk two ahead into the buffer that batch i-1 just finished with."""
    i = STREAM["taken"] = STREAM["taken"] + 1
    STREAM["futs"].pop(i).result()
    n = STREAM["next"]
    if n < len(STREAM["chunks"]):
        _, views = stage_buffers()
        STREAM["futs"][n] = readers.submit(read_into, STREAM["chunks"][n][0], STREAM["chunks"][n][1], views[n % 3], time.perf_counter())
        STREAM["next"] = n + 1
    return i, STREAM["chunks"][i][1]


def stream_end():
    for f in STREAM["futs"].values():
        f.result()
    STREAM.update(on=False, chunks=[], next=0, futs={}, taken=-1)



BP = {"wait": 0.0, "host": 0.0, "gpu": 0.0, "batches": 0, "lin": 0.0, "qsa": 0.0, "moe": 0.0}  # batched prefill, per expert chunk: waiting for its read / host prep / gpu; per layer: linear-attention, sparse-attention and MoE wall time (printed under -v)


CPROF = bool(os.environ.get("FLASH_CHUNK_PROF"))  # diagnostics: sync-timed parts of each batched-prefill expert chunk
CP = {"fields": 0.0, "qmm": 0.0, "scatter": 0.0, "rounds": 0, "n": 0}


def chunk_rounds(tok):
    """For pairs whose tokens are `tok` (host ints, in expert order): each pair's occurrence rank among the pairs of its
    token (0 for a token's first pair, 1 for its second, ...)."""
    o = np.argsort(tok, kind="stable")
    ts = tok[o]
    start = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    rank = np.empty(len(tok), dtype=np.int64)
    rank[o] = np.arange(len(tok)) - np.repeat(start, np.diff(np.r_[start, len(tok)]))
    return rank


def experts_batch(L, xf, inds, scores):
    """All T tokens' expert MLPs for layer L with the experts streamed from disk: (token, expert) pairs sorted by expert,
    each needed expert read once, sorted gather_qmm per chunk, fp32 scatter-add into y in a fixed order (chunk_rounds).
    Records are read two chunks ahead into a ring of three GPU-visible buffers and turned into the slab's fields on the GPU
    (chunk_fields), so the drive streams continuously and no byte is copied on the host. The chunks are this layer's needed experts in id
    order, or every expert in file order when the whole prefill's schedule is streamed (stream_begin). The experts the
    last `keep` tokens used are copied out as their chunk passes and landed in the slab once, so the layer ends holding
    exactly the window a token-by-token prefill would have left."""
    T = xf.shape[0]
    flat = inds.reshape(-1)
    order = np.argsort(flat, kind="stable")
    e_sorted, tok_sorted = flat[order], order // K
    xh = xf.astype(mx.float16)
    sc_pairs = scores.reshape(-1)[mx.array(order)]
    y = mx.zeros((T, H), dtype=mx.float32)
    base = st["t"]  # the batch's tokens take clocks base+1 .. base+T
    win = tok_sorted >= T - a.keep
    latest = {}
    for e_, t_ in zip(e_sorted[win].tolist(), tok_sorted[win].tolist()):
        latest[e_] = max(latest.get(e_, -1), t_)
    keep_set = sorted(sorted(latest, key=lambda e_: -latest[e_])[: CAPS[L]])  # at most one slab of window experts, the latest tokens' first
    slot_of = {e_: i for i, e_ in enumerate(keep_set)}
    bufs, views = stage_buffers()

    def union():  # this layer's experts in id order, read two chunks ahead; buffer (b+2)%3 was chunk b-1's, whose GPU work finished at its eval
        uniq = np.unique(e_sorted).tolist()
        chunks = [uniq[i : i + CAP] for i in range(0, len(uniq), CAP)]
        futs = {i: readers.submit(read_into, L, chunks[i], views[i % 3], time.perf_counter()) for i in range(min(2, len(chunks)))}
        for b, exp in enumerate(chunks):
            futs.pop(b).result()
            if b + 2 < len(chunks):
                futs[b + 2] = readers.submit(read_into, L, chunks[b + 2], views[(b + 2) % 3], time.perf_counter())
            yield b % 3, exp

    def streamed():  # the cross-layer schedule: every expert, in file order
        for _ in range((NE + CAP - 1) // CAP):
            i, exp = stream_take()
            yield i % 3, exp

    t_a = time.perf_counter()
    for r, exp in (streamed() if STREAM["on"] else union()):
        t_b = time.perf_counter()
        n = len(exp)
        F = chunk_fields(bufs[r], n)
        if CPROF:
            mx.eval(*F.values()); CP["fields"] += time.perf_counter() - t_b; t_p = time.perf_counter()
        lo, hi = np.searchsorted(e_sorted, [exp[0], exp[-1] + 1])  # this chunk's pairs: a run, since chunks are ascending slices of the sorted experts
        landed = [(slot_of[e_], j) for j, e_ in enumerate(exp) if e_ in slot_of]
        t_c = time.perf_counter()
        if hi > lo:
            tk = mx.array(tok_sorted[lo:hi])
            idx = mx.array(np.searchsorted(np.array(exp), e_sorted[lo:hi]).astype(np.uint32))  # each pair's record in the chunk (ascending: sorted gathers)
            yp = moe_qmm(mx.expand_dims(xh[tk], -2), F, idx, sorted_indices=True)
            contrib = yp.astype(mx.float32) * sc_pairs[lo:hi, None].astype(mx.float32)
            if CPROF:
                mx.eval(contrib); CP["qmm"] += time.perf_counter() - t_p; t_p = time.perf_counter()
            # Deterministic accumulation: a GPU scatter-add whose index has duplicates (a token using several of this
            # chunk's experts) adds them with atomics in a varying order, so two identical runs differed in the last fp32
            # bit, which fp16 rounding and the routing then amplify. Split the chunk's pairs into rounds in which every
            # token appears at most once (round j = each token's j-th pair in expert order): each round's scatter-add
            # touches every row at most once, so the sum is formed in one fixed order.
            t_np = tok_sorted[lo:hi]
            rank = chunk_rounds(t_np)
            for j in range(int(rank.max()) + 1):
                sel = np.flatnonzero(rank == j)
                y = y.at[mx.array(t_np[sel])].add(contrib[mx.array(sel)] if len(sel) < len(t_np) else contrib)
        for slot, j in landed:  # the window experts' raw records, from the ring row (read and final) into their slab rows on the host
            SLAB_NP[L][slot] = views[r][j]
            if FMT == "int8":
                relabel_int8(SLAB_NP[L][slot])
        mx.eval(y)
        t_d = time.perf_counter()
        if CPROF and hi > lo:
            CP["scatter"] += t_d - t_p; CP["rounds"] += int(rank.max()) + 1; CP["n"] += 1
        BP["wait"] += t_b - t_a
        BP["host"] += t_c - t_b
        BP["gpu"] += t_d - t_c
        BP["batches"] += 1
        t_a = time.perf_counter()
    SLOT_OF[L].clear()
    SLOT_BY[L].clear()
    LAST[L].clear()
    TAB[L][:] = -1
    for i, e_ in enumerate(keep_set):
        SLOT_OF[L][e_], SLOT_BY[L][i], LAST[L][e_], TAB[L][e_] = i, e_, base + 1 + latest.get(e_, T - 1), i
    FREE[L] = list(range(len(keep_set), CAPS[L]))
    st["bytes_batch"] += (NE if STREAM["on"] else len(np.unique(e_sorted))) * REC
    return y


# ============================================================================ 4. the block, the resident weights, embeddings
class StoreMoE(nn.Module):
    """The expert block served from the slab: released gate, top-k among the experts in memory, fused gather_qmm.
    The resident shared expert (and its sigmoid gate) is kept from the original block and added as released. Its input x
    is the MLP hyper-connection's mixed tensor, which is also what the lag-4 routers read (h_prev, h_lag)."""

    def __init__(self, L, orig):
        super().__init__()
        self.L = L
        self.gate = nn.Linear(H, NE, bias=False)
        if hasattr(orig, "shared_expert"):
            self.shared_expert, self.shared_expert_gate = orig.shared_expert, orig.shared_expert_gate

    def __call__(self, x):
        L, t = self.L, st["t"]
        tr(f"a{L}")
        xf = x.reshape(-1, H)
        if st.get("batch"):  # batched prefill: the released routing for every token, each needed expert streamed once
            # the last token's input, as its own buffer: a slice (xf[-1]) is a view that keeps the whole [T, H] batch input
            # alive while it sits in h_prev (measured: +34 MB per layer, +1.6 GB over a 4k batch, and the next batch on top)
            st["h_prev_next"][L] = xf[-1] * 1
            gates = mx.softmax(self.gate(xf), axis=-1, precise=True)
            inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if NORM:
                scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-9)
            mx.eval(inds, scores, st["h_prev_next"][L])  # evaluate here, catchably: an out-of-memory thrown inside np.array()'s buffer-protocol call aborts the process
            y = mx.zeros_like(x) if a.no_experts else experts_batch(L, xf, np.array(inds), scores).reshape(x.shape).astype(x.dtype)
            if hasattr(self, "shared_expert"):
                y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            mx.eval(y)
            if st.get("tick"):
                st["tick"](L + 1, NL)
            KEEPER.touch_slabs(1)
            return y
        st["h_prev_next"][L] = xf[0]
        with stage("predict+name"):
            while PENDING:  # last layer's prediction: the GPU has had a layer to finish it
                Lp, pred = PENDING.pop(0)
                name(Lp, pred.tolist())
            if ROUT and L + LAG < NL and ROUT[L + LAG]["lagged"]:
                pred = route(L + LAG, st["emb"], st["h_prev"][L + LAG], xf[0])
                if a.sync_name or a.mode == "exact":
                    name(L + LAG, pred.tolist())
                else:
                    mx.async_eval(pred)
                    PENDING.append((L + LAG, pred))
            else:  # no router to run here (the last LAG layers): start this layer's attention on the GPU now, as the router's async_eval does elsewhere
                mx.async_eval(xf)
        with stage("bookkeeping"):
            for e in [e for e, last in LAST[L].items() if last < t - a.keep]:  # keep window
                if e in INFL[L]:
                    INFL[L].pop(e)[1].result()  # finished but never landed: the slot is simply reused
                FREE[L].append(SLOT_OF[L].pop(e))
                TAB[L][e] = -1
                del LAST[L][e]
        with stage("released gate"):
            gates = mx.softmax(self.gate(xf), axis=-1, precise=True)
            inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
            rel = inds[0].tolist() if a.mode == "exact" else None
        if a.mode == "exact":  # the released choice; anything missing is read now
            t0 = time.perf_counter()
            now = []
            for e in rel:
                if e not in SLOT_OF[L]:
                    slot = take_slot(L, t)
                    SLOT_OF[L][e], SLOT_BY[L][slot] = slot, e
                    read_record(L, e, SLAB_NP[L][slot])
                    now.append(slot)
                    st["sync"] += 1
                LAST[L][e] = t
            flush(L, must=set(rel))
            land(L, now)
            st["wait"] += time.perf_counter() - t0
            slots = mx.array([[SLOT_OF[L][e] for e in rel]], dtype=mx.uint32)
        else:
            t0 = time.perf_counter()
            with stage("wait+land"):
                if st.get("touch_due") and any(not f.done() for _, f in INFL[L].values()):  # the host would block here anyway
                    KEEPER.touch_slabs(1)
                    st["touch_due"] = False
                tr(f"w{L}")
                flush(L, must=set(INFL[L]))  # everything named for this layer was requested four layers ago; a slot counts as in memory only once landed
            st["wait"] += time.perf_counter() - t0
            tr(f"f{L}")
            with stage("masked top-k"):  # all on the device: the released top-k among the landed experts, no host round trip
                tab = mx.array(TAB[L])
                st["hits_dev"] = st["hits_dev"] + (tab[inds] >= 0).sum()
                w_rel = mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32)  # the cluster's coverage: the released top-k probability mass in memory
                st["mass_dev"] = st["mass_dev"] + mx.stack([(w_rel * (tab[inds] >= 0)).sum(), w_rel.sum()])
                st["asked"] += K
                gates = mx.where(tab >= 0, gates, mx.zeros_like(gates))
                inds = mx.argpartition(-gates, kth=K - 1, axis=-1)[..., :K]
                slots = mx.maximum(tab[inds], 0).astype(mx.uint32)
        st["held"][0] += len(SLOT_OF[L])
        st["held"][1] += L == 0
        with stage("experts"):
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if NORM:
                scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-9)
            y = mx.zeros_like(x) if a.no_experts else expert_step(xf, slab_fields(L), slots, scores).reshape(x.shape).astype(x.dtype)
            if hasattr(self, "shared_expert"):
                y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            if a.trace_sync and L >= 40:
                mx.eval(y)
            tr(f"y{L}")
            mx.async_eval(y)  # queue this layer's expert work and go on: the GPU runs it while the host builds the next layer and waits on its reads
        return y


for L, layer in enumerate(model.model.layers):
    layer.mlp = StoreMoE(L, layer.mlp)  # the released block's routed experts (switch_mlp) are never materialised

import struct  # noqa: E402

RES = os.path.join(HERE, "resident.safetensors")
with open(RES, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
EMB_KEY = "model.language_model.embed_tokens.weight"
EMB = hdr[EMB_KEY]
assert EMB["dtype"] == "BF16" and EMB["shape"] == [cfg["vocab_size"], H]
EMB_BASE, EMB_FD = 8 + hdr_len + EMB["data_offsets"][0], os.open(RES, os.O_RDONLY)


class DiskEmbedding(nn.Module):
    """The input embedding of the current token (or of a whole batch), rows read from the resident file; the table never
    enters memory."""

    def __call__(self, ids):
        return st["emb_batch"] if st.get("emb_batch") is not None else st["emb"].reshape(1, 1, H)


def embed_rows(ids):
    rows = np.concatenate([np.frombuffer(os.pread(EMB_FD, 2 * H, EMB_BASE + t * 2 * H), dtype=np.uint16) for t in ids])
    return mx.array(rows).view(mx.bfloat16).astype(DT).reshape(len(ids), H)


# -- the PLE n-gram memory from disk. ngram/part_PP.bin: rows of fp16 scale | int8 codes [dim] (x = code * scale), global
# row r in part r // rows_per_part, contiguous, unpadded. A row is read as the 16 KB page(s) holding it, with F_NOCACHE like
# the experts; rows that share a page share the read.
NGM = json.load(open(os.path.join(HERE, a.ngram, "meta.json")))
NG_BPR, NG_DIM, NG_PR, NG_PAGE = NGM["bytes_per_row"], NGM["dim"], NGM["rows_per_part"], 16384
assert NGM["dtype"] == "int8" and NG_BPR == 2 + NG_DIM
NG_FDS = []
for f_ in NGM["files"]:
    fd = os.open(os.path.join(HERE, f_), os.O_RDONLY)
    if not a.cache:
        try:
            import fcntl

            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except Exception:
            pass
    NG_FDS.append(fd)
NG_SIZE = [os.fstat(fd).st_size for fd in NG_FDS]
assert NG_SIZE == [n * NG_BPR for n in NGM["part_rows"]], "n-gram parts are incomplete"
NGS = {"rows": 0, "reads": 0, "bytes": 0, "wait": 0.0, "lead": 0.0, "uses": 0}  # rows, page reads, bytes read, time the model waited, issue -> use
NG_PENDING = {}  # (context ids, new ids) -> reads in flight for that forward pass
_NG_TL = threading.local()


def ng_read(groups, recs):
    """Reader thread: each group is one read of the page(s) holding some rows; their records are copied into recs."""
    buf = getattr(_NG_TL, "buf", None)
    for p, pg, pe, items in groups:
        base, span = pg * NG_PAGE, (pe - pg + 1) * NG_PAGE
        want = min(span, NG_SIZE[p] - base)  # the last page of a part may be short
        if buf is None or len(buf) < span:
            buf = _NG_TL.buf = bytearray(max(span, 2 * NG_PAGE))
        n = os.preadv(NG_FDS[p], [memoryview(buf)[:want]], base)
        assert n == want, f"short n-gram read: {n} of {want}"
        page = np.frombuffer(buf, dtype=np.uint8, count=want)
        for i, o in items:
            recs[i] = page[o - base : o - base + NG_BPR]


def ng_issue(rows):
    """Start reading the n-gram rows `rows` (int64 [N]) on the reader pool: grouped by page in file order, at most
    4 x threads tasks. Returns a handle for ng_take."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    part = rows // NG_PR
    off = (rows - part * NG_PR) * NG_BPR
    groups = []
    for i in np.lexsort((off, part)).tolist():
        p, o = int(part[i]), int(off[i])
        pg, pe = o // NG_PAGE, (o + NG_BPR - 1) // NG_PAGE
        if groups and groups[-1][0] == p and pg <= groups[-1][2]:
            groups[-1][2] = max(groups[-1][2], pe)
            groups[-1][3].append((i, o))
        else:
            groups.append([p, pg, pe, [(i, o)]])
    recs = np.empty((len(rows), NG_BPR), dtype=np.uint8)
    n_task = min(len(groups), 4 * a.threads)
    futs = [pool.submit(ng_read, groups[k * len(groups) // n_task : (k + 1) * len(groups) // n_task], recs) for k in range(n_task)]
    NGS["rows"] += len(rows)
    NGS["reads"] += len(groups)
    NGS["bytes"] += sum((g_[2] - g_[1] + 1) * NG_PAGE for g_ in groups)
    return {"recs": recs, "futs": futs, "t": time.perf_counter()}


def ng_take(hd):
    """Wait for a handle's reads and dequantise its rows: float32 [N, dim] (code * scale, exact in float32)."""
    t0 = time.perf_counter()
    for f_ in hd["futs"]:
        f_.result()
    NGS["wait"] += time.perf_counter() - t0
    NGS["lead"] += t0 - hd["t"]
    NGS["uses"] += 1
    recs = hd["recs"]
    return recs[:, 2:].view(np.int8).astype(np.float32) * recs[:, :2].view(np.float16).astype(np.float32)


def ng_context(c):
    """The n-gram context the cache holds (the last ngram_size - 1 ids; EOS before the first token), as host ints."""
    prev = c[3] if c is not None and len(c.cache) > 3 else None
    return [NG_MOD.eos] * NG_MOD.context_len if prev is None else np.array(prev).reshape(-1).tolist()


def ng_prefetch(ids, cache):
    """Issue the reads for the rows of `ids` fed after the state in `cache` (no-op if already issued). Called the moment
    the next token is sampled, and at the start of a step or a batch."""
    ctx = ng_context(cache[PLE_L])
    key = (tuple(ctx), tuple(int(t) for t in ids))
    if key not in NG_PENDING:
        NG_PENDING[key] = ng_issue(NG_MOD.rows(ctx, key[1]))


def ng_drop():
    """Forget reads issued for forward passes that will not run (a rolled-back batch, a reset)."""
    for hd in NG_PENDING.values():
        for f_ in hd["futs"]:
            f_.result()
    NG_PENDING.clear()


class DiskNGram(nn.Module):
    """The PLE n-gram embedding with its table on disk. The hashing is the community port's NGramEmbedding, done on the
    host in int64 (same layer multipliers, per-head primes and offsets, EOS-aware shifts; products stay below 2^63 by
    construction), and so is the context cache: cache[3] holds the last ngram_size - 1 ids, written as a host-built array so
    reading it back costs no device sync. Rows come from the reads issued at sample time (ng_prefetch), else are read now."""

    def __init__(self, orig):
        super().__init__()
        self.ngram_size, self.context_len, self.heads_per_ngram = orig.ngram_size, orig.context_len, orig.heads_per_ngram
        self.eos = int(orig.eos_token_id)
        self._mult = [int(v) for v in orig.layer_multipliers.tolist()]
        self._sizes = np.array(orig.ngram_heads_vocab_sizes.tolist(), dtype=np.int64)
        self._offs = np.array(orig.ngram_heads_offsets.tolist(), dtype=np.int64)
        self._shape = (orig.ngram_embedding.rows_per_shard * len(orig.ngram_embedding.shards), orig.ngram_embedding.embedding_dim)

    def _shift(self, tokens, amount):
        if amount == 0:
            return tokens
        shifted = np.concatenate([np.full(amount, self.eos, dtype=np.int64), tokens[:-amount]])
        valid = np.ones(tokens.shape, dtype=bool)
        valid[:amount] = False
        for delta in range(1, amount + 1):
            valid &= np.concatenate([np.full(delta, self.eos, dtype=np.int64), tokens[:-delta]]) != self.eos
        return np.where(valid, shifted, self.eos)

    def rows(self, ctx, ids):
        """Row ids [len(ids), heads] of the tokens `ids` following the context `ctx`."""
        hist = np.array(list(ctx) + list(ids), dtype=np.int64)
        shifted = [self._shift(hist, i) for i in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            s0 = (ngram - 2) * self.heads_per_ngram
            mixed = shifted[0] * self._mult[0]
            for p in range(1, ngram):
                mixed = mixed ^ (shifted[p] * self._mult[p])
            blocks.append(mixed[:, None] % self._sizes[s0 : s0 + self.heads_per_ngram] + self._offs[s0 : s0 + self.heads_per_ngram])
        return np.concatenate(blocks, axis=-1)[-len(ids) :]

    def __call__(self, input_ids, cache):
        ids = np.array(input_ids).reshape(-1).tolist()  # built from a host list: no device sync
        ctx = [self.eos] * self.context_len if cache is None else ng_context(cache)
        if cache is not None:
            cache[3] = mx.array([(ctx + ids)[-self.context_len :]], dtype=input_ids.dtype)
        hd = NG_PENDING.pop((tuple(ctx), tuple(ids)), None) or ng_issue(self.rows(ctx, ids))
        return mx.array(ng_take(hd)).reshape(1, len(ids), -1).astype(DT)


PLE_L = [L for L, layer in enumerate(model.model.layers) if layer.ple is not None]
assert len(PLE_L) == 1, "the n-gram store holds one PLE table"
PLE_L = PLE_L[0]
NG_MOD = model.model.layers[PLE_L].ple.ple_embedding = DiskNGram(model.model.layers[PLE_L].ple.ple_embedding)
assert NG_MOD._shape == (NGM["rows"], NG_DIM), f"n-gram store {NGM['rows']} x {NG_DIM} does not match the model's table {NG_MOD._shape}"

# -- the resident weights: bf16 on disk, keys as released. The steps of the community port's sanitize (conv1d to MLX's
# layout; the zero-centred RMSNorm weights + 1, here in float32 as the reference computes x * (1 + w)); the hashing
# constants the release stores are checked against the port's own construction. --dtype int8 (the default): every
# nn.Linear with at least 64 outputs is quantised to MLX affine int8 (group 64) at the first load, layer by layer, and the
# result cached next to the model; kept in fp16: norms, conv1d, A_log/dt_bias, the router gates (mlp.gate: the released
# routing), and the small projections (hyper-connection block_inject_weight [4, .], shared_expert_gate [1, .], GDN
# in_proj_a/b [heads, .]).
NORM_SUFFIXES = (".hc_norm.weight", ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight")
NG_CONST = {"layer_multipliers": NG_MOD._mult, "ngram_heads_vocab_sizes": NG_MOD._sizes.tolist(), "ngram_heads_offsets": NG_MOD._offs.tolist()}
QRULE = "int8 g64 v1: nn.Linear with >= 64 outputs, inputs % 64 == 0, not mlp.gate"


def qpred(path, m):
    return isinstance(m, nn.Linear) and m.weight.shape[0] >= 64 and m.weight.shape[1] % 64 == 0 and not path.endswith("mlp.gate")


def resident_weights():
    out = []
    for k, v in mx.load(RES).items():  # lazy: a tensor is read when it is evaluated
        if k.startswith(("mtp.", "model.visual.")) or (a.embed == "disk" and k == EMB_KEY):
            continue
        key = k.replace("model.language_model.", "model.", 1)
        name_ = key.rsplit(".", 1)[-1]
        if ".ple.ple_embedding." in key and name_ in NG_CONST:  # the release stores the hashing constants: they must be the ones the model builds
            assert v.tolist() == NG_CONST[name_], f"{k} differs from the hashing constants the model builds"
            continue
        if "conv1d.weight" in key and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)
        if v.ndim == 1 and key.endswith(NORM_SUFFIXES):
            v = v.astype(mx.float32) + 1.0
        out.append((key, v.astype(DT) if mx.issubdtype(v.dtype, mx.floating) else v))
    return out


if a.embed == "disk":
    assert not model.args.tie_word_embeddings, "--embed disk needs an untied lm_head"
    model.model.embed_tokens = DiskEmbedding()
model.eval()
from mlx.utils import tree_flatten  # noqa: E402

QPATH = os.path.join(HERE, "resident_int8.safetensors")
QSRC = f"{os.path.getsize(RES)}:{int(os.path.getmtime(RES))}:{QRULE}:embed={a.embed}"  # the cache is rebuilt when any of these change
qmeta = mx.load(QPATH, return_metadata=True)[1] if a.dtype == "int8" and os.path.exists(QPATH) else {}
if a.dtype == "int8" and qmeta.get("source") == QSRC:
    raw_ = mx.load(RES)
    for k_ in raw_:  # the cached file has no hashing constants: check the release's against the model's every start
        if ".ple.ple_embedding." in k_ and k_.rsplit(".", 1)[-1] in NG_CONST:
            assert raw_[k_].tolist() == NG_CONST[k_.rsplit(".", 1)[-1]], f"{k_} differs from the hashing constants the model builds"
    del raw_
    nn.quantize(model, group_size=64, bits=8, class_predicate=qpred)  # the structure only (lazy); the cached values replace it
    model.load_weights(QPATH, strict=True)
    RES_HOW = "int8 (cached)"
else:
    model.load_weights(resident_weights(), strict=True)
    if a.dtype == "int8":
        t_q = time.time()
        nn.quantize(model, group_size=64, bits=8, class_predicate=qpred)
        for m_ in list(model.model.layers) + [model.model.hyper_connection_mixer, model.lm_head, model.model.embed_tokens]:  # one layer's bf16 in flight at a time
            mx.eval(m_.parameters())
        mx.eval(model.parameters())
        tmp = QPATH[: -len(".safetensors")] + ".tmp.safetensors"
        mx.save_safetensors(tmp, dict(tree_flatten(model.parameters())), metadata={"source": QSRC})
        os.replace(tmp, QPATH)
        log(f"resident weights quantised to int8 in {time.time() - t_q:.1f}s, cached in {os.path.basename(QPATH)} ({os.path.getsize(QPATH) / 2**30:.2f} GB)")
        RES_HOW = "int8 (quantised now, cached)"
    else:
        RES_HOW = a.dtype
mx.eval(model.parameters())
n_res = sum(int(np.prod(v["shape"])) for k, v in hdr.items() if k != "__metadata__" and not (a.embed == "disk" and k == EMB_KEY) and k.rsplit(".", 1)[-1] not in NG_CONST)  # params, as released
for L in range(NL):
    st["h_prev"][L] = mx.zeros((H,), dtype=DT)


class PageKeeper:
    """Keeps the model's pages hot for the macOS pager by touching one byte per 16 KB page from the CPU through zero-copy
    views (a full sweep of 8.4 GB costs ~360 ms). The weights and routers never change, so a background thread sweeps
    them; the slabs (replaced by grow() on the model thread) are touched only by the model thread, a slice per token and
    all of them when idle."""

    SLICES = 16  # static arrays: one slice per second (a full sweep every 16 s)
    SLAB_SLICES = 64  # slabs: one slice per generated token (a full sweep every ~12 s at 5 tok/s; ~2 ms per token)

    def __init__(self):
        from mlx.utils import tree_flatten

        self.static = [v for _, v in tree_flatten(model.parameters()) if isinstance(v, mx.array) and v.nbytes >= 1 << 20]
        self.static += [v for r in ROUT for v in r.values() if isinstance(v, mx.array) and v.nbytes >= 1 << 20]
        self.i = self.j = 0
        self.on = not a.no_keep_hot
        if self.on:
            threading.Thread(target=self._loop, name="keeper", daemon=True).start()

    @staticmethod
    def _touch(arrs):
        for v in arrs:
            try:
                np.asarray(v).view(np.uint8).reshape(-1)[::16384].sum()
            except Exception:
                pass

    def _loop(self):
        n = 0
        while True:
            time.sleep(1.0)
            self._touch(self.static[self.i :: self.SLICES])
            self.i = (self.i + 1) % self.SLICES
            n += 1
            if n % 60 == 0:  # once a minute: re-request wiring if the kernel dropped it
                ensure_wired()

    def touch_slabs(self, n_slices=1):
        """Model thread only."""
        if not self.on:
            return
        arrs = list(SLAB_BASE)
        for _ in range(n_slices):
            self._touch(arrs[self.j :: self.SLAB_SLICES])
            self.j = (self.j + 1) % self.SLAB_SLICES


KEEPER = PageKeeper()
LOADED = (f"loaded in {time.time() - t00:.1f}s | {a.store} ({FMT}) | resident {n_res / 1e6:.0f}M params, {RES_HOW} | experts on disk {NL * NE * 3 * MAT / 1e9:.2f}B ({NL * NE * REC / 2**30:.1f} GB) "
          f"| n-gram memory on disk {NGM['rows'] * NG_DIM / 1e9:.2f}B ({NGM['rows'] * NG_BPR / 2**30:.1f} GB, {NG_MOD.ngram_heads if hasattr(NG_MOD, 'ngram_heads') else len(NG_MOD._sizes)} rows per token, layer {PLE_L}) "
          f"| {a.mode}, {a.m} named, keep {a.keep}, {CAP} slots/layer ({NL * CAP * REC / 2**30:.1f} GB) | lag {LAG}{'' if ROUT else ' (no routers)'} | {a.router_dtype} routers, embeddings on {a.embed} "
          f"| in memory {mx.get_active_memory() / 2**30:.1f} GB, wired limit {WIRED / 2**30:.1f} GB")
log(LOADED if VERBOSE or not a.chat else f"{os.path.basename(HERE)} loaded in {time.time() - t00:.1f}s, {mx.get_active_memory() / 2**30:.1f} GB in memory. Type a message; /quit to exit; run with -v for statistics.")


# ======================================================================================= 5. steps: a token, a batch, a segment
def embed_hook(token_id, during=None):
    """Token start: the embedding is known; name every layer whose router needs nothing from this forward pass.
    `during` (host work, e.g. the n-gram reads) runs while the GPU evaluates those routers."""
    st["t"] += 1
    st["emb"] = embed_rows([token_id])[0] if a.embed == "disk" else model.model.embed_tokens(mx.array([[token_id]]))[0, 0]
    tr("emb")
    st["h_prev_next"] = list(st["h_prev"])
    first = [L for L in range(NL if ROUT else 0) if not ROUT[L]["lagged"]]
    preds = [route(L, st["emb"], st["h_prev"][L], None) for L in first]
    if preds:
        tr("rg")
        mx.async_eval(*preds)
    if during:
        during()
    if preds:
        mx.eval(*preds)  # one device round trip for all of them
        tr("rt")
    for L, pred in zip(first, preds):
        name(L, pred.tolist())
    tr("nm")


LTRACE = bool(os.environ.get("FLASH_LAYER_TRACE"))  # diagnostics: per-layer time and memory of a batched prefill
SSM_IDX, FA_IDX = 0, next(L for L, layer in enumerate(model.model.layers) if not layer.is_linear)


def forward(ids, cache, attn_block=0):
    """The community port's Qwen4ExpModel.__call__ + lm_head, with the embedding from disk, a whole batch's attention run
    in attn_block pieces against the cache (the PLE and the expert block see the whole batch at once), and the head on the
    last position only. Returns the last position's logits."""
    mm, T = model.model, len(ids)
    inputs = mx.array([ids], dtype=mx.int32)
    h = mx.tile(st["emb_batch"] if st.get("batch") else st["emb"].reshape(1, 1, H), (1, 1, HC))  # the 4 hyper-connection streams start equal
    AB = attn_block if 0 < attn_block < T else T
    masks = {}
    for b0 in range(0, T, AB):  # the masks the model builds before its first layer, for each piece length
        n = min(AB, T - b0)
        if n not in masks:
            masks[n] = (_create_qwen3_5_ssm_mask(h[:, :n], cache[SSM_IDX]), _create_qwen3_5_attention_mask(h[:, :n], cache[FA_IDX].caches[0]))
    batch = st.get("batch")
    for layer, c in zip(mm.layers, cache):
        if layer.ple is not None:  # DiskNGram inside: the rows were issued at sample time (or at the start of the batch)
            with stage("ple"):
                h = h + layer.ple(h, inputs, c)
        t_at = time.perf_counter()
        parts = []
        for b0 in range(0, T, AB):
            hb = h[:, b0 : b0 + AB] if AB < T else h
            ssm_mask, fa_mask = masks[hb.shape[1]]
            x_, res, inj = layer.attn_hyper_connection(hb)
            if layer.is_linear:
                blk = layer.linear_attn(x_, mask=ssm_mask, cache=c)
            else:
                blk = layer.self_attn(x_, mask=fa_mask, cache=c, position_ids=None, position_embeddings=None)
            hb = layer._inject(blk, res, inj)
            if AB < T or st.get("batch"):
                mx.eval(hb)
            parts.append(hb)
        h = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        if batch:
            compact_caches([c])  # the convolution state's view would hold this layer's block input until the batch ends
            mx.eval(h)
            BP["lin" if layer.is_linear else "qsa"] += time.perf_counter() - t_at
            if LTRACE:
                log(f"[layer] {layer.layer_idx if hasattr(layer, 'layer_idx') else '?'} {'lin' if layer.is_linear else 'qsa'} {time.perf_counter() - t_at:.2f}s active {mx.get_active_memory() / 2**30:.2f} peak {mx.get_peak_memory() / 2**30:.2f}")
                mx.reset_peak_memory()
            t_at = time.perf_counter()
        x_, res, inj = layer.mlp_hyper_connection(h)
        h = layer._inject(layer.mlp(x_), res, inj)  # StoreMoE on the mixed input: one token from the slab, or a batch streamed once
        if batch:
            mx.eval(h)
            BP["moe"] += time.perf_counter() - t_at
            if LTRACE:
                log(f"[layer]   moe {time.perf_counter() - t_at:.2f}s active {mx.get_active_memory() / 2**30:.2f} peak {mx.get_peak_memory() / 2**30:.2f}")
                mx.reset_peak_memory()
    tr("head")
    if a.trace_sync:
        mx.eval(h)
        tr("hsync")
    return model.lm_head(mm.hyper_connection_mixer(h[:, -1:]))[0, -1]


def step_begin(token_id, cache):
    """A token's start: its n-gram rows and embedding, and the reads of the layers whose routers need nothing from this
    forward pass (so the first reads are on the drive before anything else is done at the token boundary)."""
    with stage("token"):
        embed_hook(token_id, during=lambda: (ng_prefetch([token_id], cache), tr("ng")))  # the n-gram rows are issued while the GPU runs the first routers
        st["touch_due"] = True  # the page keeper's slab slice: done where the host waits on reads anyway (wait+land)


def step_finish(token_id, cache):
    """The rest of the token: the forward pass; returns the next-token logits."""
    with stage("token"):
        logits = forward([token_id], cache)
        tr("ev0")
        am = mx.argmax(logits) if not SAMPLE or TEMP <= 0 else None  # greedy: the next token in the same device round trip
        mx.eval(logits, st["hits_dev"], st["mass_dev"], *([am] if am is not None else []))
        st["argmax"] = (logits, am)
        tr("ev1")
        assert not PENDING
        st["h_prev"] = st["h_prev_next"]
    if st.pop("touch_due", False):
        KEEPER.touch_slabs(1)
    return logits


def step(token_id, cache):
    """One token through the prefetch pipeline; returns the next-token logits."""
    step_begin(token_id, cache)
    return step_finish(token_id, cache)


def prefill_batch(ids, cache, tick=None, done=0, total=None):
    """Prefill `ids` as one batch: the decoder layers run the whole sequence (attention in --attn-block pieces against the
    cache), each StoreMoE streams the experts its tokens need once, and every n-gram row of the batch is read from the start
    (their addresses follow from the ids). Leaves the same state behind as `step` would after the last token: the KV and
    recurrent caches, the n-gram context, the token clock, the routers' previous-step inputs; returns the last position's
    logits."""
    t0 = time.perf_counter()
    T = len(ids)
    assert not PENDING
    for L in range(NL):
        flush(L, must=set(INFL[L]))
    ng0 = dict(NGS)
    ng_prefetch(ids, cache)
    st["batch"], st["h_prev_next"] = True, list(st["h_prev"])
    st["tick"] = (lambda l, n: tick(done + T * l // n, total)) if tick else None
    st["emb_batch"] = embed_rows(ids).reshape(1, T, H) if a.embed == "disk" else model.model.embed_tokens(mx.array([ids]))
    if a.stream_from and T >= a.stream_from and not a.no_experts:
        stream_begin()
    try:
        logits = forward(ids, cache, attn_block=max(1, a.attn_block))
        mx.eval(logits)
    finally:
        st["batch"], st["emb_batch"], st["tick"] = False, None, None
        if STREAM["on"]:
            stream_end()
        stage_free()
    st["t"] += T
    st["h_prev"] = st["h_prev_next"]
    compact_caches(cache)
    if CPROF:
        log(f"[chunks] {CP['n']}: fields {CP['fields']:.2f}s, gather_qmm + scale {CP['qmm']:.2f}s, scatter-add rounds {CP['scatter']:.2f}s ({CP['rounds'] / max(1, CP['n']):.1f} rounds per chunk)")
        CP.update(fields=0.0, qmm=0.0, scatter=0.0, rounds=0, n=0)
    if VERBOSE:
        log(f"[batch] {T} tokens in {time.perf_counter() - t0:.1f}s ({T / (time.perf_counter() - t0):.1f} tok/s) | read {st['bytes_batch'] / 2**30:.1f} GB | memory {mx.get_active_memory() / 2**30:.1f} GB, peak {mx.get_peak_memory() / 2**30:.1f} GB"
            f" | layers: linear attention {BP['lin']:.1f}s, sparse attention {BP['qsa']:.1f}s, MoE {BP['moe']:.1f}s | expert chunks {BP['batches']}: waiting for reads {BP['wait']:.1f}s, host prep {BP['host']:.1f}s, gpu (eval) {BP['gpu']:.1f}s"
            f" | reads: {RT['n']} chunks, mean lag before start {1000 * RT['lag'] / max(1, RT['n']):.1f} ms, mean duration {1000 * RT['read'] / max(1, RT['n']):.1f} ms"
            f" | ngram rows read {NGS['rows'] - ng0['rows']} in {NGS['reads'] - ng0['reads']} page reads ({(NGS['bytes'] - ng0['bytes']) / 2**20:.1f} MB), issued at the start, waited {1000 * (NGS['wait'] - ng0['wait']):.1f} ms at layer {PLE_L}")
    BP.update(wait=0.0, host=0.0, gpu=0.0, batches=0, lin=0.0, qsa=0.0, moe=0.0)
    RT.update(lag=0.0, read=0.0, n=0)
    st["bytes_batch"] = 0
    return logits


def compact_caches(cache):
    """After a batch: the recurrent caches' arrays as buffers of their own. The convolution state is the last 3 rows of
    the block's [1, n+3, 10240] input taken as a view (mlx-vlm's update_window: mx.contiguous of a contiguous slice does
    not copy), which keeps the whole block input alive in each of the 36 linear-attention layers (~10 MB each per 512-token
    block). Values are unchanged (x * 1)."""
    out = []
    for c in cache:
        for x in (c.caches if isinstance(c, CacheList) else [c]):
            if isinstance(x, ArraysCache):
                x.cache = [None if v is None else v * 1 for v in x.cache]
                out += [v for v in x.cache if v is not None]
    mx.eval(*out)


def cache_mark(c):
    """What a rollback needs: offsets for the KV caches (a reference to the buffer would pin it, and every block's in-place
    write would then copy the whole buffer), the arrays of a recurrent cache (a copy of its list: the cache mutates the
    list in place), the QSA cache's three parts."""
    if isinstance(c, CacheList):
        return [cache_mark(x) for x in c.caches]
    if isinstance(c, KVCache):
        return c.offset
    return list(c.cache)


def cache_rollback(c, mark):
    if isinstance(c, CacheList):
        for x, m in zip(c.caches, mark):
            cache_rollback(x, m)
    elif isinstance(c, KVCache):
        c.trim(c.offset - mark)
    else:
        c.cache = list(mark)


def prefill(ids, cache, tick=None, done=0, total=None, chunk_max=None):
    """Prefill a segment (no snapshot inside it): batched when it is long enough, token by token otherwise. A batch that
    runs the GPU out of memory is rolled back and retried at half the size, down to token by token."""
    chunk_max = chunk_max or a.chunk_max
    if a.chunk and len(ids) >= a.chunk and chunk_max >= a.chunk:
        logits = None
        for i in range(0, len(ids), chunk_max):
            seg = ids[i : i + chunk_max]
            saved = ([cache_mark(c) for c in cache], st["t"], list(st["h_prev"]))
            try:
                logits = prefill_batch(seg, cache, tick, done + i, total)
            except RuntimeError as e:
                if "emory" not in str(e):
                    raise
                for c, sv in zip(cache, saved[0]):  # the batch may have appended some blocks: roll the cache back
                    cache_rollback(c, sv)
                st["t"], st["h_prev"] = saved[1], saved[2]
                st["batch"], st["emb_batch"], st["tick"] = False, None, None
                reset_slots()
                mx.clear_cache()
                log(f"[batch] {len(seg)} tokens at context {done + i}: GPU out of memory, retrying at {max(chunk_max // 2, 1)}")
                logits = prefill(seg, cache, tick, done + i, total, chunk_max=max(chunk_max // 2, 1))
            mx.clear_cache()
        return logits
    logits = None
    for j, t in enumerate(ids):
        logits = step(t, cache)
        if tick and (done + j + 1) % 8 == 0:
            tick(done + j + 1, total)
    return logits


def reset_stats():
    st.update(bytes=0, wait=0.0, hits_dev=mx.array(0), mass_dev=mx.zeros(2), asked=0, held=[0, 0], sync=0, grown=0, early=0)
    mx.eval(st["hits_dev"], st["mass_dev"])  # materialised now: a lazy mx.zeros belongs to the stream of the thread that made it (the main thread at import), and the server's model thread could not evaluate it
    NGS.update(rows=0, reads=0, bytes=0, wait=0.0, lead=0.0, uses=0)
    PROF.clear()
    RLOG.clear()
    TR.clear()
    LZ.update(dec_s=0.0, n=0, io_s=0.0)


def top5(logits):
    top = mx.argpartition(-logits, kth=4)[:5].tolist()
    return sorted(((int(i), round(float(logits[i].item()), 3)) for i in top), key=lambda z: -z[1])


def summary(n, dt):
    n = max(1, n)
    mb = st["bytes"] / 2**20
    held = st["held"][0] / max(1, st["held"][1])
    how = f"released top-{K} in memory {int(st['hits_dev'].item()) / max(1, st['asked']):.1%} by count, {float(st['mass_dev'][0]) / max(1e-9, float(st['mass_dev'][1])):.1%} by probability mass" if a.mode == "prefetch" else f"read on the spot {st['sync']}"
    return (f"{n} tokens in {dt:.1f}s = {n / dt:.2f} tok/s | read {mb / n:.0f} MB per token ({mb / dt:.0f} MB/s) | waited on experts {1000 * st['wait'] / n:.0f} ms per token "
            f"| {how} | experts held {held:.0f} ({held * REC / 2**30:.1f} GB), slots {sum(CAPS)} "
            f"| ngram rows read {NGS['rows'] / n:.0f} per token ({NGS['bytes'] / n / 2**10:.0f} KB in {NGS['reads'] / n:.0f} page reads), issued {1000 * NGS['lead'] / max(1, NGS['uses']):.1f} ms before use, wait {1000 * NGS['wait'] / n:.2f} ms per token "
            f"| memory {mx.get_active_memory() / 2**30:.1f} GB, peak {mx.get_peak_memory() / 2**30:.1f} GB")


# -- sampling: the release's settings, as every local runner does; argmax for --greedy, --bench, --check
GEN = json.load(open(os.path.join(HERE, "generation_config.json"))) if os.path.exists(os.path.join(HERE, "generation_config.json")) else {}
STOP = set(GEN.get("eos_token_id", []) if isinstance(GEN.get("eos_token_id"), list) else [GEN.get("eos_token_id")] if GEN.get("eos_token_id") is not None else [])
STOP |= {tok.eos_token_id} if tok.eos_token_id is not None else set()
SAMPLE = not (a.greedy or a.bench or a.check) and bool(GEN.get("do_sample", False))
TEMP, TOP_P, TOP_K = (a.temp if a.temp is not None else GEN.get("temperature", 1.0)), (a.top_p if a.top_p is not None else GEN.get("top_p", 1.0)), (a.top_k if a.top_k is not None else GEN.get("top_k", 0))
mx.random.seed(a.seed)


def sample(logits):
    """Temperature, then top-k, then top-p on the tempered distribution: the release's (transformers') order. mlx-lm's
    make_sampler applies top-p before the temperature, which is a different distribution at Qwen's 0.7 / 0.8 / 20."""
    if not SAMPLE or TEMP <= 0:
        pre = st.get("argmax")
        if pre is not None and pre[0] is logits and pre[1] is not None:  # evaluated with the logits in step_finish
            return int(pre[1].item())
        return int(mx.argmax(logits).item())
    lg = logits.astype(mx.float32) / TEMP
    if TOP_K and TOP_K > 0:
        kth = mx.sort(lg)[-TOP_K]
        lg = mx.where(lg < kth, -mx.inf, lg)
    if TOP_P and TOP_P < 1.0:
        order = mx.argsort(-lg)
        probs = mx.softmax(lg[order])
        cum = mx.cumsum(probs)
        keep = cum - probs < TOP_P  # the smallest set whose mass reaches top-p
        lg = mx.put_along_axis(mx.full(lg.shape, -mx.inf), order, mx.where(keep, lg[order], -mx.inf), axis=0)
    return int(mx.random.categorical(lg).item())


# ======================================================================= 6. the session: prefix reuse, snapshots, generation
def tree_arrays(s):
    return [y for x in s for y in tree_arrays(x)] if isinstance(s, (list, tuple)) else ([] if s is None else [s])


def cache_state(c):
    """A cache's state, as a structure of its own: MLX arrays are immutable, but the recurrent caches hold their arrays in a
    list they update in place (cache[0] = ...), so a snapshot that kept that list would follow the live cache."""
    if isinstance(c, CacheList):
        return [cache_state(x) for x in c.caches]
    if isinstance(c, ArraysCache):
        return list(c.cache)
    return tuple(c.state)


def set_cache_state(c, s):
    if isinstance(c, CacheList):
        for x, sx in zip(c.caches, s):
            set_cache_state(x, sx)
    elif isinstance(c, ArraysCache):
        c.cache = list(s)
    else:
        c.state = tuple(s)


def save_cache(path, cache, metadata):
    """One safetensors file: every array of every layer's cache state, plus the layout (which leaves are empty)."""
    arrays = {}

    def walk(s, key):
        if isinstance(s, (list, tuple)):
            return [walk(x, f"{key}.{i}") for i, x in enumerate(s)]
        if s is None:
            return None
        arrays[key] = s
        return key

    layout = [walk(cache_state(c), str(L)) for L, c in enumerate(cache)]
    mx.save_safetensors(path, arrays, metadata={**metadata, "layout": json.dumps(layout)})


def load_cache(path):
    arrays, md = mx.load(path, return_metadata=True)

    def build(x):
        return [build(y) for y in x] if isinstance(x, list) else (None if x is None else arrays[x])

    cache = make_prompt_cache(model)
    for c, lay in zip(cache, json.loads(md["layout"])):
        set_cache_state(c, build(lay))
    return cache, md


class Session:
    """One conversation's state over the engine: the KV cache plus the routers' previous-step inputs and token clock.
    feed() brings it to a prompt by reusing the decoded prefix, else restoring the longest snapshot (memory, then disk),
    else starting over; the rest is prefilled in segments, snapshotted at message boundaries. generate() samples tokens.
    Used as is by the chat loop and by the server; every MLX call happens on the thread that owns the session."""

    HOT_MAX = 0  # snapshots kept in memory: each pins its KV and recurrent buffers (the live cache's in-place writes then copy them: ~350 MB at 8k tokens), and a restore from disk takes 1.1-1.8 s with the slots re-read, so none (2026-11-14, Flash-Next on 16 GB)

    def __init__(self, cache_dir=None):
        self.dir = cache_dir
        self.fed, self.cache, self.logits = [], None, None
        self.restored = ("", 0.0)
        self.hot = OrderedDict()  # snapshot path -> (cache states, t, h_prev, logits)
        self.last = ("", 0, 0.0)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- snapshots
    def _path(self, ids):
        return os.path.join(self.dir, f"prefix_{hash(tuple(ids)) & 0xFFFFFFFFFFFF:012x}.safetensors")

    def _remember(self, path, ids, slots=None):
        self.hot[path] = ([cache_state(c) for c in self.cache], st["t"], list(st["h_prev"]), self.logits, slots if slots is not None else slot_state(), list(ids))
        self.hot.move_to_end(path)
        while len(self.hot) > self.HOT_MAX:
            self.hot.popitem(last=False)

    def _save(self, ids, disk=True):
        """A snapshot of the fed state at ids. On disk only where it pays (the first message boundary, an agent's
        system prompt: what every new thread and every restart starts from); the other message boundaries only in memory,
        and only with HOT_MAX > 0 (a client that re-renders the model's turn differently then resumes there instead of
        at the system prompt). Writing ~350 MB per turn while the machine is short of memory measured 34-66 s on the
        critical path."""
        path = self._path(ids)
        if os.path.exists(path):
            return
        if not disk:
            if self.HOT_MAX and path not in self.hot:
                self._remember(path, ids)
            return
        t0 = time.time()
        save_cache(path, self.cache, {"t": str(st["t"])})
        slots = slot_state()
        mx.savez(path + ".state.npz", logits=self.logits, **{str(L): st["h_prev"][L] for L in range(NL)},
                 **{f"held{L}": mx.array(h) for L, (h, _) in enumerate(slots)}, **{f"last{L}": mx.array(l.astype(np.int32)) for L, (_, l) in enumerate(slots)})
        json.dump(ids, open(path.replace(".safetensors", ".ids.json"), "w"))
        self._remember(path, ids, slots)
        if VERBOSE:
            log(f"[cache] snapshot of {len(ids)} tokens saved in {time.time() - t0:.1f}s ({sum(os.path.getsize(path.replace('.safetensors', x)) for x in ('.safetensors', '.safetensors.state.npz')) / 2**20:.0f} MB, {sum(len(h) for h, _ in slots)} held experts)")
        self._prune(keep=path)

    def _prune(self, keep=None):
        """Least recently used snapshots go once the folder holds more than --cache-gb (a restore counts as a use); a
        snapshot of an agent's long conversation is hundreds of MB, and the disk is the store's."""
        groups = {}
        for f in os.listdir(self.dir):
            if f.startswith("prefix_"):
                groups.setdefault(f.split(".")[0], []).append(os.path.join(self.dir, f))
        sizes = {k: sum(os.path.getsize(p) for p in v) for k, v in groups.items()}
        used = lambda k: max(os.path.getmtime(p) for p in groups[k])  # noqa: E731
        total = sum(sizes.values())
        for k in sorted(groups, key=used):
            if total <= a.cache_gb * 2**30:
                break
            if keep and os.path.basename(keep).split(".")[0] == k:
                continue
            for p in groups[k]:
                os.remove(p)
            self.hot.pop(os.path.join(self.dir, k + ".safetensors"), None)
            total -= sizes[k]
            log(f"[cache] pruned snapshot {k} ({sizes[k] / 2**20:.0f} MB); folder now {total / 2**30:.2f} GB")

    def _restore(self, ids):
        """The longest saved prefix of ids, from memory or disk; returns how many tokens it covers."""
        if not self.dir:
            return 0
        best = None
        cands = [h[5] for h in self.hot.values()] + [json.load(open(os.path.join(self.dir, f))) for f in os.listdir(self.dir) if f.endswith(".ids.json")]
        for saved in cands:
            if len(saved) <= len(ids) and ids[: len(saved)] == saved and (best is None or len(saved) > len(best)):
                best = saved
        if not best:
            return 0
        path = self._path(best)
        try:  # a restore is a use (the least recently used snapshots are pruned first)
            os.utime(path.replace(".safetensors", ".ids.json"))
        except OSError:
            pass
        t0 = time.time()
        if path in self.hot:
            states, t_, hp, lg, slots, _ = self.hot[path]
            self.cache = make_prompt_cache(model)
            for c, state in zip(self.cache, states):
                set_cache_state(c, state)
            self.fed, self.logits = list(best), lg
            st["t"], st["h_prev"] = t_, list(hp)
            restore_slots(slots)
            self.hot.move_to_end(path)
            self.restored = ("memory", time.time() - t0)
            return len(best)
        if not os.path.exists(path):
            return 0
        self.cache, meta_ = load_cache(path)
        hp = mx.load(path + ".state.npz")
        self.fed, self.logits = list(best), hp["logits"]
        st["t"], st["h_prev"] = int(meta_.get("t", len(best))), [hp[str(L)] for L in range(NL)]
        slots = [(np.array(hp[f"held{L}"]), np.array(hp[f"last{L}"]).astype(np.int64)) for L in range(NL)] if "held0" in hp else [(np.zeros(0, np.int32), np.zeros(0, np.int64))] * NL
        mx.eval(self.logits, *st["h_prev"], *[x for c in self.cache for x in tree_arrays(cache_state(c))])
        restore_slots(slots)
        self._remember(path, best, slots)
        self.restored = ("disk", time.time() - t0)
        return len(best)

    @staticmethod
    def _retokenized_prefix(fed, ids):
        """j such that ids[:j] spells exactly the text of fed, or 0: a client sends the model's own turn back re-rendered
        through the chat template, same text but not always the same token boundaries, and the state computed on the
        model's own tokens is the right one to continue from."""
        text = tok.decode(fed)
        lo, hi = 0, len(ids)
        while lo < hi:  # decoded length grows with the token count
            mid = (lo + hi + 1) // 2
            if len(tok.decode(ids[:mid])) <= len(text):
                lo = mid
            else:
                hi = mid - 1
        return lo if lo and tok.decode(ids[:lo]) == text else 0

    def reset(self):
        self.fed, self.cache, self.logits = [], None, None

    # -- the two operations
    def feed(self, ids, marks=(), tick=None):
        t0, how = time.time(), "live prefix"
        if self.cache is not None and self.fed and ids[: len(self.fed)] != self.fed:
            j = self._retokenized_prefix(self.fed, ids)
            if j:
                self.fed, how = list(ids[:j]), "live prefix (re-tokenized)"
        if self.cache is None or ids[: len(self.fed)] != self.fed:
            self.reset()
            reset_slots()
            n = self._restore(ids)
            how = f"restored snapshot of {n} from {self.restored[0]} in {self.restored[1]:.1f}s" if n else "cold"
            if n == 0:
                self.cache = make_prompt_cache(model)
                for L in range(NL):
                    st["h_prev"][L] = mx.zeros((H,), dtype=DT)
                st["t"] = -1
        start = len(self.fed)
        n_new = len(ids) - start
        # snapshot points: message boundaries, plus every N tokens when the segment is prefilled token by token (a batched
        # segment is one pass over the store however long it is, so it is not cut up)
        points = sorted(p for p in set(marks) if start < p < len(ids)) if self.dir else []
        if self.dir and a.snapshot_every and not (a.chunk and n_new >= a.chunk):
            points = sorted(set(points) | {p for p in range(max(64, start + 1), len(ids)) if p % a.snapshot_every == 0})
        pos, logits = start, self.logits
        for p in points + [len(ids)]:
            if p <= pos:
                continue
            logits = prefill(ids[pos:p], self.cache, tick, pos, len(ids))
            pos = p
            self.fed, self.logits = list(ids[:p]), logits
            if p < len(ids):
                self._save(ids[:p], disk=a.disk_snapshots == "all" or p == min(marks))
        self.last = (how, n_new, time.time() - t0)
        if VERBOSE:
            log(f"[feed] {how}, {n_new} of {len(ids)} tokens prefilled in {time.time() - t0:.1f}s | next-token top-5 {top5(logits)} | memory peak {mx.get_peak_memory() / 2**30:.1f} GB")
        return logits

    def generate(self, max_new, on_text=None):
        """Sample up to max_new tokens from the fed state; on_text(piece) receives the decoded text as it becomes valid UTF-8."""
        t0, new, shown, logits = time.time(), [], "", self.logits
        self.first = None  # seconds until the first text is out (the first token is sampled from the prefill's logits, so ~0 after feed)
        for _ in range(max_new):
            tr("s0")
            t = sample(logits)
            if SIMREC is not None:  # timing experiments only
                if a.sim_replay:
                    t = SIMREC["tokens"][len(new)]
                else:
                    SIMREC["tokens"].append(t)
            tr("s1")
            if t in STOP:
                break
            step_begin(t, self.cache)  # its embedding, n-gram rows and first layers' reads go out before the text work below
            new.append(t)
            if on_text:
                text = tok.decode(new)
                if not text.endswith("�") and len(text) > len(shown):
                    on_text(text[len(shown):])
                    shown = text
                    self.first = self.first if self.first is not None else time.time() - t0
            logits = step_finish(t, self.cache)
        self.fed, self.logits = self.fed + new, logits
        return new, time.time() - t0


# ============================================================================================================ 7. front-ends
if a.check:  # research: the released model's outputs against a reference (the cluster's float32 run, or test_tiny.py's)
    import torch  # noqa: E402  (only to read the reference)

    ref = torch.load(a.check, map_location="cpu")
    ids, n_prompt = [int(t) for t in ref["ids"]], int(ref["n_prompt"])
    assert (ref.get("rule", "route") == "orig") == (a.mode == "exact"), "reference and mode do not match"
    # A reference with per-position logits (all_top_ids/all_top_logits/all_argmax for every position) is compared at every
    # position the run produces, including the last; one in the cluster format (top_ids/top_logits from position n_prompt-1,
    # ids[i+1] as the expected token) at positions n_prompt-1 .. len(ids)-2. The prompt is fed through the Session (so --chunk
    # batches it, --prompt-cache snapshots and restores it): as the turns `cuts` gives (each feed extends the last), or as the
    # list of `feeds` (token lists; they may diverge from ids, to exercise snapshot restores; the last must be ids[:n_prompt]),
    # with snapshot points `marks`; then the rest token by token.
    every = "all_top_ids" in ref
    last = len(ids) - 1 if every else len(ids) - 2

    def expect(p):
        if every:
            return int(ref["all_argmax"][p]), ref["all_top_ids"][p].numpy(), ref["all_top_logits"][p].numpy()
        j = p - (n_prompt - 1)
        return ids[p + 1], ref["top_ids"][j].numpy(), ref["top_logits"][j].numpy()

    ses = Session(a.prompt_cache)
    agree, n, maxdiff, per_pos, t0, nll, nlls = 0, 0, 0.0, [], time.time(), 0.0, []

    def compare(p, logits):
        global agree, n, maxdiff, nll
        want, tids, tlg = expect(p)
        lg = np.array(logits.astype(mx.float32))
        d = float(np.abs(lg[tids] - tlg).max())
        agree, n, maxdiff = agree + int(int(lg.argmax()) == want), n + 1, max(maxdiff, d)
        z = lg.astype(np.float64) - lg.max()
        nll_p = float(np.log(np.exp(z).sum()) - z[want])  # teacher-forced loss of the expected token
        nll += nll_p
        nlls.append(round(nll_p, 5))
        per_pos.append((p, int(lg.argmax()) == want, round(d, 4)))

    cuts = sorted({int(c) for c in ref.get("cuts", []) if 0 < int(c) < n_prompt} | {n_prompt})
    feeds = [[int(t) for t in f] for f in ref["feeds"]] if "feeds" in ref else [ids[:c] for c in cuts]
    marks = [int(m) for m in ref.get("marks", cuts)]
    assert feeds[-1] == ids[:n_prompt], "the last feed must be the prompt"
    hows, saved = [], {}
    for f in feeds:  # each feed: the Session reuses the live state, else restores the longest snapshot, and prefills the rest
        logits = ses.feed(f, marks=marks)
        hows.append(f"{ses.last[0]} -> {len(f)}")
        if f == ids[: len(f)] and (every or len(f) == n_prompt):
            compare(len(f) - 1, logits)
            if a.check_save:
                saved[len(f) - 1] = np.array(logits.astype(mx.float32))
    for p in range(n_prompt, last + 1):
        logits = step(ids[p], ses.cache)
        compare(p, logits)
        if a.check_save:
            saved[p] = np.array(logits.astype(mx.float32))
    if a.check_save:
        np.savez(a.check_save, positions=np.array(sorted(saved)), logits=np.stack([saved[p] for p in sorted(saved)]))
    final = float(np.abs(np.array(logits.astype(mx.float32)) - ref["final_logits"].float().numpy()).max()) if "final_logits" in ref else float("nan")
    log(f"CHECK against {os.path.basename(a.check)} ({n} positions: {'every position' if every else 'generated positions'}; feeds: {', '.join(hows)}; --chunk {a.chunk}): "
        f"next-token agreement {agree}/{n} | max |logit diff| on the reference top-5 {maxdiff:.4f} | max |logit diff| over the whole vocabulary at the last position {final:.4f} "
        f"| {(len(ids) - 1) / (time.time() - t0):.2f} tok/s")
    log(f"teacher-forced loss of the expected token {nll / max(1, n):.4f} nats over {n} positions | " + (f"released top-{K} in memory {int(st['hits_dev'].item()) / max(1, st['asked']):.1%} by count, {float(st['mass_dev'][0]) / max(1e-9, float(st['mass_dev'][1])):.1%} by probability mass" if a.mode == "prefetch" else f"read on the spot {st['sync']}")
        + f" | read {st['bytes'] / max(1, len(ids)) / 1e6:.0f} MB per token")
    json.dump({"mode": a.mode, "positions": [p for p, _, _ in per_pos], "agree": [int(ok) for _, ok, _ in per_pos], "nll": nlls}, open(f"{a.check}.{a.mode}.nll.json", "w"))
    if VERBOSE and len(per_pos) <= 200:
        log("per position (position, agree, max top-5 diff): " + " ".join(f"{p}:{'y' if ok else 'N'}:{d}" for p, ok, d in per_pos))
    raise SystemExit(0)

if a.bench or (not a.chat and not a.serve):  # research: one prompt, timing JSON (--bench: token-by-token prefill, argmax) or the plain completion
    prompt = a.prompt or "The history of the city of Pittsburgh begins with"
    if a.chat:
        prompt = tok.render([{"role": "user", "content": prompt}])
    ids = tok.encode(prompt)
    if a.bench:
        a.chunk = 0
    ses = Session()
    t_pre = time.time()
    logits = ses.feed(ids)
    if not a.bench:
        log(f"[prefill: {len(ids)} tokens in {time.time() - t_pre:.1f}s, {'batched' if a.chunk and len(ids) >= a.chunk else 'token by token'} | next-token top-5 {top5(logits)} | memory peak {mx.get_peak_memory() / 2**30:.1f} GB]")
        print(prompt, end="", flush=True)
    reset_stats()
    new, dt = ses.generate(a.max_new, on_text=None if a.bench else (lambda s: print(s, end="", flush=True)))
    if a.bench:
        if RLOG:  # the read pipeline: how deep the queue was and how busy the drive
            t0r, t1r = min(r[0] for r in RLOG), max(r[2] for r in RLOG)
            busy = sorted((r[1], r[2]) for r in RLOG)
            cover, cur = 0.0, None
            for b, e in busy:
                if cur is None or b > cur[1]:
                    if cur:
                        cover += cur[1] - cur[0]
                    cur = [b, e]
                else:
                    cur[1] = max(cur[1], e)
            cover += cur[1] - cur[0]
            PROF["reads: mean in flight"] = sum(r[2] - r[1] for r in RLOG) / (t1r - t0r) * len(new) / 1000
            PROF["reads: drive busy %"] = 100 * cover / (t1r - t0r) * len(new) / 1000
            PROF["reads: queue wait ms"] = sum(r[1] - r[0] for r in RLOG) / len(RLOG) * len(new)
            PROF["reads: read ms"] = sum(r[2] - r[1] for r in RLOG) / len(RLOG) * len(new)
        if a.sim_record:
            json.dump(SIMREC, open(a.sim_record, "w"))
        if a.trace:
            json.dump({"reads": RLOG, "events": TR, "tokens": len(new), "dt": dt}, open(a.trace, "w"))
        log(json.dumps({"engine": "mlx", "mode": a.mode, "m": a.m, "keep": a.keep, "slab": CAP, "slots": sum(CAPS), "slots_max_layer": max(CAPS), "grown": st["grown"], "evicted_in_window_per_token": round(st["early"] / len(new), 1), "tokens": len(new), "tok_per_s": round(len(new) / dt, 3), "ms_per_token": round(1000 * dt / len(new), 1),
                        "expert_wait_ms": round(1000 * st["wait"] / len(new), 1), "mb_read_per_token": round(st["bytes"] / len(new) / 2**20, 1), "held_experts": round(st["held"][0] / max(1, st["held"][1]), 1),
                        "released_topk_in_memory": round(int(st["hits_dev"].item()) / max(1, st["asked"]), 4), "sync_reads": st["sync"], "peak_gb": round(mx.get_peak_memory() / 2**30, 2), "active_gb": round(mx.get_active_memory() / 2**30, 2), "dtype": a.dtype, "sim_reads": a.sim_reads,
                        "ngram_rows_per_token": round(NGS["rows"] / len(new), 1), "ngram_kb_per_token": round(NGS["bytes"] / len(new) / 2**10, 1), "ngram_wait_ms": round(1000 * NGS["wait"] / len(new), 3), "ngram_issued_ms_before_use": round(1000 * NGS["lead"] / max(1, NGS["uses"]), 2),
                        **({"sim_lzfse": a.sim_lzfse, "lzfse_read_bytes": LZ_RED, "lzfse_sample_ratio": round(LZ["sample_ratio"], 4), "lzfse_sample_padded_ratio": round(LZ["sample_padded_ratio"], 4),
                            "records_per_token": round(LZ["n"] / len(new), 1), "lzfse_io_ms_per_record": round(1000 * LZ["io_s"] / max(1, LZ["n"]), 2), "lzfse_decode_ms_per_record": round(1000 * LZ["dec_s"] / max(1, LZ["n"]), 2)} if a.sim_lzfse != "off" else {}),
                        "ms_per_token_by_stage": {k: round(1000 * v / len(new), 1) for k, v in PROF.items()}, "text": tok.decode(new)[:80]}))
    else:
        print(f"\n\n{summary(len(new), dt)}")
    raise SystemExit(0)

if not a.serve:  # research mode: the chat loop. Same Session as the server; --prompt-cache gives it the same snapshots.
    ses = Session(a.prompt_cache)
    msgs = []
    turns = [a.prompt] if a.prompt else None
    while True:
        if turns is not None:
            if not turns:
                break
            user = turns.pop(0)
            print(f"user> {user}", flush=True)
        else:
            try:
                user = input("\nuser> ").strip()
            except EOFError:
                break
            if user in ("", "/quit", "/exit"):
                break
        msgs.append({"role": "user", "content": user})
        ids, marks = tok.prompt_ids(msgs)
        ses.feed(ids, marks)
        reset_stats()
        print("assistant> ", end="", flush=True)
        new, dt = ses.generate(a.max_new, on_text=lambda s: print(s, end="", flush=True))
        text = tok.decode(new, skip_special_tokens=True)  # a thinking turn: reasoning, "</think>", the answer (the template re-renders the two parts)
        r, c = text.split("</think>", 1) if "</think>" in text else ("", text)
        msgs.append({"role": "assistant", "content": c.strip(), **({"reasoning_content": r.strip()} if r.strip() else {})})
        print(f"\n[prefill {ses.last[1]} tokens in {ses.last[2]:.1f}s, first text at {ses.last[2] + (ses.first or 0):.1f}s | {summary(len(new), dt)}]" if VERBOSE else "", flush=True)
    raise SystemExit(0)

# -- demo mode: the same engine and Session behind an OpenAI-compatible API. One model thread owns every MLX call (arrays
# are tied to the stream of the thread that made them); handler threads parse, enqueue, keep the socket alive, and the
# model thread streams the answer back. Tools go through the release's own chat template; tool calls come back as OpenAI
# tool_calls. OpenCode's per-thread title request is answered without the model.
import http.server, uuid  # noqa: E402

MODEL_ID = os.path.basename(HERE.rstrip("/")) or "store"
JOBS = queue.Queue()
ses = Session(a.prompt_cache)


def model_thread():
    mx.set_default_device(mx.gpu)
    set_qos_user_interactive()
    while True:
        try:
            fn, started, done = JOBS.get(timeout=0.5)
        except queue.Empty:
            KEEPER.touch_slabs(16)  # idle: a quarter of the slabs every half second
            continue
        started.set()
        try:
            fn()
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            done.set()


def on_model_thread(fn, on_wait=None):
    started, done = threading.Event(), threading.Event()
    JOBS.put((fn, started, done))
    while not started.wait(5):
        if on_wait:
            on_wait()
    done.wait()


threading.Thread(target=model_thread, name="model", daemon=True).start()
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)  # Qwen3-Instruct: a JSON object in the tags
XML_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)  # Qwen3-Coder: XML function/parameter blocks
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
BARE_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*\n(\{.*\})\s*$", re.S)


def parse_tool_calls(text, tools=()):
    """The model's tool calls as OpenAI tool_calls, from either format the Qwen templates use, plus the content around them."""
    names = {t.get("function", {}).get("name") for t in tools}
    specs = {t.get("function", {}).get("name"): (t.get("function", {}).get("parameters") or {}).get("properties") or {} for t in tools}
    calls = []

    def call(name, args):
        return {"id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    for m in TOOL_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
            calls.append(call(d.get("name", ""), d.get("arguments", {})))
        except Exception:
            pass
    for m in XML_RE.finditer(text):
        name, args = m.group(1), {}
        for pm in PARAM_RE.finditer(m.group(2)):
            k, v = pm.group(1), pm.group(2)
            typ = (specs.get(name, {}).get(k) or {}).get("type")
            if typ in ("integer", "number", "boolean", "object", "array") or (typ is None and v[:1] in "[{"):
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            args[k] = v
        calls.append(call(name, args))
    content = XML_RE.sub("", TOOL_RE.sub("", text)).strip()
    if not calls:  # a JSON call written without the tag wrapper ("bash\n{...}"), seen occasionally: accepted when the name is a known tool
        m = BARE_RE.match(content)
        if m and m.group(1) in names:
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict):
                    calls.append(call(m.group(1), args.get("arguments", args) if set(args) == {"name", "arguments"} else args))
                    content = ""
            except Exception:
                pass
    return content, calls


THINK_PROMPT = tok.encode("<think>\n")  # how the template's generation prompt ends when thinking is on
EFFORT = {"xhigh": "xhigh", "high": "xhigh", "medium": "medium", "low": "low", "minimal": "low"}  # the template knows xhigh / medium / low


def template_kwargs(req):
    """The chat template's switches from a request: vLLM's chat_template_kwargs (enable_thinking, reasoning_effort, ...)
    and OpenAI's reasoning_effort (what OpenCode sends for a reasoning variant), mapped onto the efforts this template
    accepts; "none" turns thinking off. Nothing set: the template's defaults (thinking on, effort xhigh)."""
    kw = dict(req.get("chat_template_kwargs") or {})
    eff = req.get("reasoning_effort")
    if not eff and isinstance(req.get("reasoning"), dict):
        eff = req["reasoning"].get("effort")
    eff = kw.pop("reasoning_effort", None) or eff
    if eff:
        if str(eff).lower() == "none":
            kw["enable_thinking"] = False
        else:
            kw["reasoning_effort"] = EFFORT.get(str(eff).lower(), str(eff))
    return kw


def title_of(messages):
    """A thread title from the last user message's first words."""
    text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
            break
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    return (" ".join(words[:7]) + ("..." if len(words) > 7 else "")) or "New thread"


def to_template(messages):
    """OpenAI message dicts -> what the chat template expects (tool arguments as dicts, content parts as text)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        pass
                tcs.append({"type": "function", "function": fn})
            m["tool_calls"] = tcs
            m["content"] = m.get("content") or ""
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        out.append(m)
    return out


def sse(rid, created, model, delta, finish=None):
    return f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n".encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _write(self, data):
        with self.wlock:
            self.wfile.write(data)
            self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local"} for m in (MODEL_ID, "title")]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        self.wlock = threading.Lock()  # keepalives (this thread) and the answer (model thread) share the socket
        if str(req.get("model", "")).endswith("title"):
            return self._canned(req, title_of(req.get("messages", [])))
        msgs = to_template(req.get("messages", []))
        tools = req.get("tools") or None
        try:
            kw = template_kwargs(req)
            ids, marks = tok.prompt_ids(msgs, tools, **kw)
        except Exception as e:  # the template refuses the conversation (no user message, bad reasoning effort, ...): the client's error, not a crash
            log(f"[serve] request refused by the chat template: {type(e).__name__}: {e}")
            return self._json(400, {"error": {"message": f"chat template: {e}", "type": "invalid_request_error"}})
        if a.prompt_cache and (tools or not os.path.exists(os.path.join(a.prompt_cache, "last_request.json"))):  # the agent's latest request, for --warm
            try:
                json.dump({"messages": req.get("messages", []), "tools": tools, "kwargs": kw}, open(os.path.join(a.prompt_cache, "last_request.json"), "w"))
            except Exception:
                pass
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or a.max_new or 1024)
        stream = bool(req.get("stream"))
        log(f"[serve] request: {len(ids)} prompt tokens, {len(tools or [])} tools, {'stream' if stream else 'json'}")
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time())
        self.send_response(200)  # headers at once (clients time out on headers); SSE comments keep the stream alive during a long prefill
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        last_ping = [time.time()]

        def ping(n=None, total=None):
            if stream and time.time() - last_ping[0] > 5:
                try:
                    self._write(f": prefill {n}/{total}\n\n".encode() if n else b": waiting\n\n")
                except Exception:
                    pass
                last_ping[0] = time.time()

        def job():
            try:
                ses.feed(ids, marks, tick=ping)
                reset_stats()
                self._respond(req, ids, stream, max_new, rid, created)
            except Exception as e:  # a client that hung up mid-answer leaves the cache ahead of the bookkeeping: the next request starts from a snapshot
                ses.reset()
                log(f"[serve] request aborted ({type(e).__name__}); session reset")
                if not isinstance(e, (BrokenPipeError, ConnectionError)):
                    raise

        on_model_thread(job, on_wait=ping)

    def _canned(self, req, text):
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time())
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for delta, fin in (({"role": "assistant", "content": text}, None), ({}, "stop")):
                self.wfile.write(sse(rid, created, "title", delta, fin))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": "title", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    def _respond(self, req, ids, stream, max_new, rid, created):
        model_name = str(req.get("model") or MODEL_ID)

        def send(delta, finish=None):
            self._write(sse(rid, created, model_name, delta, finish))

        # A thinking turn: the template's generation prompt ends with "<think>\n", so the turn starts inside the reasoning
        # and "</think>" (one token) ends it. Reasoning goes out as reasoning_content (what vLLM's qwen3 reasoning parser
        # sends, and what OpenCode shows as thinking and sends back), the rest as content with tool-call blocks held back
        # (emitted as tool_calls at the end). Whitespace at the edges is held: the template trims both parts and writes
        # its own separators when the turn comes back, and the reused prefix must match.
        thinking = ids[-2:] == THINK_PROMPT
        state = {"mode": "reasoning" if thinking else "content", "held": "", "lead": True}

        def on_text(piece):
            if state["mode"] == "tool":
                return
            buf = state["held"] + piece
            if state["mode"] == "reasoning":
                if "</think>" in buf:
                    head = buf[: buf.index("</think>")].rstrip()
                    if head:
                        send({"reasoning_content": head})
                    state.update(mode="content", held="", lead=True)
                    rest = buf[buf.index("</think>") + len("</think>") :]
                    return on_text(rest) if rest else None
                field = "reasoning_content"
            else:
                if "<tool_call>" in buf:
                    head = buf[: buf.index("<tool_call>")].rstrip()
                    if state["lead"]:
                        head = head.lstrip()
                    if head:
                        send({"content": head})
                    state.update(mode="tool", held="")
                    return
                field = "content"
            if state["lead"]:  # nothing sent in this part yet: leading whitespace is the template's separator
                buf = buf.lstrip()
            keep = len(buf) - len(buf.rstrip())  # trailing whitespace waits until something follows it
            if len(buf) - keep > 0:
                send({field: buf[: len(buf) - keep]})
                state["lead"] = False
            state["held"] = buf[len(buf) - keep :]

        if stream:
            send({"role": "assistant", "content": ""})
        new, gen_s = ses.generate(max_new, on_text=on_text if stream else None)
        text = tok.decode(new, skip_special_tokens=True)
        reasoning, rest = (text.split("</think>", 1) if "</think>" in text else (text, "")) if thinking else (None, text)
        content, calls = parse_tool_calls(rest, req.get("tools") or [])
        finish = "tool_calls" if calls else ("length" if len(new) >= max_new else "stop")
        mx.clear_cache()
        how, n_pre, pre_s = ses.last
        log(f"[serve] {how}: {n_pre} of {len(ids)} prompt tokens prefilled in {pre_s:.1f}s, first text at {pre_s + (ses.first or 0):.1f}s | {summary(len(new), max(gen_s, 1e-9))} | "
            f"reasoning {len(tok.encode(reasoning)) if reasoning else 0} tokens, tool calls {len(calls)}{'' if '</think>' in text or not thinking else ' (no </think>: the turn ended inside the reasoning)'}")
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(new), "total_tokens": len(ids) + len(new)}
        if stream:
            if calls:
                send({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]})
            send({}, finish)
            if (req.get("stream_options") or {}).get("include_usage"):
                self._write(f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [], 'usage': usage})}\n\n".encode())
            self._write(b"data: [DONE]\n\n")
            return
        msg = {"role": "assistant", "content": content if content else None}
        if reasoning is not None and reasoning.strip():
            msg["reasoning_content"] = reasoning.strip()
        if calls:
            msg["tool_calls"] = calls
        self._write(json.dumps({"id": rid, "object": "chat.completion", "created": created, "model": model_name, "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}).encode())


def warm(reason):
    """Replay the head of the agent's last request (the system prompt and the first user message: what a new thread starts
    from) with today's date stamped in (OpenCode writes "Today's date: <toDateString()>" into its system prompt), so the
    first message of a thread finds its prefix in memory. At start-up and when the date changes."""
    path = os.path.join(a.prompt_cache, "last_request.json")
    if not os.path.exists(path):
        return
    last = json.load(open(path))
    today = time.strftime("%a %b %d %Y")
    for m in last.get("messages", []):
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = re.sub(r"(Today's date: )[A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4}", lambda g: g.group(1) + today, m["content"])
    msgs = last.get("messages", [])
    msgs = msgs[: next((i for i, m in enumerate(msgs) if m.get("role") == "user"), len(msgs) - 1) + 1]
    ids, marks = tok.prompt_ids(to_template(msgs), last.get("tools") or None, **(last.get("kwargs") or {}))
    t0 = time.time()
    on_model_thread(lambda: ses.feed(ids, marks))
    log(f"warm ({reason}): {len(ids)} tokens in {time.time() - t0:.1f}s")


def rewarm_at_midnight():
    day = time.strftime("%Y-%m-%d")
    while True:
        time.sleep(30)
        if time.strftime("%Y-%m-%d") != day:
            day = time.strftime("%Y-%m-%d")
            try:
                warm("new day")
            except Exception as e:
                log(f"warm (new day) failed: {type(e).__name__}: {e}")


import faulthandler, signal  # noqa: E402

faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid> dumps every thread's stack
if a.warm and a.prompt_cache:
    warm("start")
    threading.Thread(target=rewarm_at_midnight, name="rewarm", daemon=True).start()
ensure_wired()
log(f"serving {MODEL_ID} at http://127.0.0.1:{a.serve}/v1 (OpenAI-compatible; tools via the release's chat template; prompt cache {a.prompt_cache or 'off'}; kernel wired {kernel_wired_gb():.1f} GB)")
http.server.ThreadingHTTPServer.allow_reuse_address = True
try:
    http.server.ThreadingHTTPServer(("127.0.0.1", a.serve), Handler).serve_forever()
except KeyboardInterrupt:
    pass
