"""Bytes read per decode token by layer group, from a preadlog.c log of an Inkling dmeng run.
  python3 readsplit.py LOG RUN.json [--looped 6-17] [--late 20-23]
Token windows: the decode's embedding-row reads (one per token, from the embeddings file) mark the token boundaries; the
last `tokens` of them are the decode's. Every expert-file read between two marks is charged to that token (reads named for
the next token before its mark are charged to the earlier one). Record index = offset // record bytes (file size / 128).
Prints one JSON line: MB/token, reads/token and reads per pick by group (looped, late, rest), the share of tokens in which
a looped layer's record was read more than once (and such re-reads per token), and the decode's total MB/token from the log
next to the engine's own counter."""
import json, os, re, sys
from collections import defaultdict
import numpy as np

log, rj = sys.argv[1], sys.argv[2]
rng = lambda s: set(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))
LOOP = rng(sys.argv[sys.argv.index("--looped") + 1]) if "--looped" in sys.argv else rng("6-17")
LATE = rng(sys.argv[sys.argv.index("--late") + 1]) if "--late" in sys.argv else rng("20-23")
J = json.load(open(rj))
names = [l.rstrip("\n") for l in open(log + ".files") if not l.startswith("#")]
R = np.fromfile(log, dtype=[("t", "<u8"), ("f", "<i4"), ("n", "<i4"), ("off", "<i8")])
R = R[np.argsort(R["t"], kind="stable")]
layer_of, rec_of, emb = {}, {}, set()
for i, p in enumerate(names):
    m = re.search(r"/(store[^/]*)/layer_(\d+)\.bin$", p)
    if m:
        layer_of[i] = int(m.group(2))
        rec_of[i] = os.path.getsize(p) // 128
    elif "embed" in os.path.basename(p) or p.endswith("resident.safetensors"):
        emb.add(i)
ntok = int(J["tokens"]) if "tokens" in J else int(J["decode_steps"])
em = R[np.isin(R["f"], list(emb))]
marks = em["t"][-ntok:]
assert len(marks) == ntok, (len(marks), ntok)
ex = R[np.isin(R["f"], list(layer_of))]
ex = ex[ex["t"] >= marks[0]]
tokw = np.searchsorted(marks, ex["t"], side="right") - 1
grp = lambda L: "looped" if L in LOOP else "late" if L in LATE else "rest"
mb = defaultdict(float)
reads = defaultdict(int)
seen = defaultdict(int)  # (token, layer, record) -> read starts
for r, k in zip(ex, tokw):
    L = layer_of[int(r["f"])]
    rec = rec_of[int(r["f"])]
    g = grp(L)
    mb[g] += r["n"] / 1e6
    if r["off"] % rec == 0:
        reads[g] += 1
        seen[(int(k), L, int(r["off"] // rec))] += 1
steps = J.get("model_steps") or 48
picks = {"looped": len(LOOP) * 3 * 8, "late": len(LATE) * 8, "rest": (24 - len(LOOP) - len(LATE)) * 8}
rere = defaultdict(int)
for (k, L, e), c in seen.items():
    if L in LOOP and c > 1:
        rere[k] += c - 1
out = {"tokens": ntok, "mb_per_token": {g: round(mb[g] / ntok, 2) for g in ("looped", "late", "rest")},
       "mb_per_token_total_log": round(sum(mb.values()) / ntok, 2), "mb_per_token_engine": round(J.get("mb_read_per_token", 0), 2),
       "reads_per_token": {g: round(reads[g] / ntok, 2) for g in ("looped", "late", "rest")},
       "reads_per_pick": {g: round(reads[g] / ntok / picks[g], 3) for g in ("looped", "late", "rest")},
       "looped_reread_token_share": round(len(rere) / ntok, 3), "looped_rereads_per_token": round(sum(rere.values()) / ntok, 3),
       "hit_rate": J.get("hit_rate"), "hit_rate_looped": J.get("hit_rate_looped"), "hit_rate_other": J.get("hit_rate_other"),
       "hold_releases": J.get("hold_releases"), "tok_s_mac": J.get("tok_s")}
ms = J.get("misses_by_step") or []
if ms:  # on-the-spot misses per token by group (the engine's per-step counter; step -> layer by the model's step_layer)
    sl = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    mg = defaultdict(float)
    for s, v in enumerate(ms):
        mg[grp(sl[s])] += v
    out["urgent_misses_per_token"] = {g: round(mg[g], 2) for g in ("looped", "late", "rest")}
print(json.dumps(out))
