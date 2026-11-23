"""Replay an engine run (dmeng --record R.bin --evict-trace T.bin) through the design study's simulator core
(jobs/babel/eviction_v3.py simulate(), copied in ./babel unchanged) for a policy and a slot count, so the engine's
MB/token and hit rate can be compared with the simulation on the same routing and the same router rows.
  python3 sim_trace.py RECORD.bin TRACE.bin START SLOTS(,SLOTS2) POLICY [caps] [--policy-json F] [--det]
POLICY: decay | hazard | v3 (the ranker file given by --policy-json). START: positions not counted (the prompt).
--det: the ranker's shortlist in (hazard score, record id) order (the engine's) instead of numpy argpartition.
Prints one JSON line: hits / picks, MB per counted token."""
import json, os, sys, tempfile, types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
REC, TRACE, START, SLOTS, POL = argv[0], argv[1], int(argv[2]), [int(x) for x in argv[3].split(",")], argv[4]
CAPS = "caps" in argv[5:]
DET = "--det" in argv
PJ = argv[argv.index("--policy-json") + 1] if "--policy-json" in argv else ""


def read_trace(p):
    b = open(p, "rb").read()
    assert b[:8] == b"DMEVTR1\0", "not an evict trace"
    o = 8
    npos, steps, NE, W, NL, K, npools = np.frombuffer(b, np.int32, 7, o)
    o += 28
    step_layer = np.frombuffer(b, np.int32, steps, o).copy()
    o += 4 * steps
    layer_pool = np.frombuffer(b, np.int32, NL, o).copy()
    o += 4 * NL
    pools = np.frombuffer(b, np.int64, 2 * npools, o).reshape(npools, 2).copy()
    o += 16 * npools
    n = npos * steps * W
    ri = np.frombuffer(b, np.int16, n, o).reshape(npos, steps, W); o += 2 * n
    rp = np.frombuffer(b, np.float32, n, o).reshape(npos, steps, W); o += 4 * n
    ti = np.frombuffer(b, np.int16, n, o).reshape(npos, steps, W); o += 2 * n
    tp = np.frombuffer(b, np.float32, n, o).reshape(npos, steps, W); o += 4 * n
    return dict(npos=int(npos), steps=int(steps), NE=int(NE), NL=int(NL), K=int(K), step_layer=step_layer, layer_pool=layer_pool,
                pools=pools, rel=(ri, rp), rt=(ti, tp))


def read_rec(p, steps, K):
    b = open(p, "rb").read()
    npos, st, k = np.frombuffer(b, np.int32, 3, 0)
    assert st == steps and k == K
    return np.frombuffer(b, np.int16, npos * st * k, 12).reshape(npos, st, k).astype(np.int64)


tr = read_trace(TRACE)
picks = read_rec(REC, tr["steps"], tr["K"])
T = min(picks.shape[0], tr["npos"])
picks = picks[:T]


def fill(idx, p):  # rows never written (idx -1): an empty row (indices 0..31, probability 0)
    idx = idx[:T].astype(np.int32).copy()
    p = p[:T].astype(np.float32).copy()
    miss = (idx < 0).any(-1)
    idx[miss] = np.arange(idx.shape[-1], dtype=np.int32)
    p[miss] = 0
    return idx, p, int(miss.sum())


rel_i, rel_p, rel_miss = fill(*tr["rel"])
rt_i, rt_p, rt_miss = fill(*tr["rt"])
sl = tr["step_layer"]
occ = {}
app_pass = []
for L in sl:
    occ[int(L)] = occ.get(int(L), 0) + 1
npass = max(occ.values())
seen = {}
for L in sl:
    L = int(L)
    if occ[L] > 1:
        app_pass.append(seen.get(L, 0))
        seen[L] = seen.get(L, 0) + 1
    else:
        app_pass.append(-1)


class Wrap:
    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


blob = {"picks": Wrap(picks[None]), "n_layers": tr["NL"], "n_experts": tr["NE"], "k": tr["K"], "record_bytes": int(tr["pools"][0][0]),
        "n_steps": tr["steps"], "app_layer": sl.tolist(), "app_pass": app_pass, "pool_of_layer": tr["layer_pool"].tolist(),
        "pool_rec": [int(x) for x in tr["pools"][:, 0]], "pool_share": [1.0 / len(tr["pools"])] * len(tr["pools"]), "half_life": 8.0,
        "routers": {"engine": {"idx": Wrap(rt_i[None]), "prob": Wrap(rt_p[None]), "lag": 2}},
        "released": {"idx": Wrap(rel_i[None]), "prob": Wrap(rel_p[None])}}
if npass > 1:
    blob["loop"] = [0, 0, npass]
if PJ:
    blob["routers"]["engine"]["lag"] = int(json.load(open(PJ)).get("router_lag", 2))

fake = types.ModuleType("torch")
fake.load = lambda path, weights_only=False: blob
sys.modules["torch"] = fake
tdir = tempfile.mkdtemp()
open(os.path.join(tdir, "wiki.pt"), "wb").close()
sys.argv = ["eviction_v3.py", tdir, os.path.join(tdir, "out.json")]
import importlib.util

BAB = os.path.join(HERE, "babel") if os.path.isdir(os.path.join(HERE, "babel")) else os.path.join(HERE, "..", "babel")
spec = importlib.util.spec_from_file_location("ev3", os.path.join(BAB, "eviction_v3.py"))
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)
EV.pool_slots = lambda gb: list(SLOTS)  # the engine's slot counts
if DET:
    def shortlist(self, cand, s, t, L):
        ks = self.base.vscore(s, t, L)[cand]
        o = np.lexsort((cand, ks))[:EV.PREFILTER]
        return cand[o], ks[o]
    EV.Ranked.shortlist = shortlist

tr_ = EV.load("wiki")
if POL == "decay":
    pol = EV.ES.Decay(8.0)
elif POL == "hazard":
    pol = EV.ES.Hazard(128.0)
elif POL == "v3":
    js = json.load(open(PJ))
    m = {"kind": js["kind"], "lb": np.array(js["layer_bias"], dtype=np.float64)}
    if js["kind"] == "lin":
        m["w"] = np.array(js["w"], dtype=np.float64)
    else:
        m.update(w1=np.array(js["w1"], dtype=np.float64), b1=np.array(js["b1"], dtype=np.float64), w2=np.array(js["w2"], dtype=np.float64))
    EV.PREFILTER = int(js.get("prefilter", 128))
    pol = EV.Ranked(EV.ES.Hazard(128.0), EV.Feats(tr_, 0, "engine"), m, "v3")
else:
    raise SystemExit("policy decay|hazard|v3")
T_ = EV.Tx(tr_["picks"][0], START)
r = EV.simulate(T_, 0.0, pol, CAPS)
print(json.dumps({"policy": POL, "caps": CAPS, "slots": SLOTS, "det": DET, "positions": T, "counted": r["tokens"], "hit_rate": round(r["hits"] / r["picks"], 6),
                  "mb_per_token": round(r["bytes"] / r["tokens"] / 1e6, 4), "picks": r["picks"], "hits": r["hits"],
                  "rel_rows_missing": rel_miss, "rt_rows_missing": rt_miss}))
