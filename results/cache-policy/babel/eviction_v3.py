"""E6 v3 (2026-11-22): the phone study's hazard keep-score and uniform per-layer caps, beside decay / LRU / Belady / learned v1 / v2,
and a ranker that takes hazard's estimate as a feature (v3). Trace-driven, CPU only, on q36_router.py / ink_router.py DUMP_TRACES.
  python eviction_v3.py TRACE_DIR OUT_JSON [--budgets ...] [--workers 32] [--windows N] [--train-windows N]
        [--v1 POLICY_JSON] [--v2 POLICY_JSON] [--validate RECORD_BIN PROMPT_IDS]
Rules come from the phone study's evict_sim.py (copied unchanged as evict_sim_phone.py; md5 in the output): its Decay, LRU, Hazard
and Belady classes are used as they are, with its module constants set to this trace's shape, and the core below is its simulate()
in engine semantics (the picks of a step are taken in order, a hit or a miss pins its slot as it is taken, so a miss may evict a later
pick of the same step; use statistics at the end of the step; hazard touches the step's picks before its misses, as the dmeng port's
pre_use()), generalised to the trace: steps are applications (Inkling: 48 over 24 layers, a looped layer's picks pinned until its last
pass of the token), slot pools with their own record bytes (Inkling: NVFP4 / MXFP8, mx_frac from the trace), and two modes:
  pool  one victim set per pool (the engine today)
  caps  uniform per-layer caps: each pool's slots split evenly over its layers (remainder to the first layers), a miss evicts one of
        its own layer's records (evict_sim.layer_caps(total, "uniform") per pool)
Policies:
  decay      the engine's default (half-life from the trace, 8)        lru     least recently used
  hazard     Store.swift Evict(kind: .hazard): per-layer online table P(used at the next token | log2 age, log2 decayed count h 128)
  belady     offline optimum (bound)
  v1         the exported first learned predictor (eviction_study.py policy JSON; its router's features from the trace)
  v2x        the exported v2 ranker (eviction_v2.py policy JSON: decay prefilter 128, model on 17 features)
  v2r_lin / v2r_mlp   v2 retrained here on every budget and both modes (snapshots on decay's decisions, decay prefilter)
  v3_lin / v3_mlp     v2's features + hazard's estimate (and its log2) for the candidate, hazard prefilter 128, trained on
                      hazard's decisions (the Belady victim among the sampled candidates as the target)
MB per token counts every position from a cold window start (pitt validation: the prompt replayed first, not counted);
gap = (MB decay one pool - MB policy) / (MB decay one pool - MB Belady one pool), the same reference for both modes."""
import hashlib, importlib.util, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("evict_sim_phone", os.path.join(HERE, "evict_sim_phone.py"))
ES = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ES)
ES_MD5 = hashlib.md5(open(os.path.join(HERE, "evict_sim_phone.py"), "rb").read()).hexdigest()

args = sys.argv[1:]
TDIR, OUT = args[0], args[1]


def opt(name, default, cast=float, many=False):
    if name in args:
        i = args.index(name)
        vals = []
        for v in args[i + 1:]:
            if v.startswith("--"):
                break
            vals.append(cast(v))
        return vals if many else vals[0]
    return default


WORKERS = opt("--workers", os.cpu_count() or 8, int)
MAXWIN = opt("--windows", 0, int)
TRWIN = opt("--train-windows", 0, int)
V1_PATH = opt("--v1", "", str)
V2_PATH = opt("--v2", "", str)
VALIDATE = opt("--validate", [], str, many=True)
PREFILTER = int(os.environ.get("PREFILTER", "128"))
SNAP_P = float(os.environ.get("SNAP_P", "0.01"))
SNAP_CAND = int(os.environ.get("SNAP_CAND", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
HAZ_H = float(os.environ.get("HAZ_H", "128"))
INF = np.inf
HLS = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])

DOMAINS = [d for d in ("wiki", "web", "chat", "code", "agent") if os.path.exists(os.path.join(TDIR, f"{d}.pt"))]
b0 = torch.load(os.path.join(TDIR, f"{DOMAINS[0]}.pt"), weights_only=False)
NL, NE, K, REC = b0["n_layers"], b0["n_experts"], b0["k"], b0["record_bytes"]
NR = NL * NE
NSTEP = int(b0.get("n_steps", NL))
APP_LAYER = np.array(b0.get("app_layer", list(range(NL))))
APP_PASS = np.array(b0.get("app_pass", [-1] * NL))
NPASS = int(b0["loop"][2]) if "loop" in b0 else 1
POOL_OF = np.array(b0.get("pool_of_layer", [0] * NL))
POOL_REC = list(b0.get("pool_rec", [REC]))
POOL_SHARE = list(b0.get("pool_share", [1.0]))
HL = float(b0.get("half_life", 8.0))
ROUTERS = list(b0["routers"])
LAGS = {r: b0["routers"][r]["lag"] for r in ROUTERS}
MODEL = "inkling" if NSTEP != NL else "q36"
del b0
BUDGETS = opt("--budgets", [0.8, 1.27, 2.0, 3.3, 5.0, 6.6] if MODEL == "q36" else [4.0, 6.0, 7.8], float, many=True)
ES.NL, ES.NE, ES.NR, ES.K = NL, NE, NR, K  # the phone classes read these at reset / call time
REC_OF_LAYER = np.array([POOL_REC[POOL_OF[L]] for L in range(NL)], dtype=np.float64)
V2_FEATS = ["log2_c1", "log2_c2", "log2_c4", "log2_c8", "log2_c16", "log2_c32", "log2_c64", "log2_1p_since_use", "log2_1p_since_use2",
            "rel_p_now", "rel_p_ema2", "rel_p_ema8", "rt_p_now", "rt_p_ema4", "rt_p_ema16", "rt_p_max8", "log2_1p_pos"]
V3_FEATS = V2_FEATS + ["hazard_p", "log2_hazard_p"]
DATA = {}


def load(d):
    if d not in DATA:
        b = torch.load(os.path.join(TDIR, f"{d}.pt"), weights_only=False)
        DATA[d] = {"picks": b["picks"].numpy().astype(np.int64), "part": b["part"].numpy() if "part" in b else None, "parts": b.get("parts"),
                   "rel": (b["released"]["idx"].numpy().astype(np.int32), b["released"]["prob"].numpy().astype(np.float32)) if "released" in b else None,
                   "routers": {r: (v["idx"].numpy().astype(np.int32), v["prob"].numpy().astype(np.float32)) for r, v in b["routers"].items()}}
    return DATA[d]


# --------------------------------------------------------------------------------------------------- slots, caps, text
def pool_slots(gb):
    if len(POOL_REC) == 1 and POOL_REC[0] == ES.REC_BYTES and NL == 40:
        return [ES.slots_of(gb)]  # the phone study's sizes (1.27 GB = 720 slots)
    return [min(int(gb * 1e9 * POOL_SHARE[q] // POOL_REC[q]), int((POOL_OF == q).sum()) * NE) for q in range(len(POOL_REC))]


def groups(gb, caps):
    """-> (group of layer [NL], cap per group, record mask per group or None for per-layer slices)"""
    Sp = pool_slots(gb)
    if not caps:
        g = POOL_OF.copy()
        masks = [np.repeat(POOL_OF == q, NE) for q in range(len(Sp))] if len(Sp) > 1 else [None]
        return g, list(Sp), masks
    cap = [0] * NL
    for q, tot in enumerate(Sp):
        Ls = [L for L in range(NL) if POOL_OF[L] == q]
        c = ES.layer_caps(tot, "uniform") if len(Ls) == ES.NL and tot >= len(Ls) else None
        if c is None:  # ES.layer_caps splits over ES.NL layers; for a pool of a subset split the same way here
            c = [tot // len(Ls)] * len(Ls)
            for i in range(tot - sum(c)):
                c[i] += 1
        for L, v in zip(Ls, c):
            cap[L] = v
    return np.arange(NL), cap, None


class Tx:
    """the phone core's Text interface over a trace window: steps = applications, record = (layer, expert)"""

    def __init__(self, picks, start=0):
        self.T = picks.shape[0]
        self.start = start
        recs = (APP_LAYER[None, :, None] * NE + picks).reshape(-1, K)
        self.steps = [row.tolist() for row in recs]
        S = len(self.steps)
        nx = np.full((S, K), 10 ** 9, dtype=np.int64)
        seen = {}
        for s in range(S - 1, -1, -1):
            row = self.steps[s]
            for j in range(K):
                nx[s, j] = seen.get(row[j], 10 ** 9)
            for j in range(K):
                seen[row[j]] = s
        self.nxt = nx
        first = np.full(NR, 10 ** 9, dtype=np.int64)
        for r, s in seen.items():
            first[r] = s
        self.first_use = first
        self.names = None


# ------------------------------------------------------------------------------------------- feature state (learned)
class Feats:
    """v1 / v2 / v3 features, updated as eviction_v2.replay does: counts and last uses at every application's end, the released
    router's row after its application, the early router's row lag applications ahead (lag < first applications at token start)"""

    def __init__(self, tr, w, router):
        self.rel = (tr["rel"][0][w], tr["rel"][1][w]) if tr["rel"] is not None else None
        self.q = (tr["routers"][router][0][w], tr["routers"][router][1][w])
        self.lag = LAGS[router]

    def reset(self):
        N = NR
        self.cnt = np.zeros((len(HLS), N))
        self.cnt_t = np.zeros(N)
        self.last1 = np.full(N, -1e9)
        self.last2 = np.full(N, -1e9)
        self.rel_now, self.rel2, self.rel8 = np.zeros(N), np.zeros(N), np.zeros(N)
        self.rt_now, self.rt4, self.rt16, self.rtmax = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        self.a2, self.a8, self.a4, self.a16, self.am = (2 ** (-1.0 / h) for h in (2, 8, 4, 16, 8))

    def rt_row(self, t, a_):
        L = APP_LAYER[a_]
        row = np.zeros(NE)
        row[self.q[0][t, a_]] = self.q[1][t, a_]
        s = slice(L * NE, (L + 1) * NE)
        self.rt_now[s] = row
        self.rt4[s] = self.rt4[s] * self.a4 + (1 - self.a4) * row
        self.rt16[s] = self.rt16[s] * self.a16 + (1 - self.a16) * row
        self.rtmax[s] = np.maximum(self.rtmax[s] * self.am, row)
        return L

    def rel_row(self, t, a_):
        if self.rel is None:
            return
        L = APP_LAYER[a_]
        row = np.zeros(NE)
        row[self.rel[0][t, a_]] = self.rel[1][t, a_]
        s = slice(L * NE, (L + 1) * NE)
        self.rel_now[s] = row
        self.rel2[s] = self.rel2[s] * self.a2 + (1 - self.a2) * row
        self.rel8[s] = self.rel8[s] * self.a8 + (1 - self.a8) * row

    def token_start(self, t):
        return [self.rt_row(t, a_) for a_ in range(min(self.lag, NSTEP))]

    def step_end(self, s, t, picks):
        a_ = s % NSTEP
        ids = np.array(picks)
        dt = t - self.cnt_t[ids]
        self.cnt[:, ids] = self.cnt[:, ids] * np.exp2(-dt[None, :] / HLS[:, None]) + 1
        self.cnt_t[ids] = t
        self.last2[ids] = self.last1[ids]
        self.last1[ids] = t
        self.rel_row(t, a_)
        if a_ + self.lag < NSTEP:
            return self.rt_row(t, a_ + self.lag)
        return None

    def v2(self, cid, t):
        dt = t - self.cnt_t[cid]
        c = self.cnt[:, cid] * np.exp2(-dt[None, :] / HLS[:, None])
        f = np.empty((len(cid), 17))
        f[:, 0:7] = np.log2(c.T + 1e-3)
        f[:, 7] = np.log2(1 + np.maximum(0, t - self.last1[cid]))
        f[:, 8] = np.log2(1 + np.minimum(1e6, np.maximum(0, t - self.last2[cid])))
        f[:, 9], f[:, 10], f[:, 11] = self.rel_now[cid], self.rel2[cid], self.rel8[cid]
        f[:, 12], f[:, 13], f[:, 14], f[:, 15] = self.rt_now[cid], self.rt4[cid], self.rt16[cid], self.rtmax[cid]
        f[:, 16] = np.log2(1 + t)
        return f

    def v1(self, cid, t):
        """eviction_study.py: log2 c2 / c8 / c32, log2(1 + since last use), router p_now / ema4 / ema16 (= rt_now / rt4 / rt16)"""
        dt = t - self.cnt_t[cid]
        f = np.empty((len(cid), 7))
        for j, h in enumerate((1, 3, 5)):  # HLS index of 2 / 8 / 32
            f[:, j] = np.log2(self.cnt[h, cid] * np.exp2(-dt / HLS[h]) + 1e-3)
        f[:, 3] = np.log2(1 + np.maximum(0, t - self.last1[cid]))
        f[:, 4], f[:, 5], f[:, 6] = self.rt_now[cid], self.rt4[cid], self.rt16[cid]
        return f


def model_score(m, f, lay):
    if m["kind"] == "lin":
        return f @ m["w"] + m["lb"][lay]
    return np.maximum(0.0, f @ m["w1"] + m["b1"]) @ m["w2"] + m["lb"][lay]


def model_macs(m, n):
    return n * (len(m["w"]) if m["kind"] == "lin" else m["w1"].shape[0] * m["w1"].shape[1] + m["w1"].shape[1])


class V1(ES.Policy):
    """exported eviction_study.py predictor: priority = w . features + b (layer one-hot weights), evict the lowest; a static score
    over every record, recomputed at token start and, after each step, for its picks and the layer whose early-router row moved"""

    def __init__(self, js, fs):
        self.w = np.array(js["w"], dtype=np.float64)
        self.b = float(js["b"])
        self.fs = fs
        self.lb = np.repeat(self.w[7:7 + NL], NE)
        self.macs = 0

    def reset(self, T):
        self.fs.reset()
        self.sc = np.zeros(NR)

    def _re(self, idx, t):
        self.sc[idx] = self.fs.v1(idx, t) @ self.w[:7] + self.b + self.lb[idx]
        self.macs += len(idx) * 8

    def token_start(self, t):
        self.fs.token_start(t)
        self._re(np.arange(NR), float(t))

    def step_end(self, s, t, L, picks):
        L2 = self.fs.step_end(s, t, picks)
        self._re(np.array(picks), float(t))
        if L2 is not None:
            self._re(np.arange(L2 * NE, (L2 + 1) * NE), float(t))


class Ranked(ES.Policy):
    """prefilter PREFILTER candidates by a base order (decay key or hazard estimate), evict the lowest model score among them;
    with model None the base order decides (used to take training snapshots on the base policy's own decisions)"""
    dynamic_pick = True

    def __init__(self, base, fs, model=None, feats="v2", hz=None):
        self.base, self.fs, self.model, self.kind = base, fs, model, feats
        self.hz = hz if hz is not None else (base if isinstance(base, ES.Hazard) else None)
        self.macs = 0

    def reset(self, T):
        self.base.reset(T)
        if self.hz is not None and self.hz is not self.base:
            self.hz.reset(T)
        self.fs.reset()

    def token_start(self, t):
        self.base.token_start(t)
        if self.hz is not None and self.hz is not self.base:
            self.hz.token_start(t)
        self.fs.token_start(t)

    def step_begin(self, s, t, L, picks):
        self.base.step_begin(s, t, L, picks)
        if self.hz is not None and self.hz is not self.base:
            self.hz.step_begin(s, t, L, picks)

    def step_end(self, s, t, L, picks):
        self.base.step_end(s, t, L, picks)
        if self.hz is not None and self.hz is not self.base:
            self.hz.step_end(s, t, L, picks)
        self.fs.step_end(s, t, picks)

    def on_insert(self, r, s, t, by_name=False):
        self.base.on_insert(r, s, t, by_name)

    def shortlist(self, cand, s, t, L):
        ks = self.base.vscore(s, t, L)[cand]
        if len(cand) > PREFILTER:
            sel = np.argpartition(ks, PREFILTER - 1)[:PREFILTER]
            return cand[sel], ks[sel]
        return cand, ks

    def features(self, idx, t):
        f = self.fs.v2(idx, float(t))
        if self.kind == "v3":
            hp = self.hz.sc[idx]
            f = np.concatenate([f, hp[:, None], np.log2(hp + 1e-4)[:, None]], 1)
        return f

    def pick(self, cand, s, t, L):
        idx, ks = self.shortlist(cand, s, t, L)
        if self.model is None:
            return int(idx[int(np.argmin(ks))])
        v = model_score(self.model, self.features(idx, t), idx // NE)
        self.macs += model_macs(self.model, len(idx))
        return int(idx[int(np.argmin(v))])


# ------------------------------------------------------------------------------------------------------------- core
def simulate(T_, gb, pol, caps, snap=None, seed=0):
    """evict_sim.simulate (engine semantics, cache only) over applications, pools and caps; snap = list to append training
    snapshots to (Ranked base policy: features of up to SNAP_CAND shortlisted candidates, target = the farthest next use)"""
    g_of, cap, masks = groups(gb, caps)
    pol.reset(T_)
    held = np.zeros(NR, bool)
    avail = np.zeros(NR, bool)
    pins = np.zeros(NR, np.int32)
    cnt = [0] * len(cap)
    hold = {}
    nu = T_.first_use.astype(np.float64).copy() if snap is not None else None
    rng = np.random.default_rng(seed)
    dyn = getattr(pol, "dynamic_pick", False)
    picks_n = hits = 0
    mbytes = 0.0
    steps = T_.steps
    t_ops = 0

    def cands(L):
        if caps:
            a = L * NE
            return np.flatnonzero(avail[a:a + NE]) + a
        m = masks[g_of[L]]
        return np.flatnonzero(avail if m is None else (avail & m))

    def victim(s, t, L):
        if dyn:
            idx = cands(L)
            return pol.pick(idx, s, t, L) if len(idx) else -1
        if caps:
            a = L * NE
            c = np.where(avail[a:a + NE], pol.vscore(s, t, L)[a:a + NE], INF)
            v = int(c.argmin())
            return -1 if c[v] == INF else v + a
        m = masks[g_of[L]]
        c = np.where(avail if m is None else (avail & m), pol.vscore(s, t, L), INF)
        v = int(c.argmin())
        return -1 if c[v] == INF else v

    for t in range(T_.T):
        counted = t >= T_.start
        pol.token_start(t)
        for a_ in range(NSTEP):
            L = int(APP_LAYER[a_])
            g = int(g_of[L])
            s = t * NSTEP + a_
            picks = steps[s]
            pol.step_begin(s, t, L, picks)
            for r in picks:
                if counted:
                    picks_n += 1
                if held[r]:
                    if counted:
                        hits += 1
                    avail[r] = False
                    continue
                if counted:
                    mbytes += REC_OF_LAYER[L]
                if cnt[g] >= cap[g]:
                    if snap is not None and rng.random() < SNAP_P:
                        idx = cands(L)
                        if len(idx) >= 2:
                            idx, _ = pol.shortlist(idx, s, t, L)
                            if len(idx) > SNAP_CAND:
                                idx = idx[rng.choice(len(idx), size=SNAP_CAND, replace=False)]
                            snap.append((pol.features(idx, t).astype(np.float32), (idx // NE).astype(np.int64), int(np.argmax(nu[idx]))))
                    v = victim(s, t, L)
                    if v < 0:
                        raise RuntimeError(f"no slot: gb {gb} caps {caps} step {s}")
                    held[v] = False
                    avail[v] = False
                    cnt[int(g_of[v // NE])] -= 1
                    pol.on_evict(v)
                held[r] = True
                cnt[g] += 1
                pol.on_insert(r, s, t, False)
                avail[r] = False
            pol.step_end(s, t, L, picks)
            if nu is not None:
                nx = T_.nxt[s]
                for j, r in enumerate(picks):
                    nu[r] = nx[j]
            rel = picks
            if APP_PASS[a_] >= 0:  # loop hold: pinned until the layer's last pass of the token
                if APP_PASS[a_] < NPASS - 1:
                    for r in picks:
                        pins[r] += 1
                    hold.setdefault(L, []).extend(picks)
                else:
                    rel = hold.pop(L, []) + list(picks)
                    for r in rel[:len(rel) - len(picks)]:
                        pins[r] -= 1
            for r in rel:
                if pins[r] == 0 and held[r]:
                    avail[r] = True
        if isinstance(pol, ES.Hazard) or (isinstance(pol, Ranked) and pol.hz is not None):
            t_ops += NR + NSTEP * NE  # the per-token table pass + each step's key refresh
    ntok = T_.T - T_.start
    macs = getattr(pol, "macs", 0)
    return {"picks": picks_n, "hits": hits, "bytes": mbytes, "tokens": ntok, "macs": macs, "table_ops": t_ops}


# ---------------------------------------------------------------------------------------------------------- policies
MODELS = {}  # name -> (model dict, router, feats)
V1JS = json.load(open(V1_PATH)) if V1_PATH else None
V2JS = json.load(open(V2_PATH)) if V2_PATH else None


def v2x_model():
    js = V2JS
    m = {"kind": js["kind"], "lb": np.array(js["layer_bias"], dtype=np.float64)}
    if js["kind"] == "lin":
        m["w"] = np.array(js["w"], dtype=np.float64)
    else:
        m.update(w1=np.array(js["w1"]), b1=np.array(js["b1"]), w2=np.array(js["w2"]))
    return m


def make(name, tr, w):
    if name == "decay":
        return ES.Decay(HL)
    if name == "lru":
        return ES.LRU()
    if name == "hazard":
        return ES.Hazard(HAZ_H)
    if name == "belady":
        return ES.Belady()
    if name == "v1":
        return V1(V1JS, Feats(tr, w, V1JS["router"]))
    if name == "v2x":
        return Ranked(ES.Decay(HL), Feats(tr, w, V2JS["router"]), v2x_model(), "v2")
    if name == "snap_v2":
        return Ranked(ES.Decay(HL), Feats(tr, w, ROUTER_L), None, "v2")
    if name == "snap_v3":
        return Ranked(ES.Hazard(HAZ_H), Feats(tr, w, ROUTER_L), None, "v3")
    m, r, kind = MODELS[name]
    base = ES.Hazard(HAZ_H) if kind == "v3" else ES.Decay(HL)
    return Ranked(base, Feats(tr, w, r), m, kind)


ROUTER_L = os.environ.get("ROUTER_L", "full_k2_h512" if "full_k2_h512" in ROUTERS else ROUTERS[0])


def _job(a):
    d, w, gb, caps, name, snap = a
    tr = load(d)
    T_ = Tx(tr["picks"][w])
    pol = make(name, tr, w)
    sn = [] if snap else None
    r = simulate(T_, gb, pol, caps, snap=sn, seed=w * 7 + int(gb * 100) + (1 if caps else 0))
    if snap:
        r["snaps"] = sn
    return (d, w, gb, caps, name), r


def train(snaps, kind, FD):
    m = SNAP_CAND
    n = len(snaps)
    X = np.zeros((n, m, FD), np.float32)
    Lc = np.zeros((n, m), np.int64)
    M = np.zeros((n, m), bool)
    Y = np.zeros(n, np.int64)
    for i, (x, l, y) in enumerate(snaps):
        X[i, :len(x)], Lc[i, :len(x)], M[i, :len(x)], Y[i] = x, l, True, y
    X, Lc, M, Y = map(torch.from_numpy, (X, Lc, M, Y))
    flat = X[M]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xn = (X - mu) / sd
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    nv = max(1, n // 10)
    vi, ti = perm[:nv], perm[nv:]
    torch.manual_seed(0)
    net = torch.nn.Linear(FD, 1) if kind == "lin" else torch.nn.Sequential(torch.nn.Linear(FD, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    lb = torch.nn.Parameter(torch.zeros(NL))
    o = torch.optim.Adam(list(net.parameters()) + [lb], lr=3e-3)

    def scores(idx):
        sc = net(Xn[idx])[..., 0] + lb[Lc[idx]]
        return sc.masked_fill(~M[idx], 1e9)

    for ep in range(EPOCHS):
        for b in torch.split(ti[torch.randperm(len(ti))], 4096):
            o.zero_grad()
            torch.nn.functional.cross_entropy(-scores(b), Y[b]).backward()
            o.step()
    with torch.no_grad():
        top1 = float((scores(vi).argmin(1) == Y[vi]).float().mean())
        chance = float((1.0 / M[vi].sum(1).float()).mean())
    mod = {"kind": kind}
    if kind == "lin":
        wr = net.weight.detach().numpy()[0].astype(np.float64)
        mod["w"] = wr / sd.numpy()
        mod["lb"] = lb.detach().numpy().astype(np.float64) + float(net.bias.detach()) - float((wr * mu.numpy() / sd.numpy()).sum())
    else:
        W1 = net[0].weight.detach().numpy().astype(np.float64)
        mod["w1"] = W1.T / sd.numpy()[:, None]
        mod["b1"] = net[0].bias.detach().numpy().astype(np.float64) - (mu.numpy() / sd.numpy()) @ W1.T
        mod["w2"] = net[2].weight.detach().numpy()[0].astype(np.float64)
        mod["lb"] = lb.detach().numpy().astype(np.float64) + float(net[2].bias.detach())
    return mod, {"snapshots": n, "val_belady_top1": top1, "chance": chance}


def validate(rec_path, ids_path):
    """the phone study's pitt cells, through its own simulate() and through this core (q36 only)"""
    assert MODEL == "q36"
    rec = ES.load_rec(rec_path)
    start = len(ES.ids(ids_path))
    T_es = object.__new__(ES.Text)
    T_es.name, T_es.rec, T_es.start, T_es.src, T_es.T, T_es.names, T_es._nxt = "pitt", rec, start, rec_path, rec.shape[0], None, None
    T_es.recs = (rec + (np.arange(NL) * NE)[None, :, None]).reshape(-1, K)
    T_es.steps = [row.tolist() for row in T_es.recs]
    T_ = Tx(rec, start)
    out = []
    for spec, gb, caps in (("decay", 0.8, False), ("decay", 1.27, False), ("decay", 2.0, False), ("decay", 3.3, False), ("lru", 1.27, False),
                           ("belady", 1.27, False), ("hazard", 0.8, True), ("hazard", 1.27, True), ("hazard", 2.0, True), ("hazard", 3.3, False)):
        sl = ES.slots_of(gb)
        es_spec = {"decay": "decay:8", "lru": "lru", "belady": "belady", "hazard": f"hazard:{int(HAZ_H)}"}[spec]
        a = ES.simulate(T_es, sl, es_spec, caps=ES.layer_caps(sl, "uniform") if caps else None)
        b = simulate(T_, gb, make(spec, None, 0), caps)
        mb_b = b["bytes"] / b["tokens"] / 1e6
        row = {"policy": spec, "gb": gb, "caps": caps, "phone_core_mb": round(a["mb_per_token"], 2), "this_core_mb": round(mb_b, 2),
               "phone_core_hit": round(a["hit"], 4), "this_core_hit": round(b["hits"] / b["picks"], 4)}
        print("VALIDATE", json.dumps(row), flush=True)
        out.append(row)
    return out


def main():
    t0 = time.time()
    out = {"trace_dir": TDIR, "model": MODEL, "budgets": BUDGETS, "evict_sim_md5": ES_MD5, "hazard_h": HAZ_H, "prefilter": PREFILTER,
           "router_learned": ROUTER_L, "v1": V1_PATH, "v2": V2_PATH, "pool_rec": POOL_REC, "pool_share": POOL_SHARE,
           "slots": {str(gb): pool_slots(gb) for gb in BUDGETS}, "models": {}, "results": {}}
    if VALIDATE:
        out["validate"] = validate(*VALIDATE)
    for d in DOMAINS + ["train"]:
        load(d)
    nwin = {d: load(d)["picks"].shape[0] for d in DOMAINS}
    ntr = load("train")["picks"].shape[0]
    if MAXWIN:
        nwin = {d: min(v, MAXWIN) for d, v in nwin.items()}
    if TRWIN:
        ntr = min(ntr, TRWIN)
    out.update(windows=nwin, train_windows=ntr, tokens_per_window=int(load(DOMAINS[0])["picks"].shape[1]))
    print("START", json.dumps({k: out[k] for k in ("model", "budgets", "slots", "windows", "train_windows", "tokens_per_window", "evict_sim_md5")}), flush=True)
    # 1. train v2r (decay's decisions) and v3 (hazard's decisions) on every budget and both modes
    for tag, FD in (("v2", 17), ("v3", 19)):
        with Pool(WORKERS) as pool:
            got = pool.map(_job, [("train", w, gb, caps, f"snap_{tag}", True) for w in range(ntr) for gb in BUDGETS for caps in (False, True)])
        snaps = [x for g in got for x in g[1]["snaps"]]
        for kind in ("lin", "mlp"):
            m_, info = train(snaps, kind, FD)
            MODELS[f"{tag}{'r' if tag == 'v2' else ''}_{kind}"] = (m_, ROUTER_L, tag)
            out["models"][f"{tag}{'r' if tag == 'v2' else ''}_{kind}"] = info
            print("TRAINED", tag, kind, json.dumps(info), round(time.time() - t0), flush=True)
    # 2. every policy, both modes, every held-out window
    pols = ["decay", "lru", "hazard", "belady"] + (["v1"] if V1JS else []) + (["v2x"] if V2JS else []) + list(MODELS)
    jobs = [(d, w, gb, caps, p, False) for d in DOMAINS for w in range(nwin[d]) for gb in BUDGETS for caps in (False, True) for p in pols]
    jobs.sort(key=lambda j: j[2])  # small budgets (slowest) first
    with Pool(WORKERS) as pool:
        got = pool.map(_job, jobs, chunksize=1)
    agg = {}
    for (d, w, gb, caps, p), r in got:
        a = agg.setdefault((d, gb, caps, p), {"picks": 0, "hits": 0, "bytes": 0.0, "tokens": 0, "macs": 0, "table_ops": 0})
        for k in a:
            a[k] += r[k]
    for (d, gb, caps, p), a in agg.items():
        out["results"].setdefault("caps" if caps else "pool", {}).setdefault(d, {}).setdefault(str(gb), {})[p] = {
            "hit_rate": a["hits"] / a["picks"], "mb_per_token": a["bytes"] / a["tokens"] / 1e6,
            "model_macs_per_token": a["macs"] / a["tokens"], "table_ops_per_token": a["table_ops"] / a["tokens"]}
    for mode in out["results"]:
        for d in out["results"][mode]:
            for gb, ps in out["results"][mode][d].items():
                ref = out["results"]["pool"][d][gb]
                m0, mb = ref["decay"]["mb_per_token"], ref["belady"]["mb_per_token"]
                for p, x in ps.items():
                    x["gap_recovered"] = (m0 - x["mb_per_token"]) / (m0 - mb) if m0 > mb else None
    # 3. the default per budget: lowest mean MB/token over the domains among online policies (Belady excluded), per mode and overall
    out["choice"] = {}
    for gb in map(str, BUDGETS):
        rows = []
        for mode in ("pool", "caps"):
            for p in pols:
                if p == "belady":
                    continue
                mbs = [out["results"][mode][d][gb][p]["mb_per_token"] for d in DOMAINS]
                rows.append((float(np.mean(mbs)), mode, p))
        rows.sort()
        haz = min(r for r in rows if r[2] == "hazard")
        learned = [r for r in rows if r[2] not in ("decay", "lru", "hazard")]
        out["choice"][gb] = {"best": {"policy": rows[0][2], "mode": rows[0][1], "mean_mb": rows[0][0]},
                             "best_hazard": {"mode": haz[1], "mean_mb": haz[0]},
                             "best_learned": {"policy": learned[0][2], "mode": learned[0][1], "mean_mb": learned[0][0]} if learned else None,
                             "decay_pool_mean_mb": float(np.mean([out["results"]["pool"][d][gb]["decay"]["mb_per_token"] for d in DOMAINS])),
                             "belady_pool_mean_mb": float(np.mean([out["results"]["pool"][d][gb]["belady"]["mb_per_token"] for d in DOMAINS]))}
        print("CHOICE", gb, json.dumps(out["choice"][gb]), flush=True)
    for mode in ("pool", "caps"):
        for d in DOMAINS:
            for gb in map(str, BUDGETS):
                ps = out["results"][mode][d][gb]
                print("EVICT3", mode, d, gb, json.dumps({p: [round(x["hit_rate"], 4), round(x["mb_per_token"], 1), round(x["gap_recovered"], 3)] for p, x in ps.items()}), flush=True)
    out["s"] = round(time.time() - t0)
    json.dump(out, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    # 4. export the best v3 model (hazard feature) for the engine, with its prefilter and the hazard rule it reads
    v3s = [p for p in MODELS if p.startswith("v3")]
    if v3s:
        best = min(v3s, key=lambda p: sum(min(out["results"][mo][d][str(gb)][p]["mb_per_token"] for mo in ("pool", "caps")) for d in DOMAINS for gb in BUDGETS))
        m_, r, _ = MODELS[best]
        pol = {"policy": "learned eviction v3: at an eviction take the PREFILTER unpinned candidates (of the pool, or of the layer with per-layer caps) "
                         "with the lowest hazard estimate, score each with the model on the features below, evict the lowest score",
               "name": best, "prefilter": PREFILTER, "prefilter_order": f"hazard (Store.swift Evict(kind: .hazard), h {HAZ_H}, evict_sim_phone.py Hazard)",
               "router": r, "router_lag": LAGS[r], "kind": m_["kind"], "features": V3_FEATS, "layer_bias": [float(v) for v in m_["lb"]],
               "feature_notes": "as eviction_v2 (c_h decayed use counts, since_use(2), rel_p released router top-32 probability + EMAs, rt_p early router "
                                "probability + EMAs + decayed max, pos); hazard_p = the hazard table's estimate for the record at the next token (hzU/hzN at "
                                "its age / decayed-count bucket), log2_hazard_p = log2(hazard_p + 1e-4)"}
        if m_["kind"] == "lin":
            pol["w"] = [float(v) for v in m_["w"]]
        else:
            pol.update(w1=[[float(v) for v in row] for row in m_["w1"]], b1=[float(v) for v in m_["b1"]], w2=[float(v) for v in m_["w2"]],
                       formula="score = relu(f @ w1 + b1) @ w2 + layer_bias[layer]")
        json.dump(pol, open(OUT.replace(".json", "_policy.json"), "w"), indent=1)
    print("EVICTION_V3_DONE", OUT, out["s"], flush=True)


if __name__ == "__main__":
    main()
