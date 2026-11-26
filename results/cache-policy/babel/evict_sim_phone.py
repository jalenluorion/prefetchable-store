"""Keep-or-evict policies for the expert cache at phone sizes, offline on recorded routing (no model run).

One simulator: a cache of N records of 1,769,472 B (one pool over the 40 layers, or per-layer caps) replays recorded
top-8 routing step by step (token t, layer L) and a policy chooses the victim on every miss. Reports hit rate and MB read
per counted token (decimal MB, as stage4_sizing / P0), and, where the trained router's names are recorded, which misses
were named before their layer ran (could be read early) and which were not (read on the spot).

Two step semantics:
  engine  dmeng's Core::lossless: the picks of a step are taken in order, a hit or a miss pins its slot as it is taken,
          so a miss may evict a later pick of the same step; use statistics are updated at the end of the step.
  phone   PhoneStore Store.swift exact mode: the layer's picks are touched first (all of them protected), then the
          misses take slots in order; optional keep-window drop at token start (--keep 31 without --no-drop).

Policies (score = lower is evicted; ties by the lowest record id unless the policy folds its own tie-break in):
  decay:H       the engine's default: decayed use count, half-life H tokens, key reset on insert, recency tie-break
                (stage4_sizing.sim_ours); decayh:H keeps the count across evictions
  lru, lfu, lfuh (in-cache LFU / LFU with history; recency tie-break)
  arc, 2q, s3fifo   recency + frequency with ghost lists (own structures, pooled)
  hazard:H      Store.swift Evict(kind: .hazard): per-layer online table P(used at the next token | log2 age bucket,
                log2 decayed-count bucket, half-life H tokens), victim = lowest estimate for the next token
  learned       the engine's keep_score() with babel's q36_d_evict_policy.json, router features 0 (cache-only mode:
                router_probs() is only called with a router loaded); learned2 = q36_d_evict2_policy.json the same way
                (prefilter 128 by the decay key; not in the engine; log2 eps 1e-3 assumed)
  belady        offline optimum (stage4_sizing.sim_belady)
  la:K:BASE     Belady restricted to K layers of lookahead (exact future, upper bound): a record used within the next K
                steps is protected (nearer = more protected), a record of a layer that runs within K steps but is not
                used there is demoted, everything else is ordered by BASE
  nm:SRC:LAG:N:BASE  names as the only knowledge of the future, inside the token: at step (t, L) the names of
                (t, L+1 .. L+LAG) are known; named records are protected, held records of those layers that are not
                named are demoted. SRC = exact (recorded routing, upper bound) or r8 (the trained k2 router's top N,
                recorded only for the agent / code / prose teacher-forced texts)
Modes: cache only (default); named early (pf): the router's top N for (t, L+LAG) are read into the cache when layer L
runs and pinned until their step (Core's names: a name leaves 2K slots), misses of the released picks are read on the
spot; two tiers: a policy tier + an LRU tier (exclusive: catches what the first evicts; inclusive: the OS cache filled
by buffered reads of the first tier's misses).

  .venv/bin/python results/ios/evict/evict_sim.py <job> ...   (see main)
"""
import heapq, json, math, os, struct, sys, time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.environ.get("ENGINE_DIR", "engine")
PHONE = os.environ.get("PHONE_DIR", "phone-store")
REC_BYTES = 1769472
NL, NE, K = 40, 256, 8
NR = NL * NE
INF = np.inf
BIG = 1e12
SIZES = [0.8, 1.0, 1.27, 1.6, 2.0, 2.5, 3.3, 5.0]


def slots_of(gb):
    return 720 if gb == 1.27 else int(gb * 1e9 // REC_BYTES)  # 1.27 GB = the phone's 720 slots


# ------------------------------------------------------------------------------------------------------------ texts
def load_rec(p):
    raw = open(p, "rb").read()
    npos, nl, k = struct.unpack("<3i", raw[:12])
    return np.frombuffer(raw[12:12 + npos * nl * k * 2], dtype=np.int16).reshape(npos, nl, k).astype(np.int64)


def ids(p):
    return [int(x) for x in open(p).read().replace("\n", ",").split(",") if x.strip()]


TEXTS = {  # name: (kind, path, prompt positions not counted or None, names file)
    "pitt": ("bin", "results/stage2/runs/record_400.bin", "results/stage2/pitt_ids.txt", None),
    "coding": ("bin", "results/stage3/runs/record_code_400.bin", "results/stage3/code_ids.txt", None),
    "agent": ("bin", "results/stage3c/runs/record_tf_agent.bin", None, "results/stage3c/runs/c_r8_agent_gb6.6.names"),
    "code_text": ("bin", "results/stage3c/runs/record_tf_code.bin", None, "results/stage3c/runs/c_r8_code_gb6.6.names"),
    "prose_text": ("bin", "results/stage3c/runs/record_tf_prose.bin", None, "results/stage3c/runs/c_r8_prose_gb6.6.names"),
    "ph_chat": ("npz", "docs/evict_runs/chat.npz", None, None),
    "ph_wiki": ("npz", "docs/evict_runs/wiki.npz", None, None),
    "ph_code": ("npz", "docs/evict_runs/code.npz", None, None),
}


class Text:
    def __init__(self, name):
        kind, rel, prompt, names = TEXTS[name]
        self.name = name
        if kind == "bin":
            self.rec = load_rec(os.path.join(ENG, rel))
            self.start = len(ids(os.path.join(ENG, prompt))) if prompt else 0
            self.src = rel
        else:
            d = np.load(os.path.join(PHONE, rel))
            self.rec = d["rel"].astype(np.int64)
            self.start = 0  # teacher-forced / prompt-fed: every position counted, cold start counted (as stage4's TF texts)
            self.src = "phone-store/" + rel
        self.T = self.rec.shape[0]
        self.names = None
        if names:
            n = np.fromfile(os.path.join(ENG, names), dtype=np.int16).reshape(-1, NL, 16).astype(np.int64)
            assert n.shape[0] == self.T
            self.names = n
        lay = (np.arange(NL) * NE)[None, :, None]
        self.recs = (self.rec + lay).reshape(-1, K)  # [steps, 8] record ids
        self.steps = [row.tolist() for row in self.recs]
        self._nxt = None

    @property
    def nxt(self):  # next step at which each (step, pick) record is used again, 1e9 if never
        if self._nxt is None:
            S = len(self.steps)
            nx = np.full((S, K), 10 ** 9, dtype=np.int64)
            seen = {}
            for s in range(S - 1, -1, -1):
                row = self.steps[s]
                for j in range(K):
                    r = row[j]
                    nx[s, j] = seen.get(r, 10 ** 9)
                for j in range(K):
                    seen[row[j]] = s
            self._nxt = nx
            first = np.full(NR, 10 ** 9, dtype=np.int64)
            for r, s in seen.items():
                first[r] = s
            self.first_use = first
        return self._nxt

    def named(self, src, N):
        """[steps] -> list of record ids named for that step"""
        if src == "exact":
            return self.steps
        assert self.names is not None, f"{self.name}: no recorded router names"
        lay = (np.arange(NL) * NE)[None, :, None]
        return [row.tolist() for row in (self.names[:, :, :N] + lay).reshape(-1, N)]

    def in_named(self, N):
        """[steps, 8] bool: the released pick was in the router's top N"""
        if self.names is None:
            return None
        return (self.rec[:, :, :, None] == self.names[:, :, None, :N]).any(-1).reshape(-1, K)


# --------------------------------------------------------------------------------------------------------- policies
class Policy:
    """score-array policy over all 10,240 records; the core takes argmin over held, unpinned records"""
    dynamic_pick = False

    def reset(self, T): ...
    def token_start(self, t): ...
    def step_begin(self, s, t, L, picks): ...
    def step_end(self, s, t, L, picks): ...
    def on_insert(self, r, s, t, by_name=False): ...
    def on_evict(self, r): ...
    def vscore(self, s, t, L): return self.sc


class Decay(Policy):
    def __init__(self, hl=8.0, hist=False):
        self.hl, self.hist = float(hl), hist

    def reset(self, T):
        self.key = np.full(NR, -INF)
        self.last = np.full(NR, -1.0)
        self.sc = np.full(NR, -INF)

    def on_insert(self, r, s, t, by_name=False):
        if not self.hist:
            self.key[r] = -INF
            self.last[r] = -1.0
            self.sc[r] = -INF

    def step_end(self, s, t, L, picks):
        hl = self.hl
        key, last, sc = self.key, self.last, self.sc
        tt = t / hl
        for r in picks:
            k = key[r]
            v = 0.0 if k == -INF else 2.0 ** (k - tt)
            k = math.log2(v + 1.0) + tt
            key[r] = k
            last[r] = s
            sc[r] = k + s * 1e-12


class LRU(Policy):
    def reset(self, T):
        self.sc = np.zeros(NR)
        self.c = 0

    def on_insert(self, r, s, t, by_name=False):
        self.c += 1
        self.sc[r] = self.c

    def step_end(self, s, t, L, picks):
        for r in picks:
            self.c += 1
            self.sc[r] = self.c


class LRUTok(Policy):
    """LRU on the token clock, ties by admission order (Store.swift argminLast, evict_sim.py lru / lru_keep)"""
    def reset(self, T):
        self.lt = np.full(NR, -1.0)
        self.adm = np.zeros(NR)
        self.sc = np.zeros(NR)
        self.c = 0

    def on_insert(self, r, s, t, by_name=False):
        self.c += 1
        self.adm[r] = self.c
        self.lt[r] = t
        self.sc[r] = t * 1e7 + self.c

    def step_end(self, s, t, L, picks):
        for r in picks:
            self.lt[r] = t
            self.sc[r] = t * 1e7 + self.adm[r]


class LFU(Policy):
    def __init__(self, hist=False):
        self.hist = hist

    def reset(self, T):
        self.n = np.zeros(NR)
        self.sc = np.zeros(NR)
        self.c = 0

    def on_insert(self, r, s, t, by_name=False):
        if not self.hist:
            self.n[r] = 0
        self.c += 1
        self.sc[r] = self.n[r] + self.c * 1e-9

    def step_end(self, s, t, L, picks):
        for r in picks:
            self.c += 1
            self.n[r] += 1
            self.sc[r] = self.n[r] + self.c * 1e-9


class Hazard(Policy):
    """Store.swift Evict(kind: .hazard) (= evict_sim.py hazard_h128), the tables per layer, the clock per token"""
    NA, NF = 13, 10

    def __init__(self, h=128.0, at_end=False):
        self.fac = 2.0 ** (-1.0 / h)
        self.at_end = at_end  # touch where the engine's Cache::use() runs (end of the step) instead of before the misses

    def reset(self, T):
        nb = self.NA * self.NF
        self.last = np.full(NR, -1.0)
        self.dec = np.zeros(NR)
        self.hzU = np.ones((NL, nb))
        self.hzN = np.full((NL, nb), 2.0)
        self.B = np.zeros(NR, dtype=np.int64)
        self.bOK = False
        self.tl = None
        self.cur = np.zeros(NR, bool)
        self.sc = np.zeros(NR)
        self.off = (np.arange(NL) * nb).repeat(NE)

    def bins(self, last, dec, t):
        age = np.where(last >= 0, t - last, float(1 << 14))
        ab = np.minimum(np.floor(np.log2(np.maximum(age, 1.0))).astype(np.int64), self.NA - 1)
        fb = np.minimum(np.log2(1 + 4 * dec).astype(np.int64), self.NF - 1)
        return ab * self.NF + fb

    def keys(self, a, b, t):
        kb = self.bins(self.last[a:b], self.dec[a:b], t + 1)
        L = a // NE
        self.sc[a:b] = self.hzU[L, kb] / self.hzN[L, kb] + self.last[a:b] * 1e-9

    def token_start(self, t):
        nb = self.NA * self.NF
        if self.tl is not None:
            self.B = self.bins(self.last, self.dec, t)
            self.hzN += np.bincount(self.B + self.off, minlength=NL * nb).reshape(NL, nb)
            self.bOK = True
        g = 1 if self.tl is None else t - self.tl
        self.dec *= self.fac ** g
        self.cur[:] = False
        self.tl = t
        kb = self.bins(self.last, self.dec, t + 1)
        self.sc = self.hzU.ravel()[kb + self.off] / self.hzN.ravel()[kb + self.off] + self.last * 1e-9

    def step_begin(self, s, t, L, picks):
        if not self.at_end:
            self.touch(t, L, picks)

    def step_end(self, s, t, L, picks):
        if self.at_end:
            self.touch(t, L, picks)

    def touch(self, t, L, picks):  # Store.swift Evict.touch, then layer L's keys
        for r in picks:
            if self.cur[r]:
                continue
            self.cur[r] = True
            self.last[r] = t
            self.dec[r] += 1
            if self.bOK:
                self.hzU[L, self.B[r]] += 1
        self.keys(L * NE, (L + 1) * NE, t)


class Learned(Policy):
    """dmeng Cache::keep_score with load_policy(q36_d_evict_policy.json), router features 0 (no router loaded)"""
    HL = (2.0, 8.0, 32.0)

    def __init__(self, path):
        js = json.load(open(path))
        self.w = np.array(js["w"], dtype=np.float64)
        self.b = float(js["b"])
        assert len(self.w) == 7 + NL

    def reset(self, T):
        self.lc = [np.zeros(NR) for _ in self.HL]
        self.lc_t = np.zeros(NR)
        self.l_last = np.full(NR, -1e9)
        self.lb = np.repeat(self.w[7:], NE)
        self.sc = np.zeros(NR)
        self.t = 0

    def score(self, idx, t):
        v = self.b + self.lb[idx]
        for h, hl in enumerate(self.HL):
            v = v + self.w[h] * np.log2(self.lc[h][idx] * np.exp2(-(t - self.lc_t[idx]) / hl) + 1e-3)
        v = v + self.w[3] * np.log2(1.0 + np.maximum(0.0, t - self.l_last[idx]))
        return v

    def token_start(self, t):
        self.t = t
        self.sc = self.score(slice(None), float(t))

    def step_end(self, s, t, L, picks):
        p = np.array(picks)
        for h, hl in enumerate(self.HL):
            self.lc[h][p] = self.lc[h][p] * np.exp2(-(t - self.lc_t[p]) / hl) + 1.0
        self.lc_t[p] = t
        self.l_last[p] = t
        self.sc[p] = self.score(p, float(t))


class Learned2(Policy):
    """q36_d_evict2_policy.json (lin): prefilter = the 128 unpinned slots with the lowest decay key, evict the lowest
    w . f + layer_bias; router features 0, log2 eps 1e-3 (assumed); not implemented in the engine"""
    HL = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    dynamic_pick = True

    def __init__(self, path):
        js = json.load(open(path))
        self.w = np.array(js["w"], dtype=np.float64)
        self.lb = np.repeat(np.array(js["layer_bias"]), NE)
        self.pre = int(js["prefilter"])
        assert js["features"][:9] == ["log2_c1", "log2_c2", "log2_c4", "log2_c8", "log2_c16", "log2_c32", "log2_c64",
                                      "log2_1p_since_use", "log2_1p_since_use2"] and js["features"][16] == "log2_1p_pos"
        self.dk = Decay(8.0)

    def reset(self, T):
        self.dk.reset(T)
        self.c = np.zeros((len(self.HL), NR))
        self.ct = np.zeros(NR)
        self.l1 = np.full(NR, -1e9)
        self.l2 = np.full(NR, -1e9)
        self.sc = self.dk.sc

    def on_insert(self, r, s, t, by_name=False):
        self.dk.on_insert(r, s, t, by_name)

    def step_end(self, s, t, L, picks):
        self.dk.step_end(s, t, L, picks)
        p = np.array(picks)
        hl = np.array(self.HL)[:, None]
        self.c[:, p] = self.c[:, p] * np.exp2(-(t - self.ct[p])[None, :] / hl) + 1.0
        self.ct[p] = t
        self.l2[p] = self.l1[p]
        self.l1[p] = t

    def pick(self, cand_idx, s, t, L):
        ks = self.dk.sc[cand_idx]
        if len(cand_idx) > self.pre:
            sel = np.argpartition(ks, self.pre - 1)[:self.pre]
            idx = cand_idx[sel]
        else:
            idx = cand_idx
        hl = np.array(self.HL)[:, None]
        c = self.c[:, idx] * np.exp2(-(t - self.ct[idx])[None, :] / hl)
        f = np.log2(c + 1e-3)
        v = self.lb[idx] + (self.w[:7, None] * f).sum(0)
        v += self.w[7] * np.log2(1 + np.minimum(1e6, t - self.l1[idx]))
        v += self.w[8] * np.log2(1 + np.minimum(1e6, t - self.l2[idx]))
        v += self.w[16] * math.log2(1 + t)
        return int(idx[int(np.argmin(v))])


class Belady(Policy):
    def reset(self, T):
        T.nxt
        self.nu = T.first_use.astype(np.float64).copy()
        self.nx = T.nxt
        self.sc = -self.nu

    def step_end(self, s, t, L, picks):
        nx = self.nx[s]
        for j, r in enumerate(picks):
            self.nu[r] = nx[j]
            self.sc[r] = -nx[j]


class Lookahead(Policy):
    """la: Belady restricted to k steps (exact); nm: names inside the token (exact or router), lag steps"""

    def __init__(self, base, k=None, names=None, lag=None):
        self.base, self.k, self.nmode, self.lag = base, k, names, lag
        self.dynamic_pick = getattr(base, "dynamic_pick", False)

    def reset(self, T):
        self.base.reset(T)
        T.nxt
        self.nu = T.first_use.astype(np.float64).copy()
        self.nx = T.nxt
        if self.nmode is not None:
            self.nlist = T.named(*self.nmode)
        if self.k is not None:
            d = (np.arange(NL)[None, :] - np.arange(NL)[:, None]) % NL
            d[d == 0] = NL
            self.inwin = np.repeat(d <= self.k, NE, axis=1)  # [L now, record]: its layer runs within k steps
        self._s = -1

    def token_start(self, t): self.base.token_start(t)
    def step_begin(self, s, t, L, picks): self.base.step_begin(s, t, L, picks)
    def on_insert(self, r, s, t, by_name=False): self.base.on_insert(r, s, t, by_name)

    def step_end(self, s, t, L, picks):
        self.base.step_end(s, t, L, picks)
        nx = self.nx[s]
        for j, r in enumerate(picks):
            self.nu[r] = nx[j]

    def offset(self, s, t, L):
        if self._s == s:
            return self._off
        if self.k is not None:
            dist = self.nu - s
            prot = dist <= self.k
            off = np.where(prot, BIG * (2.0 - dist / (self.k + 1.0)), np.where(self.inwin[L], -BIG, 0.0))
        else:
            off = np.zeros(NR)
            hi = min(L + self.lag, NL - 1)
            if hi > L:
                off[(L + 1) * NE:(hi + 1) * NE] = -BIG
                for s2 in range(s + 1, s + 1 + hi - L):
                    off[self.nlist[s2]] = BIG
        self._s, self._off = s, off
        return off

    def vscore(self, s, t, L):  # the offset only changes between steps (nu moves at a step's end)
        return self.base.vscore(s, t, L) + self.offset(s, t, L)

    def pick(self, cand_idx, s, t, L):  # dynamic base (learned2): restrict to the least protected class first
        off = self.offset(s, t, L)[cand_idx]
        m = off == off.min()
        return self.base.pick(cand_idx[m], s, t, L)


def make_policy(spec):
    n, _, a = spec.partition(":")
    if n == "decay":
        return Decay(float(a or 8))
    if n == "decayh":
        return Decay(float(a or 8), hist=True)
    if n == "lru":
        return LRU()
    if n == "lrut":
        return LRUTok()
    if n == "lfu":
        return LFU()
    if n == "lfuh":
        return LFU(hist=True)
    if n == "hazard":
        return Hazard(float(a or 128))
    if n == "hazarde":
        return Hazard(float(a or 128), at_end=True)
    if n == "learned":
        return Learned(os.path.join(HERE, "babel", "q36_d_evict_policy.json"))
    if n == "learned2":
        return Learned2(os.path.join(HERE, "babel", "q36_d_evict2_policy.json"))
    if n == "belady":
        return Belady()
    if n == "la":  # la:K:BASE...
        k, _, base = a.partition(":")
        return Lookahead(make_policy(base or "decay:8"), k=int(k))
    if n == "nm":  # nm:SRC:LAG:N:BASE...
        src, lag, N, base = a.split(":", 3)
        return Lookahead(make_policy(base), names=(src, int(N)), lag=int(lag))
    raise ValueError(spec)


# ------------------------------------------------------------------------------------------------------------ core
def layer_caps(total, shape):
    if shape == "uniform":
        c = [total // NL] * NL
        for i in range(total - sum(c)):
            c[i] += 1
        return c
    if shape == "table":  # phone-store LayerCapsTable (auto:TOTAL), interpolated between rows to the exact total
        rows = caps_table()
        lo = max(0, min((total - 400) // 40, len(rows) - 1))
        a = list(rows[lo])
        b = rows[min(lo + 1, len(rows) - 1)]
        rem = total - sum(a)
        i = 0
        while rem > 0 and i < 4 * NL:
            L = i % NL
            if i >= NL or b[L] > a[L]:
                a[L] += 1
                rem -= 1
            i += 1
        assert sum(a) == total, (total, sum(a))
        return a
    raise ValueError(shape)


_CT = None


def caps_table():
    global _CT
    if _CT is None:
        src = open(os.path.join(PHONE, "native/Sources/PhoneStore/LayerCapsTable.swift")).read()
        body = src.split("rows: [[Int]] = [", 1)[1].split("\n    ]", 1)[0]
        _CT = [[int(x) for x in row.split(",")] for row in body.split("[")[1:] if row.split("]")[0].strip()
               for row in [row.split("]")[0]]]
    return _CT


def simulate(T, slots, spec, sem="engine", caps=None, keep=None, pf=None, tier2=None, excl64=False, log=None):
    """Returns a dict of counts. pf = (src, N, lag) named-early mode. tier2 = (slots2, 'excl'|'incl')."""
    pol = make_policy(spec) if isinstance(spec, str) else spec
    pol.reset(T)
    held = np.zeros(NR, bool)
    avail = np.zeros(NR, bool)
    pins = np.zeros(NR, np.int32)
    ltok = np.full(NR, -1, np.int64)
    pf_unused = np.zeros(NR, bool)
    cnt = [0] * NL
    nheld = 0
    capL = caps
    start = 64 if excl64 else T.start
    st = dict(picks=0, hits=0, misses=0, early=0, wasted=0, dropped_names=0, t2_hits=0, drive=0)
    in8 = T.in_named(8)
    in12 = T.in_named(12)
    m8 = m12 = 0
    steps = T.steps
    nlist = T.named(pf[0], pf[1]) if pf else None
    lag = pf[2] if pf else 0
    t2 = OrderedDict() if tier2 else None
    t2cap = tier2[0] if tier2 else 0
    t2mode = tier2[1] if tier2 else None
    dyn = getattr(pol, "dynamic_pick", False)

    def victim(s, t, L, reserve=0):
        if capL is not None:
            a, b = L * NE, (L + 1) * NE
            m = avail[a:b]
            if dyn:
                idx = np.flatnonzero(m) + a
                if len(idx) <= reserve:
                    return -1
                return pol.pick(idx, s, t, L)
            c = np.where(m, pol.vscore(s, t, L)[a:b], INF)
            v = int(c.argmin())
            if c[v] == INF:
                return -1
            return v + a
        if reserve and int(avail.sum()) <= reserve:
            return -1
        if dyn:
            idx = np.flatnonzero(avail)
            return pol.pick(idx, s, t, L) if len(idx) else -1
        c = np.where(avail, pol.vscore(s, t, L), INF)
        v = int(c.argmin())
        return -1 if c[v] == INF else v

    def evict(v, counted):
        nonlocal nheld
        held[v] = False
        avail[v] = False
        cnt[v // NE] -= 1
        nheld -= 1
        pol.on_evict(v)
        if pf_unused[v]:
            if counted:
                st["wasted"] += 1
            pf_unused[v] = False
        if t2 is not None and t2mode == "excl":
            t2[v] = 1
            t2.move_to_end(v)
            if len(t2) > t2cap:
                t2.popitem(last=False)

    def full(L):
        return cnt[L] >= capL[L] if capL is not None else nheld >= slots

    def insert(r, s, t, by_name):
        nonlocal nheld
        held[r] = True
        avail[r] = pins[r] == 0
        cnt[r // NE] += 1
        nheld += 1
        pol.on_insert(r, s, t, by_name)

    def read(r, counted):  # a record that must come in: from the second tier or the drive
        if t2 is not None:
            if r in t2:
                if t2mode == "excl":
                    del t2[r]
                else:
                    t2.move_to_end(r)
                if counted:
                    st["t2_hits"] += 1
                return
            if t2mode == "incl":
                t2[r] = 1
                if len(t2) > t2cap:
                    t2.popitem(last=False)
        if counted:
            st["drive"] += 1

    def name_step(s2, s, t, L, counted):
        for r in nlist[s2]:
            pins[r] += 1
            if held[r]:
                avail[r] = False
                continue
            if full(r // NE):
                v = victim(s, t, r // NE if capL is not None else L, reserve=2 * K)
                if v < 0:
                    pins[r] -= 1
                    st["dropped_names"] += counted
                    continue
                evict(v, counted)
            insert(r, s, t, True)
            avail[r] = False
            pf_unused[r] = True
            read(r, counted)
            if counted:
                st["early"] += 1

    def unname_step(s2):
        for r in nlist[s2]:
            if pins[r] > 0:
                pins[r] -= 1
                if pins[r] == 0 and held[r]:
                    avail[r] = True

    named_at = {}  # step -> issued (for unpinning)
    for t in range(T.T):
        counted = t >= start
        pol.token_start(t)
        if keep is not None:  # the keep-window drop (Store.swift drop(), the MLX reference implementation's bookkeeping)
            d = np.flatnonzero(held & (ltok < t - keep) & (pins == 0))
            for v in d.tolist():
                held[v] = False
                avail[v] = False
                cnt[v // NE] -= 1
                nheld -= 1
                pol.on_evict(v)
        if pf:
            for L2 in range(min(lag, NL)):
                s2 = t * NL + L2
                name_step(s2, s2, t, L2, counted)
                named_at[s2] = True
        for L in range(NL):
            s = t * NL + L
            picks = steps[s]
            pol.step_begin(s, t, L, picks)
            if sem == "phone":
                for r in picks:
                    if held[r]:
                        avail[r] = False
            for j, r in enumerate(picks):
                if counted:
                    st["picks"] += 1
                if held[r]:
                    if counted:
                        st["hits"] += 1
                    avail[r] = False
                    pf_unused[r] = False
                    continue
                if log is not None:
                    log["miss"] += 1
                if counted:
                    st["misses"] += 1
                    if in8 is not None:
                        m8 += bool(in8[s, j])
                        m12 += bool(in12[s, j])
                if full(L):
                    v = victim(s, t, L)
                    if v < 0:
                        raise RuntimeError(f"no slot: {spec} slots {slots} step {s}")
                    if log is not None:
                        log["ev"].append((s, v))
                    evict(v, counted)
                insert(r, s, t, False)
                avail[r] = False
                read(r, counted)
            if pf and L + lag < NL:  # Core: after_router names a later step, behind this step's misses
                name_step(s + lag, s, t, L, counted)
                named_at[s + lag] = True
            pol.step_end(s, t, L, picks)
            for r in picks:
                ltok[r] = t
                pf_unused[r] = False
                if pins[r] == 0 and held[r]:
                    avail[r] = True
            if pf and named_at.pop(s, False):
                unname_step(s)
    ntok = T.T - start
    out = {"tokens": ntok, "hit": st["hits"] / st["picks"],
           "misses_per_token": st["misses"] / ntok,
           "mb_per_token": st["misses"] / ntok * REC_BYTES / 1e6}
    if in8 is not None and st["misses"]:
        out["miss_in_named8"] = m8 / st["misses"]
        out["miss_in_named12"] = m12 / st["misses"]
    if pf:
        out["early_per_token"] = st["early"] / ntok
        out["spot_per_token"] = st["misses"] / ntok
        out["wasted_per_token"] = st["wasted"] / ntok
        out["dropped_names_per_token"] = st["dropped_names"] / ntok
        out["mb_per_token"] = (st["early"] + st["misses"]) / ntok * REC_BYTES / 1e6
        out["mb_early"] = st["early"] / ntok * REC_BYTES / 1e6
        out["mb_spot"] = st["misses"] / ntok * REC_BYTES / 1e6
    if tier2:
        out["drive_mb_per_token"] = st["drive"] / ntok * REC_BYTES / 1e6
        out["tier2_hits_per_token"] = st["t2_hits"] / ntok
    return out


# ------------------------------------------------------------------------------------------------ ARC / 2Q / S3-FIFO
def simulate_dict(T, slots, algo, tier2=None):
    """recency + frequency policies with their own lists; pooled; pins = the step's picks taken so far"""
    start = T.start
    hits = picks_n = misses = drive = t2h = 0
    c = slots
    t2 = OrderedDict() if tier2 else None
    t2cap = tier2[0] if tier2 else 0

    def spill(v):
        if t2 is not None:
            t2[v] = 1
            t2.move_to_end(v)
            if len(t2) > t2cap:
                t2.popitem(last=False)

    def fetch(r, counted):
        nonlocal drive, t2h
        if t2 is not None and r in t2:
            del t2[r]
            t2h += counted
            return
        drive += counted

    def pop_lru(od, pinned):  # least recent unpinned
        for k in od:
            if k not in pinned:
                del od[k]
                return k
        return None

    if algo == "arc":
        T1, T2, B1, B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        p = 0.0

        def replace(x_in_b2, pinned):
            if T1 and (len(T1) > p or (x_in_b2 and len(T1) == p)):
                v = pop_lru(T1, pinned)
                if v is not None:
                    B1[v] = 1
                    return v
                v = pop_lru(T2, pinned)
                B2[v] = 1
                return v
            v = pop_lru(T2, pinned)
            if v is not None:
                B2[v] = 1
                return v
            v = pop_lru(T1, pinned)
            B1[v] = 1
            return v

        for s, row in enumerate(T.steps):
            counted = s // NL >= start
            pinned = set()
            for x in row:
                picks_n += counted
                if x in T1:
                    del T1[x]
                    T2[x] = 1
                    hits += counted
                elif x in T2:
                    T2.move_to_end(x)
                    hits += counted
                else:
                    misses += counted
                    if x in B1:
                        p = min(c, p + max(len(B2) / len(B1), 1.0))
                        spill(replace(False, pinned))
                        del B1[x]
                        T2[x] = 1
                    elif x in B2:
                        p = max(0.0, p - max(len(B1) / len(B2), 1.0))
                        spill(replace(True, pinned))
                        del B2[x]
                        T2[x] = 1
                    else:
                        if len(T1) + len(B1) == c:
                            if len(T1) < c:
                                B1.popitem(last=False)
                                spill(replace(False, pinned))
                            else:
                                v = pop_lru(T1, pinned)
                                if v is None:
                                    v = pop_lru(T2, pinned)
                                spill(v)
                        elif len(T1) + len(B1) < c and len(T1) + len(T2) + len(B1) + len(B2) >= c:
                            if len(T1) + len(T2) + len(B1) + len(B2) >= 2 * c:
                                if B2:
                                    B2.popitem(last=False)
                                else:
                                    B1.popitem(last=False)
                            if len(T1) + len(T2) >= c:
                                spill(replace(False, pinned))
                        T1[x] = 1
                    fetch(x, counted)
                pinned.add(x)
                assert len(T1) + len(T2) <= c
    elif algo == "2q":
        kin, kout = max(1, c // 4), max(1, c // 2)
        A1in, A1out, Am = OrderedDict(), OrderedDict(), OrderedDict()

        def reclaim(pinned):
            if len(A1in) + len(Am) < c:
                return
            if len(A1in) > kin:
                v = pop_lru(A1in, pinned)
                if v is not None:
                    A1out[v] = 1
                    if len(A1out) > kout:
                        A1out.popitem(last=False)
                    spill(v)
                    return
            v = pop_lru(Am, pinned)
            if v is None:
                v = pop_lru(A1in, pinned)
                A1out[v] = 1
                if len(A1out) > kout:
                    A1out.popitem(last=False)
            spill(v)

        for s, row in enumerate(T.steps):
            counted = s // NL >= start
            pinned = set()
            for x in row:
                picks_n += counted
                if x in Am:
                    Am.move_to_end(x)
                    hits += counted
                elif x in A1in:
                    hits += counted
                else:
                    misses += counted
                    reclaim(pinned)
                    if x in A1out:
                        del A1out[x]
                        Am[x] = 1
                    else:
                        A1in[x] = 1
                    fetch(x, counted)
                pinned.add(x)
    elif algo == "s3fifo":  # Yang et al. 2023: small FIFO 10 %, main FIFO with 2-bit counts, ghost of main's size
        ssz = max(1, c // 10)
        S, M, G = OrderedDict(), OrderedDict(), OrderedDict()  # first = oldest; value = freq
        gcap = c - ssz

        def make_room(pinned):
            while True:
                if len(S) >= ssz and any(q not in pinned for q in S):
                    k = next(q for q in S if q not in pinned)
                    f = S.pop(k)
                    if f > 1:
                        M[k] = 0
                        continue
                    G[k] = 1
                    if len(G) > gcap:
                        G.popitem(last=False)
                    return k
                k = next(iter(M))
                f = M.pop(k)
                if k in pinned:
                    M[k] = f
                    continue
                if f > 0:
                    M[k] = f - 1
                    continue
                return k

        for s, row in enumerate(T.steps):
            counted = s // NL >= start
            pinned = set()
            for x in row:
                picks_n += counted
                if x in S:
                    S[x] = min(3, S[x] + 1)
                    hits += counted
                elif x in M:
                    M[x] = min(3, M[x] + 1)
                    hits += counted
                else:
                    misses += counted
                    if len(S) + len(M) >= c:
                        spill(make_room(pinned))
                    if x in G:
                        del G[x]
                        M[x] = 0
                    else:
                        S[x] = 0
                    fetch(x, counted)
                pinned.add(x)
    else:
        raise ValueError(algo)
    ntok = T.T - start
    out = {"tokens": ntok, "hit": hits / picks_n, "misses_per_token": misses / ntok,
           "mb_per_token": misses / ntok * REC_BYTES / 1e6}
    if tier2:
        out["drive_mb_per_token"] = drive / ntok * REC_BYTES / 1e6
        out["tier2_hits_per_token"] = t2h / ntok
    return out


def run_cell(T, gb, spec, **kw):
    slots = kw.pop("slots", None) or slots_of(gb)
    if spec in ("arc", "2q", "s3fifo"):
        return simulate_dict(T, slots, spec, tier2=kw.get("tier2"))
    shape = kw.pop("caps_shape", None)
    caps = layer_caps(slots, shape) if shape else None
    return simulate(T, slots, spec, caps=caps, **kw)


# ------------------------------------------------------------------------------------------------------------ jobs
def rnd(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def job(name, texts, sizes, cells, out_path):
    """cells: list of (label, spec, kwargs)"""
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for tn in texts:
        T = Text(tn)
        res.setdefault(tn, {})
        for gb in sizes:
            res[tn].setdefault(str(gb), {})
            for label, spec, kw in cells:
                if label in res[tn][str(gb)]:
                    continue
                if (kw.get("pf") or "nm:r8" in spec) and T.names is None:
                    continue
                t0 = time.time()
                try:
                    r = run_cell(T, gb, spec, **dict(kw))
                except AssertionError as e:
                    print("skip", tn, gb, label, e, flush=True)
                    continue
                res[tn][str(gb)][label] = rnd(r)
                print(f"{name} {tn} {gb} {label} hit {r['hit']:.4f} MB/tok {r['mb_per_token']:.1f}"
                      + (f" drive {r['drive_mb_per_token']:.1f}" if "drive_mb_per_token" in r else "")
                      + f" ({time.time() - t0:.0f}s)", flush=True)
                json.dump(res, open(out_path, "w"), indent=1)
    json.dump(res, open(out_path, "w"), indent=1)


MAIN3 = ["pitt", "coding", "agent"]
OTHER = ["code_text", "prose_text", "ph_chat", "ph_wiki", "ph_code"]


def cells_for(group):
    E = {}
    if group == "base":
        c = [("decay:8", "decay:8", E), ("lru", "lru", E), ("lfu", "lfu", E), ("lfuh", "lfuh", E),
             ("decayh:8", "decayh:8", E), ("hazard", "hazard:128", E), ("hazard_drop31", "hazard:128", {"keep": 31}),
             ("learned", "learned", E), ("learned2", "learned2", E), ("arc", "arc", E), ("2q", "2q", E),
             ("s3fifo", "s3fifo", E), ("belady", "belady", E)]
        return c
    if group == "hl":
        return [(f"decay:{h}", f"decay:{h}", E) for h in (1, 2, 4, 16, 32, 64)] + \
               [(f"decayh:{h}", f"decayh:{h}", E) for h in (2, 32, 128)]
    if group == "caps":
        c = []
        for p, sp in (("decay:8", "decay:8"), ("lru", "lru"), ("hazard", "hazard:128")):
            for shape in ("uniform", "table"):
                c.append((f"{p}|caps_{shape}", sp, {"caps_shape": shape}))
        # the phone engine as implemented: per-layer caps (table), touch first, hazard, with / without the keep window
        c.append(("phone_hazard_nodrop", "hazard:128", {"caps_shape": "table", "sem": "phone"}))
        c.append(("phone_hazard_keep31", "hazard:128", {"caps_shape": "table", "sem": "phone", "keep": 31}))
        c.append(("phone_lru_keep31", "lrut", {"caps_shape": "table", "sem": "phone", "keep": 31}))
        return c
    if group == "look":
        c = []
        for k in (2, 4, 8, 40, 80, 160):
            c.append((f"la:{k}|decay", f"la:{k}:decay:8", E))
        for k in (2, 4, 40):
            c.append((f"la:{k}|lru", f"la:{k}:lru", E))
        for lag in (2, 4):
            c.append((f"nm:exact:{lag}|decay", f"nm:exact:{lag}:8:decay:8", E))
        for N in (8, 12, 16):
            c.append((f"nm:r8:2:{N}|decay", f"nm:r8:2:{N}:decay:8", E))
        return c
    if group == "look2":  # the same on the best online base
        return [("la:2|hazard", "la:2:hazard:128", E), ("la:40|hazard", "la:40:hazard:128", E),
                ("nm:exact:2|hazard", "nm:exact:2:8:hazard:128", E), ("nm:r8:2:12|hazard", "nm:r8:2:12:hazard:128", E),
                ("nm:exact:2|lru", "nm:exact:2:8:lru", E), ("nm:r8:2:12|lru", "nm:r8:2:12:lru", E)]
    if group == "pf":  # named early: the router's top 8 / 12 read two layers ahead (Core's lossless router mode)
        c = []
        for p, sp in (("decay:8", "decay:8"), ("lru", "lru"), ("hazard", "hazard:128")):
            for N in (8, 12):
                c.append((f"pf_r8_{N}|{p}", sp, {"pf": ("r8", N, 2)}))
            c.append((f"pf_exact_8|{p}", sp, {"pf": ("exact", 8, 2)}))
        return c
    if group == "tier":
        c = []
        for p, sp in (("decay:8", "decay:8"), ("lru", "lru"), ("hazard", "hazard:128"), ("arc", "arc"),
                      ("belady", "belady")):
            c.append((f"A_excl|{p}", sp, {"tier2": (1186, "excl")}))
            if sp not in ("arc",):
                c.append((f"A_incl|{p}", sp, {"tier2": (1186, "incl")}))
        return c
    raise ValueError(group)


def validate(out_path):
    """stage4_sizing's numbers (decay half-life 8 and Belady at 3.3 / 5 / 6.6 GB, pitt and coding), P0's 1.27 GB rows,
    and phone-store evict_sim.py (per-layer cap 48, excl. first 64 tokens, MiB) for the hazard / keep-window port"""
    s4 = json.load(open(os.path.join(ENG, "results/stage4/sizing.json")))["belady"]
    out = {"stage4": {}, "phone_evict_sim": {}}
    for tn in ("pitt", "coding"):
        T = Text(tn)
        for gb in ("3.3", "5.0", "6.6"):
            d = simulate(T, slots_of(float(gb)), "decay:8")["hit"]
            b = simulate(T, slots_of(float(gb)), "belady")["hit"]
            ref = s4[tn][gb]
            out["stage4"][f"{tn} {gb}"] = {"decay": round(d, 4), "stage4_ours": ref["ours_sim"],
                                           "belady": round(b, 4), "stage4_belady": ref["belady"]}
            print(tn, gb, out["stage4"][f"{tn} {gb}"], flush=True)
    ev = json.load(open(os.path.join(PHONE, "docs/evict_runs/evict_results.json")))
    for tn, key in (("ph_chat", "chat"), ("ph_wiki", "wiki"), ("ph_code", "code")):
        T = Text(tn)
        for spec, kw, ref in (("hazard:128", {}, "hazard_h128"), ("lrut", {"keep": 31}, "lru_keep"), ("lrut", {}, "lru")):
            r = simulate(T, 48 * NL, spec, caps=[48] * NL, sem="phone", excl64=True, **kw)
            mib = r["misses_per_token"] * REC_BYTES / 2 ** 20
            out["phone_evict_sim"][f"{tn} cap48 {ref}"] = {"ours_MiB": round(mib, 2), "evict_sim": find_ref(ev, key, ref, 48)}
            print(tn, ref, out["phone_evict_sim"][f"{tn} cap48 {ref}"], flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


def find_ref(ev, key, pol, cap):
    return round(ev[key]["rows"][pol]["warm"][ev["_caps"].index(cap)], 2)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "validate":
        validate(os.path.join(HERE, "validate.json"))
    elif what == "log":  # log TEXT SLOTS SPEC OUT [uniform]: every eviction as "step record" (tests/evict_check's format)
        T = Text(sys.argv[2])
        lg = {"ev": [], "miss": 0}
        caps = layer_caps(int(sys.argv[3]), "uniform") if len(sys.argv) > 6 and sys.argv[6] == "uniform" else None
        simulate(T, int(sys.argv[3]), sys.argv[4], caps=caps, log=lg)
        with open(sys.argv[5], "w") as fo:
            for s_, v_ in lg["ev"]:
                fo.write(f"{s_} {v_}\n")
            fo.write(f"misses {lg['miss']} picks {T.T * NL * K}\n")
        print(sys.argv[2], sys.argv[4], sys.argv[3], "misses", lg["miss"])
    else:  # job group texts out
        grp = what
        texts = sys.argv[2].split(",") if len(sys.argv) > 2 else MAIN3
        sizes = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else SIZES
        out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, f"res_{grp}.json")
        job(grp, texts, sizes, cells_for(grp), out)
