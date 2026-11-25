#!/usr/bin/env python3
"""Bytes read per token for Flash-Next from one recorded exact-mode run, offline (audit tool, no model).

Input: the files of one `dmeng --mode exact --record R --evict-trace T` run (prefill as single steps, so every position
has its routing): T (DMEVTR1) holds per (position, step) the released router's top 32 (ids, softmax probabilities; the
first 10 are the picks) and the early router's top 32 (lag 4, the shipped mlp_k4). R is only used to check T.

The simulator replays the positions in order. Per token, per layer step s:
  1. names for step s are issued at the start of step max(0, s - 4) (the lag-4 routers; steps 0-3 at the token's start):
     a named record not in memory is read (one record) into a slot; it is pinned until step s has run;
  2. the step's 10 picks: a pick not in memory is read (exact mode), or (fast mode) the heaviest missing picks are read
     until the missing weight left is at most MASS (any missing pick over W is read): the rest are left out (no read);
  3. a slot for a read evicts the lowest keep-score among unpinned held records (one pool, no per-layer caps).
Names always land in time (the bytes view: lateness costs time, not bytes).
Policies: hazard (the engine's default keep-score: an online per-layer table of P(use next token | age bucket, decayed
count bucket), half-life 8, as Cache::keep_score), lru, freq (decayed count, half-life 8), belady (evict the farthest next
use; demand only: the bound for any names).
Names: none | router:M (the trained router's top M) | recall:R:M (a router whose top M holds a share R of the picks, the
rest of its M filled by the trained router's wrong names) | perfect (names = picks).
Counts only the decode positions (from --first, default the prompt length); earlier positions warm the cache.

usage: trace_sim.py TRACE [--first P] [--grid] [--reuse] [--json OUT]
"""
import argparse, json, math, sys
from collections import OrderedDict

import numpy as np

REC = 2768896
NL, NE, K, LAG = 48, 512, 10, 4
HZ_NA, HZ_NF = 13, 10


def load(path):
    b = open(path, "rb").read()
    assert b[:7] == b"DMEVTR1", "not a DMEVTR1 trace"
    h = np.frombuffer(b, np.int32, 7, 8)
    npos, steps, ne, W, nl, k, npools = (int(x) for x in h)
    o = 8 + 28 + 4 * steps + 4 * nl + 16 * npools
    n = npos * steps * W
    ri = np.frombuffer(b, np.int16, n, o).reshape(npos, steps, W).astype(np.int32); o += 2 * n
    rp = np.frombuffer(b, np.float32, n, o).reshape(npos, steps, W); o += 4 * n
    ti = np.frombuffer(b, np.int16, n, o).reshape(npos, steps, W).astype(np.int32); o += 2 * n
    tp = np.frombuffer(b, np.float32, n, o).reshape(npos, steps, W)
    return ri, rp, ti, tp


class Sim:
    def __init__(self, ri, rp, ti, slots, policy, names, mode="exact", w=0.13, mass=0.13, first=0, seed=0, caps=False,
                 name_touch=False, use_touch=True):
        # caps: the slots split evenly over the layers (a layer evicts only its own); name_touch: a name counts as a use in
        # the keep-score's statistics (the near-future evidence a router gives); use_touch = False: picks do not (the old
        # runner); mode prefetch: picks never read (the released top-k among the held: hits only)
        self.caps, self.name_touch, self.use_touch = caps, name_touch, use_touch
        self.lheld = np.zeros(NL, np.int64)
        self.cur_L = 0
        self.hits = 0
        self.ri, self.rp, self.ti = ri, rp, ti
        self.npos = ri.shape[0]
        self.C, self.policy, self.names, self.mode, self.W, self.MASS, self.first = slots, policy, names, mode, w, mass, first
        self.rng = np.random.default_rng(seed)
        self.look = None  # look:H:R = hazard + the next H tokens' uses known for a share R of them (a cross-token predictor)
        self.hl = 8.0
        # named:D[:X] = hazard, with a floor for what the routers have named for the coming steps: a record among the lag
        # router's top D for a step s+1 .. s+4 (steps 0-3 from the token's start) keeps at least P(a pick | its rank) (this
        # trace's calibration); X = 1 adds the next token's steps 0-3 (an upper bound: that router does not exist)
        self.floor = None
        if policy.startswith("named:"):
            f = policy.split(":")
            self.floor = (int(f[1]), int(f[2]) if len(f) > 2 else 0)
            pk = ri[:, :, :K]
            self.prank = np.array([(ti[:, :, r : r + 1] == pk).any(-1).mean() for r in range(32)])
            policy = self.policy = "hazard"
        # hzr = hazard whose table also bins the record's rank in its layer's latest released row (top 5, 10, 16, 32,
        # none): the routing just seen is the estimate of the next token's
        self.nr = 1
        self.redges = (5, 10, 16, 32)
        self.r2 = False  # hzr2: the layer's two latest released rows (this token's and the one before)
        if policy.startswith("hzr2"):
            self.r2 = True
            policy = "hzr" + policy[4:]
        if policy.startswith("hzr"):
            if policy.startswith("hzr:"):
                self.redges = tuple(int(x) for x in policy.split(":")[1].split("-"))
            self.nr = len(self.redges) + 1
            self.nr1 = self.nr
            if self.r2:
                self.nr = self.nr1 * self.nr1
            policy = self.policy = "hazard"
        if policy.startswith("look:"):
            _, h, r = policy.split(":")
            self.look = (int(h), float(r))
            policy = self.policy = "hazard"
        elif policy.startswith("freq:"):
            self.hl = float(policy.split(":")[1])
            policy = self.policy = "freq"
        N = NL * NE
        self.N = N
        self.held = np.zeros(N, bool)
        self.pin = np.full(N, -1, np.int64)  # pinned while pin >= the current event id
        self.unused = np.zeros(N, bool)  # read by a name, not yet used by a pick
        self.nheld = 0
        self.seen = np.zeros(N, bool)
        # stats over the counted positions
        self.per_layer = np.zeros((NL, 4))  # demand reads, name reads used, name reads wasted, left-out picks
        self.left_w = []
        self.reads_counted = 0
        self.compulsory = 0
        self.score = np.zeros(N)
        self.nu = np.full(N, float(1 << 60))
        self.nuflag = np.ones(N)
        self.ev = 0
        self.t = -1
        if policy == "hazard":
            self.fac = 2 ** (-1 / 8)
            self.last = np.full(N, -1.0)
            self.dec = np.zeros(N)
            self.u = np.ones((NL, HZ_NA * HZ_NF * self.nr))
            self.n = np.full((NL, HZ_NA * HZ_NF * self.nr), 2.0)
            self.rb = np.full(N, self.nr - 1, np.int64)
            self.kb = self._bin(np.full(N, -1.0), np.zeros(N), 0, self.rb)
            self.b = np.zeros(N, np.int64)
            self.bok = False
            self.cur = np.zeros(N, bool)
            self.layer_of = np.repeat(np.arange(NL), NE)
        elif policy == "freq":
            self.fac = 2 ** (-1 / self.hl)
            self.dec = np.zeros(N)
            self.last = np.full(N, -1.0)
        if policy == "belady" or self.look:
            # every key's pick events (event id = pos * NL + step); score = -(next use at or after now)
            uses = {}
            for p in range(self.npos):
                for s in range(NL):
                    for e in ri[p, s, :K]:
                        uses.setdefault(s * NE + int(e), []).append(p * NL + s)
            self.uses = {k: np.array(v) for k, v in uses.items()}

    # ---------------------------------------------------------------- hazard table (Cache::hz_bin / hz_tick / hz_touch)
    def _bin(self, last, dec, t, rb=None):
        age = np.maximum(1, t - last)
        ab = np.where(last >= 0, np.minimum(np.floor(np.log2(age)), HZ_NA - 1), min(14, HZ_NA - 1))
        fb = np.minimum(np.floor(np.log2(1.0 + 4.0 * dec + 0.0)), HZ_NF - 1)
        b = (ab * HZ_NF + fb).astype(np.int64)
        return b * self.nr + (rb if rb is not None and self.nr > 1 else 0)

    def see_released(self, p, s):
        """hzr: layer s's released row of this token re-bins its records"""
        rb = np.full(NE, self.nr1 - 1, np.int64)
        for r, e in enumerate(self.ri[p, s, :32]):
            rb[int(e)] = int(np.searchsorted(self.redges, r, side="right"))
        sl = slice(s * NE, (s + 1) * NE)
        if self.r2:
            rb = rb * self.nr1 + self.rb[sl] // self.nr1
        self.rb[sl] = rb
        self.kb[sl] = self._bin(self.last[sl], self.dec[sl], self.t + 1, rb)
        self.score[sl] = self.u[s, self.kb[sl]] / self.n[s, self.kb[sl]] + self.last[sl] * 1e-9

    def tick(self, t):
        if self.policy == "hazard":
            if self.t >= 0:
                np.add.at(self.n, (self.layer_of, self.kb), 1.0)
                self.b = self.kb.copy()
                self.bok = True
            self.dec *= self.fac
            self.kb = self._bin(self.last, self.dec, t + 1, self.rb)
            self.cur[:] = False
            self.score = self.u[self.layer_of, self.kb] / self.n[self.layer_of, self.kb] + self.last * 1e-9
        elif self.policy == "freq":
            self.dec *= self.fac
            self.score = self.dec + self.last * 1e-9
        self.t = t

    def _setnu(self, k, after):
        v = self._nu(k, after)
        self.nu[k] = v
        self.nuflag[k] = ((int(v) * 2654435761 + k * 40503) % 1000003) / 1000003.0

    def touch(self, k):
        if self.look:
            self._setnu(k, self.ev + 1)
        if self.policy == "hazard":
            if self.cur[k]:
                return
            self.cur[k] = True
            if self.bok:
                self.u[k // NE, self.b[k]] += 1.0
            self.last[k] = self.t
            self.dec[k] += 1.0
            self.kb[k] = self._bin(np.array([self.last[k]]), np.array([self.dec[k]]), self.t + 1, self.rb[k : k + 1])[0]
            self.score[k] = self.u[k // NE, self.kb[k]] / self.n[k // NE, self.kb[k]] + self.last[k] * 1e-9
        elif self.policy == "freq":
            self.last[k] = self.t
            self.dec[k] += 1.0
            self.score[k] = self.dec[k] + self.t * 1e-9
        elif self.policy == "lru":
            self.score[k] = self.ev + 1e-6 * (k % 1000)  # recency (ties broken by id)
        elif self.policy == "belady":
            self.score[k] = -self._nu(k, self.ev + 1)

    def _nu(self, k, after):
        arr = self.uses.get(k)
        if arr is None:
            return float(1 << 60)
        i = int(np.searchsorted(arr, after, side="left"))
        return float(arr[i]) if i < len(arr) else float(1 << 60)

    def victim(self):
        if self.look:
            H, R = self.look
            ok = self.held & (self.pin < self.ev)
            d = self.nu - self.ev
            seen = (d <= H * NL) & (self.nuflag < R)
            sc = self.score + np.where(seen, 10.0 + 1.0 - d / (H * NL + 1.0), 0.0)
            return int(np.argmin(np.where(ok, sc, np.inf)))
        sc = self.score
        if self.floor:
            D, X = self.floor
            p, s = divmod(self.ev, NL)
            sc = sc.copy()
            tg = [(p, s2) for s2 in range(s + 1, min(s + LAG, NL - 1) + 1)]
            if X and s >= NL - LAG and p + 1 < self.npos:
                tg += [(p + 1, s2) for s2 in range(LAG)]
            for p2, s2 in tg:
                ks = s2 * NE + self.ti[p2, s2, :D]
                sc[ks] = np.maximum(sc[ks], self.prank[:D] + 1.0 * 0)
        ok = self.held & (self.pin < self.ev)
        if self.caps:
            ok = ok & (np.arange(self.N) // NE == self.cur_L)
        cand = np.where(ok, sc, np.inf)
        return int(np.argmin(cand))

    def insert(self, k, pin_until, counted, L, by_name):
        self.cur_L = k // NE
        full = self.lheld[k // NE] >= self.C // NL if self.caps else self.nheld >= self.C
        if full:
            v = self.victim()
            self.lheld[v // NE] -= 1
            if self.unused[v] and counted:
                self.per_layer[v // NE, 2] += 1  # a named read evicted before any use: wasted
            self.held[v] = False
            self.unused[v] = False
            self.nheld -= 1
        self.held[k] = True
        self.nheld += 1
        self.lheld[k // NE] += 1
        self.pin[k] = pin_until
        self.unused[k] = by_name
        if self.policy == "lru":
            self.score[k] = self.ev + 1e-6 * (k % 1000)
        elif self.policy == "belady":
            self.score[k] = -self._nu(k, self.ev if by_name else self.ev + 1)
        if self.look:
            self._setnu(k, self.ev if by_name else self.ev + 1)
        if counted:
            self.reads_counted += 1

    def name_list(self, p, s):
        nm = self.names
        picks = self.ri[p, s, :K]
        if nm == "none":
            return []
        if nm == "perfect":
            return list(picks)
        kind, *a = nm.split(":")
        if kind == "router":
            return list(self.ti[p, s, : int(a[0])])
        if kind == "recall":
            r, m = float(a[0]), int(a[1])
            x = r * K
            kk = int(math.floor(x)) + (self.rng.random() < x - math.floor(x))
            good = list(self.rng.permutation(picks)[:kk])
            ps = set(int(e) for e in picks)
            wrong = [int(e) for e in self.ti[p, s] if int(e) not in ps][: max(0, m - kk)]
            return good + wrong
        raise ValueError(nm)

    def run(self):
        for p in range(self.npos):
            counted = p >= self.first
            self.tick(p)
            for s in range(NL):
                self.ev = p * NL + s
                # names for steps whose naming point is this step
                targets = [s + LAG] if s + LAG < NL else []
                if s == 0:
                    targets = list(range(0, min(LAG, NL))) + targets
                for st in targets:
                    for e in self.name_list(p, st):
                        k = st * NE + int(e)
                        if not self.held[k]:
                            self.insert(k, p * NL + st, counted, st, True)
                        else:
                            self.pin[k] = max(self.pin[k], p * NL + st)
                        if self.name_touch:
                            self.touch(k)
                picks = self.ri[p, s, :K]
                wts = self.rp[p, s, :K].astype(np.float64)
                wts = wts / wts.sum()
                keys = [s * NE + int(e) for e in picks]
                if counted:
                    self.hits += sum(bool(self.held[k]) for k in keys)
                for k in keys:
                    if self.use_touch:
                        self.touch(k)
                    self.pin[k] = max(self.pin[k], self.ev)
                if self.policy == "hazard" and self.nr > 1:
                    self.see_released(p, s)
                miss = [(wts[j], keys[j]) for j in range(K) if not self.held[keys[j]]]
                read = miss if self.mode != "prefetch" else []
                left = []
                if self.mode == "fast" and miss:
                    miss.sort(reverse=True)
                    rest = sum(w for w, _ in miss)
                    read = []
                    for w, k in miss:
                        if w > self.W or rest > self.MASS:
                            read.append((w, k))
                            rest -= w
                        else:
                            left.append((w, k))
                    if counted:
                        self.left_w.append(sum(w for w, _ in left))
                elif counted and self.mode == "fast":
                    self.left_w.append(0.0)
                for w, k in read:
                    if counted:
                        self.per_layer[s, 0] += 1
                        if not self.seen[k]:
                            self.compulsory += 1
                    self.insert(k, self.ev, counted, s, False)
                for k in keys:
                    if self.held[k] and self.unused[k]:
                        self.unused[k] = False
                        if counted:
                            self.per_layer[s, 1] += 1
                            if not self.seen[k]:
                                self.compulsory += 1
                    if (not self.mode == "fast") or self.held[k]:
                        self.seen[k] = True
                if counted:
                    self.per_layer[s, 3] += len(left)
        # named reads still unused at the end count as wasted
        T = self.npos - self.first
        unused_end = int((self.unused & self.held).sum())
        pl = self.per_layer
        need = pl[:, 0].sum() + pl[:, 1].sum()
        waste = pl[:, 2].sum() + unused_end
        out = {
            "records_per_token": self.reads_counted / T, "top10_held": self.hits / (T * NL * K),
            "mb_per_token": self.reads_counted * REC / 1e6 / T,
            "needed_per_token": need / T,
            "demand_per_token": pl[:, 0].sum() / T,
            "named_used_per_token": pl[:, 1].sum() / T,
            "wasted_per_token": waste / T,
            "compulsory_per_token": self.compulsory / T,
            "left_out_picks_per_token": pl[:, 3].sum() / T,
            "left_out_weight_per_step": float(np.mean(self.left_w)) if self.left_w else 0.0,
            "per_layer_needed_wasted": [[round(float(pl[L, 0] + pl[L, 1]) / T, 2), round(float(pl[L, 2]) / T, 2)] for L in range(NL)],
        }
        return out


def old_runner(ri, ti, first, slots=24, keep=3, m=10, lag=LAG):
    """The old MLX runner's slot rule (reference/flash/mlx_run.py, prefetch mode): per layer `slots` slots; the lag router's
    top m are named (steps < lag at the token's start, step s + lag at step s); a named expert not held is read into a free
    slot, else over the least recently NAMED one outside the keep window (named < t - keep), else over the least recently
    named one not named this token; at each step the layer's experts not named within `keep` tokens are freed. Picks are
    never read and never refresh an expert (the released top-k runs among the experts held). After the prompt each layer
    holds the experts its last `keep` prompt positions used, the latest first (the runner's batched prefill).
    Returns decode MB per token and the share of the released top 10 held when its layer runs."""
    npos = ri.shape[0]
    last = [dict() for _ in range(NL)]  # layer -> expert -> token last named
    for L in range(NL):
        lat = {}
        for p in range(max(0, first - keep), first):
            for e in ri[p, L, :K]:
                lat[int(e)] = p
        for e in sorted(lat, key=lambda e: -lat[e])[:slots]:
            last[L][e] = lat[e]
    reads = hits = asked = 0

    def name(L, t, pred, counted):
        nonlocal reads
        H = last[L]
        for c in pred:
            if c in H:
                H[c] = t
        for c in pred:
            if c in H:
                continue
            if len(H) >= slots:
                old = [e for e, l in H.items() if l < t - keep]
                if not old:
                    old = [e for e, l in H.items() if l < t]
                if not old:
                    raise RuntimeError("all slots named this token")
                del H[min(old, key=lambda e: H[e])]
            H[c] = t
            reads += counted

    for p in range(first, npos):
        for L in range(min(lag, NL)):
            name(L, p, [int(e) for e in ti[p, L, :m]], True)
        for L in range(NL):
            if L + lag < NL:
                name(L + lag, p, [int(e) for e in ti[p, L + lag, :m]], True)
            H = last[L]
            for e in [e for e, l in H.items() if l < p - keep]:
                del H[e]
            hits += sum(int(e) in H for e in ri[p, L, :K])
            asked += K
    T = npos - first
    return {"old_runner": {"slots": slots, "keep": keep, "m": m}, "mb_per_token": reads * REC / 1e6 / T,
            "records_per_token": reads / T, "top10_held": hits / asked}


def reuse(ri, first):
    """LRU stack distance of every pick use (distinct records touched since its last use; global over layers), decode
    positions only, and the per-layer chance that a pick was a pick of the same layer within the last d tokens."""
    seq = [s * NE + int(e) for p in range(ri.shape[0]) for s in range(NL) for e in ri[p, s, :K]]
    pos_of = [p for p in range(ri.shape[0]) for s in range(NL) for _ in range(K)]
    n = len(seq)
    bit = np.zeros(n + 1, np.int64)

    def add(i, v):
        i += 1
        while i <= n:
            bit[i] += v
            i += i & -i

    def pre(i):
        i += 1
        r = 0
        while i > 0:
            r += bit[i]
            i -= i & -i
        return r

    last = {}
    dist = []
    for i, k in enumerate(seq):
        j = last.get(k)
        if j is not None:
            d = pre(i - 1) - pre(j)
            add(j, -1)
            if pos_of[i] >= first:
                dist.append(d)
        elif pos_of[i] >= first:
            dist.append(-1)
        add(i, 1)
        last[k] = i
    dist = np.array(dist)
    gb = lambda x: x * REC / 1e9
    cold = float((dist < 0).mean())
    hist = {}
    for g in (0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32):
        hist[g] = float(((dist >= 0) & (dist < g * 1e9 / REC)).mean())
    # token-level stickiness
    tok = {}
    for d in (1, 2, 4, 8, 16, 32, 64):
        hit = tot = 0
        for p in range(max(first, d), ri.shape[0]):
            for s in range(NL):
                prev = set(ri[max(0, p - d) : p, s, :K].ravel().tolist())
                hit += sum(int(e) in prev for e in ri[p, s, :K])
                tot += K
        tok[d] = hit / tot
    return {"first_use_share": cold, "lru_hit_at_gb": hist, "in_last_d_tokens_same_layer": tok,
            "median_stack_distance_gb": float(gb(np.median(dist[dist >= 0])))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--first", type=int, default=-1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--caps", action="store_true")
    ap.add_argument("--name-touch", action="store_true")
    ap.add_argument("--no-use-touch", action="store_true")
    ap.add_argument("--old", default=None, help="the old runner's rule: SLOTS,KEEP,M")
    ap.add_argument("--one", default=None, help="one run: GB,policy,names,mode[,mass]")
    ap.add_argument("--grid", default="names", help="names | policy | fast")
    a = ap.parse_args()
    ri, rp, ti, tp = load(a.trace)
    first = a.first
    res = {}
    if a.reuse:
        res["reuse"] = reuse(ri, first)
        print(json.dumps(res["reuse"]))
    if a.old:
        f = [int(x) for x in a.old.split(",")]
        print(json.dumps(old_runner(ri, ti, first, *f)))
    if a.one:
        f = a.one.split(",")
        gb, pol, nm, mode = float(f[0]), f[1], f[2], f[3]
        mass = float(f[4]) if len(f) > 4 else 0.13
        r = Sim(ri, rp, ti, int(gb * 1e9 // REC), pol, nm, mode, w=mass, mass=mass, first=first, caps=a.caps, name_touch=a.name_touch,
                use_touch=not a.no_use_touch).run()
        r.pop("per_layer_needed_wasted")
        print(json.dumps({"gb": gb, "policy": pol, "names": nm, "mode": mode, "mass": mass, "caps": a.caps, "name_touch": a.name_touch,
                          "use_touch": not a.no_use_touch, **r}))
