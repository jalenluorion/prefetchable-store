#!/usr/bin/env python3
"""Token time for Flash-Next's exact-mode decode from one recorded run, offline (audit tool, no model): trace_sim.py's
cache replay with a clock, a drive and the engine's read schedule, to choose a source of low-priority reads for the
drive's idle time before building one.

Input: the --evict-trace file of one exact-mode run (trace_sim.py): per (position, step) the released picks and the lag-4
router's top 32.

The main thread, per token: names for steps 0-3 at the token's start; per step s: PRE ms (attention, router), then the
step's misses go to the urgent queue and the names for step s + 4 (the router's top M) to the speculative one; the experts
in memory run (X ms each), and for a pick still in flight the step waits; before its first wait the run-ahead (RA ms)
names step s + 1's picks (each found with probability RA_FIND, one wrong name per RA_WRONG found) unless the read lands
first. HEAD ms at the token's end. The hazard keep-score (trace_sim.Sim) chooses every victim among the unpinned, landed
records (names leave 2K slots). Names pin until their step has run.

The drive: R readers, each one read in flight; total rate R1 GB/s with one read in flight, R2 with two or more, shared
equally (processor sharing; drive_sweep.c: 2.76 / 3.49). A read submitted to an idle drive starts WAKE ms later (the
reader's wake-up; the engine's reader_idle_queued time). Urgent reads first, then speculative, then background.

Background reads (--fill): started only while nothing urgent or speculative is queued and at most GATE reads are in flight,
one record at a time in PIECES pieces (so an urgent read waits at most one piece), admitted only over a record whose
keep-score is below the candidate's value, never pinned. A pick that finds its record queued moves it to the urgent queue.
Sources (comma-separated, tried in order):
  rank:D       the router's ranks M+1..D for the steps already named (s+1 .. s+4), nearest step first; value = this
               trace's P(a pick | rank)
  recent:W:A   the picks of the same layer in the last W tokens that are not held, for steps s+1 .. s+A (A > 47 - s
               reaches the next token's first steps), most recent first; value = the hazard keep-score
  next:D       upper bound for names of the next token's steps 0-3 before its token starts: its router's top D from step
               44 on (the router does not exist; this says whether one would be worth having)
Prints one JSON line per run: ms per token, wait, drive idle ms (nothing in flight: [0] nothing queued, [1] queued), ms
with exactly one read in flight, MB per token (urgent / speculative / background, background used / wasted), effective
GB/s (bytes over wall time).

usage: time_sim.py TRACE --first P [--gb 2.0] [--fill SRC] [--json OUT] [params]
"""
import argparse, json, math, os, sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_sim import load, Sim, REC, NL, NE, K, LAG  # noqa: E402

MB = REC / 1e6
INF = float("inf")


class Drive:
    def __init__(self, a):
        self.R, self.r1, self.r2, self.wake = a.readers, a.r1, a.r2, a.wake  # GB/s = MB/ms
        self.t = 0.0
        self.fl = []  # in flight: [remaining MB, key, cls, last piece]
        self.q = [deque(), deque(), deque()]  # (key, MB, last piece, t_ready)
        self.dms = np.zeros(6)  # [0] nothing in flight or queued, [1] queued only, [2..5] 1, 2, 3, 4+ in flight
        self.mb = np.zeros(3)
        self.max_bg, self.gate, self.no_urgent = a.max_bg, a.gate, a.no_urgent
        self.fill = None
        self.on_land = None
        self.count = True

    def submit(self, key, mb, cls, pieces=1):
        tr = self.t + (self.wake if not self.fl else 0.0)
        for i in range(pieces):
            self.q[cls].append((key, mb / pieces, i == pieces - 1, tr))

    def promote(self, key):
        moved = False
        for c in (1, 2):
            keep = deque()
            for j in self.q[c]:
                if j[0] == key:
                    self.q[0].append((j[0], j[1], j[2], min(j[3], self.t)))
                    moved = True
                else:
                    keep.append(j)
            self.q[c] = keep
        return moved

    def _start(self):
        while len(self.fl) < self.R:
            cls = -1
            for c in (0, 1):
                if self.q[c] and self.q[c][0][3] <= self.t:
                    cls = c
                    break
            if cls < 0:
                if self.q[0] or self.q[1]:
                    break
                nbg = sum(1 for f in self.fl if f[2] == 2)
                if nbg >= self.max_bg or len(self.fl) - nbg > self.gate:
                    break
                if self.no_urgent and any(f[2] == 0 for f in self.fl):
                    break
                if not self.q[2] and self.fill:
                    self.fill()
                if not self.q[2] or self.q[2][0][3] > self.t:
                    break
                cls = 2
            k, m, last, _ = self.q[cls].popleft()
            self.fl.append([m, k, cls, last])
            if self.count:
                self.mb[cls] += m

    def _state(self):
        n = len(self.fl)
        return 1 + min(n, 4) if n else (1 if any(self.q) else 0)

    def step(self, t_lim):
        """advance to the next completion or t_lim, whichever is first"""
        self._start()
        n = len(self.fl)
        each = (self.r1 if n == 1 else self.r2) / n if n else 0.0
        dt_c = min(f[0] for f in self.fl) / each if n else INF
        dt_r = INF
        for c in range(3):
            if self.q[c] and self.q[c][0][3] > self.t:
                dt_r = min(dt_r, self.q[c][0][3] - self.t)
        dt = max(0.0, min(dt_c, dt_r, t_lim - self.t))
        if self.count:
            self.dms[self._state()] += dt
        self.t += dt
        if n:
            done = []
            for f in self.fl:
                f[0] -= each * dt
                if f[0] <= 1e-9:
                    done.append(f)
            for f in done:
                self.fl.remove(f)
                if f[3] and self.on_land:
                    self.on_land(f[1], f[2])
        return dt_c == INF and dt_r == INF and self.t >= t_lim

    def advance(self, t_to):
        while self.t < t_to - 1e-12:
            self.step(t_to)
        self._start()

    def run_until(self, cond):
        while not cond():
            if not self.fl and not any(self.q):
                raise RuntimeError("waiting for a read that was never issued")
            self.step(INF)


class TimeSim:
    def __init__(self, ri, ti, a):
        self.ri, self.ti, self.a = ri, ti, a
        self.npos = ri.shape[0]
        self.C = int(a.gb * 1e9 // REC)
        self.pol = Sim(ri, np.zeros_like(ri, dtype=np.float32), ti, self.C, "hazard", "none")  # the hazard statistics
        N = NL * NE
        self.held = np.zeros(N, bool)
        self.landed = np.zeros(N, bool)
        self.pins = np.zeros(N, np.int32)
        self.bg_unused = np.zeros(N, bool)
        self.boost = np.zeros(N)  # a background read's value, kept as its keep-score until its step has run
        self.boost_until = np.full(N, -1, np.int64)
        self.nheld = 0
        self.d = Drive(a)
        self.d.on_land = self.land
        self.d.fill = self.fill
        self.rng = np.random.default_rng(a.seed)
        self.p = self.s = 0
        self.counted = False
        self.st = dict(bg_used=0, bg_wasted=0, bg_issued=0, bg_refused=0, wait=0.0, misses=0, late=0, hits=0, ra_names=0)
        # P(a pick | router rank), from this trace (the engine's routers calibrate the same online)
        pk = ri[:, :, :K]
        self.prank = np.array([(ti[:, :, r : r + 1] == pk).any(-1).mean() for r in range(32)])
        self.fills = []
        for f in (a.fill.split(",") if a.fill else []):
            kind, *x = f.split(":")
            self.fills.append((kind, [float(v) for v in x]))

    # ---------------------------------------------------------------- cache
    def land(self, k, cls):
        self.landed[k] = True

    def victim(self, reserve):
        if self.nheld < self.C:
            return -2
        ok = self.held & (self.pins == 0) & self.landed
        if reserve > 0 and int(ok.sum()) <= reserve:
            return -1
        sc = self.pol.score
        if self.a.protect:
            ev = self.p * NL + self.s
            sc = np.where(self.boost_until >= ev, np.maximum(sc, self.boost), sc)
        sc = np.where(ok, sc, np.inf)
        v = int(np.argmin(sc))
        return v if np.isfinite(sc[v]) else -1

    def evict(self, v):
        if v >= 0:
            if self.bg_unused[v] and self.counted:
                self.st["bg_wasted"] += 1
            self.held[v] = self.landed[v] = False
            self.bg_unused[v] = False
            self.nheld -= 1

    def read(self, k, cls, reserve, pieces=1):
        v = self.victim(reserve)
        if v == -1:
            return False
        self.evict(v)
        self.held[k] = True
        self.landed[k] = False
        self.nheld += 1
        self.d.submit(k, MB, cls, pieces)
        return True

    # ---------------------------------------------------------------- background source
    def candidates(self):
        p, s = self.p, self.s
        for kind, x in self.fills:
            if kind == "rank":
                D = int(x[0])
                for s2 in range(s + 1, min(s + LAG, NL - 1) + 1):
                    for r in range(self.a.m, D):
                        yield s2 * NE + int(self.ti[p, s2, r]), self.prank[r], p * NL + s2
            elif kind == "recent":
                W, A = int(x[0]), int(x[1])
                for d in range(1, A + 1):
                    s2, p2 = s + d, p
                    if s2 >= NL:
                        s2, p2 = s2 - NL, p + 1
                    if s2 >= NL:
                        break
                    for back in range(1, W + 1):
                        pb = p2 - back
                        if pb < 0:
                            break
                        for e in self.ri[pb, s2, :K]:
                            k = s2 * NE + int(e)
                            yield k, self.pol.score[k], p2 * NL + s2
            elif kind == "next":
                D = int(x[0])
                if s >= NL - LAG and p + 1 < self.npos:
                    for s2 in range(0, LAG):
                        for r in range(D):
                            yield s2 * NE + int(self.ti[p + 1, s2, r]), self.prank[r], (p + 1) * NL + s2

    def fill(self):
        tried = 0
        for k, val, until in self.candidates():
            if self.held[k]:
                continue
            tried += 1
            if tried > 64:
                return
            v = self.victim(2 * K)
            if v == -1:
                return
            vs = self.pol.score[v] if v >= 0 else -1.0
            if v >= 0 and self.a.protect and self.boost_until[v] >= self.p * NL + self.s:
                vs = max(vs, self.boost[v])
            if v >= 0 and vs >= val - self.a.margin:
                if self.counted:
                    self.st["bg_refused"] += 1
                continue
            self.read(k, 2, 2 * K, self.a.pieces)
            self.bg_unused[k] = True
            self.boost[k], self.boost_until[k] = val, until
            if self.counted:
                self.st["bg_issued"] += 1
            return

    # ---------------------------------------------------------------- the decode loop
    def name(self, p, s2, ks):
        for k in ks:
            if self.held[k]:
                self.pins[k] += 1
                self.named.setdefault(s2, []).append(k)
            elif self.read(k, 1, 2 * K):
                self.pins[k] += 1
                self.named.setdefault(s2, []).append(k)

    def router_names(self, p, s2):
        return [s2 * NE + int(e) for e in self.ti[p, s2, : self.a.m]]

    def run(self):
        a, d = self.a, self.d
        t = 0.0
        tok_ms = []
        first = a.first
        self.named = {}
        st = self.st
        for p in range(self.npos):
            self.counted = d.count = p >= first
            t0 = t
            self.pol.tick(p)
            self.p, self.s = p, 0
            d.advance(t)
            for s2 in range(LAG):
                self.name(p, s2, self.router_names(p, s2))
            for s in range(NL):
                self.p, self.s = p, s
                self.pol.ev = p * NL + s
                t += a.pre
                d.advance(t)
                keys = [s * NE + int(e) for e in self.ri[p, s, :K]]
                for k in keys:
                    self.pol.touch(k)
                for k in keys:
                    if self.held[k]:
                        if self.counted:
                            st["hits" if self.landed[k] else "late"] += 1
                        if self.bg_unused[k]:
                            self.bg_unused[k] = False
                            if self.counted:
                                st["bg_used"] += 1
                    else:
                        self.read(k, 0, 0)
                        if self.counted:
                            st["misses"] += 1
                    self.pins[k] += 1
                if s + LAG < NL:
                    self.name(p, s + LAG, self.router_names(p, s + LAG))
                # experts: what has landed runs; wait for the rest in pick order (run-ahead before the first wait)
                tc = t + a.x  # the shared expert
                d.advance(tc)
                left = list(keys)
                ra = False
                while left:
                    now = [k for k in left if self.landed[k]]
                    if now:
                        left = [k for k in left if not self.landed[k]]
                        tc += a.x * len(now)
                        d.advance(tc)
                        continue
                    k0 = left[0]
                    d.promote(k0)
                    if a.ra and not ra and s + 1 < NL:
                        ra = True
                        tr = tc + a.ra_ms
                        d.advance(tr)
                        if not self.landed[k0]:
                            nxt = [(s + 1) * NE + int(e) for e in self.ri[p, s + 1, :K] if self.rng.random() < a.ra_find]
                            nf = len(nxt)
                            wrong = [(s + 1) * NE + int(e) for e in self.ti[p, s + 1, :32] if int(e) not in set(self.ri[p, s + 1, :K].tolist())]
                            nxt += wrong[: int(round(nf * a.ra_wrong))]
                            self.name(p, s + 1, [k for k in nxt if not self.held[k]])
                            if self.counted:
                                st["ra_names"] += len(nxt)
                            tc = tr
                    w0 = tc
                    d.run_until(lambda: self.landed[k0])
                    tc = max(tc, d.t)
                    if self.counted:
                        st["wait"] += tc - w0
                t = tc
                d.advance(t)
                for k in keys:
                    self.pins[k] -= 1
                for k in self.named.pop(s, []):
                    self.pins[k] -= 1
            t += a.head
            d.advance(t)
            if self.counted:
                tok_ms.append(t - t0)
        T = len(tok_ms)
        dms = d.dms / T
        mb = d.mb / T
        tot = mb.sum()
        ms = float(np.mean(tok_ms))
        return {
            "ms_per_token": round(ms, 2), "tok_s": round(1000 / ms, 3), "wait_ms": round(st["wait"] / T, 2),
            "drive_idle_ms": round(float(dms[0] + dms[1]), 2), "idle_nothing_queued_ms": round(float(dms[0]), 2),
            "one_read_ms": round(float(dms[2]), 2), "two_plus_ms": round(float(dms[3:].sum()), 2),
            "mb_per_token": round(float(tot), 1), "mb_urgent": round(float(mb[0]), 1), "mb_spec": round(float(mb[1]), 1),
            "mb_bg": round(float(mb[2]), 1), "bg_used_per_token": round(st["bg_used"] / T, 2),
            "bg_wasted_per_token": round(st["bg_wasted"] / T, 2), "bg_refused_per_token": round(st["bg_refused"] / T, 2),
            "effective_gb_s": round(float(tot / ms), 3), "hit": round(st["hits"] / (T * NL * K), 4),
            "late": round(st["late"] / (T * NL * K), 4), "miss": round(st["misses"] / (T * NL * K), 4),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--gb", type=float, default=2.0)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--fill", default="")
    ap.add_argument("--pieces", type=int, default=8)
    ap.add_argument("--gate", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=1)
    ap.add_argument("--no-urgent", type=int, default=0, help="no background read starts while an urgent read is in flight")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--protect", type=int, default=1, help="a background read keeps its value as its keep-score until its step")
    ap.add_argument("--readers", type=int, default=4)
    ap.add_argument("--r1", type=float, default=2.76)
    ap.add_argument("--r2", type=float, default=3.49)
    ap.add_argument("--wake", type=float, default=0.6)
    ap.add_argument("--pre", type=float, default=2.25)
    ap.add_argument("--x", type=float, default=0.074)
    ap.add_argument("--head", type=float, default=6.5)
    ap.add_argument("--ra", type=int, default=1)
    ap.add_argument("--ra-ms", type=float, default=1.34)
    ap.add_argument("--ra-find", type=float, default=0.94)
    ap.add_argument("--ra-wrong", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ri, rp, ti, tp = load(a.trace)
    r = TimeSim(ri, ti, a).run()
    print(json.dumps({"tag": a.tag, "gb": a.gb, "fill": a.fill, "pieces": a.pieces, "gate": a.gate, "wake": a.wake, **r}))
