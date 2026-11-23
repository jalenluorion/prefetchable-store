"""P0 item 1: offline cache tiers at phone sizes, from recorded routing (no model run).

Inputs (read-only): the routing records that deep-memory/engine/bench/stage4_sizing.py reads
  results/stage2/runs/record_400.bin        Pittsburgh prompt, 400 generated tokens (prompt positions not counted)
  results/stage3/runs/record_code_400.bin   coding prompt, generated (prompt positions not counted)
  results/stage3c/runs/record_tf_agent.bin  agent transcript, teacher-forced (all positions counted, cold start)
  results/stage3c/runs/record_tf_code.bin   code text, teacher-forced
  results/stage3c/runs/record_tf_prose.bin  prose text, teacher-forced
Policies: the engine's decayed-count policy (half-life 8 tokens, smallest key evicted, the layer's own picks pinned)
exactly as stage4_sizing.sim_ours, implemented with a lazy heap (same scores, same tie-break by lowest slot), and Belady
(stage4_sizing.sim_belady, unchanged). One pool over all 40 layers.
  python3 tier_sim.py [ENGINE_DIR] -> tier_sim.json beside this file
"""
import heapq, json, math, os, struct, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = sys.argv[1] if len(sys.argv) > 1 else "<work>/engine"
REC_BYTES = 1769472
NL, K = 40, 8


def load_rec(p):
    raw = open(p, "rb").read()
    npos, nl, k = struct.unpack("<3i", raw[:12])
    return np.frombuffer(raw[12:12 + npos * nl * k * 2], dtype=np.int16).reshape(npos, nl, k)


def ids(p):
    return [int(x) for x in open(p).read().replace("\n", ",").split(",") if x.strip()]


RECS = {
    "pitt": ("results/stage2/runs/record_400.bin", len(ids(os.path.join(ENG, "results/stage2/pitt_ids.txt")))),
    "coding": ("results/stage3/runs/record_code_400.bin", len(ids(os.path.join(ENG, "results/stage3/code_ids.txt")))),
    "agent_text": ("results/stage3c/runs/record_tf_agent.bin", 0),
    "code_text": ("results/stage3c/runs/record_tf_code.bin", 0),
    "prose_text": ("results/stage3c/runs/record_tf_prose.bin", 0),
}


def sim_ours(rec, slots, start, hl=8.0):
    """stage4_sizing.sim_ours with a heap instead of argmin: score = key + last * 1e-12, smallest evicted, pinned skipped"""
    key = [-math.inf] * slots
    last = [-1.0] * slots
    ver = [0] * slots
    who, owner = {}, [None] * slots
    free = list(range(slots - 1, -1, -1))
    heap = []
    pinned = [False] * slots
    hit = tot = 0
    for t in range(rec.shape[0]):
        rt = rec[t].tolist()
        for L in range(NL):
            used = []
            for e in rt[L]:
                s = who.get((L, e))
                if t >= start:
                    tot += 1
                    hit += s is not None
                if s is None:
                    if free:
                        s = free.pop()
                    else:
                        stash = []
                        while True:
                            sc, ss, vv = heapq.heappop(heap)
                            if vv != ver[ss]:
                                continue
                            if pinned[ss]:
                                stash.append((sc, ss, vv))
                                continue
                            s = ss
                            break
                        for x in stash:
                            heapq.heappush(heap, x)
                        del who[owner[s]]
                    who[(L, e)] = s
                    owner[s] = (L, e)
                    key[s] = -math.inf
                    last[s] = -1.0
                    ver[s] += 1  # no heap entry while pinned with key -inf
                pinned[s] = True
                used.append(s)
            for s in used:
                v = 0.0 if key[s] == -math.inf else 2.0 ** (key[s] - t / hl)
                key[s] = math.log2(v + 1.0) + t / hl
                last[s] = t * NL + L
                pinned[s] = False
                ver[s] += 1
                heapq.heappush(heap, (key[s] + last[s] * 1e-12, s, ver[s]))
    return hit / tot, tot


def sim_belady(rec, slots, start):
    """stage4_sizing.sim_belady, unchanged"""
    seq = [(t * NL + L, (L, int(e))) for t in range(rec.shape[0]) for L in range(NL) for e in rec[t, L]]
    nxt = [0] * len(seq)
    seen = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = seen.get(seq[i][1], 10 ** 9)
        seen[seq[i][1]] = seq[i][0]
    inc, heap, cur, cur_step = {}, [], set(), -1
    hit = tot = 0
    for i, (step, k) in enumerate(seq):
        if step != cur_step:
            cur, cur_step = set(), step
        if step // NL >= start:
            tot += 1
            hit += k in inc
        if k not in inc and len(inc) >= slots:
            stash = []
            while True:
                nu, kk = heapq.heappop(heap)
                if inc.get(kk) != -nu:
                    continue
                if kk in cur:
                    stash.append((nu, kk))
                    continue
                del inc[kk]
                break
            for x in stash:
                heapq.heappush(heap, x)
        inc[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
        cur.add(k)
    return hit / tot


SIZES = [0.8, 1.27, 1.6, 2.0, 2.5, 3.3, 4.0, 5.0, 6.6, 8.0]


def slots_of(gb):
    return 720 if gb == 1.27 else int(gb * 1e9 // REC_BYTES)  # 1.27 GB = today's phone cache, 720 slots


out = {"record_bytes": REC_BYTES, "sizes_gb": SIZES, "slots": {str(g): slots_of(g) for g in SIZES}, "texts": {}}
for name, (rel, start) in RECS.items():
    p = os.path.join(ENG, rel)
    if not os.path.exists(p):
        continue
    rec = load_rec(p)
    counted = rec.shape[0] - start
    out["texts"][name] = {"record": rel, "positions": int(rec.shape[0]), "uncounted_prompt_positions": start, "rows": {}}
    for gb in SIZES:
        s = slots_of(gb)
        t0 = time.time()
        h_o, tot = sim_ours(rec, s, start)
        h_b = sim_belady(rec, s, start)
        row = {"slots": s, "gb_exact": round(s * REC_BYTES / 1e9, 3)}
        for pol, h in (("ours", h_o), ("belady", h_b)):
            miss = (1 - h) * NL * K
            row[pol] = {"hit": round(h, 4), "misses_per_token": round(miss, 2), "mb_per_token": round(miss * REC_BYTES / 1e6, 1)}
        out["texts"][name]["rows"][str(gb)] = row
        print(name, gb, s, row["ours"], row["belady"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "tier_sim.json"), "w"), indent=1)
json.dump(out, open(os.path.join(HERE, "tier_sim.json"), "w"), indent=1)
print("wrote", os.path.join(HERE, "tier_sim.json"))
