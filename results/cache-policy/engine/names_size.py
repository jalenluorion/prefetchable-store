"""Sizing: a longer-lead router set run only for Inkling's last six applications (42-47).
From one run's routing (cache-only replay with hazard at the run's slots) and each lead's names (dmeng --dump-names):
per application recall of the released picks in the top 8 / 12 / 16, share of the cache's misses named in the top m,
names that would be read and not used; then a drive-queue model of the wait per application (reads FIFO at the
measured drive rate, a name read issued when its router runs = the start of application a - lead, a miss not named
read at its own step) for k2 everywhere vs k2 + lead L on 42-47, calibrated against the measured k2 run.
  python3 names_size.py RUNS_DIR"""
import json, os, subprocess, sys
from collections import defaultdict

R = sys.argv[1]
W = os.environ.get("CP_WORK", os.path.dirname(os.path.abspath(__file__)))  # the folder with bin/evict_replay
REP = os.path.join(W, "bin", "evict_replay")
BASE = "i_hz_6"
LEADS = {1: "i_names_k1_6", 2: "i_v3_6", 4: "i_names_k4_6", 8: "i_names_k8_6", 16: "i_names_k16_6"}
SLOTS = ",".join(str(x) for x in json.load(open(os.path.join(R, BASE + ".json")))["slots"])
START, M = 10, 8
LAST = list(range(42, 48))
REC = {0: 4489216, 1: 8224768}
step_layer = [0, 1, 2, 3, 4, 5] + list(range(6, 18)) * 3 + list(range(18, 24))
pool = lambda L: 1 if L >= 20 else 0


def picks_of(rec):
    b = open(rec, "rb").read()
    import struct
    n, st, k = struct.unpack("3i", b[:12])
    v = struct.unpack(f"{n * st * k}h", b[12:12 + 2 * n * st * k])
    return [[list(v[(t * st + a) * k:(t * st + a + 1) * k]) for a in range(st)] for t in range(n)]


def names_of(path, T, steps=48):
    b = open(path, "rb").read()
    import struct
    v = struct.unpack(f"{len(b) // 2}h", b)
    n = len(v) // (steps * 16)
    return {START + i: [list(v[(i * steps + a) * 16:(i * steps + a + 1) * 16]) for a in range(steps)] for i in range(n)}


picks = picks_of(os.path.join(R, BASE + ".rec"))
T = len(picks)
miss = {}   # lead -> {(t, a): [(e, rank)]}
waste = {}  # lead -> {(t, a): (w8, w12, w16)}
for k, run in LEADS.items():
    out = os.path.join(R, f"misses_k{k}.txt")
    env = dict(os.environ, NAMES=os.path.join(R, run + ".names"), NAMES_START=str(START), MISSES_OUT=out)
    subprocess.run([REP, os.path.join(R, BASE + ".rec"), os.path.join(R, BASE + ".trace"), SLOTS, "hazard", "/dev/null", str(START)],
                   env=env, capture_output=True, check=True)
    mi, wa = defaultdict(list), {}
    for ln in open(out):
        f = ln.split()
        if f[0] == "M":
            mi[(int(f[1]), int(f[2]))].append((int(f[3]), int(f[4])))
        else:
            wa[(int(f[1]), int(f[2]))] = tuple(int(x) for x in f[3:6])
    miss[k], waste[k] = mi, wa
names = {k: names_of(os.path.join(R, run + ".names"), T) for k, run in LEADS.items()}
toks = [t for t in range(START, T) if t in names[2]]
nt = len(toks)

print(f"tokens {nt} (positions {START}..{T - 1}), slots {SLOTS}, cache-only hazard replay; per token")
print("app | lead | recall@8 / 12 / 16 | misses | named in top 8 / 12 / 16 | wasted names top 8 (MB)")
tab = {}
for a in LAST:
    for k in LEADS:
        rc = [0, 0, 0]
        for t in toks:
            nn = names[k][t][a]
            for q, m in enumerate((8, 12, 16)):
                rc[q] += len(set(picks[t][a]) & set(nn[:m])) / 8
        ms = sum(len(miss[k].get((t, a), [])) for t in toks)
        cov = [sum(1 for t in toks for e, r in miss[k].get((t, a), []) if 0 <= r < m) for m in (8, 12, 16)]
        w8 = sum(waste[k].get((t, a), (0, 0, 0))[0] for t in toks)
        tab[(a, k)] = (ms / nt, cov[0] / max(1, ms), w8 / nt)
        print(f"{a} | k{k} | {rc[0] / nt:.3f} / {rc[1] / nt:.3f} / {rc[2] / nt:.3f} | {ms / nt:.2f} | "
              f"{cov[0] / max(1, ms):.2f} / {cov[1] / max(1, ms):.2f} / {cov[2] / max(1, ms):.2f} | "
              f"{w8 / nt:.2f} ({w8 / nt * REC[pool(step_layer[a])] / 1e6:.1f})")

# drive-queue model
# model inputs: the drive at 3.4 GB/s (the Mac's sequential rate) and the compute per application of the timing campaigr's
# k2 run at 6 GB in the old engine (results/inkling_draft/runs/pittx_k2_gb6_rt1: 60.0 ms compute per token, waits
# 12.2 / 9.7 / 9.8 ms in the first six / looped / last six), which the model is checked against
gbs = float(os.environ.get("DRIVE_GBS", "3.4"))
t_app = float(os.environ.get("COMPUTE_MS", "60.0")) / 48
MEAS = (12.2, 9.7, 9.8)


def model(lead_last):
    tot = [0.0] * 48
    for t in toks:
        free = 0.0
        start = [0.0] * 48
        clock = 0.0
        pend = defaultdict(list)  # issue step -> reads
        for a in range(48):
            k = lead_last if a in LAST else 2
            src = a - k
            mb = REC[pool(step_layer[a])] / 1e6
            named = [(e, r) for e, r in miss[k].get((t, a), []) if 0 <= r < M]
            un = [(e, r) for e, r in miss[k].get((t, a), []) if not (0 <= r < M)]
            wasted = waste[k].get((t, a), (0, 0, 0))[0]
            pend[max(0, src)] += [(mb, a)] * (len(named) + wasted)
            pend[a] += [(mb, a, "u")] * len(un)
        done = {}
        for a in range(48):
            start[a] = clock
            for r in pend[a]:  # FIFO at the drive rate; a read is needed by step r[1]
                free = max(free, clock) + r[0] / gbs  # ms: MB / (GB/s)
                done[r[1]] = max(done.get(r[1], 0.0), free)
            need = done.get(a, 0.0)
            w = max(0.0, need - (clock + t_app))
            tot[a] += w
            clock += t_app + w
    return [x / nt for x in tot]


base = model(2)
print(f"\nmodel: {t_app:.2f} ms compute per application, drive {gbs:.2f} GB/s, names = top {M}")
print("wait ms/token, apps 0-5 / looped 6-41 / 42-47: measured k2 (old engine, freq) "
      f"{MEAS[0]} / {MEAS[1]} / {MEAS[2]}; model k2 "
      f"{sum(base[:6]):.1f} / {sum(base[6:42]):.1f} / {sum(base[42:]):.1f}")
for k in (4, 8, 16):
    m = model(k)
    extra_mb = sum(tab[(a, k)][2] * REC[1] / 1e6 for a in LAST) - sum(tab[(a, 2)][2] * REC[1] / 1e6 for a in LAST)
    print(f"k2 + k{k} on 42-47: model wait 42-47 {sum(m[42:]):.1f} ms (k2 {sum(base[42:]):.1f}), total {sum(m):.1f} (k2 {sum(base):.1f}); "
          f"wasted-name reads {extra_mb:+.1f} MB/token vs k2")
