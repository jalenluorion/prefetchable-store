"""Cache-size curve for an agent turn (and prose for contrast) from one recorded run per model and text: the run's routing
and early-router rows replayed through dm::Cache (tests/evict_check/evict_replay) per cache size and policy.
  python3 agent_curve.py RUNS_DIR > table.md"""
import json, os, subprocess, sys

R = sys.argv[1]
W = os.environ.get("CP_WORK", os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(W, "bin", "evict_replay")
Q_REC, I_REC = [1769472], [4489216, 8224768]
I_SHARE, I_MAX = [0.8, 0.2], [20 * 128, 4 * 128]


def slots(model, gb):
    if model == "q36":
        return [min(int(gb * 1e9 // Q_REC[0]), 40 * 256)]
    return [min(int(gb * 1e9 * I_SHARE[p] // I_REC[p]), I_MAX[p]) for p in range(2)]


TEXTS = [("35B", "q36", "agent turn (file rewrite)", "ag_q36", 869, [3.3, 5, 6.6, 8, 9.5, 11], 120),
         ("35B", "q36", "prose (pitt)", "pitt_v3_3.3", 9, [3.3, 5, 6.6, 8, 9.5, 11], 120),
         ("Inkling", "ink", "agent turn (file rewrite)", "ag_ink", 839, [4, 6, 7.8, 9, 10.5], 180),
         ("Inkling", "ink", "prose (pitt)", "i_v3_6", 10, [4, 6, 7.8, 9, 10.5], 180)]
ARMS = [("a", "cache only, old default (freq)", "freq", 0, 0), ("b", "cache only, hazard", "hazard", 0, 0),
        ("c", "k2 names m 8, hazard", "hazard", 8, 0), ("d", "k2 names m 8, hazard, name budget", "hazard", 8, 1),
        ("e", "k2 names m 8, old default (the timing run's setup)", "freq", 8, 0)]


def run(name, start, sl, pol, m, budget):
    env = dict(os.environ, NAME_M=str(m))
    if budget:
        env["NAME_BUDGET_MB"] = str(budget)
    out = subprocess.run([REP, os.path.join(R, name + ".rec"), os.path.join(R, name + ".trace"), ",".join(map(str, sl)), pol,
                          "/dev/null", str(start)], env=env, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


for mod, key, text, name, start, sizes, budget in TEXTS:
    if not os.path.exists(os.path.join(R, name + ".trace")):
        continue
    print(f"\n### {mod}, {text}: hit rate / MB read per token (urgent + named, named-but-unused in brackets)\n")
    print("| GB | slots | " + " | ".join(f"({a}) {d}" for a, d, *_ in ARMS) + " |")
    print("|---|---|" + "---|" * len(ARMS))
    for gb in sizes:
        sl = slots(key, gb)
        cells = []
        for a, d, pol, m, bud in ARMS:
            j = run(name, start, sl, pol, m, budget if bud else 0)
            c = f"{j['hit_rate']:.3f} / {j['mb_per_token']:.0f}"
            if m:
                c += f" ({j['urgent_mb_per_token']:.0f} + {j['named_mb_per_token']:.0f} [{j['named_unused_mb_per_token']:.0f}])"
            if bud:
                c += f", {j['names_skipped_per_token']:.0f} names skipped"
            cells.append(c)
        print(f"| {gb:g} | {'+'.join(map(str, sl))} | " + " | ".join(cells) + " |")
    sys.stdout.flush()
