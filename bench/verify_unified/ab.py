"""Same-session A/B of two dmeng builds on a real model: arms alternate cell by cell (A, B, A, B), every run through the
memory guard, one at a time. Decode cells only (prompt-file, greedy).

  python3 ab.py OUT_DIR GUARD_PY MODEL_DIR PROMPT_DIR --arm main=BIN --arm unified=BIN [--gb 6.6] [--gb2 5.0]
                [--modes exact,fast] [--prompts pitt,code] [--reps 2] [--n 400] [--extra "..."]
  python3 ab.py table OUT_DIR

Cells: at --gb every mode x prompt x rep; at --gb2 (ids check) exact only, one rep. Each run writes NAME.json (the engine's
statistics line plus the guard's summary as "guard") and NAME.ids.
"""
import argparse, glob, json, os, subprocess, sys

FIELDS = ("tok_s", "compute_ms_per_token", "wait_ms_per_token", "mb_read_per_token", "hit_rate")


def run(out, guard, name, binp, args, env_extra=None):
    js, ids = f"{out}/{name}.json", f"{out}/{name}.ids"
    if os.path.exists(js):
        return json.load(open(js))
    env = dict(os.environ, DISK_GB="8", FOOT_GB="13.5", SWAP_MB="2000", POLL_S="2")
    env.update(env_extra or {})
    g = subprocess.run(["python3", guard, f"{out}/{name}.log", "2400", "--", binp, *args, "--json", js, "--out-ids", ids],
                       capture_output=True, text=True, env=env).stdout
    gl = [l for l in g.splitlines() if l.startswith("GUARD ")]
    gd = json.loads(gl[-1][6:]) if gl else {}
    if not os.path.exists(js):
        print(f"{name}: FAILED {gd.get('killed_by_guard')} rc={gd.get('rc')}", flush=True)
        return None
    r = json.load(open(js))
    r["guard"] = gd
    json.dump(r, open(js, "w"))
    print(f"{name}: " + " | ".join(f"{k} {r.get(k, 0):.2f}" for k in FIELDS)
          + f" | others {gd.get('others_cpu_pct_mean')} | {(gd.get('before') or {}).get('power')}", flush=True)
    return r


def main():
    if sys.argv[1] == "table":
        return table(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("guard"); ap.add_argument("model"); ap.add_argument("prompts_dir")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--gb", default="6.6"); ap.add_argument("--gb2", default="")
    ap.add_argument("--modes", default="exact,fast"); ap.add_argument("--prompts", default="pitt,code")
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--n", default="400")
    ap.add_argument("--extra", default=""); ap.add_argument("--tag", default="d")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arms = [x.split("=", 1) for x in a.arm]
    base = lambda p, gb, mode: ["--model", a.model, "--prompt-file", f"{a.prompts_dir}/{p}.ids", "--n", a.n, "--cache-gb", gb,
                                "--temp", "0", "--mode", mode, *a.extra.split()]
    for rep in range(1, a.reps + 1):
        for p in a.prompts.split(","):
            for mode in a.modes.split(","):
                for an, b in (arms if rep % 2 else arms[::-1]):
                    run(a.out, a.guard, f"{a.tag}_{p}_gb{a.gb}_{mode}_{an}_r{rep}", b, base(p, a.gb, mode))
    if a.gb2:
        for p in a.prompts.split(","):
            for an, b in arms:
                run(a.out, a.guard, f"{a.tag}_{p}_gb{a.gb2}_exact_{an}_r1", b, base(p, a.gb2, "exact"))


def table(out):
    rows = {}
    for f in sorted(glob.glob(f"{out}/*.json")):
        n = os.path.basename(f)[:-5]
        key, rep = n.rsplit("_r", 1)
        rows.setdefault(key, []).append((rep, json.load(open(f))))
    print("| cell | arm | tok/s per rep | compute ms | wait ms | MB/token | hit | ids = other arm | others CPU % | power |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows):
        cell, arm = key.rsplit("_", 1)
        v = [r for _, r in sorted(rows[key])]
        mine = [open(f"{out}/{key}_r{rp}.ids").read().strip() for rp, _ in sorted(rows[key])]
        other = [g for g in glob.glob(f"{out}/{cell}_*_r*.ids") if f"{cell}_{arm}_" not in g]
        same = all(open(g).read().strip() == m for g in other for m in mine) if other else None
        mean = lambda k: sum(x.get(k) or 0 for x in v) / len(v)
        g = [x.get("guard") or {} for x in v]
        print(f"| {cell} | {arm} | {', '.join('%.2f' % x.get('tok_s', 0) for x in v)} | {mean('compute_ms_per_token'):.1f} | "
              f"{mean('wait_ms_per_token'):.1f} | {mean('mb_read_per_token'):.1f} | {mean('hit_rate'):.3f} | {same} | "
              f"{', '.join(str(round(x.get('others_cpu_pct_mean') or 0)) for x in g)} | {(g[0].get('before') or {}).get('power', '?')[:7]} |")


if __name__ == "__main__":
    main()
