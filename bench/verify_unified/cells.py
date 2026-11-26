"""Named dmeng cells, alternated rep by rep (order reversed on even reps), every run through the memory guard.
  python3 cells.py OUT GUARD_PY --common "flags shared by every cell" --cell NAME=BIN::flags ... [--reps 2]
  python3 cells.py table OUT [REF_CELL]     (ids of every cell compared with REF_CELL's first rep)
"""
import argparse, glob, json, os, subprocess, sys


def main():
    if sys.argv[1] == "table":
        return table(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("guard"); ap.add_argument("--common", default="")
    ap.add_argument("--cell", action="append", required=True); ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = []
    for c in a.cell:
        n, rest = c.split("=", 1)
        b, fl = rest.split("::", 1) if "::" in rest else (rest, "")
        cells.append((n, b, fl.split()))
    for rep in range(1, a.reps + 1):
        for n, b, fl in (cells if rep % 2 else cells[::-1]):
            name = f"{n}_r{rep}"
            js = f"{a.out}/{name}.json"
            if os.path.exists(js):
                continue
            env = dict(os.environ, DISK_GB="8", FOOT_GB="13.5", SWAP_MB="2000", POLL_S="2")
            g = subprocess.run(["python3", a.guard, f"{a.out}/{name}.log", "2400", "--", b, *a.common.split(), *fl, "--json", js,
                                "--out-ids", f"{a.out}/{name}.ids"], capture_output=True, text=True, env=env).stdout
            gl = [l for l in g.splitlines() if l.startswith("GUARD ")]
            gd = json.loads(gl[-1][6:]) if gl else {}
            if not os.path.exists(js):
                print(f"{name}: FAILED {gd.get('killed_by_guard')} rc={gd.get('rc')}", flush=True)
                continue
            r = json.load(open(js)); r["guard"] = gd; json.dump(r, open(js, "w"))
            d = r.get("draft") or {}
            print(f"{name}: tok/s {r.get('tok_s', 0):.2f} | tokens/step {d.get('tokens_per_step', 1)} | MB/tok {r.get('mb_read_per_token', 0):.1f} | "
                  f"wait {r.get('wait_ms_per_token', 0):.1f} | tokens {r.get('tokens')} | others {gd.get('others_cpu_pct_mean')} | {(gd.get('before') or {}).get('power')}", flush=True)


def table(out, ref):
    rows = {}
    for f in sorted(glob.glob(f"{out}/*_r[0-9].json")):
        n = os.path.basename(f)[:-5]
        cell, rep = n.rsplit("_r", 1)
        rows.setdefault(cell, []).append((int(rep), json.load(open(f))))
    refids = open(f"{out}/{ref}_r1.ids").read().strip() if ref and os.path.exists(f"{out}/{ref}_r1.ids") else None
    print("| cell | tok/s per rep | tokens/step | offered / declined / drafted / accepted | MB/token | wait ms | ids = ref | others CPU % |")
    print("|---|---|---|---|---|---|---|---|")
    for c in sorted(rows):
        v = [r for _, r in sorted(rows[c], key=lambda x: x[0])]
        d = v[0].get("draft") or {}
        ids = [open(f"{out}/{c}_r{k}.ids").read().strip() for k, _ in sorted(rows[c], key=lambda x: x[0])]
        same = all(i == refids for i in ids) if refids else "-"
        print(f"| {c} | {', '.join('%.2f' % r.get('tok_s', 0) for r in v)} | {d.get('tokens_per_step', '-')} | "
              f"{d.get('offered', '-')} / {d.get('declined', '-')} / {d.get('drafted', '-')} / {d.get('accepted', '-')} | "
              f"{sum(r.get('mb_read_per_token', 0) for r in v) / len(v):.1f} | {sum(r.get('wait_ms_per_token', 0) for r in v) / len(v):.1f} | {same} | "
              f"{', '.join(str(round((r.get('guard') or {}).get('others_cpu_pct_mean') or 0)) for r in v)} |")


if __name__ == "__main__":
    main()
