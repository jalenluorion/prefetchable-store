"""Inkling at phone-sized caches, measured on the Mac: bytes read per decode token by layer group, per expert set, cache
size, prompt and variant. Each run: dmeng under the memory guard with the read log (preadlog.c) inserted, then
readsplit.py on its log; the log is deleted after the split.

  python3 phone_bytes.py OUT GUARD_PY DMENG MODEL_DIR SHIM_DIR --prompt NAME=IDS ... [--gbs 2,3,4,6] [--n 300]
                         [--sets released,nv4] [--var NAME="flags"] ...
Sets: released = --profile phone (8-bit in-memory weights, layers 20-23 MXFP8 as released); nv4 = --profile phone-small
--mode exact with the 8-bit in-memory file (every expert 4-bit: the store_nv4 copies of layers 20-23).
A variant adds flags on top of a set (e.g. --var fit="--layer-caps auto:fit"); "base" = no flags.
  python3 phone_bytes.py table OUT
"""
import argparse, glob, json, os, subprocess, sys

H = os.path.dirname(os.path.abspath(__file__))


def main():
    if sys.argv[1] == "table":
        return table(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("guard"); ap.add_argument("dmeng"); ap.add_argument("model"); ap.add_argument("shim")
    ap.add_argument("--prompt", action="append", required=True); ap.add_argument("--gbs", default="2,3,4,6")
    ap.add_argument("--n", default="300"); ap.add_argument("--sets", default="released,nv4")
    ap.add_argument("--var", action="append", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q8 = glob.glob(f"{a.model}/derived/resident_q8.*.bin")[0]
    sets = {"released": ["--profile", "phone"], "nv4": ["--profile", "phone-small", "--mode", "exact", "--resident", q8]}
    vars_ = [("base", [])] + [(v.split("=", 1)[0], v.split("=", 1)[1].split()) for v in a.var]
    for pn, pf in (p.split("=", 1) for p in a.prompt):
        for gb in a.gbs.split(","):
            for sn in a.sets.split(","):
                for vn, vf in vars_:
                    name = f"{sn}_{vn}_{pn}_gb{gb}"
                    js, lg = f"{a.out}/{name}.json", f"{a.out}/{name}.preads"
                    if os.path.exists(f"{a.out}/{name}.split"):
                        continue
                    cmd = [f"{a.shim}/preadrun", f"{a.shim}/preadlog.dylib", a.dmeng, "--model", a.model, *sets[sn], "--prompt-file", pf,
                           "--n", a.n, "--cache-gb", gb, "--temp", "0", *vf, "--json", js, "--out-ids", f"{a.out}/{name}.ids"]
                    env = dict(os.environ, DISK_GB="8", FOOT_GB="13.5", SWAP_MB="2000", POLL_S="2", PREADLOG=lg)
                    g = subprocess.run(["python3", a.guard, f"{a.out}/{name}.log", "2400", "--", *cmd], capture_output=True, text=True, env=env).stdout
                    gl = [l for l in g.splitlines() if l.startswith("GUARD ")]
                    gd = json.loads(gl[-1][6:]) if gl else {}
                    if not os.path.exists(js) or not os.path.exists(lg):
                        print(f"{name}: FAILED {gd.get('killed_by_guard')} rc={gd.get('rc')}", flush=True)
                        continue
                    r = json.load(open(js)); r["guard"] = gd; json.dump(r, open(js, "w"))
                    sp = subprocess.run(["/usr/bin/python3", f"{H}/readsplit.py", lg, js], capture_output=True, text=True)
                    if sp.returncode:
                        print(f"{name}: split failed {sp.stderr[-300:]}", flush=True)
                        continue
                    open(f"{a.out}/{name}.split", "w").write(sp.stdout)
                    for f in (lg, lg + ".files"):
                        os.remove(f)
                    s = json.loads(sp.stdout)
                    print(f"{name}: {s['mb_per_token']} total {s['mb_per_token_total_log']} (engine {s['mb_per_token_engine']}) | "
                          f"tok/s {r.get('tok_s', 0):.2f} | others {gd.get('others_cpu_pct_mean')}", flush=True)


def table(out):
    print("| set | variant | prompt | cache GB | MB/token looped | 20-23 | rest | total (engine) | reads/pick looped / 20-23 / rest | hit looped / other | looped re-read tokens | hold releases | tok/s Mac | others CPU |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{out}/*.split")):
        n = os.path.basename(f)[:-6]
        sn, vn, pn, gb = n.split("_", 3)
        s = json.loads(open(f).read())
        r = json.load(open(f[:-6] + ".json"))
        m, rp = s["mb_per_token"], s["reads_per_pick"]
        print(f"| {sn} | {vn} | {pn} | {gb[2:]} | {m['looped']} | {m['late']} | {m['rest']} | {s['mb_per_token_total_log']} ({s['mb_per_token_engine']}) | "
              f"{rp['looped']} / {rp['late']} / {rp['rest']} | {s['hit_rate_looped']:.3f} / {s['hit_rate_other']:.3f} | {s['looped_reread_token_share']} | "
              f"{s['hold_releases']} | {r.get('tok_s', 0):.2f} | {(r.get('guard') or {}).get('others_cpu_pct_mean')} |")


if __name__ == "__main__":
    main()
