#!/usr/bin/env python3
"""The draft controller's decisive timing (results/drafts/DRAFT-CTL.md): off vs drafts (the measured gate), two interleaved
runs per cell, each through engine/bench/guard.py with campaign.py's power / QUIET / PRIORITY rules; resumable.
  35B at 5 and 8 GB, Inkling at 6 GB; file rewrite (toolrw), new file (toolnew), coding answer (code).
  Routers: the 35B's linear k2 (--m 8), Inkling's ink_r_best k2 (--rt-tau 0.5). Drafts: 35B --draft lookup, Inkling
  --draft both (lookup + its draft head); --early 4, the tool fragments on the tool cases, --verify-names 0.
  python3 ctl_timing.py [run] [table]"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign as C

C.RUNS = os.path.join(C.OUT, "ctl_runs")
os.makedirs(C.RUNS, exist_ok=True)
C.ROUTERS["q36"] = ["--prefetch", "router", "--routers", os.path.join(C.Q36, "routers", "lin", "q36_router_lin_k2.dmr"), "--m", "8"]
C.BASE["ink"] = ["--model", C.INK, "--resident", os.path.join(C.ENG, "build_ink", "resident_real_b8.bin"), "--prefill-chunk", "64"]
CELLS = [("q36", 5.0), ("q36", 8.0), ("ink", 6.0)]

# two guards that poll at the same moment both see a free slot and start together (it happened twice in this campaign):
# start only after the slot has been free on three checks 4 s apart
import re, subprocess, time
_wait0 = C.wait_turn


def model_busy():
    ps = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True).stdout
    return any(re.search(r"/(dmeng|dmserve|dmink|bmoe-cli|llama-[a-z-]+)( |$)", l) and "guard.py" not in l for l in ps.splitlines())


def wait_turn():
    while True:
        _wait0()
        free = 0
        while free < 3:
            if model_busy():
                free = 0
                time.sleep(10)
            else:
                free += 1
                time.sleep(4)
        if not model_busy():
            return


C.wait_turn = wait_turn
CASES = os.environ.get("CASES", "toolrw,toolnew,code").split(",")
N = {"toolrw": 800, "toolnew": 800, "code": 300}


def cmd(model, case, gb, mode, name):
    c = [C.DMENG] + C.BASE[model] + ["--prompt-file", os.path.join(C.PR, f"{model}_{case}_ids.txt"), "--n", str(N[case]), "--cache-gb", str(gb),
                                     "--ctx", "16384", "--temp", "0", "--mem", "mlock", "--stop-eos", str(C.EOS[model])] + C.ROUTERS[model]
    if mode == "on":
        c += ["--draft", "lookup" if model == "q36" else "both", "--early", "4", "--verify-names", "0"]
        sk = os.path.join(C.PR, f"{model}_{case}_skel.txt")
        if os.path.exists(sk):
            c += ["--skeleton", sk]
    return c + ["--json", os.path.join(C.RUNS, name + ".json.tmp"), "--out-ids", os.path.join(C.RUNS, name + ".ids")]


def run():
    for model, gb in CELLS:
        for case in CASES:
            for rep in (1, 2):
                for mode in ("off", "on"):
                    name = f"{model}_{case}_gb{gb}_{mode}_r{rep}"
                    C.run(name, cmd(model, case, gb, mode, name))
                    if mode == "on":  # the ids gate on every pair
                        a, b = (open(os.path.join(C.RUNS, f"{model}_{case}_gb{gb}_{m}_r{rep}.ids")).read() for m in ("off", "on"))
                        print("IDS", name, "IDENTICAL" if a == b else "DIFFER", flush=True)
                        if a != b:
                            sys.exit("ids differ: stopping")


REDO_PCT = float(os.environ.get("REDO_PCT", 140))


def others(name):
    m = os.path.join(C.RUNS, name + ".meta.json")
    return ((json.load(open(m)).get("guard") or {}).get("others_cpu_pct_mean") or 0) if os.path.exists(m) else 0


def redo():
    """a pair where either run saw others' CPU above REDO_PCT is set aside (*.contended*) and run again, up to twice"""
    for model, gb in CELLS:
        for case in CASES:
            for rep in (1, 2):
                for attempt in range(2):
                    names = [f"{model}_{case}_gb{gb}_{m}_r{rep}" for m in ("off", "on")]
                    if not all(os.path.exists(os.path.join(C.RUNS, n + ".json")) for n in names):
                        break
                    if max(others(n) for n in names) <= REDO_PCT:
                        break
                    for n in names:
                        for ext in (".json", ".meta.json", ".ids", ".log"):
                            f = os.path.join(C.RUNS, n + ext)
                            if os.path.exists(f):
                                os.replace(f, os.path.join(C.RUNS, f"{n}.contended{attempt}{ext}"))
                    for m, n in zip(("off", "on"), names):
                        C.run(n, cmd(model, case, gb, m, n))
                    a, b = (open(os.path.join(C.RUNS, n + ".ids")).read() for n in names)
                    print("IDS", names[1], "IDENTICAL" if a == b else "DIFFER", flush=True)


def table():
    rows = ["| model | GB | case | rep | off tok/s | drafts tok/s | on/off | tokens/step | wait ms/tok off / on | MB/tok off / on | verify passes | peak GB on | power | others CPU % off / on |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, gb in CELLS:
        for case in CASES:
            for rep in (1, 2):
                r = {}
                for m in ("off", "on"):
                    n = f"{model}_{case}_gb{gb}_{m}_r{rep}"
                    p = os.path.join(C.RUNS, n + ".json")
                    if os.path.exists(p):
                        r[m] = (json.load(open(p)), json.load(open(os.path.join(C.RUNS, n + ".meta.json"))))
                if len(r) < 2:
                    continue
                (a, am), (b, bm) = r["off"], r["on"]
                d = b.get("draft", {})
                oc = lambda m: (m.get("guard") or {}).get("others_cpu_pct_mean")
                rows.append(f"| {model} | {gb:g} | {case} | {rep} | {a['tok_s']:.2f} | {b['tok_s']:.2f} | {b['tok_s'] / a['tok_s']:.3f} | {d.get('tokens_per_step', 0):.2f} | "
                            f"{a['wait_ms_per_token']:.1f} / {b['wait_ms_per_token']:.1f} | {a['mb_read_per_token']:.0f} / {b['mb_read_per_token']:.0f} | {d.get('verifies')} | "
                            f"{b['footprint_peak_gb']:.2f} | {bm['power_before'][2]} {bm['power_before'][0]}% | {oc(am)} / {oc(bm)} |")
    open(os.path.join(C.OUT, "ctl_table.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    for s in sys.argv[1:] or ["run", "redo", "table"]:
        {"run": run, "redo": redo, "table": table}[s]()
