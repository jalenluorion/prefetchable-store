"""Inkling's draft head (src/models/inkling2_loop/ink_draft.cpp): the port check, the exactness gate and the timing campaign.
Every model process goes through the research tree's guard (one model process at a time); before each run the machine
rules are checked: battery >= 30%, or >= 20% and charging; no other job's QUIET file fresher than 12 minutes; no
PRIORITY file fresher than 30 minutes. Resumable: a run whose JSON exists is skipped.

  python3 bench/ink_draft/run.py port      # gate 1: C++ head vs the MLX reference on recorded features (pitt, code)
  python3 bench/ink_draft/run.py exact     # gate 2: ids with the head on = one-position decoding (pitt, code, edit, tool)
  python3 bench/ink_draft/run.py timing    # off / lookup / head / both, 6 and 7.8 GB, two runs per cell, interleaved
  python3 bench/ink_draft/run.py table     # results/ink_draft/ tables from the runs

Paths (environment): MODEL (default ../../../inkling/model from the repo), DM_GUARD (../../../../engine/bench/guard.py),
INK_PY (the MLX venv's python, ../../../inkling/.venv/bin/python), FEATS (recorded features and ids,
../../../../engine/results/inkling_mtp; the port check only), BIN (build/dmeng). Prompts: bench/prompts/inkling2_loop/ (pitt,
code) and results/ink_draft/prompts/ (edit, tool, newfile).
"""
import glob, json, os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
up = lambda *p: os.path.normpath(os.path.join(REPO, *p))
MODEL = os.environ.get("MODEL", up("..", "..", "..", "inkling", "model"))
GUARD = os.environ.get("DM_GUARD", up("..", "..", "..", "..", "engine", "bench", "guard.py"))
ENG = os.path.dirname(os.path.dirname(GUARD))
INK_PY = os.environ.get("INK_PY", up("..", "..", "..", "inkling", ".venv", "bin", "python"))
FEATS = os.environ.get("FEATS", up("..", "..", "..", "..", "engine", "results", "inkling_mtp"))
BIN = os.environ.get("BIN", os.path.join(REPO, "build", "dmeng"))
OUT = os.path.join(REPO, "results", "ink_draft")
RUNS = os.path.join(OUT, "runs")
SCR = os.environ.get("SCRATCH", os.path.join(REPO, "build", "ink_draft"))  # big outputs (logits), not committed
QUIET = os.path.join(ENG, "bench", "QUIET")
PRIORITY = os.path.join(ENG, "bench", "PRIORITY")
for d in (RUNS, SCR):
    os.makedirs(d, exist_ok=True)


def batt():
    s = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%;\s*([^;]+);", s)
    pct, state = (int(m.group(1)), m.group(2).strip()) if m else (0, "unknown")
    src = "AC" if "AC Power" in s else "battery"
    return pct >= 30 or (pct >= 20 and state == "charging"), f"{src} {pct}% {state}"


def wait_rules():
    while True:
        ok, txt = batt()
        if not ok:
            print(f"power {txt}: waiting", flush=True)
            time.sleep(60)
            continue
        now = time.time()
        if os.path.exists(QUIET) and now - os.path.getmtime(QUIET) < 720:
            time.sleep(20)
            continue
        if os.path.exists(PRIORITY) and now - os.path.getmtime(PRIORITY) < 1800:
            time.sleep(30)
            continue
        return txt


def guarded(name, cmd, timeout=1800, stdout_json=False):
    js = os.path.join(RUNS, name + ".json")
    if os.path.exists(js):
        return json.load(open(js))
    power = wait_rules()
    log = os.path.join(RUNS, name + ".log")
    r = subprocess.run(["python3", GUARD, log, str(timeout), "--", *cmd], capture_output=True, text=True, env=dict(os.environ, POLL_S="10"))
    g = json.loads(r.stdout.strip().splitlines()[-1][6:])
    if g["rc"] != 0 or g["killed_by_guard"]:
        print(f"{name}: FAILED rc {g['rc']} {g['killed_by_guard']} (log {log})", flush=True)
        return None
    if stdout_json:
        res = None
        for l in open(log):
            if l.startswith("{"):
                res = json.loads(l)
        if res is None:
            return None
    else:
        res = json.load(open(js)) if os.path.exists(js) else None
        if res is None:
            return None
    res["guard"] = {k: g.get(k) for k in ("wall_s", "waited_s", "peak_footprint_gb_time", "others_cpu_pct_mean")}
    res["power"] = [power, g["after"]["power"] + " " + g["after"]["battery"].split("\t")[-1].split(";")[0] if g.get("after") else "?"]
    json.dump(res, open(js, "w"))
    return res


# ------------------------------------------------------------------------------------------------ gate 1: the port
TEXTS = {"pitt": ("taps_pitt_seq.bin", "pitt_seq_ids.txt", 9, "accept1_pitt_seq.json"),
         "code": ("taps_code_seq.bin", "code_seq_ids.txt", 461, "accept1_code_seq.json")}


def port():
    for t, (taps, ids, start, acc) in TEXTS.items():
        args = ["--taps", os.path.join(FEATS, taps), "--ids", os.path.join(FEATS, ids), "--start", str(start)]
        guarded(f"port_{t}_ref", [INK_PY, os.path.join(REPO, "tests", "inkling2_loop", "draft_ref_mlx_run.py"), "--model", MODEL, *args,
                                  "--out", os.path.join(SCR, f"{t}_ref"), "--check-accept", os.path.join(FEATS, acc)], stdout_json=True)
        for ex in (1, 0):
            guarded(f"port_{t}_cpp_exact{ex}", [BIN, "--model", MODEL, "--cache-gb", "1.5", "--mem", "plain", "--ctx", "2048",
                                                "--draft-check", os.path.join(FEATS, taps), "--prompt-file", os.path.join(FEATS, ids),
                                                "--draft-check-start", str(start), "--draft-exact", str(ex), "--draft-check-out",
                                                os.path.join(SCR, f"{t}_cpp{ex}"), "--json", os.path.join(RUNS, f"port_{t}_cpp_exact{ex}.json")])
        for ex in (1, 0):
            r = subprocess.run([INK_PY, os.path.join(REPO, "tests", "inkling2_loop", "draft_compare.py"), os.path.join(SCR, f"{t}_cpp{ex}"),
                                os.path.join(SCR, f"{t}_ref")], capture_output=True, text=True)
            open(os.path.join(RUNS, f"port_{t}_compare_exact{ex}.json"), "w").write(r.stdout)
            c = json.loads(r.stdout)
            print(t, "exact" if ex else "decoding", {k: c[k] for k in ("positions", "draft1_equal", "all7_equal", "draft1_logit_max_abs_diff",
                                                                         "draft1_disagreements_near_tie")}, flush=True)


# ------------------------------------------------------------------------------------------------ decoding runs
def prompt_file(p):  # pitt / code: the repo's standard prompts; edit / tool (engine/ink/make_draft_prompts.py) and newfile
    return {"pitt": os.path.join(REPO, "bench", "prompts", "inkling2_loop", "pitt.ids"),          # (make_prompts.py) copied here
            "code": os.path.join(REPO, "bench", "prompts", "inkling2_loop", "code.ids")}.get(p, os.path.join(OUT, "prompts", f"{p}_ids.txt"))


ROUTERS = ["--prefetch", "router", "--routers", os.path.join(MODEL, "routers", "ink_r_fast", "ink_router_full_k2_h512_b8.dmr"), "--rt-tau", "0.5"]
MODES = {
    "off": [],
    "lookup": ["--draft", "lookup"],
    "head": ["--draft", "head"],
    "both": ["--draft", "both"],
}


NGEN = {"pitt": 400, "code": 400, "edit": 400, "tool": 640, "newfile": 640}
COMMON = ["--prefill-chunk", "64"]  # the prompt through the batched prefill (exact; keeps the runs short)


def run(prompt, mode, gb, tag, n=None, extra=(), cache_gb=None):
    name = f"{prompt}_{mode}_gb{gb}_{tag}"
    n = n or NGEN[prompt]
    cmd = [BIN, "--model", MODEL, "--cache-gb", str(cache_gb or gb), "--mem", "mlock", "--prompt-file", prompt_file(prompt), "--n", str(n), *COMMON,
           "--out-ids", os.path.join(RUNS, name + ".ids"), "--json", os.path.join(RUNS, name + ".json"),
           "--steps-out", os.path.join(RUNS, name + ".steps"), *MODES[mode], *extra]
    r = guarded(name, cmd)
    if r:
        d = r.get("draft", {})
        print(f"{name}: {r['tok_s']:.2f} tok/s, {d.get('tokens_per_step', 1):.2f} tok/step, wait {r['wait_ms_per_token']:.1f} ms/tok, "
              f"{r['mb_read_per_token']:.0f} MB/tok, foot {r['footprint_peak_gb']:.2f} GB, {r['power']}, others {r['guard']['others_cpu_pct_mean']}%",
              flush=True)
    return r


def ids_of(name):
    p = os.path.join(RUNS, name + ".ids")
    return open(p).read().strip().split(",") if os.path.exists(p) else None


EXACT_PROMPTS = ("pitt", "code", "edit", "tool")
EXACT_N = int(os.environ.get("EXACT_N", 200))


def exact():
    for p in EXACT_PROMPTS:
        run(p, "off", 6, "x", n=EXACT_N, extra=ROUTERS)
        for m in ("head", "both"):
            run(p, m, 6, "x", n=EXACT_N, extra=ROUTERS)
        base = ids_of(f"{p}_off_gb6_x")
        for m in ("head", "both"):
            mine = ids_of(f"{p}_{m}_gb6_x")
            same = base is not None and mine == base
            first = None if same or not base or not mine else next((i for i, (a, b) in enumerate(zip(mine, base)) if a != b), min(len(mine), len(base)))
            print(f"{p} {m}: {len(mine or [])} ids, identical to one-position: {same}{'' if same else f' (first difference at {first})'}", flush=True)


TIMING_PROMPTS = ("pitt", "code", "edit", "tool", "newfile")
HEAD_CUT = {7.8: 7.05}


def timing():
    for gb in (6, 7.8):
        for p in TIMING_PROMPTS:
            for r, order in (("t1", ("off", "lookup", "head", "both")), ("t2", ("both", "head", "lookup", "off"))):
                for m in order:
                    # at 7.8 GB the head's 0.73 GB (weights + state) would take the footprint over 9.5 GB: its cells run with
                    # the slots cut by that much (the same total memory as drafting off)
                    run(p, m, gb, r, extra=ROUTERS, cache_gb=HEAD_CUT.get(gb) if m in ("head", "both") else None)


def table():
    cells = {}
    for js in sorted(glob.glob(os.path.join(RUNS, "*_gb*_t?.json"))):
        p, m, gb, t = os.path.basename(js)[:-5].split("_")
        cells.setdefault((p, gb[2:], m), []).append(json.load(open(js)))
    print("| prompt | cache GB | drafting | runs | tok/s (each) | vs off | tokens/step | wait ms / token | MB read / token | footprint GB | power (start) | others' CPU % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (p, gb, m), rs in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0], list(MODES).index(x[0][2]))):
        off = cells.get((p, gb, "off"))
        t = sum(r["tok_s"] for r in rs) / len(rs)
        base = sum(r["tok_s"] for r in off) / len(off) if off else float("nan")
        tps = sum(r.get("draft", {}).get("tokens_per_step", 1.0) for r in rs) / len(rs)
        print(f"| {p} | {gb} | {m} | {len(rs)} | {t:.2f} ({', '.join(f'{r['tok_s']:.2f}' for r in rs)}) | {t / base:.3f} | {tps:.2f} | "
              f"{sum(r['wait_ms_per_token'] for r in rs) / len(rs):.1f} | {sum(r['mb_read_per_token'] for r in rs) / len(rs):.0f} | "
              f"{max(r['footprint_peak_gb'] for r in rs):.2f} | {'; '.join(r['power'][0] for r in rs)} | "
              f"{', '.join(str(round(r['guard']['others_cpu_pct_mean'] or 0)) for r in rs)} |")


def bench(tag="b1"):
    r = guarded(f"kernel_bench_{tag}", [BIN, "--draft-bench", "20"], timeout=600, stdout_json=True)
    print(json.dumps(r), flush=True)


def smoke(prompt="code", mode="both", n="80"):
    r = run(prompt, mode, 6, "smoke", n=int(n), extra=ROUTERS)
    print(json.dumps({k: r.get(k) for k in ("draft", "model", "footprint_peak_gb")} if r else None), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gates_and_timing":
        exact()
        timing()
        sys.exit(0)
    if sys.argv[1] == "smoke":
        smoke(*sys.argv[2:])
        sys.exit(0)
    {"port": port, "exact": exact, "timing": timing, "table": table, "bench": lambda: bench(*sys.argv[2:])}[sys.argv[1]]()
