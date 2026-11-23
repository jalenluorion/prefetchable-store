#!/usr/bin/env python3
"""Draft-and-verify gates and timing for both models (results/drafts/runs/), every model run through engine/bench/guard.py.
Resumable: a run whose JSON exists is skipped. Before each run: battery >= 30%, or >= 20% and charging; no fresh
engine/bench/QUIET (< 12 min) and no fresh engine/bench/PRIORITY (< 30 min); 20 s after the previous run. Each run's
power state and the others' CPU (guard) go into runs/NAME.meta.json.

  python3 campaign.py check     # real-model verify bit-check (both models, routers on) + the batched prefill check (35B)
  python3 campaign.py gates     # edit prompt off / on, the fixed-draft stress (every step the whole draft) on the rewrite
  python3 campaign.py timing    # 2 models x 2 caches x 5 cases x off / on x 2 reps, interleaved per cell
  python3 campaign.py server    # the server worker, greedy, drafting off / on: the 35B's recorded OpenCode requests, Inkling's tool turns
  python3 campaign.py table     # results/drafts/table.md (+ ids comparison of every off / on pair)
ENV: REPS (default 1,2), CASES, MODELS, GBS (e.g. "q36:5,6.6;ink:6,7.8")"""
import glob, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DM = os.path.abspath(os.path.join(REPO, "..", "..", "..", ".."))
ENG = os.path.join(DM, "engine")
GUARD = os.path.join(ENG, "bench", "guard.py")
QUIET, PRIORITY = os.path.join(ENG, "bench", "QUIET"), os.path.join(ENG, "bench", "PRIORITY")
OUT = os.path.join(REPO, "results", "drafts")
RUNS = os.path.join(OUT, "runs")
PR = os.path.join(OUT, "prompts")
os.makedirs(RUNS, exist_ok=True)
DMENG, DMSERVE = os.path.join(REPO, "build", "dmeng"), os.path.join(REPO, "build", "dmserve")
Q36, INK = os.path.join(DM, "local", "q36", "model"), os.path.join(DM, "local", "inkling", "model")
BASE = {"q36": ["--model", Q36, "--prefill-chunk", "512"],
        "ink": ["--model", INK, "--resident", os.path.join(ENG, "build_ink", "resident_real_b8.bin"), "--prefill-chunk", "256"]}
ROUTERS = {"q36": ["--prefetch", "router", "--routers", os.path.join(Q36, "routers", "q36_router_full_k2_h512.dmr"), "--m", "8"],
           "ink": ["--prefetch", "router", "--routers", os.path.join(INK, "routers", "ink_r_best", "ink_router_full_k2_h512_b8.dmr"), "--rt-tau", "0.5"]}
EOS = {"q36": 248046, "ink": 200006}
N = {"prose": 200, "code": 300, "edit": 400, "toolnew": 800, "toolrw": 800, "toolshort": 120}
GATE_N = int(os.environ.get("GATE_N", 200))  # the ids gates: short runs
CASES = os.environ.get("CASES", "prose,code,toolnew,toolrw,toolshort").split(",")
MODELS = os.environ.get("MODELS", "q36,ink").split(",")
GBS = {m: [float(x) for x in g.split(",")] for m, g in (p.split(":") for p in os.environ.get("GBS", "q36:5,6.6;ink:6,7.8").split(";"))}
REPS = [int(x) for x in os.environ.get("REPS", "1,2").split(",")]
EARLY = os.environ.get("EARLY", "4")


def power():
    s = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%;\s*([^;]+);", s)
    pct, st = (int(m.group(1)), m.group(2).strip()) if m else (0, "?")
    src = "AC" if "AC Power" in s else "battery"
    return pct, st, src


def power_ok():
    pct, st, src = power()
    return pct >= 30 or (pct >= 20 and st == "charging")


def fresh(path, s):
    try:
        return time.time() - os.path.getmtime(path) < s
    except OSError:
        return False


LAST = [0.0]


def wait_turn():
    while True:
        if not power_ok():
            print(time.strftime("%H:%M:%S"), "waiting: power", power(), flush=True)
            time.sleep(120)
            continue
        if fresh(QUIET, 720) or fresh(PRIORITY, 1800):
            print(time.strftime("%H:%M:%S"), "waiting: QUIET / PRIORITY", flush=True)
            time.sleep(60)
            continue
        gap = 20 - (time.time() - LAST[0])
        if gap > 0:
            time.sleep(gap)
            continue
        return


def run(name, cmd, timeout=3600, js=True):
    """one guarded run; NAME.json is the command's JSON (dmeng --json) or its last stdout line"""
    out = os.path.join(RUNS, name + ".json")
    if os.path.exists(out):
        return json.load(open(out))
    wait_turn()
    log = os.path.join(RUNS, name + ".log")
    pw0 = power()
    t0 = time.time()
    print(time.strftime("%H:%M:%S"), "run", name, "power", pw0, flush=True)
    r = subprocess.run(["python3", GUARD, log, str(timeout), "--"] + cmd, capture_output=True, text=True)
    LAST[0] = time.time()
    pw1 = power()
    txt = open(log).read() if os.path.exists(log) else ""
    g = [l for l in txt.splitlines() if l.startswith("GUARD")]
    meta = {"name": name, "cmd": cmd, "rc": r.returncode, "power_before": pw0, "power_after": pw1, "wall_s": round(time.time() - t0, 1),
            "guard": json.loads(g[-1][len("GUARD"):].strip(" :")) if g else None, "started": time.ctime(t0), "stderr": r.stderr[-2000:]}
    json.dump(meta, open(os.path.join(RUNS, name + ".meta.json"), "w"), indent=1)
    res = None
    if js and os.path.exists(out + ".tmp"):
        try:
            res = json.load(open(out + ".tmp"))
        except Exception:
            res = None
        os.replace(out + ".tmp", out) if res is not None else None
    if res is None:
        lines = [l for l in txt.splitlines() if l.startswith("{")]
        try:
            res = json.loads(lines[-1]) if lines else None
        except Exception:
            res = None
        if res is not None:
            json.dump(res, open(out, "w"))
    print(time.strftime("%H:%M:%S"), "done", name, "rc", r.returncode, flush=True)
    return res


def gen(model, case, gb, mode, rep, extra=(), n=None, tag=""):
    name = f"{model}_{case}_gb{gb}_{mode}_{tag}r{rep}"
    cmd = [DMENG] + BASE[model] + ["--prompt-file", os.path.join(PR, f"{model}_{case}_ids.txt"), "--n", str(n or N[case]), "--cache-gb", str(gb),
                                   "--ctx", "16384", "--temp", "0", "--mem", "mlock", "--stop-eos", str(EOS[model])] + ROUTERS[model]
    if mode != "off":
        cmd += ["--draft", "fixed" if mode == "fixed" else "lookup"]
        if mode != "fixed" and EARLY != "0":
            cmd += ["--early", EARLY]
        sk = os.path.join(PR, f"{model}_{case}_skel.txt")
        if os.path.exists(sk):
            cmd += ["--skeleton", sk]
    cmd += list(extra) + ["--json", os.path.join(RUNS, name + ".json.tmp"), "--out-ids", os.path.join(RUNS, name + ".ids")]
    return run(name, cmd)


def check():
    for model, gb, case in (("q36", 5.0, "code"), ("ink", 6.0, "code")):
        name = f"{model}_vcheck_{case}_gb{gb}"
        cmd = [DMENG] + BASE[model] + ["--prompt-file", os.path.join(PR, f"{model}_{case}_ids.txt"), "--n", "8", "--cache-gb", str(gb), "--ctx", "16384",
                                       "--temp", "0", "--mem", "mlock"] + ROUTERS[model] + ["--verify-check", "200", "--json", os.path.join(RUNS, name + ".json.tmp")]
        run(name, cmd)
    ids = os.path.join(ENG, "results", "agentic", "check", "ids_1100.txt")
    run("q36_prefill_check_1100", [DMSERVE, "--model", Q36, "--cache-gb", "5", "--mem", "mlock", "--ctx", "2048", "--chunk", "256", "--span", "8192",
                                   "--check", ids, "--gen", "16"], js=False)


def gates():
    # ids with drafting on (lookup + early names + skeleton, routers on) equal off, every case, GATE_N tokens; and the whole
    # draft every step (most drafts rejected part way: rollback hundreds of times) on the rewrite and the edit
    for model in MODELS:
        gb = GBS[model][0]
        for case in ("prose", "code", "edit", "toolrw", "toolnew", "toolshort"):
            for mode in ("off", "on"):
                gen(model, case, gb, mode, 1, n=GATE_N, tag="g")
        for case in ("toolrw", "edit"):
            gen(model, case, gb, "fixed", 1, n=GATE_N, tag="g")


def timing():
    for model in MODELS:
        for gb in GBS[model]:
            for case in CASES:
                for rep in REPS:
                    for mode in ("off", "on"):
                        gen(model, case, gb, mode, rep)


def server():
    # the 35B: the recorded OpenCode requests (front end's tokenisation, tool fragments), greedy; Inkling: its tool turns
    reqs = {"q36": os.path.join(PR, "q36_server_reqs.json"), "ink": os.path.join(PR, "ink_server_reqs.json")}
    for model in MODELS:
        gb = GBS[model][0]
        wargs = (["--model", Q36] if model == "q36" else ["--model", INK, "--resident", os.path.join(ENG, "build_ink", "resident_real_b8.bin")])
        wargs += ["--cache-gb", str(gb), "--mem", "mlock", "--ctx", "16384", "--routers", ROUTERS[model][3]]
        if model == "q36":
            wargs += ["--m", "8"]
        for mode in ("off", "on"):
            name = f"{model}_server_gb{gb}_{mode}"
            d = ["--draft", "lookup", "--early", EARLY] if mode == "on" else []
            run(name, ["python3", os.path.join(HERE, "worker_drive.py"), os.path.join(RUNS, name + ".json.tmp"), reqs[model], "--", DMSERVE] + wargs + d, timeout=5400)


def replay():
    # the stock path (sampling, as OpenCode sends it): the unified-wip worker and front end vs this branch's, drafting off
    # and on (on must change nothing: sampled requests do not draft); first requests, seed 1, 64 tokens each
    main = os.path.join(DM, "local", "flash", "github", "Qwen3.8-Flash-Next-moved-store")
    wa = ["--cache-gb", "5", "--mem", "mlock", "--chunk", "512", "--span", "8192", "--stream", "3"]
    for name, repo, extra in (("replay_base", main, []), ("replay_off", REPO, []), ("replay_on", REPO, ["--draft", "lookup", "--early", EARLY])):
        if os.path.exists(os.path.join(RUNS, name + ".server.jsonl")) and os.path.exists(os.path.join(RUNS, name + ".meta.json")):
            continue
        run(name, ["zsh", os.path.join(HERE, "replay.sh"), repo, os.path.join(repo, "build", "dmserve"), os.path.join(RUNS, name), "64", "6"] + wa + extra,
            timeout=3600, js=False)

    def reqs(n):
        try:
            return [r for r in map(json.loads, open(os.path.join(RUNS, n + ".server.jsonl"))) if r.get("event") == "request"]
        except OSError:
            return []
    A = reqs("replay_base")
    for n in ("replay_off", "replay_on"):
        B = reqs(n)
        same = bool(A) and len(A) == len(B) and all(a.get("prompt_ids") == b.get("prompt_ids") and a.get("gen_ids") == b.get("gen_ids") for a, b in zip(A, B))
        print(f"REPLAY {n} vs base: {len(A)} / {len(B)} requests, ids", "IDENTICAL" if same else "DIFFER")


def load(name):
    p = os.path.join(RUNS, name + ".json")
    m = os.path.join(RUNS, name + ".meta.json")
    return (json.load(open(p)) if os.path.exists(p) else None), (json.load(open(m)) if os.path.exists(m) else None)


def ids_of(name):
    p = os.path.join(RUNS, name + ".ids")
    return open(p).read().strip() if os.path.exists(p) else None


def table():
    rows = ["| model | cache GB | case | mode | rep | tokens | tok/s | tokens/step | wait ms/token | MB read/token | footprint peak GB | power | others CPU % | ids = off |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(RUNS, "*r[0-9].json"))):
        name = os.path.basename(f)[:-5]
        model, case, gbs, mode, rep = name.split("_")
        tag = rep[:-2] if len(rep) > 2 else ""
        r, m = load(name)
        if not r or "tok_s" not in r:
            continue
        off = ids_of(f"{model}_{case}_{gbs}_off_{rep}")
        same = "-" if mode == "off" else ("yes" if ids_of(name) == off else "NO" if off else "?")
        d = r.get("draft", {})
        pw = m["power_before"] if m else ["?", "?", "?"]
        oc = (m.get("guard") or {}).get("others_cpu_pct_mean") if m else None
        rows.append(f"| {model} | {gbs[2:]} | {case} | {mode} | {rep} | {r['tokens']} | {r['tok_s']:.2f} | {d.get('tokens_per_step', 1.0):.2f} | "
                    f"{r['wait_ms_per_token']:.1f} | {r['mb_read_per_token']:.1f} | {r['footprint_peak_gb']:.2f} | {pw[2]} {pw[0]}% {pw[1]} | {oc} | {same} |")
    open(os.path.join(OUT, "table.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"check": check, "gates": gates, "timing": timing, "server": server, "replay": replay, "table": table}[step]()
