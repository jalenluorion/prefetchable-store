"""Names inside verify passes (Core::chunk_start, Hook::chunk_*): the exactness gate and the read counts, both models.
Every model run through the research tree's guard (one model process at a time); before each run: battery >= 30% or
>= 20% and charging, no QUIET fresher than 12 minutes, no PRIORITY fresher than 30. Resumable (a run whose JSON exists
is skipped).

  python3 bench/verify_names/run.py gates     # per model x prompt (pitt, code, edit, tool): off, drafts names off, names on
  python3 bench/verify_names/run.py table     # ids identical? and the read counts per accepted token / per verify pass
  python3 bench/verify_names/run.py timing    # the clean campaign's cells (not run here)

Environment: INK (the Inkling model folder, default ../../../inkling/model), Q36 (../../../q36/model), DM_GUARD
(../../../../engine/bench/guard.py), BIN (build/dmeng), N (200).
"""
import glob, json, os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
up = lambda *p: os.path.normpath(os.path.join(REPO, *p))
INK = os.environ.get("INK", up("..", "..", "..", "inkling", "model"))
Q36 = os.environ.get("Q36", up("..", "..", "..", "q36", "model"))
GUARD = os.environ.get("DM_GUARD", up("..", "..", "..", "..", "engine", "bench", "guard.py"))
ENG = os.path.dirname(os.path.dirname(GUARD))
BIN = os.environ.get("BIN", os.path.join(REPO, "build", "dmeng"))
N = int(os.environ.get("N", 200))
OUT = os.path.join(REPO, "results", "verify_names")
RUNS = os.path.join(OUT, "runs")
os.makedirs(RUNS, exist_ok=True)
QUIET, PRIORITY = os.path.join(ENG, "bench", "QUIET"), os.path.join(ENG, "bench", "PRIORITY")
P = os.path.join(REPO, "results", "drafts", "prompts")

MODELS = {
    "ink": {"dir": INK, "gb": 6, "base": ["--prefill-chunk", "64", "--stop-eos", "200006", "--prefetch", "router", "--routers",
                                         os.path.join(INK, "routers", "ink_r_best", "ink_router_full_k2_h512_b8.dmr"), "--rt-tau", "0.5"],
            "draft": ["--draft", "both", "--early", "4"]},  # lookup + the draft head (the head's drafts have no source span)
    "q36": {"dir": Q36, "gb": 5, "base": ["--prefill-chunk", "512", "--stop-eos", "248046", "--prefetch", "router", "--routers",
                                         os.path.join(Q36, "routers", "q36_router_full_k2_h512.dmr"), "--m", "8"],
            "draft": ["--draft", "lookup", "--early", "4"]},
}
FAM = {"ink": "inkling2_loop", "q36": "qwen3_5_moe"}


def prompt(m, p):
    if p in ("pitt", "code"):
        return os.path.join(REPO, "bench", "prompts", FAM[m], p + ".ids")
    return os.path.join(P, f"{m}_{ {'tool': 'toolrw'}.get(p, p) }_ids.txt")


def batt():
    s = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    mm = re.search(r"(\d+)%;\s*([^;]+);", s)
    pct, state = (int(mm.group(1)), mm.group(2).strip()) if mm else (0, "unknown")
    return pct >= 30 or (pct >= 20 and state == "charging"), f"{'AC' if 'AC Power' in s else 'battery'} {pct}% {state}"


def wait_rules():
    while True:
        ok, txt = batt()
        now = time.time()
        if not ok:
            time.sleep(60)
        elif os.path.exists(QUIET) and now - os.path.getmtime(QUIET) < 720:
            time.sleep(20)
        elif os.path.exists(PRIORITY) and now - os.path.getmtime(PRIORITY) < 1800:
            time.sleep(30)
        else:
            return txt


def run(m, p, mode, n=N):
    name = f"{m}_{p}_{mode}"
    js = os.path.join(RUNS, name + ".json")
    if os.path.exists(js):
        return json.load(open(js))
    M = MODELS[m]
    cmd = [BIN, "--model", M["dir"], "--cache-gb", str(M["gb"]), "--mem", "mlock", "--ctx", "16384", "--temp", "0", "--prompt-file",
           prompt(m, p), "--n", str(n), *M["base"], "--json", js, "--out-ids", os.path.join(RUNS, name + ".ids")]
    if mode != "off":
        d = [x for x in M["draft"]]
        if mode == "namesx":  # names inside the pass, no early names from the lookup's source span
            i = d.index("--early")
            d = d[:i] + d[i + 2:]
        cmd += d + ["--verify-names", "0" if mode == "plain" else "1"]
    power = wait_rules()
    log = os.path.join(RUNS, name + ".log")
    r = subprocess.run(["python3", GUARD, log, "1800", "--", *cmd], capture_output=True, text=True, env=dict(os.environ, POLL_S="2"))
    g = json.loads(r.stdout.strip().splitlines()[-1][6:])
    if g["rc"] != 0 or g["killed_by_guard"] or not os.path.exists(js):
        print(f"{name}: FAILED rc {g['rc']} {g['killed_by_guard']}", flush=True)
        return None
    res = json.load(open(js))
    res["guard"] = {k: g.get(k) for k in ("wall_s", "waited_s", "peak_footprint_gb_time", "others_cpu_pct_mean")}
    res["power"] = power
    json.dump(res, open(js, "w"))
    print(f"{name}: done ({power})", flush=True)
    return res


PROMPTS = ("pitt", "code", "edit", "tool")


def ids(name):
    f = os.path.join(RUNS, name + ".ids")
    return open(f).read().strip() if os.path.exists(f) else None


def gates():
    for m in ("ink", "q36"):
        for p in PROMPTS:
            for mode in ("off", "plain", "names"):
                run(m, p, mode)
            b = ids(f"{m}_{p}_off")
            for mode in ("plain", "names"):
                print(f"{m} {p} {mode}: ids identical to off: {b is not None and ids(f'{m}_{p}_{mode}') == b}", flush=True)


def noearly():
    """names on without the lookup's early names (does --early crowd out the routers' names?)"""
    for m in ("ink", "q36"):
        for p in ("code", "tool"):
            run(m, p, "namesx")
            print(f"{m} {p} namesx: ids identical to off: {ids(f'{m}_{p}_namesx') == ids(f'{m}_{p}_off')}", flush=True)


def table():
    print("| model | prompt | drafts | ids = off | tokens / step | verify passes | misses / verify pass | MB urgent / verify pass | "
          "misses / token | MB urgent / token | MB total / token | names refused / token | verify names: lead ms | named top-8 recall pos 0 / 1 / 3 / 7 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ("ink", "q36"):
        for p in PROMPTS:
            b = ids(f"{m}_{p}_off")
            for mode in ("off", "plain", "names", "namesx"):
                f = os.path.join(RUNS, f"{m}_{p}_{mode}.json")
                if not os.path.exists(f):
                    continue
                r = json.load(open(f))
                d = r.get("draft", {})
                rd = d.get("reads", {})
                v, a = rd.get("verify", {}), rd.get("all", {})
                rt = r.get("routers", {})
                lead = rt.get("lead", {})
                rec = rt.get("verify_top_k_in_named8_by_position", [])
                tok = r.get("decode_steps", 1)
                same = "-" if mode == "off" else ("yes" if ids(f"{m}_{p}_{mode}") == b else "NO")
                # per generated (= accepted) token over the whole decode (the drafter's in-pass split leaves out the early
                # names issued before a pass)
                misses_tok = r["read_queue"]["misses_per_token"]
                urg_tok = r["mb_read_urgent_per_token"]
                tot_tok = r["mb_read_per_token"]
                print(f"| {m} | {p} | {mode} | {same} | {d.get('tokens_per_step', 1.0):.2f} | {v.get('steps', 0):.0f} | "
                      f"{v.get('misses_per_step', 0):.1f} | {v.get('mb_urgent_per_step', 0):.0f} | {misses_tok:.1f} | {urg_tok:.0f} | {tot_tok:.0f} | {r['pf_dropped_per_token']:.1f} | "
                      f"{lead.get('verify_lead_ms', 0) if mode != 'plain' else 0:.0f} | "
                      f"{' / '.join(f'{rec[i]:.2f}' for i in (0, 1, 3, 7)) if mode in ('names', 'namesx') and rec else '-'} |")


def timing_cmds():
    """the clean campaign: off / drafts without names / drafts with names; Inkling 6 GB, the 35B 5 GB; code and the two
    file-writing tool calls; two runs per cell, interleaved (r1: off, plain, names; r2: names, plain, off)"""
    for m in ("ink", "q36"):
        for p, n in (("code", 300), ("toolnew", 800), ("toolrw", 800)):
            f = prompt(m, p) if p == "code" else os.path.join(P, f"{m}_{p}_ids.txt")
            for mode in ("off", "plain", "names"):
                M = MODELS[m]
                c = ["build/dmeng", "--model", "$" + m.upper(), "--cache-gb", str(M["gb"]), "--mem", "mlock", "--ctx", "16384", "--temp", "0",
                     "--prompt-file", os.path.relpath(f, REPO), "--n", str(n),
                     *[x.replace(M["dir"], "$" + m.upper()) for x in M["base"]]]
                if mode != "off":
                    c += M["draft"] + ["--verify-names", "1" if mode == "names" else "0"]
                    if p != "code":
                        c += ["--skeleton", os.path.relpath(os.path.join(P, f"{m}_{p}_skel.txt"), REPO)]
                c += ["--json", f"{m}_{p}_{mode}_rN.json", "--out-ids", f"{m}_{p}_{mode}_rN.ids"]
                print(" ".join(c))


if __name__ == "__main__":
    {"gates": gates, "noearly": noearly, "table": table, "timing": timing_cmds}[sys.argv[1]]()
