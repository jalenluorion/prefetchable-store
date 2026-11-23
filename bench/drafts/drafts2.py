#!/usr/bin/env python3
"""Drafting under the per-pass cost model, sampling and fast mode (results/drafts/DRAFTS2.md). Every model run goes through
engine/bench/guard.py (campaign.run: power / QUIET / PRIORITY rules, the guard's queue); resumable (a run whose JSON exists
is skipped); a failed run is reported and skipped.

  python3 drafts2.py BATCH...   batches: probe, gates, timing, server, table
  probe   one interleaved pair per model (35B file rewrite, Inkling coding answer): exact vs exact + drafts
  gates   sampled ids, drafts on vs off, three seeds per model (release sampler: temperature 1.0, top-p 0.95, top-k 20);
          fast mode with --read-w 0 (no pick left out) + drafts vs exact
  timing  per model: coding answer and file rewrite x exact / exact + drafts / fast / fast + drafts, one run per cell
          (REPS=1,2 for repeats); judged by counters (tokens per step, MB per token, compute and wait per token)
  server  the worker (dmserve) over one sampled request (seed 1) on the file rewrite, drafts off vs on
  table   results/drafts/drafts2_table.md and drafts2_rows.json (numbers only)
Profile defaults throughout (--model DIR gives routers, resident and the rest); --drafts 1, --mode fast, --cache-gb G and
--prefill-chunk on top."""
import json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign as C

C.RUNS = os.path.join(C.OUT, "runs2")
os.makedirs(C.RUNS, exist_ok=True)
GB = {"q36": 6.6, "ink": 7.8}
PC = {"q36": "512", "ink": "256"}
FAM = {"q36": "qwen3_5_moe", "ink": "inkling2_loop"}
SAMPLER = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"]
CASE_N = {"code": 300, "toolrw": int(os.environ.get("TOOLRW_N", 300))}  # (this table ran the file rewrite at 800: it stops at 451 / 108)
GATE_N = 160
REPS = [int(x) for x in os.environ.get("REPS", "1").split(",")]  # one run per cell; the wall-clock table is taken separately


def model_busy():
    ps = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True).stdout
    return any(re.search(r"/(dmeng|dmserve|dmink|bmoe-cli|llama-[a-z-]+)( |$)", l) and "guard.py" not in l for l in ps.splitlines())


_wait0 = C.wait_turn


def wait_turn():  # two guards polling at the same moment both see a free slot: start after three free checks 4 s apart
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


# the guard's own queue is first come, first served: enter it at once (campaign.wait_turn keeps the power / QUIET rules)
FAILED = []


def prompt(model, case):
    return os.path.join(C.REPO, "bench", "prompts", FAM[model], "code.ids") if case == "code" else os.path.join(C.PR, f"{model}_{case}_ids.txt")


def dmeng(name, model, case, n, extra, timeout=1800):
    cmd = [C.DMENG, "--model", C.Q36 if model == "q36" else C.INK, "--prompt-file", prompt(model, case), "--n", str(n), "--cache-gb", str(GB[model]),
           "--prefill-chunk", PC[model], "--stop-eos", str(C.EOS[model])] + extra
    if "--drafts" in extra and case.startswith("tool"):
        cmd += ["--skeleton", os.path.join(C.PR, f"{model}_{case}_skel.txt")]
    cmd += ["--json", os.path.join(C.RUNS, name + ".json.tmp"), "--out-ids", os.path.join(C.RUNS, name + ".ids")]
    try:
        r = C.run(name, cmd, timeout=timeout)
    except Exception as e:  # never stop the batch
        r = None
        print("ERROR", name, e, flush=True)
    if r is None or "tok_s" not in r:
        FAILED.append(name)
        print("FAILED", name, flush=True)
    return r


CONF = {"exact": ["--temp", "0"], "exd": ["--temp", "0", "--drafts", "1"], "fast": ["--temp", "0", "--mode", "fast"],
        "fd": ["--temp", "0", "--mode", "fast", "--drafts", "1"]}


def ids(name):
    p = os.path.join(C.RUNS, name + ".ids")
    return open(p).read().strip().split(",") if os.path.exists(p) else None


def same(a, b, prefix=False):
    x, y = ids(a), ids(b)
    if x is None or y is None:
        return "missing"
    if prefix:
        k = min(len(x), len(y))
        return "IDENTICAL" if x[:k] == y[:k] else "DIFFER"
    return "IDENTICAL" if x == y else "DIFFER"


def probe():
    tag = os.environ.get("PROBE", "p1")
    for model, case in (("q36", "toolrw"), ("ink", "code")):
        for c in ("exd",):
            dmeng(f"{model}_{case}_{c}_{tag}", model, case, CASE_N[case], CONF[c])
        print("IDS", model, case, "exact vs exact+drafts:", same(f"{model}_{case}_exact_p0", f"{model}_{case}_{c}_{tag}"), flush=True)


def gates():
    for model in ("q36", "ink"):
        for s in (1, 2, 3):
            for c in ("off", "on"):
                dmeng(f"{model}_toolrw_samp_s{s}_{c}", model, "toolrw", GATE_N, SAMPLER + ["--seed", str(s)] + (["--drafts", "1"] if c == "on" else []))
            print("IDS sampled", model, "seed", s, same(f"{model}_toolrw_samp_s{s}_off", f"{model}_toolrw_samp_s{s}_on"), flush=True)
        dmeng(f"{model}_toolrw_fastrw0", model, "toolrw", GATE_N, ["--temp", "0", "--mode", "fast", "--read-w", "0", "--drafts", "1"])
        ref = f"{model}_toolrw_exact_p0" if ids(f"{model}_toolrw_exact_p0") else f"{model}_toolrw_exact_r1"
        print("IDS fast read-w 0 + drafts vs", ref, "(prefix):", same(ref, f"{model}_toolrw_fastrw0", prefix=True), flush=True)


def timing():
    for model in ("ink", "q36"):
        for case in ("code", "toolrw"):
            for rep in REPS:
                for c in ("exact", "exd", "fast", "fd"):
                    dmeng(f"{model}_{case}_{c}_r{rep}", model, case, CASE_N[case], CONF[c])
                print("IDS", model, case, f"r{rep} exact vs exact+drafts:", same(f"{model}_{case}_exact_r{rep}", f"{model}_{case}_exd_r{rep}"), flush=True)


def server():
    for model in ("q36", "ink"):
        ids0 = [int(x) for x in open(prompt(model, "toolrw")).read().replace(",", " ").split()]
        skel = [[int(x) for x in l.split()] for l in open(os.path.join(C.PR, f"{model}_toolrw_skel.txt")) if l.strip()]
        # one sampled request (seed 1; the 35B's run here also had a greedy request and seeds 2 and 3)
        reqs = [{"ids": ids0, "max_new": GATE_N, "temp": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1, "stop": [C.EOS[model]], "skel": skel}]
        rq = os.path.join(C.RUNS, f"{model}_server_reqs.json")
        json.dump(reqs, open(rq, "w"))
        for c in ("off", "on"):
            name = f"{model}_server_{c}"
            w = [C.DMSERVE, "--model", C.Q36 if model == "q36" else C.INK, "--cache-gb", str(GB[model]), "--ctx", "16384"] + (["--drafts", "1"] if c == "on" else [])
            try:
                r = C.run(name, ["python3", os.path.join(C.HERE, "worker_drive.py"), os.path.join(C.RUNS, name + ".json.tmp"), rq, "--"] + w, timeout=2400)
            except Exception as e:
                r = None
                print("ERROR", name, e, flush=True)
            if not r:
                FAILED.append(name)
                print("FAILED", name, flush=True)
        a, b = (C.load(f"{model}_server_{c}") if hasattr(C, "load") else None for c in ("off", "on"))
        try:
            a, b = (json.load(open(os.path.join(C.RUNS, f"{model}_server_{c}.json"))) for c in ("off", "on"))
            for k in range(len(a)):
                print("IDS server", model, "request", k + 1,  "IDENTICAL" if a[k]["gen"] == b[k]["gen"] else "DIFFER",
                      "draft", json.dumps((b[k]["E"]["json"] or {}).get("draft", {}).get("tokens_per_step")), flush=True)
        except Exception as e:
            print("server compare:", e, flush=True)


def row(name):
    p = os.path.join(C.RUNS, name + ".json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    m = os.path.join(C.RUNS, name + ".meta.json")
    g = (json.load(open(m)).get("guard") or {}) if os.path.exists(m) else {}
    d = j.get("draft") or {}
    a = j.get("approx") or {}
    return {"name": name, "tokens": j.get("tokens"), "tok_s": j.get("tok_s"), "compute_ms": j.get("compute_ms_per_token"), "wait_ms": j.get("wait_ms_per_token"),
            "mb_per_tok": j.get("mb_read_per_token"), "mb_urgent_per_tok": j.get("mb_read_urgent_per_token"), "tps": d.get("tokens_per_step"),
            "verifies": d.get("verifies"), "drafted": d.get("drafted"), "accepted": d.get("accepted"), "left_out_per_tok": a.get("replaced_per_token"),
            "others_cpu": g.get("others_cpu_pct_mean"), "footprint_gb": j.get("footprint_peak_gb"), "widths": (d.get("model") or {}).get("widths"),
            "head": {k: (d.get("head") or {}).get(k) for k in ("steps", "accepted", "runs", "lost")} if d.get("head") else None}


def table():
    rows = []
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    out = ["| model | case | config | tok/s (r1 / r2) | vs exact (r1 / r2) | compute ms/tok | wait ms/tok | MB/tok | urgent MB/tok | tokens/step | kept / drafted | left out /tok | others' CPU % (r1 / r2) |",
           "|" + "---|" * 13]
    names = {"exact": "exact", "exd": "exact + drafts", "fast": "fast", "fd": "fast + drafts"}
    for model in ("ink", "q36"):
        for case in ("code", "toolrw"):
            for c in ("exact", "exd", "fast", "fd"):
                rs = [row(f"{model}_{case}_{c}_r{rep}") for rep in (1, 2)]
                ex = [row(f"{model}_{case}_exact_r{rep}") for rep in (1, 2)]
                if not any(rs):
                    continue
                rows += [r for r in rs if r]
                g = [r for r in rs if r]
                m = lambda k: sum(r[k] for r in g if r[k] is not None) / max(1, len([r for r in g if r[k] is not None])) if any(r[k] is not None for r in g) else None
                ts = " / ".join(f(r["tok_s"], 2) if r else "-" for r in rs)
                vs = " / ".join(f(r["tok_s"] / e["tok_s"], 2) if r and e else "-" for r, e in zip(rs, ex))
                oc = " / ".join(f(r["others_cpu"], 0) if r else "-" for r in rs)
                kd = f"{sum(r['accepted'] or 0 for r in g)} / {sum(r['drafted'] or 0 for r in g)}" if any(r["drafted"] is not None for r in g) else "-"
                out.append(f"| {model} | {case} | {names[c]} | {ts} | {vs} | {f(m('compute_ms'))} | {f(m('wait_ms'))} | {f(m('mb_per_tok'), 0)} | {f(m('mb_urgent_per_tok'), 0)} | {f(m('tps'), 2)} | {kd} | {f(m('left_out_per_tok'), 1)} | {oc} |")
    for n in sorted(os.listdir(C.RUNS)):
        if n.endswith(".json") and ("samp" in n or "fastrw0" in n):
            r = row(n[:-5])
            if r:
                rows.append(r)
    open(os.path.join(C.OUT, "drafts2_table.md"), "w").write("\n".join(out) + "\n")
    json.dump(rows, open(os.path.join(C.OUT, "drafts2_rows.json"), "w"), indent=0)
    print("\n".join(out))


if __name__ == "__main__":
    for b in sys.argv[1:]:
        {"probe": probe, "gates": gates, "timing": timing, "server": server, "table": table}[b]()
    print("FAILED RUNS:", FAILED if FAILED else "none", flush=True)
