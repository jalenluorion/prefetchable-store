#!/usr/bin/env python3
"""The benchmark driver of dmeng: one campaign of cells x cache sizes x prompts x repetitions for one model, every run
through a guard (one model process at a time, under /usr/bin/time -l), resumable (a run whose JSON exists is skipped).

  python bench/run.py --model DIR --out DIR [--cells cache,routers,policy,approx] [--gb 4,6] [--prompts pitt,code]
                      [--routers F.dmr] [--evict-policy F.json] [--hazard] [--approx drop] [--reps 2]
                      [--bin build/dmeng] [--count-bin build-counters/dmeng | --no-counts] [--guard GUARD]
                      [--dry-run] [--report]

Each cell runs twice: the timed run with the default binary, then the counting run with the counting build
(cmake -DDM_COUNTERS=ON, run with DM_COUNTS=1), which gives the instructions and cycles of the useful work. See
bench/README.md for the cells and the columns.
"""
import argparse, csv, glob, json, os, re, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROMPTS = os.path.join(HERE, "prompts")
STD_CELLS = ("cache", "routers", "policy", "approx")


# ------------------------------------------------------------------------------------------------ campaign
def parse_args(argv):
    p = argparse.ArgumentParser(prog="bench/run.py", description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the model folder (its moved-store.json gives the family)")
    p.add_argument("--out", required=True, help="the campaign folder: runs/ (logs, JSON, ids), results.jsonl, results.csv")
    p.add_argument("--family", help="the model family when the folder has no moved-store.json")
    p.add_argument("--resident", help="--resident for dmeng (default: the model's own)")
    p.add_argument("--cells", default=None, help="comma list of cache,routers,policy,approx and names given with --cell "
                   "(default: every standard cell whose inputs are given, then the --cell ones)")
    p.add_argument("--cell", action="append", default=[], metavar="NAME=FLAGS", help="an extra cell: dmeng flags added to the common ones; "
                   "{routers} in FLAGS expands to the routers' flags (repeatable)")
    p.add_argument("--gb", default="5", help="comma list of cache sizes, --cache-gb (default 5)")
    p.add_argument("--prompts", default="pitt,code", help="comma list: the standard prompts (pitt, code) or token-id files (default pitt,code)")
    p.add_argument("--reps", type=int, default=1, help="repetitions of every run (default 1)")
    p.add_argument("--routers", help="router file (engine format) for the routers / policy / approx cells")
    p.add_argument("--router-args", default="", help='naming rule for the routers, e.g. "--m 12" or "--rt-tau 0.5" (default: dmeng\'s)')
    p.add_argument("--evict-policy", help="learned keep-score (JSON) for the policy cell")
    p.add_argument("--hazard", action="store_true", help="the policy cell evicts with --evict hazard")
    p.add_argument("--policy-args", default="", help='more flags of the policy cell, e.g. "--layer-caps uniform"')
    p.add_argument("--approx", default="drop", help="the approx cell's leave-out rule, --mode approx:RULE (default drop; drop:w=0.12 ...)")
    p.add_argument("--n", type=int, default=400, help="tokens per run (default 400)")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--mem", default="mlock")
    p.add_argument("--extra", default="", help="flags added to every run")
    p.add_argument("--bin", default=os.path.join(REPO, "build", "dmeng"), help="the timed binary (default build/dmeng)")
    p.add_argument("--count-bin", default=os.path.join(REPO, "build-counters", "dmeng"), help="the counting build (default build-counters/dmeng)")
    p.add_argument("--no-counts", action="store_true", help="timed runs only")
    p.add_argument("--guard", default=os.environ.get("DM_GUARD"), help="guard script: GUARD LOG TIMEOUT_S -- CMD (default $DM_GUARD)")
    p.add_argument("--guard-python", default=os.environ.get("DM_GUARD_PYTHON", sys.executable), help="interpreter of the guard (default this one)")
    p.add_argument("--timeout", type=int, default=1800, help="per run, seconds (default 1800)")
    p.add_argument("--ref-ids", action="append", default=[], metavar="PROMPT=FILE", help="reference ids of a prompt (default: the first "
                   "lossless cell's run at the first cache size, repetition 1)")
    p.add_argument("--redo-others", type=float, default=150, help="redo a timed run once when the others' CPU mean exceeds this %% (0: never)")
    p.add_argument("--busy", type=float, default=50, help="flag wall-clock figures when the others' CPU mean exceeds this %% (default 50)")
    p.add_argument("--batt-floor", type=int, default=25, help="on battery, wait while the charge is below this %% (default 25)")
    p.add_argument("--no-steps", action="store_true", help="do not write the per-step files (--steps-out)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    p.add_argument("--report", action="store_true", help="only rebuild results.jsonl / results.csv and the table from runs/")
    p.add_argument("--symbolize", action="store_true", help="with --report: name the counting runs' phase functions (atos)")
    a = p.parse_args(argv)
    for k in ("model", "resident", "routers", "evict_policy", "bin", "count_bin", "guard"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    return a


def model_family(a):
    mf = os.path.join(a.model, "moved-store.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        return a.family or m["family"], m.get("name", os.path.basename(os.path.normpath(a.model)))
    if not a.family:
        sys.exit(f"{mf} not found: give --family")
    return a.family, os.path.basename(os.path.normpath(a.model))


def prompt_files(a, family):
    out = {}
    for p in [x for x in a.prompts.split(",") if x]:
        std = os.path.join(PROMPTS, family, p + ".ids")
        if os.path.exists(std):
            out[p] = std
        elif os.path.exists(p):
            out[os.path.splitext(os.path.basename(p))[0]] = os.path.abspath(p)
        else:
            sys.exit(f"prompt {p!r}: neither a standard prompt of {family} ({std}) nor a file")
    return out


def cell_flags(a):
    """name -> (dmeng flags, lossless)"""
    rt = ["--prefetch", "router", "--routers", a.routers] + shlex.split(a.router_args) if a.routers else []
    pol = (["--evict-policy", a.evict_policy] if a.evict_policy else []) + (["--evict", "hazard"] if a.hazard else []) + shlex.split(a.policy_args)
    have = {"cache": ([], True)}
    if rt:
        have["routers"] = (rt, True)
        have["approx"] = (rt + ["--mode", "approx:" + (a.approx or "drop")], False)
    if pol and (rt or not a.evict_policy):  # the learned keep-score takes its features from the routers
        have["policy"] = (rt + pol, True)
    for c in a.cell:
        if "=" not in c:
            sys.exit(f"--cell {c!r}: NAME=FLAGS")
        nm, fl = c.split("=", 1)
        toks = []
        for t in shlex.split(fl):
            toks += rt if t == "{routers}" else [t]
        have[nm] = (toks, "approx" not in toks and not any(t.startswith("approx") for t in toks))
    names = [x for x in a.cells.split(",") if x] if a.cells else [c for c in STD_CELLS if c in have] + [c.split("=", 1)[0] for c in a.cell]
    for nm in names:
        if nm not in have:
            need = {"routers": "--routers", "approx": "--routers", "policy": "--evict-policy (with --routers) or --hazard"}.get(nm, "--cell NAME=FLAGS")
            sys.exit(f"cell {nm!r} needs {need}")
    return {nm: have[nm] for nm in names}


def plan(a, family):
    cells, prompts = cell_flags(a), prompt_files(a, family)
    gbs = [float(x) for x in a.gb.split(",") if x]
    runs = []
    for rep in range(1, a.reps + 1):
        for p, pf in prompts.items():
            for gb in gbs:
                for c, (fl, lossless) in cells.items():
                    base = f"{c}_{p}_gb{gb}_r{rep}"
                    meta = dict(cell=c, prompt=p, cache_gb=gb, rep=rep, lossless=lossless, base=base)
                    common = ["--model", os.path.abspath(a.model)] + (["--resident", a.resident] if a.resident else []) + [
                        "--prompt-file", pf, "--n", str(a.n), "--cache-gb", str(gb), "--ctx", str(a.ctx), "--temp", "0", "--mem", a.mem]
                    tail = fl + shlex.split(a.extra)
                    runs.append(dict(meta, kind="timed", name=base, cmd=[a.bin] + common + tail))
                    if not a.no_counts:
                        runs.append(dict(meta, kind="counts", name="cnt_" + base, cmd=[a.count_bin] + common + tail))
    return cells, prompts, gbs, runs


# ------------------------------------------------------------------------------------------------ running
def sh(*c):
    return subprocess.run(list(c), capture_output=True, text=True).stdout


def power_ok(floor):
    out = sh("pmset", "-g", "batt")
    m = re.search(r"(\d+)%", out)
    return "AC Power" in out or not m or int(m.group(1)) >= floor


def is_counting_build(path):
    return os.path.exists(path) and "_dmc_now" in sh("nm", "-g", path)


def guarded(a, runs_dir, r):
    """one run through the guard; returns the dmeng JSON with the guard summary, or None"""
    js = os.path.join(runs_dir, r["name"] + ".json")
    if os.path.exists(js):
        return json.load(open(js))
    cmd = r["cmd"] + ["--json", js, "--out-ids", os.path.join(runs_dir, r["name"] + ".ids")]
    if r["kind"] == "timed" and not a.no_steps:
        cmd += ["--steps-out", os.path.join(runs_dir, r["name"] + ".steps")]
    log = os.path.join(runs_dir, r["name"] + ".log")
    env = {**os.environ, "POLL_S": os.environ.get("POLL_S", "5")}
    if r["kind"] == "counts":
        env["DM_COUNTS"] = "1"
    else:
        env.pop("DM_COUNTS", None)
    attempt, mlock_tries = 0, 0
    while attempt < 2:
        attempt += 1
        waited = 0
        while not power_ok(a.batt_floor):
            if waited == 0:
                print(f"{r['name']}: on battery below {a.batt_floor}%, waiting", flush=True)
            time.sleep(60)
            waited += 60
        out = subprocess.run([a.guard_python, a.guard, log, str(a.timeout), "--", *cmd], capture_output=True, text=True, env=env)
        lines = [l for l in out.stdout.splitlines() if l.startswith("GUARD ")]
        if not lines:
            print(f"{r['name']}: no guard summary: {out.stdout[-300:]} {out.stderr[-300:]}", flush=True)
            return None
        g = json.loads(lines[-1][6:])
        text = open(log, errors="replace").read() if os.path.exists(log) else ""
        if g["rc"] != 0 and not g["killed_by_guard"] and "mlock failed" in text and mlock_tries < 20:
            mlock_tries += 1  # another process holds the wired memory: wait for it, not a failure
            print(f"{r['name']}: mlock refused, retry {mlock_tries} in 60 s", flush=True)
            time.sleep(60)
            attempt -= 1
            continue
        if g["rc"] != 0 or g["killed_by_guard"]:
            print(f"{r['name']}: FAILED {g['killed_by_guard'] or g['rc']} (see {log})", flush=True)
            if os.path.exists(js):
                os.replace(js, js + ".failed")
            return None
        busy = g.get("others_cpu_pct_mean") or 0
        if r["kind"] == "timed" and a.redo_others and busy > a.redo_others and attempt == 1:  # redo once, then keep and flag
            aside = os.path.join(runs_dir, "contended")
            os.makedirs(aside, exist_ok=True)
            for f in glob.glob(os.path.join(runs_dir, r["name"] + ".*")):
                os.replace(f, os.path.join(aside, os.path.basename(f)))
            print(f"{r['name']}: others used {busy}% CPU, set aside, redoing once", flush=True)
            continue
        break
    try:
        res = json.load(open(js))
    except (OSError, ValueError) as e:
        print(f"{r['name']}: FAILED: the run's JSON does not parse: {e}", flush=True)
        if os.path.exists(js):
            os.replace(js, js + ".invalid")
        return None
    res["guard"] = g
    res["bench"] = {k: r[k] for k in ("cell", "prompt", "cache_gb", "rep", "lossless", "kind")}
    json.dump(res, open(js, "w"))
    if r["kind"] == "counts" and "DMCOUNTS " not in text:
        print(f"{r['name']}: no DMCOUNTS line: is {r['cmd'][0]} a counting build (-DDM_COUNTERS=ON)?", flush=True)
    print(f"{r['name']:34s} {res.get('tok_s', 0):6.2f} tok/s | compute {res.get('compute_ms_per_token', 0):6.1f} wait {res.get('wait_ms_per_token', 0):5.1f} ms/tok"
          f" | {res.get('mb_read_per_token', 0):6.1f} MB/tok | others {g.get('others_cpu_pct_mean')}% | {power_of(g)}", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ results
def power_of(g):
    b, f = g.get("before", {}), g.get("after", {})
    if "AC" in b.get("power", "") and "AC" in f.get("power", ""):
        return "AC"
    m = re.search(r"(\d+)%", f.get("battery", ""))
    return "batt " + (m.group(1) + "%" if m else "?")


def num(t, pat, cast=float):
    m = re.search(pat, t)
    return cast(m.group(1)) if m else None


def log_counts(log):
    """the counting build's DMCOUNTS / DMFN lines and /usr/bin/time -l's totals of one run's log"""
    t = open(log, errors="replace").read() if os.path.exists(log) else ""
    d = {}
    m = re.findall(r"DMCOUNTS (\{.*\})", t)
    if m:
        d = json.loads(m[-1])
        d["useful_ins"] = d["work_ins"] + d["main_serial_ins"]
        d["useful_cyc"] = d["work_cyc"] + d["main_serial_cyc"]
    d["proc_ins"] = num(t, r"(\d+)\s+instructions retired", int)
    d["proc_cyc"] = num(t, r"(\d+)\s+cycles elapsed", int)
    d["reclaims"] = num(t, r"(\d+)\s+page reclaims", int)
    d["page_faults"] = num(t, r"(\d+)\s+page faults", int)
    d["peak_footprint_time"] = num(t, r"(\d+)\s+peak memory footprint", int)
    d["fn"] = []
    for l in re.findall(r"^DMFN (.*)$", t, re.M):
        img, base, addr, i, c, n = l.rsplit(" ", 5)
        d["fn"].append(dict(image=img, base=base, addr=addr, offset=hex(int(addr, 16) - int(base, 16)) if base != "0x0" else addr,
                            ins=int(i), cyc=int(c), runs=int(n)))
    if not d["fn"]:
        del d["fn"]
    if t and "peak memory footprint" not in t:
        d["no_time_l"] = True
    return d


def ids_of(runs_dir, name):
    p = os.path.join(runs_dir, name + ".ids")
    return open(p).read().strip() if os.path.exists(p) else None


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row_of(a, runs_dir, base_meta, refs):
    """one result row: the timed run's wall clock, the counting run's instructions, ids against the prompt's reference"""
    t_js, c_js = os.path.join(runs_dir, base_meta["base"] + ".json"), os.path.join(runs_dir, "cnt_" + base_meta["base"] + ".json")
    if not os.path.exists(t_js) and not os.path.exists(c_js):
        return None
    t = json.load(open(t_js)) if os.path.exists(t_js) else {}
    g = t.get("guard", {})
    tl = log_counts(os.path.join(runs_dir, base_meta["base"] + ".log")) if t else {}
    rq, ap, rt, dr = t.get("read_queue", {}), t.get("approx", {}), t.get("routers", {}), t.get("draft")
    named = rt.get("all", rt)  # trained routers report by group; the cross-token router at its top level
    row = {k: base_meta[k] for k in ("cell", "prompt", "cache_gb", "rep", "lossless")}
    row.update(name=base_meta["base"], family=t.get("family"), accuracy=t.get("accuracy"), prefetch=t.get("prefetch"),
               tokens=t.get("tokens"), tok_s=t.get("tok_s"), ms_per_token=t.get("ms_per_token"),
               compute_ms_per_token=t.get("compute_ms_per_token"), wait_ms_per_token=t.get("wait_ms_per_token"),
               policy_ms_per_token=t.get("policy_ms_per_token"), mb_read_per_token=t.get("mb_read_per_token"),
               mb_read_urgent_per_token=t.get("mb_read_urgent_per_token"), mb_read_prefetch_per_token=t.get("mb_read_prefetch_per_token"),
               hit_rate=t.get("hit_rate"), late_rate=t.get("late_rate"), miss_rate=t.get("miss_rate"),
               hit_rate_looped=t.get("hit_rate_looped"), hit_rate_other=t.get("hit_rate_other"),
               misses_per_token=rq.get("misses_per_token"), late_per_token=rq.get("late_per_token"),
               wait_miss_ms_per_token=rq.get("wait_miss_ms_per_token"), wait_late_ms_per_token=rq.get("wait_late_ms_per_token"),
               drive_gb_s_while_busy=(t.get("drive") or {}).get("gb_s_while_busy", rq.get("drive_gb_s_while_busy")),
               top_k_in_named8=named.get("top_k_in_named8"), top_k_in_named12=named.get("top_k_in_named12"),
               top_k_in_named16=named.get("top_k_in_named16"), router_ms_per_token=rt.get("router_ms_per_token", rt.get("ms_per_token")),
               weight_replaced=ap.get("weight_replaced") if ap.get("on") else None,
               steps_with_a_replacement=ap.get("steps_with_a_replacement") if ap.get("on") else None,
               weight_read_now=ap.get("weight_read_now") if ap.get("on") else None,
               draft_tokens_per_step=get(dr, "tokens_per_step"), draft_accepted=get(dr, "accepted"), draft_offered=get(dr, "offered"),
               footprint_peak_gb_engine=t.get("footprint_peak_gb"),
               peak_footprint_gb=g.get("peak_footprint_gb_time") if g.get("peak_footprint_gb_time") is not None else g.get("peak_footprint_gb_guard"),
               peak_footprint_gb_guard=g.get("peak_footprint_gb_guard"), peak_swap_growth_mb=g.get("peak_swap_growth_mb"),
               others_cpu_pct=g.get("others_cpu_pct_mean"), power=power_of(g) if g else None,
               mem_free_pct_before=get(g, "before", "mem_free_pct"), reclaims=tl.get("reclaims"),
               proc_ins=tl.get("proc_ins"), proc_cyc=tl.get("proc_cyc"))
    if os.path.exists(c_js):
        c = json.load(open(c_js))
        cl = log_counts(os.path.join(runs_dir, "cnt_" + base_meta["base"] + ".log"))
        row.update(useful_ins=cl.get("useful_ins"), useful_cyc=cl.get("useful_cyc"), work_ins=cl.get("work_ins"), work_cyc=cl.get("work_cyc"),
                   main_serial_ins=cl.get("main_serial_ins"), main_serial_cyc=cl.get("main_serial_cyc"), spin_ins=cl.get("spin_ins"),
                   spin_cyc=cl.get("spin_cyc"), pool_runs=cl.get("runs"), cnt_tokens=c.get("tokens"), cnt_reclaims=cl.get("reclaims"),
                   cnt_proc_ins=cl.get("proc_ins"), cnt_proc_cyc=cl.get("proc_cyc"), cnt_others_cpu_pct=get(c, "guard", "others_cpu_pct_mean"),
                   cnt_tok_s=c.get("tok_s"), cnt_fn=cl.get("fn"))
        if row["useful_ins"] and c.get("tokens"):
            row["useful_ins_per_token"] = row["useful_ins"] / c["tokens"]
            row["useful_cyc_per_token"] = row["useful_cyc"] / c["tokens"]
            row["ipc"] = row["useful_ins"] / max(1.0, row["useful_cyc"])
    ref = refs.get(base_meta["prompt"])
    for kind, nm in (("ids", base_meta["base"]), ("cnt_ids", "cnt_" + base_meta["base"])):
        mine = ids_of(runs_dir, nm)
        row[kind + "_same"] = None if mine is None or ref is None else mine == ref
    row["is_reference"] = base_meta.get("is_ref", False)
    return row


def flag_pressure(rows):
    """a counting run whose page reclaims exceed 1.5x the least of its batch (same cache size): memory pressure, the kernel's
    page-ins counted inside the run's own threads; the timed runs the same way"""
    for key, flag in (("cnt_reclaims", "cnt_pressure"), ("reclaims", "pressure")):
        by = {}
        for r in rows:
            if r.get(key) is not None:
                by.setdefault(r["cache_gb"], []).append(r[key])
        for r in rows:
            lo = min(by.get(r["cache_gb"], [0]) or [0])
            r[flag] = r.get(key) is not None and lo > 0 and r[key] > 1.5 * lo


def references(a, runs_dir, cells, prompts, gbs):
    refs, names = {}, {}
    for spec in a.ref_ids:
        p, f = spec.split("=", 1)
        refs[p] = open(f).read().strip()
        names[p] = f
    lossless = [c for c, (_, ll) in cells.items() if ll]
    for p in prompts:
        if p in refs or not lossless:
            continue
        nm = f"{lossless[0]}_{p}_gb{gbs[0]}_r1"
        names[p] = nm
        ids = ids_of(runs_dir, nm)
        if ids is not None:
            refs[p] = ids
    return refs, names


def fmt(v, f="{:.2f}", none="-"):
    return none if v is None else f.format(v)


def table(rows, busy):
    hdr = (f"{'cell':10s} {'prompt':7s} {'GB':>5s} {'r':>2s} | {'useful':>8s} {'useful':>8s} {'IPC':>5s} {'spin':>7s} {'mem':>4s} | {'ids':>5s} || "
           f"{'tok/s':>7s} {'comp':>7s} {'wait':>7s} {'MB/tok':>7s} {'peak':>6s} {'others':>7s} {'power':>8s}")
    hdr2 = (f"{'':10s} {'':7s} {'':>5s} {'':>2s} | {'G ins':>8s} {'G cyc':>8s} {'':>5s} {'G ins':>7s} {'':>4s} | {'':>5s} || "
            f"{'':>7s} {'ms/tok':>7s} {'ms/tok':>7s} {'1e6 B':>7s} {'GB':>6s} {'CPU %':>7s} {'':>8s}")
    out = [hdr, hdr2, "-" * len(hdr)]
    for r in rows:
        busy_flag = "!" if (r.get("others_cpu_pct") or 0) > busy else " "
        w = lambda k, f="{:.2f}": fmt(r.get(k), f) + (busy_flag if r.get(k) is not None else " ")
        ids = "ref" if r["is_reference"] else {True: "same", False: "DIFF", None: "?"}[r["ids_same"]]
        if r.get("cnt_ids_same") is False:
            ids += "/c!"
        if r["ids_same"] is False and not r["lossless"]:
            ids = "apx"  # an approximate cell's ids are not expected to match
        mem = ("P" if r.get("cnt_pressure") else "ok") if r.get("cnt_reclaims") is not None else "-"
        g = lambda k: None if r.get(k) is None else r[k] / 1e9
        ipc = r.get("ipc")
        out.append(f"{r['cell'][:10]:10s} {r['prompt'][:7]:7s} {r['cache_gb']:5.2f} {r['rep']:2d} | {fmt(g('useful_ins'), '{:8.2f}'):>8s} "
                   f"{fmt(g('useful_cyc'), '{:8.2f}'):>8s} {fmt(ipc, '{:5.2f}'):>5s} {fmt(g('spin_ins'), '{:7.2f}'):>7s} {mem:>4s} | {ids:>5s} || "
                   f"{w('tok_s'):>8s}{w('compute_ms_per_token', '{:.1f}'):>8s}{w('wait_ms_per_token', '{:.1f}'):>8s}"
                   f"{fmt(r.get('mb_read_per_token'), '{:.1f}'):>7s} {fmt(r.get('peak_footprint_gb')):>6s} {fmt(r.get('others_cpu_pct'), '{:.0f}'):>7s} "
                   f"{(r.get('power') or '-'):>8s}" + (" P" if r.get("pressure") else ""))
    out.append(f"useful = pool work + main-thread serial work, spin-waits excluded (counting build); mem P = page reclaims > 1.5x the least at "
               f"that cache size; ! = others' CPU mean above {busy:.0f}%, wall clock not quotable; ids vs the prompt's reference (/c! = the "
               f"counting run differs; apx = approximate cell)")
    return "\n".join(out)


CSV_SKIP = ("cnt_fn",)


def write_results(out_dir, rows):
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys and k not in CSV_SKIP]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def symbolize(a, rows):
    """the counting runs' phase functions, named with atos against the counting binary"""
    for r in rows:
        fns = r.get("cnt_fn") or []
        if not fns:
            continue
        names = {}
        for img in sorted({f["image"] for f in fns}):  # the image the run loaded, else the counting binary of this campaign
            own = [f for f in fns if f["image"] == img]
            exe = img if os.path.exists(img) else a.count_bin if os.path.basename(img) == os.path.basename(a.count_bin) else None
            if exe and os.path.exists(exe):
                out = sh("atos", "-o", exe, "-l", own[0]["base"], *[f["addr"] for f in own]).splitlines()
                names.update({f["addr"]: n.split(" (in ")[0] for f, n in zip(own, out)})
        tot = sum(f["ins"] for f in fns) or 1
        print(f"\n{r['name']}: phase functions by instructions (pool work)")
        for f in sorted(fns, key=lambda f: -f["ins"])[:12]:
            print(f"  {f['ins'] / tot * 100:5.1f}%  {f['ins'] / 1e9:8.2f} G ins  {f['cyc'] / 1e9:8.2f} G cyc  {f['runs']:8d} runs  {names.get(f['addr'], f['offset'])}")


# ------------------------------------------------------------------------------------------------ main
def main(argv):
    a = parse_args(argv)
    family, mname = model_family(a)
    cells, prompts, gbs, runs = plan(a, family)
    runs_dir = os.path.join(os.path.abspath(a.out), "runs")
    if not a.report:
        problems = []
        if not a.guard:
            problems.append("no guard: give --guard or set DM_GUARD (bench/README.md says what it must do)")
        if not os.path.exists(a.bin):
            problems.append(f"{a.bin}: not built (cmake --build build)")
        if not a.no_counts and not is_counting_build(a.count_bin):
            problems.append(f"{a.count_bin}: not a counting build (cmake -S . -B build-counters -DDM_COUNTERS=ON; or --no-counts)")
        if problems and not a.dry_run:
            sys.exit("\n".join(problems))
        print(f"# {mname} ({family}): {len(cells)} cells x {len(gbs)} cache sizes x {len(prompts)} prompts x {a.reps} reps = {len(runs)} runs"
              + ("" if a.no_counts else " (timed + counting)"))
        for p in problems:
            print("# warning:", p)
        if a.dry_run:
            for r in runs:
                done = os.path.exists(os.path.join(runs_dir, r["name"] + ".json"))
                cmd = r["cmd"] + ["--json", os.path.join(runs_dir, r["name"] + ".json"), "--out-ids", os.path.join(runs_dir, r["name"] + ".ids")]
                if r["kind"] == "timed" and not a.no_steps:
                    cmd += ["--steps-out", os.path.join(runs_dir, r["name"] + ".steps")]
                env = "DM_COUNTS=1 " if r["kind"] == "counts" else ""
                print(("# done, skipped: " if done else "") + env + shlex.join([a.guard_python, a.guard or "$DM_GUARD", os.path.join(runs_dir, r["name"] + ".log"), str(a.timeout), "--", *cmd]))
            return
        os.makedirs(runs_dir, exist_ok=True)
        for r in runs:
            guarded(a, runs_dir, r)
    refs, ref_names = references(a, runs_dir, cells, prompts, gbs)
    rows, seen = [], set()
    for r in runs:
        if r["base"] in seen:
            continue
        seen.add(r["base"])
        meta = dict(r, is_ref=ref_names.get(r["prompt"]) == r["base"])
        row = row_of(a, runs_dir, meta, refs)
        if row:
            rows.append(row)
    flag_pressure(rows)
    if rows:
        write_results(os.path.abspath(a.out), rows)
        print(table(rows, a.busy))
        print(f"results: {os.path.join(a.out, 'results.jsonl')}, {os.path.join(a.out, 'results.csv')}")
        if a.symbolize:
            symbolize(a, rows)
    else:
        print("no results yet in", runs_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
