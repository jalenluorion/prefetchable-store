"""The same checks on a tiny model of any family (tests/tiny_all.sh runs them on all three): nothing here names a family;
what a family supports is read from the engine itself (the "capabilities" of its JSON) and from the folder's manifest.

  python tests/family_check.py TINY_DIR OUT_DIR [--bin-dir build] [--ref-python PY]    (python: the standard library)

TINY_DIR holds model/ (the folder the engine runs, with moved-store.json) and release/ (the synthetic release the
reference reads). Checks, one verdict each (PASS, FAIL, or "not supported for this family" with the engine's refusal
checked):
  reference   exact decode against the independent reference: tests/<family>/tiny_check.py (its own python, --ref-python)
  invariance  exact mode: logits and greedy ids identical, bit for bit, with every expert held / few slots (evictions and
              re-reads), 1 / 4 / 5 compute threads, the folder's router naming, run-ahead, and an oracle replaying the
              recorded routing at leads 1 and one token; names + run-ahead with the names' reads held in flight
              (DMENG_TEST_SPEC_READ_US: a miss finds every free slot still being read); the spare tier on (--spare-mem 1:
              probation budgets that overflow and that hold every record, 1 thread, names + run-ahead, names in flight,
              greedy ids), each run checked to have promoted records and served reuses from the file cache
  prefill     the batched prefill equals single steps: dmserve --check (logits, fixed state, KV, greedy continuation; CPU,
              and the GPU's share on a Metal build), and dmeng's ids after a batched and a token-by-token prompt
  checkpoint  dmserve --check-ckpt (a long request, the same again, one diverging at 60%, each equal to a cold computation)
              and dmeng --state-check (state saved, restored into a fresh one, the same ids)
  server      the server worker's greedy and sampled ids equal dmeng's for the same prompts and seeds
  drafting    dmeng --verify-check (verify passes with rollbacks equal to single steps), --draft fixed / lookup equal to plain
              decoding (greedy and sampled), the server worker the same; the same with drafting as the profile's declared
              default (a folder whose profiles declare drafts.lookup, no flag: ids, --verify-check, the server); a declared
              draft head the folder cannot load refused at open; the C API (dm_gate, dm_step_n): fixed / lookup drafts with
              fragments (the reply as a given text, every 6th token changed) and a regenerated reply (the lookup copying the
              session's earlier one) equal to plain decoding (dm_step + dm_sample), greedy and sampled with penalties, tokens
              per step above 1, the draft counters reported; dm_step_n without drafting equal to plain decoding with no
              draft block; the profile's declared drafts on through it; the slab on iOS giving drafting's memory; without a
              verify pass: every front end refuses, also a declared default
  presets     the precedence (explicit > profile > manifest > built-in) in dmeng, the C API (dm_gate) and dmserve, each
              read back from its "resolved" JSON, and the three front ends printing the same resolved settings (the
              profile's drafts and the pass width too, a declared default and an explicit request alike)
  files       every engine-ready file the engine uses is listed in the manifest with its sha256 and checked at load: a
              copy with one byte changed is refused; the manifest declares its in-memory rule (resident.q8_rule)
  caps        per-layer caps of 2K + 4 slots with the routers naming K per step: the :fit reserve refuses almost no name (under
              1%: reads in flight; the fixed one refuses most), and the ids are the same with caps, fit caps and one pool
  lock        the cache's slots locked one by one under a memory-lock limit below the start size: the refused lock stops the
              budget there, the JSON reports what is locked, the ids are an unlimited run's
  cache       the automatic cache rule on one given machine state in dmeng, dmserve and the C API (mac profile): the start
              rule the manifest declares, and the printed ceiling and start follow from the printed inputs in all three
  counters    the stages (one set of names, Times::stages_json, in dmeng and the server) sum to the token's time within 1%,
              compute = time - wait - run-ahead, and the server's MB per token equals dmeng's (MB = 10^6 bytes in both)
Writes OUT_DIR/family_check.json and prints one line per check; exit 1 if any check fails.
"""
import argparse, json, os, random, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("tiny")
ap.add_argument("out")
ap.add_argument("--bin-dir", default=os.path.join(REPO, "build"))
ap.add_argument("--ref-python", default=None, help="the python of the family's reference check (default: skip it)")
ap.add_argument("--only", default=None, help="comma-separated checks")
a = ap.parse_args()
TINY, OUT, B = os.path.abspath(a.tiny), os.path.abspath(a.out), os.path.abspath(a.bin_dir)
M = os.path.join(TINY, "model")
os.makedirs(OUT, exist_ok=True)
T = os.path.join(OUT, "tmp")
shutil.rmtree(T, ignore_errors=True)
os.makedirs(T)
DMENG, DMSERVE, DMGATE = (os.path.join(B, x) for x in ("dmeng", "dmserve", "dm_gate"))
man = json.load(open(os.path.join(M, "moved-store.json")))
FAM = man["family"]
V = man["model"].get("vocab_unpadded") or man["model"]["vocab"]
K = man["experts"]["top_k"]
meta = json.load(open(os.path.join(M, "store", "meta.json")))
rb = meta["record_bytes"] if isinstance(meta["record_bytes"], list) else [meta["record_bytes"]] * meta["layers"]
ALL_GB = 1.25 * sum(rb) * meta["experts"] / 1e9  # every record held
FEW_GB = 0.5 * sum(rb) * meta["experts"] / 1e9   # evictions and re-reads every token (above every pool's floor)
TOK_GB = 0.6 * sum(rb) * meta["experts"] / 1e9   # a whole token of oracle names held
LO = 10  # ids from 10 up (every family's specials are below)
results, lines = {}, []


def say(s):
    print(s, flush=True)
    lines.append(s)


def verdict(name, status, detail=""):
    results[name] = {"status": status, "detail": detail}
    say(f"{status:26s} {name:11s} {detail}")


def run(cmd, env=None, inp=None, timeout=900):
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, input=inp, timeout=timeout)


def last_json(r):
    for l in reversed(r.stdout.strip().splitlines()):
        if l.startswith("{"):
            return json.loads(l)
    raise ValueError((r.stdout + r.stderr)[-800:])


rnd = random.Random(11)
ids = {n: [rnd.randrange(LO, V) for _ in range(n)] for n in (40, 300, 1100)}
ids[120] = ids[300][:120]
for n, p in ids.items():
    open(os.path.join(T, f"ids{n}.txt"), "w").write(",".join(map(str, p)))
BASE = ["--profile", "none", "--threads", "4", "--readers", "2", "--mem", "plain", "--res-mem", "plain", "--ctx", "2048"]
routers = sorted(os.path.join(M, ".dmeng", f) for f in os.listdir(os.path.join(M, ".dmeng")) if f.startswith("router.")) if os.path.isdir(os.path.join(M, ".dmeng")) else []
RT = ["--routers", routers[0], "--m", str(K)] if routers else []


def dmeng(*flags, name="r", env=None):
    r = run([DMENG, "--model", M, *BASE, *flags], env=env)
    if r.returncode:
        raise RuntimeError(f"dmeng {' '.join(flags)}: {(r.stdout + r.stderr)[-600:]}")
    return last_json(r)


def logits(name, n, *flags, env=None):
    lp = os.path.join(T, name + ".f32")
    js = dmeng("--teacher", os.path.join(T, f"ids{n}.txt"), "--dump-logits", lp, *flags, env=env)
    return open(lp, "rb").read(), js


def greedy(name, *flags, n=32, prompt=300):
    op = os.path.join(T, name + ".ids")
    js = dmeng("--prompt-file", os.path.join(T, f"ids{prompt}.txt"), "--n", str(n), "--out-ids", op, *flags)
    return open(op).read().strip(), js


caps = dmeng("--prompt-file", os.path.join(T, "ids40.txt"), "--n", "2", "--cache-gb", f"{ALL_GB:.6f}").get("capabilities", {})
want = set(a.only.split(",")) if a.only else None


def wanted(n):
    return want is None or n in want


# ---------------------------------------------------------------------------------------------- reference
if wanted("reference"):
    chk = os.path.join(HERE, FAM, "tiny_check.py")
    if not os.path.exists(chk):
        verdict("reference", "FAIL", f"no tests/{FAM}/tiny_check.py")
    elif not a.ref_python:
        verdict("reference", "SKIPPED", "no --ref-python given")
    else:
        r = run([a.ref_python, chk], env={"TINY": TINY, "DMENG_BIN": DMENG, "DMSERVE_BIN": DMSERVE, "TINY_OUT": os.path.join(OUT, "reference")}, timeout=3600)
        tail = [l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL", "TINY CHECK", "NOT SUPPORTED"))]
        open(os.path.join(OUT, "reference.txt"), "w").write(r.stdout + r.stderr)
        ok = r.returncode == 0 and any("TINY CHECK PASSED" in l for l in tail)
        npass = sum(l.startswith("PASS") for l in tail)
        nns = [l[14:] for l in tail if l.startswith("NOT SUPPORTED")]
        status = "FAIL" if not ok else "not supported for this family" if nns else "PASS"
        verdict("reference", status, (f"not supported: {'; '.join(nns)}; " if nns else "") + f"{npass} comparisons passed (the rest information)"
                + ("" if ok else " | " + (r.stdout + r.stderr)[-400:].replace("\n", " ")))

# ---------------------------------------------------------------------------------------------- invariance
if wanted("invariance"):
    try:
        rec = os.path.join(T, "rec.bin")
        base, _ = logits("base", 300, "--cache-gb", f"{ALL_GB:.6f}", "--record", rec)
        FEW = ["--cache-gb", f"{FEW_GB:.6f}"]
        variants = [("few slots", FEW), ("few slots, 1 thread", FEW + ["--threads", "1"]), ("few slots, 5 threads", FEW + ["--threads", "5"]),
                    ("few slots, run-ahead", FEW + ["--run-ahead", str(K)])]
        if RT:
            variants += [("few slots, router names", FEW + RT), ("few slots, router names + run-ahead", FEW + RT + ["--run-ahead", str(K)])]
        variants += [("oracle lead 1", FEW + ["--prefetch", "oracle", "--oracle", rec, "--lead", "1"]),
                     ("oracle lead token", ["--cache-gb", f"{TOK_GB:.6f}", "--prefetch", "oracle", "--oracle", rec, "--lead", "token"])]
        # the spare tier on (store.h): reads land in probation, reuse promotes into the slab; a probation budget the
        # reads overflow (the oldest dropped) and one that holds every record the slab does not
        SPARE_LO = ["--spare-mem", "1", "--spare-gb", f"{0.2 * sum(rb) * meta['experts'] / 1e9:.6f}"]
        SPARE_HI = ["--spare-mem", "1", "--spare-gb", f"{0.8 * sum(rb) * meta['experts'] / 1e9:.6f}"]
        spare = [("few slots, spare tier (small budget)", FEW + SPARE_LO), ("few slots, spare tier (every record)", FEW + SPARE_HI),
                 ("few slots, spare tier, 1 thread", FEW + SPARE_LO + ["--threads", "1"])]
        if RT:
            spare += [("few slots, spare tier, router names + run-ahead", FEW + SPARE_LO + RT + ["--run-ahead", str(K)])]
        variants += spare
        bad, reads, tier, pageins = [], [], [], []
        for nm, fl in variants:
            g, js = logits("v", 300, *fl)
            reads.append(js.get("reads", 0))
            if g != base:
                bad.append(nm)
            if "--spare-mem" in fl:  # the tier must have run: records promoted, reuses served from the file cache
                sm = js.get("spare_memory") or {}
                tier.append(sm.get("promotions_per_token", 0))
                pageins.append(sm.get("pageins", -1))
                if not (sm.get("promotions_per_token", 0) > 0 and sm.get("reuses_per_token", 0) > 0):
                    bad.append(f"{nm}: the tier did not run as expected ({json.dumps(sm)[:300]})")
        if RT:  # names' reads held 3 ms in flight (one reader), as on a busy machine: a miss waits for one to land
            b40, _ = logits("base40", 40, "--cache-gb", f"{ALL_GB:.6f}")
            g, _ = logits("v40", 40, *FEW, *RT, "--run-ahead", str(K), "--readers", "1", env={"DMENG_TEST_SPEC_READ_US": "3000"})
            if g != b40:
                bad.append("few slots, router names + run-ahead, names' reads held in flight (40 positions)")
            g, _ = logits("v40s", 40, *FEW, *SPARE_LO, *RT, "--run-ahead", str(K), "--readers", "1", env={"DMENG_TEST_SPEC_READ_US": "3000"})
            if g != b40:
                bad.append("few slots, spare tier, router names + run-ahead, names' reads held in flight (40 positions)")
        g0, _ = greedy("g0", "--cache-gb", f"{ALL_GB:.6f}")
        g1, _ = greedy("g1", *FEW, *RT, "--threads", "1")
        if g0 != g1:
            bad.append("greedy ids, few slots / names / 1 thread")
        g2, _ = greedy("g2", *FEW, *SPARE_LO, *RT)
        if g0 != g2:
            bad.append("greedy ids, few slots / spare tier / names")
        verdict("invariance", "FAIL" if bad else "PASS",
                (f"differ: {', '.join(bad)}" if bad else f"{len(variants) + 1} runs identical (300 positions; reads per run {min(reads)}-{max(reads)})"
                           + ("; names' reads held in flight: identical (40 positions)" if RT else "")
                           + f"; spare tier on in {len(tier)} (promotions per token {min(tier):.2f}-{max(tier):.2f}, process page-ins {min(pageins)}-{max(pageins)}) and the greedy ids")
                + ("" if RT else "; the folder has no router file"))
    except Exception as e:
        verdict("invariance", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- prefill
if wanted("prefill"):
    try:
        bad, done = [], []
        for nm, extra in (("CPU", []), ("CPU, sub-chunks of 64", ["--chunk", "64"]), ("GPU share 0.4 / 0.4", ["--gpu-frac", "0.4", "--gpu-experts", "0.4"]),
                          ("few slots", ["--cache-gb", f"{FEW_GB:.6f}"])):
            cmd = [DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--mem", "plain", "--ctx", "2048", "--chunk", "256",
                   "--span", "8192", "--check", os.path.join(T, "ids1100.txt"), "--gen", "16", "--threads", "4", "--readers", "2", *extra]
            r = run(cmd, env={"DMENG_QSA_PAR_MIN": "16"})
            try:
                js = last_json(r)
                ok = js["logit_bits_differ"] == 0 and js["fixed_state_bytes_differ"] == 0 and js["kv_bytes_differ"] == 0 and js["greedy_identical"]
            except Exception as e:
                ok, js = False, {"error": str(e)[-300:]}
            (done if ok else bad).append(nm)
        gb, _ = greedy("pb", "--cache-gb", f"{ALL_GB:.6f}", "--prefill-chunk", "64", prompt=1100)
        gt, _ = greedy("pt", "--cache-gb", f"{ALL_GB:.6f}", "--prefill-chunk", "1", prompt=1100)
        (done if gb == gt else bad).append("dmeng ids batched == token by token")
        verdict("prefill", "FAIL" if bad else "PASS", ("differ: " + ", ".join(bad) + " | " if bad else "") + f"equal: {', '.join(done)} (1,100 positions)")
    except Exception as e:
        verdict("prefill", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- checkpoint
if wanted("checkpoint"):
    try:
        ck = os.path.join(T, "ck.txt")
        r3 = random.Random(5)
        open(ck, "w").write(",".join(str(r3.randrange(LO, V)) for _ in range(1200)))
        r = run([DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--mem", "plain", "--ctx", "2048", "--chunk", "128",
                 "--span", "8192", "--ckpt-every", "256", "--ckpt-gb", "0", "--check-ckpt", ck, "--gen", "8", "--threads", "4", "--readers", "2"])
        js = last_json(r)
        rows = ", ".join(f"{k} {js[k]['source']} reused {js[k]['reused']}" for k in ("long", "identical", "diverging"))
        sc = []
        for pc in ("1", "64"):
            j = dmeng("--prompt-file", os.path.join(T, "ids1100.txt"), "--n", "24", "--state-check", "--prefill-chunk", pc, "--cache-gb", f"{ALL_GB:.6f}")
            sc.append(j.get("state_check", {}).get("same_ids") is True)
        # a caller whose first mark moves with every call (a typing position): through the C API, 10 syncs of a growing
        # prefix, each marked first at its own end; one pinned checkpoint at a time (each new pin releases the last) and
        # the checkpoints' memory within the budget (ckpt_gb tiny: the budget is ckpt_min fixed states)
        jm = os.path.join(T, "moving_marks.json")
        r = run([DMGATE, "--model", M, "--profile", "none", "--prompt-ids", os.path.join(T, "ids1100.txt"), "--moving-marks", "10", "--ckpt-gb", "1e-9",
                 "--cache-gb", f"{ALL_GB:.6f}", "--mem", "plain", "--ctx", "2048", "--threads", "2", "--json", jm])
        mv = [x["prefix_cache"] for x in json.load(open(jm))["moving"]] if r.returncode == 0 else []
        pins = [x["pinned_checkpoints"] for x in mv]
        within = all(x["checkpoint_mb"] <= x["checkpoint_budget_mb"] + 1e-9 for x in mv)
        moving_ok = len(mv) == 10 and pins == [1] * 10 and within and mv[-1]["pins_released"] == 9
        mrow = (f"moving first mark through the C API: pinned {pins[-1] if pins else '-'} after {len(mv)} syncs (max {max(pins) if pins else '-'}), "
                f"released {mv[-1]['pins_released'] if mv else '-'}, checkpoints {mv[-1]['checkpoint_mb'] if mv else '-'} of "
                f"{mv[-1]['checkpoint_budget_mb'] if mv else '-'} MB" if mv else f"moving first mark: {r.stderr[-200:]}")
        ok = bool(js["passed"]) and all(sc) and moving_ok
        verdict("checkpoint", "PASS" if ok else "FAIL", f"dmserve --check-ckpt {'passed' if js['passed'] else 'FAILED'} ({rows}); dmeng --state-check same ids: {sc}; {mrow}")
    except Exception as e:
        verdict("checkpoint", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- server
DRIVE = os.path.join(REPO, "bench", "drafts", "worker_drive.py")


def worker(name, reqs, *flags, model=None, profile="none"):
    rq, op = os.path.join(T, name + ".reqs.json"), os.path.join(T, name + ".json")
    json.dump(reqs, open(rq, "w"))
    r = run([sys.executable, DRIVE, op, rq, "--", DMSERVE, "--model", model or M, "--profile", profile, "--cache-gb", f"{FEW_GB:.6f}", "--ctx", "2048",
             "--threads", "2", "--readers", "2", "--mem", "plain", "--res-mem", "plain",
             # (under a profile the server keeps its prefix cache, hot set and pass costs in the engine folder: not here)
             *(["--disk", "none", "--hotset", "none", "--draft-cost", "none"] if profile != "none" else []), *flags])
    if r.returncode:
        raise RuntimeError(f"worker {' '.join(flags)}: {(r.stdout + r.stderr)[-500:]}")
    return json.load(open(op))


if wanted("server"):
    try:
        p = ids[120]
        w = worker("srv", [{"ids": p, "max_new": 40}, {"ids": p, "max_new": 40, "temp": 1, "top_p": 0.95, "top_k": 20, "seed": 3}])
        g, _ = greedy("sg", "--cache-gb", f"{FEW_GB:.6f}", n=40, prompt=120)
        s, _ = greedy("ss", "--cache-gb", f"{FEW_GB:.6f}", "--temp", "1", "--top-p", "0.95", "--top-k", "20", "--seed", "3", n=40, prompt=120)
        sg, ss = ",".join(map(str, w[0]["gen"])), ",".join(map(str, w[1]["gen"]))
        same_g, same_s = sg == g, ss == s
        verdict("server", "PASS" if same_g and same_s else "FAIL", f"worker ids == dmeng ids: greedy {same_g}, sampled {same_s} (40 tokens after 120)")
    except Exception as e:
        verdict("server", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- drafting
def declared(name, drafts=None, edit=None):
    """The tiny folder (symlinks) with a manifest whose every profile declares `drafts` (the resolved-default path), or
    edited by `edit` (a function of the manifest)."""
    F = os.path.join(T, name)
    shutil.rmtree(F, ignore_errors=True)
    os.makedirs(F)
    for e in os.listdir(M):
        if e == ".dmeng":  # its own engine folder (the server writes its prefix cache and hot set there), the derived files linked
            os.makedirs(os.path.join(F, e))
            for d in os.listdir(os.path.join(M, e)):
                if d.endswith(".bin"):
                    os.symlink(os.path.join(M, e, d), os.path.join(F, e, d))
        elif e != "moved-store.json":
            os.symlink(os.path.join(M, e), os.path.join(F, e))
    m2 = json.loads(json.dumps(man))
    if drafts is not None:
        for pv in m2["profiles"].values():
            pv["drafts"] = drafts
    if edit:
        edit(m2)
    json.dump(m2, open(os.path.join(F, "moved-store.json"), "w"), indent=1)
    return F


PB = ["--profile", "mac"] + BASE[2:]  # BASE under the mac profile (the folder's declared defaults apply)
LOOKUP, HEAD = {"lookup": True, "head": False}, {"lookup": False, "head": True}
if wanted("drafting"):
    if not caps.get("verify"):
        refusals = []
        for nm, cmd in (("dmeng --draft lookup", [DMENG, "--model", M, *BASE, "--prompt-ids", "11,12,13", "--n", "4", "--draft", "lookup", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmeng --drafts 1", [DMENG, "--model", M, *BASE, "--prompt-ids", "11,12,13", "--n", "4", "--drafts", "1", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmeng --verify-check", [DMENG, "--model", M, *BASE, "--prompt-ids", "11,12,13", "--n", "4", "--verify-check", "8", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmserve --draft lookup", [DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--draft", "lookup"]),
                        ("dmserve --drafts 1", [DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--drafts", "1"]),
                        ("dmeng, the profile declaring lookup", [DMENG, "--model", declared("dl", LOOKUP), *PB, "--prompt-ids", "11,12,13", "--n", "4", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmserve, the profile declaring lookup", [DMSERVE, "--model", declared("dl", LOOKUP), "--profile", "mac", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("the C API run:draft=lookup", [DMGATE, "--model", M, "--profile", "none", "--prompt-ids", os.path.join(T, "ids40.txt"), "--n", "4",
                                                        "--cache-gb", f"{ALL_GB:.6f}", "--draft", "lookup"]),
                        ("the C API, the profile declaring lookup", [DMGATE, "--model", declared("dl", LOOKUP), "--profile", "mac", "--prompt-ids",
                                                                     os.path.join(T, "ids40.txt"), "--n", "4", "--cache-gb", f"{ALL_GB:.6f}"])):
            r = run(cmd, inp="")
            refusals.append((nm, r.returncode != 0 and "not supported for this family" in r.stderr))
        ok = all(x for _, x in refusals)
        verdict("drafting", "not supported for this family" if ok else "FAIL",
                "no multi-position verify pass; refused loudly by " + ", ".join(n for n, x in refusals if x) + ("" if ok else " | NOT refused: " + ", ".join(n for n, x in refusals if not x)))
    else:
        try:
            bad, okd = [], []
            P = [str(rnd.randrange(LO, V)) for _ in range(60)]
            for nm, fl in (("i16", []), ("f16", ["--prec", "f16"]), ("screened head", ["--fast-head", "1"]), ("previous pass", ["--verify-old", "1"])) + \
                          ((("f32", ["--prec", "f32"]),) if caps.get("ref_path") else ()):
                r = run([DMENG, "--model", M, *BASE, "--prompt-ids", ",".join(P), "--n", "8", "--cache-gb", f"{FEW_GB:.6f}", *fl, "--verify-check", "150"])
                l = [x for x in r.stderr.splitlines() if "verify_check" in x]
                (okd if r.returncode == 0 and l and '"ok":true' in l[-1] else bad).append(f"verify-check {nm}")
            for sm, S in (("greedy", []), ("sampled", ["--temp", "1", "--top-k", "20", "--top-p", "0.95", "--seed", "7"])):
                plain, _ = greedy("dp", "--cache-gb", f"{FEW_GB:.6f}", *S, n=100, prompt=40)
                g = plain.split(",")
                frag = os.path.join(T, "frag.txt")  # tool-call fragments made of plain decoding's own output, every 6th token changed
                open(frag, "w").write(" ".join([str(x) for x in ids[40][-6:]] + [t if i % 6 != 5 else str(LO + (int(t) + 1 - LO) % (V - LO)) for i, t in enumerate(g)]))
                for d in ("fixed", "lookup"):
                    for dm in ("1", "7"):
                        out, js = greedy("dd", "--cache-gb", f"{FEW_GB:.6f}", *S, "--draft", d, "--draft-max", dm, "--skeleton", frag, n=100, prompt=40)
                        nv = js.get("draft", {}).get("verifies", 0)
                        (okd if out == plain and (nv > 0 or d == "lookup") else bad).append(f"{sm} --draft {d} --draft-max {dm} ({nv} passes)")
            # --drafts 1 means the same in every front end (dmserve: --draft lookup), also in a folder holding a
            # model's draft head file (dmeng used to switch to --draft both there, by the model's name)
            F = os.path.join(T, "drafts1")
            shutil.rmtree(F, ignore_errors=True)
            os.makedirs(F)
            for e in os.listdir(M):
                os.symlink(os.path.join(M, e), os.path.join(F, e))
            if not os.path.exists(os.path.join(F, "mtp.safetensors")):
                os.symlink(os.path.join(M, "moved-store.json"), os.path.join(F, "mtp.safetensors"))
            plain, _ = greedy("d1p", "--cache-gb", f"{FEW_GB:.6f}", n=60, prompt=40)
            r1 = run([DMENG, "--model", F, *BASE, "--prompt-file", os.path.join(T, "ids40.txt"), "--n", "60", "--cache-gb", f"{FEW_GB:.6f}",
                      "--drafts", "1", "--out-ids", os.path.join(T, "d1.ids")])
            d1 = last_json(r1).get("draft", {}) if r1.returncode == 0 else {}
            ok1 = r1.returncode == 0 and d1.get("mode") == "lookup" and open(os.path.join(T, "d1.ids")).read().strip() == plain
            (okd if ok1 else bad).append(f"--drafts 1 = lookup beside a draft head file (mode {d1.get('mode')}, rc {r1.returncode})")
            r5 = random.Random(5)
            q = [r5.randrange(LO, V) for _ in range(50)]
            reqs = [{"ids": q, "max_new": 60}, {"ids": [LO, LO + 1, LO + 2] + q[:20], "max_new": 40, "continue": True},
                    {"ids": q[:30], "max_new": 40, "skel": [q[10:30], [LO + 1, LO + 2, LO + 3]]}, {"ids": q, "max_new": 60},
                    {"ids": q, "max_new": 60, "temp": 1, "top_p": 0.95, "top_k": 20, "seed": 3}]
            base = worker("wnone", reqs, "--draft", "none")
            for d in ("fixed", "lookup"):
                w = worker("w" + d, reqs, "--draft", d)
                acc = sum(x["E"]["json"].get("draft", {}).get("accepted", 0) for x in w)
                ([okd, bad][not ([x["gen"] for x in w] == [x["gen"] for x in base] and acc > 0)]).append(f"server --draft {d} ({acc} accepted)")
            # drafting as the profile's declared default (no flag): the same ids as plain decoding under the same profile,
            # greedy and sampled, in dmeng and the server; the verify check on that path; the resolved value reported
            FD = declared("dl", LOOKUP)
            for sm, S in (("greedy", []), ("sampled", ["--temp", "1", "--top-k", "20", "--top-p", "0.95", "--seed", "7"])):
                outs = {}
                for nm, mdl, fl in (("plain", M, ["--drafts", "0"]), ("declared", FD, [])):
                    op = os.path.join(T, f"dd_{nm}.ids")
                    r = run([DMENG, "--model", mdl, *PB, "--prompt-file", os.path.join(T, "ids40.txt"), "--n", "100", "--cache-gb", f"{FEW_GB:.6f}",
                             *S, *fl, "--out-ids", op])
                    outs[nm] = (open(op).read().strip() if r.returncode == 0 else f"rc {r.returncode}: {r.stderr[-200:]}", last_json(r) if r.returncode == 0 else {})
                js = outs["declared"][1]
                ok = (outs["declared"][0] == outs["plain"][0] and js.get("draft", {}).get("mode") == "lookup" and js["resolved"]["draft"] == "lookup"
                      and js["resolved"]["drafts"] == LOOKUP and js["preset"]["drafts"] == LOOKUP and outs["plain"][1]["resolved"]["draft"] == "none")
                (okd if ok else bad).append(f"{sm}, the profile declaring lookup ({js.get('draft', {}).get('verifies', 0)} passes)")
            r = run([DMENG, "--model", FD, *PB, "--prompt-ids", ",".join(P), "--n", "8", "--cache-gb", f"{FEW_GB:.6f}", "--verify-check", "150"])
            l = [x for x in r.stderr.splitlines() if "verify_check" in x]
            (okd if r.returncode == 0 and l and '"ok":true' in l[-1] else bad).append("verify-check, the profile declaring lookup")
            plain_w = worker("wdp", reqs, "--draft", "none", profile="mac")
            w = worker("wdd", reqs, model=FD, profile="mac")
            acc = sum(x["E"]["json"].get("draft", {}).get("accepted", 0) for x in w)
            ([okd, bad][not ([x["gen"] for x in w] == [x["gen"] for x in plain_w] and acc > 0)]).append(f"server, the profile declaring lookup ({acc} accepted)")
            # the C API (dm_gate: dm_step + dm_sample as the app decodes, against dm_step_n): the same ids with drafts off and on,
            # greedy and sampled (the C sampler, penalties included), tokens per step above 1 where drafts are kept, dmeng's
            # counters in the stats' draft block; fragments: the reply as a given text (every 6th token changed); repeat: the
            # same prompt regenerated in one engine, the lookup copying the session's earlier reply
            CB = ["--profile", "none", "--threads", "4", "--readers", "2", "--mem", "plain", "--ctx", "2048", "--cache-gb", f"{FEW_GB:.6f}",
                  "--model-options", "run:draft-cost=none"]

            def gate(name, *flags, model=None, prompt=40, n=100, env=None):
                op, jp = os.path.join(T, name + ".ids"), os.path.join(T, name + ".json")
                r = run([DMGATE, "--model", model or M, "--prompt-ids", os.path.join(T, f"ids{prompt}.txt"), "--n", str(n), *flags, "--out", op, "--json", jp], env=env)
                if r.returncode:
                    raise RuntimeError(f"dm_gate {' '.join(flags)}: rc {r.returncode} {(r.stdout + r.stderr)[-400:]}")
                return open(op).read().strip(), json.load(open(jp))
            COUNTERS = ("offered", "declined", "drafted", "accepted", "verifies", "tokens_per_step")
            for sm, S in (("greedy", []), ("sampled", ["--temp", "1", "--top-k", "20", "--top-p", "0.95", "--seed", "7"]),
                          ("sampled with penalties", ["--temp", "0.8", "--top-k", "40", "--seed", "5", "--repetition-penalty", "1.1",
                                                      "--presence-penalty", "0.3", "--frequency-penalty", "0.1"])):
                plain, pj = gate("cp", *CB, *S)
                g = plain.split(",")
                frag = os.path.join(T, "cfrag.txt")
                open(frag, "w").write(" ".join([str(x) for x in ids[40][-6:]] + [t if i % 6 != 5 else str(LO + (int(t) + 1 - LO) % (V - LO)) for i, t in enumerate(g)]) + "\n")
                off, oj = gate("cn", *CB, *S, "--step-n", "1")  # dm_step_n without drafting: dm_step + dm_sample, no draft block
                (okd if off == plain and "draft" not in oj["stats"] and "draft" not in oj["stats"]["policy"] and "draft" not in pj["stats"] else bad).append(
                    f"C API {sm}: dm_step_n without drafting = dm_step + dm_sample, no draft block")
                for d in ("fixed", "lookup"):
                    out, js = gate("cd", *CB, *S, "--draft", d, "--skeleton", frag)
                    dj = js["stats"].get("draft", {})
                    ok = out == plain and all(k in dj for k in COUNTERS) and js["stats"]["policy"].get("draft") == d and js["stats"]["resolved"]["draft"] == d
                    if d == "fixed":
                        ok = ok and dj["tokens_per_step"] > 1 and js["tokens_per_call"] > 1
                    (okd if ok else bad).append(f"C API {sm} drafts {d} with fragments ({dj.get('accepted')} kept, {js['tokens_per_call']:.2f} tokens per step)")
                out, js = gate("cr", *CB, *S, "--draft", "lookup", "--repeat", "2")
                rp = (js.get("repeats") or [{}])[0]
                rd = rp.get("draft", {})
                ok = out == plain and js.get("repeats_identical") and rp.get("tokens_per_call", 0) > 1 and rd.get("verifies_by_source", {}).get("lookup", 0) > 0
                (okd if ok else bad).append(f"C API {sm} lookup drafts on a regenerated reply ({rd.get('accepted')} kept, {rp.get('tokens_per_call', 0):.2f} tokens per step)")
            # the profile's declared drafting on through the C API (no option): the same ids as drafts off under that profile
            FDc = declared("dlc", LOOKUP)
            PC = ["--profile", "mac"] + CB[2:]
            plain, _ = gate("cdp", *PC, "--model-options", "run:drafts=0;run:draft-cost=none", model=FDc)
            out, js = gate("cdd", *PC, "--step-n", "1", "--repeat", "2", model=FDc)
            rp = (js.get("repeats") or [{}])[0]
            ok = out == plain and js["stats"]["resolved"]["draft"] == "lookup" and js["stats"]["policy"]["preset"]["drafts"] == LOOKUP and js.get("repeats_identical") and rp.get("tokens_per_call", 0) > 1
            (okd if ok else bad).append(f"C API, the profile declaring lookup ({rp.get('tokens_per_call', 0):.2f} tokens per step on the regenerated reply)")
            # memory: on iOS (DMENG_TEST_AVAIL_OPEN) the slab rule gives drafting's memory out of the slab; drafts off: no reserve
            AVd = {"DMENG_TEST_AVAIL_OPEN": str(int(4e9))}
            sl = {}
            for d in ("none", "lookup"):
                _, js = gate("cs" + d, "--profile", "phone" if "phone" in man["profiles"] else "mac", "--threads", "2", "--mem", "plain", "--cache-gb", "auto",
                             "--model-options", "run:draft-cost=none", "--draft", d, n=8, env=AVd)
                sl[d] = js["stats"]["policy"]
            rv = sl["lookup"]["slab_rule"].get("draft_reserve_gb", 0)
            ok = "draft_reserve_gb" not in sl["none"]["slab_rule"] and rv > 0 and abs(sl["none"]["slab_rule"]["gb"] - sl["lookup"]["slab_rule"]["gb"] - rv) < 1e-4
            (okd if ok else bad).append(f"C API on iOS: the slab gives drafting {rv * 1e3:.1f} MB")
            # what a value needs and the folder lacks fails at open: a declared draft head without the head's files
            # (no tiny folder carries one; the 35B and Flash-Next have none at all)
            FH = declared("dh", HEAD)
            for nm, cmd in (("dmeng", [DMENG, "--model", FH, *PB, "--prompt-ids", "11,12,13", "--n", "4", "--cache-gb", f"{ALL_GB:.6f}"]),
                            ("dmserve", [DMSERVE, "--model", FH, "--profile", "mac", "--cache-gb", f"{ALL_GB:.6f}"]),
                            ("the C API", [DMGATE, "--model", FH, "--profile", "mac", "--prompt-ids", os.path.join(T, "ids40.txt"), "--n", "4", "--cache-gb", f"{ALL_GB:.6f}"]),
                            ("the C API run:draft=head", [DMGATE, "--model", M, "--profile", "none", "--prompt-ids", os.path.join(T, "ids40.txt"), "--n", "4",
                                                          "--cache-gb", f"{ALL_GB:.6f}", "--draft", "head"])):
                r = run(cmd, inp="")
                (okd if r.returncode != 0 and "draft head" in r.stderr else bad).append(f"{nm} refuses a declared head it cannot load (rc {r.returncode})")
            verdict("drafting", "FAIL" if bad else "PASS", ("differ: " + "; ".join(bad) + " | " if bad else "") + f"{len(okd)} equal to plain decoding: " + "; ".join(okd))
        except Exception as e:
            verdict("drafting", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- presets
if wanted("presets"):
    try:
        ip = os.path.join(T, "pids.txt")
        open(ip, "w").write(",".join(str(LO + i) for i in range(8)))

        def resolved(front, prof, flags):
            j = os.path.join(T, "pr.json")
            if os.path.exists(j):
                os.remove(j)
            if front == "dmeng":
                r = run([DMENG, "--model", M, "--profile", prof, "--prompt-file", ip, "--n", "4", "--threads", "2", *flags, "--json", j])
                return json.load(open(j))["resolved"] if r.returncode == 0 else {"error": r.stderr[-300:]}
            if front == "capi":
                r = run([DMGATE, "--model", M, "--profile", prof, "--prompt-ids", ip, "--n", "4", "--threads", "2", *flags, "--json", j])
                return json.load(open(j))["stats"]["resolved"] if r.returncode == 0 else {"error": r.stderr[-300:]}
            r = run([DMSERVE, "--model", M, "--profile", prof, "--threads", "2", *flags], inp="", timeout=120)
            ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
            return json.loads(ready[0][6:])["resolved"] if ready else {"error": r.stderr[-300:]}

        def spell(front, flags):
            out, i, mo = [], 0, []
            while i < len(flags):
                f, v = flags[i], (flags[i + 1] if i + 1 < len(flags) else None)
                if f == "--gpu":
                    # one spelling in every front end (presets.h run_flag): dmeng is given the old alias, dmserve the name
                    out += [{"dmeng": "--gpu-frac", "dmserve": "--prefill-gpu-frac"}.get(front, "--gpu-frac"), v]; i += 2
                elif f == "--gpu-off":
                    out += ["--gpu-frac", "-1", "--gpu-experts", "-1"] if front == "capi" else ["--prefill-gpu", "off"]; i += 1
                elif f == "--ra-off":
                    out += ["--run-ahead", "-1" if front == "capi" else "0"]; i += 1
                elif f in ("--drafts", "--draft-max") and front == "capi":  # (no field of its own: the run-flag passthrough)
                    mo.append(f"run:{f[2:]}={v}"); i += 2
                else:
                    out += [f, v]; i += 2
            if mo:
                out += ["--model-options", ";".join(mo)]
            if front == "capi" and "--cache-gb" not in out:
                out += ["--cache-gb", "auto"]
            return out

        def sub(e, g, k=""):
            if isinstance(e, dict):
                if not isinstance(g, dict):
                    return [f"{k}: {g!r} != {e!r}"]
                return [x for kk, v in e.items() for x in sub(v, g.get(kk), f"{k}.{kk}")]
            if isinstance(e, float) or isinstance(g, float):
                return [] if g is not None and abs(float(g) - float(e)) < 1e-6 else [f"{k}: {g!r} != {e!r}"]
            return [] if g == e else [f"{k}: {g!r} != {e!r}"]

        bad, ncell, cross, n_capi_prio = [], 0, [], 0
        FRONT_SPECIFIC = set()  # (the prompt's sub-chunk is the model's declared one in all three: printed as 0)

        def slab_rule(prof):
            """The slab rule a profile resolves to, from the manifest (presets.h SlabRule): its cache_slab, else the
            built-in available rule."""
            sl = man["profiles"][prof].get("cache_slab") or {"rule": "available"}
            e = {"rule": sl["rule"], "min_gb": sl.get("min_gb", 0.6), "max_gb": sl.get("max_gb", 8.0)}
            if sl["rule"] == "footprint_target":
                e.update(target_gb=sl["target_gb"], non_cache_gb=sl["non_cache_gb"])
            else:
                e.update(state_gb=sl.get("state_gb", 0.3), margin_gb=sl.get("margin_gb", 0.9))
            return e

        def slab_exp(prof):  # as resolved.slab prints it: only when the profile's cache is a slab
            return slab_rule(prof) if man["profiles"][prof].get("cache") == "slab" else None
        for prof in man["profiles"]:
            r = run([DMENG, "--model", M, "--profile", prof, "--print-defaults"])
            P = json.loads(r.stdout)
            names = P["names"]
            exp_default = {"mode": P["engine_mode"], "eviction": P["eviction"], "prefill_gpu": P["prefill_gpu"],
                           "run_ahead": {k: P["run_ahead"][k] for k in ("picks", "depth", "budget", "max_tok", "sat_skip")},
                           "names": names if P["routers"] else None}
            if P["readers"] > 0:
                exp_default["readers"] = P["readers"]
            # the profile's declared drafting; the tiny folders' profiles declare it as the published manifests do (a family
            # without what it needs would fail at open: the drafting check)
            exp_default["drafts"] = P["drafts"]
            exp_default["slab"] = slab_exp(prof)  # (the mac profile: not printed, the block unchanged)
            # the profile's declared context positions (profiles.<p>.ctx: N or {"positions": N}; not printed when undeclared)
            cx = man["profiles"][prof].get("ctx")
            ctx_decl = (cx.get("positions") if isinstance(cx, dict) else cx) or None
            exp_default["ctx"] = ctx_decl
            # the profile's declared prompt GPU share, from the manifest (a build without Metal: CPU only, with a note)
            pg = man["profiles"][prof].get("prefill_gpu")
            if pg and not any("without Metal" in n for n in P["notes"]):
                exp_default["prefill_gpu"] = {"projections": pg.get("projections", 0), "experts": pg.get("experts", 0)}
            gpu = P["prefill_gpu"]
            cells = [("default", [], exp_default),
                     ("mode fast", ["--mode", "fast"], {"mode": json.loads(run([DMENG, "--model", M, "--profile", prof, "--mode", "fast", "--print-defaults"]).stdout)["engine_mode"]}),
                     ("top m 4", ["--m", "4"], {"names": {"rule": "top", "m": 4}} if P["routers"] else {"names": None}),
                     ("names off", ["--routers", "none"], {"routers": "", "names": None}),
                     ("cache 0.05", ["--cache-gb", "0.05"], {"cache_gb": 0.05, "cache_auto": False, "slab": None}),
                     ("evict freq", ["--evict", "freq"], {"eviction": "freq"}),
                     ("gpu 0.2", ["--gpu", "0.2"], {"prefill_gpu": {"projections": 0.2, "experts": gpu["experts"]}}),
                     ("gpu off", ["--gpu-off"], {"prefill_gpu": {"projections": 0, "experts": 0}}),
                     ("run-ahead off", ["--ra-off"], {"run_ahead": {"picks": 0}}),
                     ("run-ahead 4", ["--run-ahead", "4"], {"run_ahead": {"picks": 4, "budget": P["run_ahead"]["budget"]}}),
                     ("run-ahead limits", ["--run-ahead-budget", "0.5", "--run-ahead-max", "3", "--run-ahead-sat-skip", "1"],
                      {"run_ahead": {"budget": 0.5, "max_tok": 3, "sat_skip": True}}),
                     ("readers 2", ["--readers", "2"], {"readers": 2}),
                     ("ctx 2048", ["--ctx", "2048"], {"ctx": 2048 if ctx_decl else None}),  # an explicit ctx over the declared one
                     ("drafts 0", ["--drafts", "0"], {"drafts": {"lookup": False, "head": False}, "draft": "none"})]
            if caps.get("verify"):  # (the pass width: printed while drafting is on)
                cells.append(("drafts 1 max 3", ["--drafts", "1", "--draft-max", "3"], {"drafts": {"lookup": True, "head": False}, "draft": "lookup", "draft_max": 3}))
            if routers:
                cells.append(("router file", ["--routers", routers[0]], {"routers": routers[0], "names": {"rule": "top", "m": 8}}))
            for nm, fl, exp in cells:
                got = {fe: resolved(fe, prof, spell(fe, fl)) for fe in ("dmeng", "capi", "dmserve")}
                ncell += 1
                for fe, g in got.items():
                    e = sub(exp, g)
                    if e:
                        bad.append(f"{prof} {nm} {fe}: {'; '.join(e)}")
                # priority reads in force: the C API's own rule (dmeng_c.cpp) leaves them off in exact mode with in-token
                # routers only, where dmeng and dmserve apply defaults.priority_reads (left as it runs until measured)
                capi_prio = all(isinstance(g, dict) for g in got.values()) and got["capi"].get("mode") == "lossless" and not got["capi"].get("cross_token") \
                    and (got["capi"].get("priority_reads"), got["dmeng"].get("priority_reads"), got["dmserve"].get("priority_reads")) == (0, 1, 1)
                n_capi_prio += capi_prio
                norm = {fe: json.dumps({k: v for k, v in g.items() if k not in FRONT_SPECIFIC and not (k == "cache_gb" and g.get("cache_auto"))
                                        and not (capi_prio and k == "priority_reads")}, sort_keys=True)
                        for fe, g in got.items()}
                if len(set(norm.values())) > 1:
                    keys = set().union(*[g.keys() for g in got.values()]) - FRONT_SPECIFIC
                    cross.append(f"{prof} {nm}: " + ", ".join(f"{k} {{{', '.join(f'{fe} {json.dumps(got[fe].get(k))}' for fe in got)}}}" for k in sorted(keys)
                                                             if len({json.dumps(got[fe].get(k)) for fe in got}) > 1))
        # the slab on iOS (the C API as on an iPhone, DMENG_TEST_AVAIL_OPEN: 4.0 GB available at open), every profile: the
        # profile's declared rule (cache_slab), else the built-in available rule (the mac profile too, as the C API sizes
        # a slab wherever iOS reports the memory): policy.slab_rule prints the rule, its inputs and the slab, and the
        # slab follows from them; resolved.slab is the rule, resolved.cache_gb the slab
        AV = 4.0e9
        for prof in man["profiles"]:
            if os.path.exists(j := os.path.join(T, "pr.json")):
                os.remove(j)
            r = run([DMGATE, "--model", M, "--profile", prof, "--prompt-ids", ip, "--n", "2", "--threads", "2", "--cache-gb", "auto", "--json", j],
                    env={"DMENG_TEST_AVAIL_OPEN": str(int(AV))})
            st = json.load(open(j))["stats"] if r.returncode == 0 else None
            if not st:
                bad.append(f"{prof} capi iOS slab: rc {r.returncode} {r.stderr[-200:]}")
                continue
            sr, rs = st["policy"].get("slab_rule") or {}, st["resolved"]
            e = slab_rule(prof)
            clamp = lambda x: min(e["max_gb"], max(e["min_gb"], x))
            want_gb = clamp(e["target_gb"] - e["non_cache_gb"]) if e["rule"] == "footprint_target" else \
                clamp(sr.get("available_gb", -9) - sr.get("resident_gb", 9) - sr.get("router_gb", 9) - e["state_gb"] - e["margin_gb"])
            # the start rule printed is the slab's own (the C API applies no automatic start rule and runs no watcher)
            errs = sub(e, sr) + sub(e, rs.get("slab")) + sub("slab:" + e["rule"], rs.get("cache_start"))
            # (slab_rule.gb and resolved.cache_gb are printed to 6 digits, dm_stats_json's slab_auto_gb to 3)
            for k, g, tol in (("policy.slab_rule.gb", sr.get("gb"), 1e-4), ("resolved.cache_gb", rs.get("cache_gb"), 1e-4),
                              ("policy.slab_auto_gb", st["policy"].get("slab_auto_gb"), 5e-3)):
                if g is None or abs(g - want_gb) > tol:
                    errs.append(f"{k}: {g!r} != {want_gb:.6f}")
            # the profile's declared prompt GPU share through the C API as on an iPhone (phone and phone-small)
            pg = man["profiles"][prof].get("prefill_gpu")
            if pg and not any("without Metal" in n for n in json.loads(run([DMENG, "--model", M, "--profile", prof, "--print-defaults"]).stdout)["notes"]):
                errs += sub({"projections": pg.get("projections", 0), "experts": pg.get("experts", 0)}, rs.get("prefill_gpu"))
            if e["rule"] == "available":
                errs += sub(AV / 1e9, sr.get("available_gb")) + ([] if sr.get("resident_gb", 0) > 0 else ["resident_gb not > 0"])
            if errs:
                bad.append(f"{prof} capi iOS slab: {'; '.join(errs)}")
            ncell += 1
        # a profile declaring drafting and its pass width (drafts.max): resolved the same in the three front ends (the
        # family's drafting check runs it; a family without a verify pass refuses it at open, the drafting check)
        if caps.get("verify"):
            FD = declared("pd", {"lookup": True, "head": False, "max": 4})
            j = os.path.join(T, "pr.json")
            gotd = {}
            r = run([DMENG, "--model", FD, "--profile", "mac", "--prompt-file", ip, "--n", "4", "--threads", "2", "--json", j])
            gotd["dmeng"] = json.load(open(j))["resolved"] if r.returncode == 0 else {"error": r.stderr[-200:]}
            r = run([DMGATE, "--model", FD, "--profile", "mac", "--prompt-ids", ip, "--n", "4", "--threads", "2", "--cache-gb", "auto",
                     "--model-options", "run:draft-cost=none", "--json", j])
            gotd["capi"] = json.load(open(j))["stats"]["resolved"] if r.returncode == 0 else {"error": r.stderr[-200:]}
            r = run([DMSERVE, "--model", FD, "--profile", "mac", "--threads", "2", "--draft-cost", "none", "--disk", "none", "--hotset", "none"], inp="", timeout=120)
            ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
            gotd["dmserve"] = json.loads(ready[0][6:])["resolved"] if ready else {"error": r.stderr[-200:]}
            for fe, g in gotd.items():
                e = sub({"draft": "lookup", "drafts": {"lookup": True, "head": False}, "draft_max": 4}, g)
                if e:
                    bad.append(f"the profile declaring lookup, max 4, {fe}: {'; '.join(e)}")
            ncell += 1
        # the prompt path's values under each profile: the profile's own prefill block over the model-level one, the same
        # in all three front ends (their "prefill_declared")
        for prof in man["profiles"]:
            # (want_pf: not `want`, the --only filter every later check reads)
            want_pf = {**{k: v for k, v in man.get("prefill", {}).items() if isinstance(v, (int, float, str, bool)) and k not in ("notes", "note")},
                    **{k: v for k, v in man["profiles"][prof].get("prefill", {}).items() if isinstance(v, (int, float, str, bool)) and k not in ("notes", "note")}}
            want_pf = {k: str(int(v)) if isinstance(v, bool) else (f"{v:.17g}" if isinstance(v, (int, float)) else v) for k, v in want_pf.items()}
            got = {}
            r = run([DMENG, "--model", M, "--profile", prof, "--prompt-file", ip, "--n", "2", "--threads", "2", "--json", j])
            got["dmeng"] = json.load(open(j)).get("prefill_declared") if r.returncode == 0 else r.stderr[-200:]
            r = run([DMGATE, "--model", M, "--profile", prof, "--prompt-ids", ip, "--n", "2", "--threads", "2", "--cache-gb", "auto", "--json", j])
            got["capi"] = json.load(open(j))["stats"]["policy"].get("prefill_declared") if r.returncode == 0 else r.stderr[-200:]
            r = run([DMSERVE, "--model", M, "--profile", prof, "--threads", "2"], inp="", timeout=120)
            ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
            got["dmserve"] = json.loads(ready[0][6:]).get("prefill_declared") if ready else r.stderr[-200:]
            for fe, g in got.items():
                if g != want_pf:
                    bad.append(f"{prof} prefill {fe}: {g} != {want_pf}")
            ncell += 1
        # a profile's own mode declaration (profiles.<p>.modes.fast) over the model-level modes.fast, in a temporary
        # manifest: that profile resolves to its string, every other profile to the folder's own declaration if it has one
        # (none of the published manifests does: the model-level mode, modes.fast), in all three front ends ("preset":
        # engine_mode, engine_mode_from; "resolved": mode); an explicit engine string is kept as given
        OVER = "approx:drop:w=0.12"
        op = "phone" if "phone" in man["profiles"] else next(iter(man["profiles"]))

        PER_MODE = {"exact": {"rule": "top", "m": 5}, "fast": {"rule": "confidence", "tau": 0.25, "candidates": 12}, "calibration": {"measured": "(a test value)"}}

        def over(m2):
            m2["profiles"][op]["modes"] = {"fast": {"definition": "the fast mode with this profile's own leave-out threshold", "engine": OVER,
                                                    "read_if": {"missing_routing_weight_over": 0.12}, "calibration": {"measured": "(a test value)"}}}
            m2["profiles"][op]["names"] = PER_MODE  # a names rule per mode (profiles.<p>.names: {"exact": rule, "fast": rule})

        FO = declared("pm", edit=over)
        model_fast = man["modes"]["fast"]["engine"]

        def preset_of(fe, prof, mode):
            if os.path.exists(j):
                os.remove(j)
            if fe == "dmeng":
                r = run([DMENG, "--model", FO, "--profile", prof, "--mode", mode, "--prompt-file", ip, "--n", "2", "--threads", "2", "--json", j])
                d = json.load(open(j)) if r.returncode == 0 else None
                return (d["preset"], d["resolved"]) if d else r.stderr[-200:]
            if fe == "capi":
                r = run([DMGATE, "--model", FO, "--profile", prof, "--mode", mode, "--prompt-ids", ip, "--n", "2", "--threads", "2", "--cache-gb", "auto", "--json", j])
                d = json.load(open(j))["stats"] if r.returncode == 0 else None
                return (d["policy"]["preset"], d["resolved"]) if d else r.stderr[-200:]
            r = run([DMSERVE, "--model", FO, "--profile", prof, "--mode", mode, "--threads", "2"], inp="", timeout=120)
            ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
            d = json.loads(ready[0][6:]) if ready else None
            return (d["preset"], d["resolved"]) if d else r.stderr[-200:]

        def own_fast(prof):  # the folder's own declaration of a profile's fast mode (the 35B's phone-small), else the model-level one
            e = man["profiles"][prof].get("modes", {}).get("fast", {}).get("engine")
            return (e, f"profiles.{prof}.modes.fast") if e else (model_fast, "modes.fast")

        cases = [(prof, "fast", (OVER, f"profiles.{op}.modes.fast") if prof == op else own_fast(prof)) for prof in man["profiles"]]
        cases.append((op, "approx:drop:w=0.2", ("approx:drop:w=0.2", "given")))  # an explicit engine string wins
        for prof, mode, (s, src) in cases:
            for fe in ("dmeng", "capi", "dmserve"):
                g = preset_of(fe, prof, mode)
                if not isinstance(g, tuple) or (g[0].get("engine_mode"), g[0].get("engine_mode_from"), g[1].get("mode")) != (s, src, s):
                    got_s = g if not isinstance(g, tuple) else (g[0].get("engine_mode"), g[0].get("engine_mode_from"), g[1].get("mode"))
                    bad.append(f"{prof} --mode {mode} with {op}'s own fast {fe}: {got_s} != {(s, src, s)}")
            ncell += 1
        # the names rule per mode: a profile's names may be one rule or {"exact": rule, "fast": rule}, taken for the mode in
        # force; resolved.names in all three front ends, in the folder (its own declarations: a rule per mode on the 35B's
        # phone-small and Inkling's mac) and in the temporary manifest above (op's per-mode rule), for --mode exact / fast
        # and an engine string
        m_over = json.loads(json.dumps(man))
        over(m_over)

        def names_exp(mm, prof, mode):
            for nb in (mm["profiles"][prof].get("names"), mm.get("defaults", {}).get("names")):
                if isinstance(nb, dict):
                    r = nb if "rule" in nb else nb.get(mode)
                    if r:
                        return {"rule": "confidence", "tau": r.get("tau", 0.5), "candidates": r.get("candidates", 16)} if r["rule"] == "confidence" \
                            else {"rule": "top", "m": r.get("m", 8)}
            return {"rule": "top", "m": 8}

        for fold, mm in ((M, man), (FO, m_over)):
            for prof in man["profiles"]:
                for mode, nmode in (("exact", "exact"), ("fast", "fast"), ("approx:drop:w=0.2", "fast")):
                    want_n = names_exp(mm, prof, nmode)
                    for fe in ("dmeng", "capi", "dmserve"):
                        if os.path.exists(j):
                            os.remove(j)
                        if fe == "dmeng":
                            r = run([DMENG, "--model", fold, "--profile", prof, "--mode", mode, "--prompt-file", ip, "--n", "2", "--threads", "2", "--json", j])
                            g = json.load(open(j))["resolved"] if r.returncode == 0 else None
                        elif fe == "capi":
                            r = run([DMGATE, "--model", fold, "--profile", prof, "--mode", mode, "--prompt-ids", ip, "--n", "2", "--threads", "2", "--cache-gb", "auto", "--json", j])
                            g = json.load(open(j))["stats"]["resolved"] if r.returncode == 0 else None
                        else:
                            r = run([DMSERVE, "--model", fold, "--profile", prof, "--mode", mode, "--threads", "2"], inp="", timeout=120)
                            ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
                            g = json.loads(ready[0][6:])["resolved"] if ready else None
                        if g is None:
                            bad.append(f"{prof} names --mode {mode} {fe}: {r.stderr[-200:]}")
                            continue
                        e = sub(want_n if g.get("routers") else None, g.get("names"))
                        if e:
                            bad.append(f"{prof} names --mode {mode} {fe}{' (per-mode test manifest)' if fold == FO else ''}: {'; '.join(e)}")
                    ncell += 1
        # the in-memory weights opened: where phone-small declares the 4-bit file (resident_q4) and the folder holds it, the
        # C API's resident_bits 0 (the profile decides) opens it and reports it (policy.resident_bits / policy.resident,
        # resolved.resident_bits / resolved.resident), as dmeng and dmserve do; an explicit resident_bits 8 still wins;
        # without the file the profile falls back to the 8-bit one, and says so
        if man["profiles"].get("phone-small", {}).get("resident") == "resident_q4":
            RQ = declared("rq4")
            r = run([os.path.join(B, "ms_derive"), "resident", RQ, "--bits", "4"])
            if r.returncode:
                bad.append(f"ms_derive resident --bits 4: {r.stderr[-200:]}")
            for fold, bits, wbits in ((RQ, 0, 4), (RQ, 8, 8), (M, 0, 8)):
                if os.path.exists(j):
                    os.remove(j)
                r = run([DMGATE, "--model", fold, "--profile", "phone-small", "--resident-bits", str(bits), "--prompt-ids", ip, "--n", "2", "--threads", "2",
                         "--cache-gb", "auto", "--json", j])
                d = json.load(open(j))["stats"] if r.returncode == 0 else None
                got = (d["policy"].get("resident_bits"), d["policy"].get("resident"), d["resolved"].get("resident_bits"), d["resolved"].get("resident")) if d else r.stderr[-200:]
                wid = "resident_q4" if wbits == 4 else "resident_q8"
                if got != (wbits, wid, wbits, wid):
                    bad.append(f"phone-small resident_bits {bits}{' (no 4-bit file)' if fold == M else ''} capi: {got} != {(wbits, wid, wbits, wid)}")
                ncell += 1
            for fe in ("dmeng", "dmserve"):  # the profile's file, opened and reported the same way
                if os.path.exists(j):
                    os.remove(j)
                if fe == "dmeng":
                    r = run([DMENG, "--model", RQ, "--profile", "phone-small", "--prompt-file", ip, "--n", "2", "--threads", "2", "--json", j])
                    g = json.load(open(j))["resolved"] if r.returncode == 0 else None
                else:
                    r = run([DMSERVE, "--model", RQ, "--profile", "phone-small", "--threads", "2"], inp="", timeout=120)
                    ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
                    g = json.loads(ready[0][6:])["resolved"] if ready else None
                got = (g.get("resident_bits"), g.get("resident")) if g else r.stderr[-200:]
                if got != (4, "resident_q4"):
                    bad.append(f"phone-small resident {fe}: {got} != (4, 'resident_q4')")
            ncell += 1
        detail = f"{ncell} cells x 3 front ends ({', '.join(man['profiles'])}); priority reads off in the C API's exact mode only (in-token routers): {n_capi_prio} cells"
        verdict("presets", "FAIL" if bad or cross else "PASS", detail + ("" if not bad else " | precedence: " + " || ".join(bad[:6]))
                + ("" if not cross else " | front ends differ: " + " || ".join(cross[:6])))
    except Exception as e:
        verdict("presets", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- files
if wanted("files"):
    try:
        bad = []
        listed = {d["id"]: d for d in man["derived"]}
        held = sorted(f for f in os.listdir(os.path.join(M, ".dmeng")) if f.endswith(".bin"))
        for f in held:
            did = f.split(".dmr1.")[0].split(".dmres1.")[0]
            if did not in listed or not listed[did].get("sha256"):
                bad.append(f"{f}: not listed with a sha256")
        q8 = man["resident"].get("q8_rule") or {}
        if not (q8.get("quantiser") == "mlx_affine" and q8.get("bits") == 8 and q8.get("group") == 64 and q8.get("input_cast") in ("f16", "bf16")):
            bad.append("resident.q8_rule not declared")
        # a corrupted copy of the resident: same size, one byte changed, in a fresh cache folder
        F = os.path.join(T, "files")
        shutil.rmtree(F, ignore_errors=True)
        os.makedirs(os.path.join(F, ".dmeng"))
        for e in os.listdir(M):
            if e != ".dmeng":
                os.symlink(os.path.join(M, e), os.path.join(F, e))
        res = [f for f in held if f.startswith("resident_q8.")][0]
        b = bytearray(open(os.path.join(M, ".dmeng", res), "rb").read())
        b[len(b) // 2] ^= 1
        open(os.path.join(F, ".dmeng", res), "wb").write(b)
        r = run([DMENG, "--model", F, *BASE, "--prompt-ids", "11,12,13", "--n", "2", "--cache-gb", f"{ALL_GB:.6f}"])
        refused = r.returncode != 0 and "sha256" in r.stderr
        if not refused:
            bad.append("a corrupted resident was not refused")
        verdict("files", "FAIL" if bad else "PASS", ("; ".join(bad) + " | " if bad else "") +
                f"{len(held)} engine-ready files listed with sha256; q8 rule {q8.get('input_cast')} cast; a corrupted resident refused: {refused}")
    except Exception as e:
        verdict("files", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- caps
if wanted("caps"):
    if not routers:
        verdict("caps", "not supported for this family", "no router file: nothing names experts")
    else:
        try:
            # per-layer caps of 2K + 4 slots, the routers naming K per step: the fixed reserve (2K + holds unpinned) refuses
            # most names; :fit (min(2K + holds, cap - names per step - the layer's holds)) admits the names the layer can
            # hold; ids unchanged
            cap_gb = (2 * K + 4) * sum(rb) / 1e9
            got = {}
            for caps in ("none", "uniform", "uniform:fit"):
                out, js = greedy("caps_" + caps.replace(":", "_"), "--cache-gb", f"{cap_gb:.6f}", "--routers", routers[0], "--m", str(K), "--layer-caps", caps, n=64)
                got[caps] = (out, js["pf_dropped_per_token"], js["pf_issued_per_token"])
            bad = []
            if len({v[0] for v in got.values()}) != 1:
                bad.append("ids differ between cap settings")
            # (a slot whose read is still in flight cannot be given: timing leaves a few refusals, under 1% of the names)
            if got["uniform:fit"][1] > 0.01 * got["none"][2]:
                bad.append(f"fit still refuses {got['uniform:fit'][1]:.2f} names per token")
            verdict("caps", "FAIL" if bad else "PASS", ("; ".join(bad) + " | " if bad else "") + "names refused / issued per token at 2K + 4 slots per layer: " +
                    ", ".join(f"{k} {v[1]:.2f} / {v[2]:.2f}" for k, v in got.items()) + "; ids identical")
        except Exception as e:
            verdict("caps", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- background
if wanted("background"):
    if not routers:
        verdict("background", "not supported for this family", "no router file: nothing names experts")
    else:
        try:
            # background reads (the routers' next ranks in the drive's idle time) at 2K + 4 slots per layer, with and
            # without per-layer caps: exact ids identical to no background reads, and the reads issued under caps too
            cap_gb = (2 * K + 4) * sum(rb) / 1e9
            got, bad = {}, []
            for caps in ("none", "uniform", "uniform:fit"):
                for bg in ("0", str(min(16, 2 * K + 2))):
                    out, js = greedy(f"bg_{caps.replace(':', '_')}_{bg}", "--cache-gb", f"{cap_gb:.6f}", "--routers", routers[0], "--m", str(K),
                                     "--layer-caps", caps, "--bg-ranks", bg, n=64)
                    b = js.get("drive", {}).get("background", {})
                    got[(caps, bg)] = (out, b.get("issued_per_token", 0.0), b.get("used_per_token", 0.0))
            # the confidence rule (Inkling's names) under caps: the ranks it left unnamed
            out, js = greedy("bg_uniform_conf", "--cache-gb", f"{cap_gb:.6f}", "--routers", routers[0], "--rt-tau", "0.5", "--layer-caps", "uniform",
                             "--bg-ranks", "16", n=64)
            b = js.get("drive", {}).get("background", {})
            got[("uniform, confidence rule", "16")] = (out, b.get("issued_per_token", 0.0), b.get("used_per_token", 0.0))
            if len({v[0] for v in got.values()}) != 1:
                bad.append("ids differ with background reads")
            # (how many are issued depends on the drive's idle time and the policy's values, so it is reported, not required;
            # under caps a background read leaves K unpinned in its layer, a name 2K + holds)
            verdict("background", "FAIL" if bad else "PASS", ("; ".join(bad) + " | " if bad else "") + "issued / used per token: " +
                    ", ".join(f"{c} {v[1]:.2f} / {v[2]:.2f}" for (c, bg), v in got.items() if bg != "0") + "; ids identical")
        except Exception as e:
            verdict("background", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- lock
if wanted("lock"):
    try:
        # the cache's slots locked one by one (--cache-start-gb: the serving worker's automatic rule does this) under a
        # memory-lock limit (ulimit -l) below the start size: the lock the OS refuses stops the budget there, the JSON
        # reports what is locked, and the ids are those of an unlimited run
        start_gb = 0.8 * ALL_GB
        lim_kb = int(0.9 * start_gb * 1e9 / 1024)  # (above every pool's floor: slots below it are used even unlocked, reported)
        res = {}
        for nm, lim in (("unlimited", "unlimited"), ("limited", str(lim_kb))):
            op = os.path.join(T, f"lock_{nm}.ids")
            jp = os.path.join(T, f"lock_{nm}.json")
            cmd = " ".join([DMENG, "--model", M, *BASE, "--mem", "mlock", "--prompt-file", os.path.join(T, "ids300.txt"), "--n", "32",
                            "--cache-gb", f"{ALL_GB:.6f}", "--cache-start-gb", f"{start_gb:.6f}", "--out-ids", op, "--json", jp])
            r = run(["/bin/zsh", "-c", f"ulimit -l {lim}; exec {cmd}"])
            if r.returncode:
                raise RuntimeError(f"{nm}: {(r.stdout + r.stderr)[-400:]}")
            res[nm] = (open(op).read(), json.load(open(jp))["cache_lock"])
        bad = []
        u, l = res["unlimited"][1], res["limited"][1]
        if u["refused"] or abs(u["locked_gb"] - u["budget_gb"]) > 1e-6:
            bad.append(f"unlimited: {u}")
        if not l["refused"] or l["locked_gb"] * 1e9 > lim_kb * 1024 + max(rb) or abs(l["locked_gb"] - l["budget_gb"]) > 1e-6:
            bad.append(f"limited: {l}")
        if res["unlimited"][0] != res["limited"][0]:
            bad.append("ids differ")
        verdict("lock", "FAIL" if bad else "PASS", ("; ".join(bad) + " | " if bad else "") +
                f"start {start_gb * 1e3:.1f} MB under a {lim_kb / 1e3:.1f} MB lock limit: locked {l['locked_gb'] * 1e3:.1f} MB = the budget, "
                f"{l['refused']} lock refused; unlimited: locked {u['locked_gb'] * 1e3:.1f} MB; ids identical")
    except Exception as e:
        verdict("lock", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- cache
if wanted("cache"):
    try:
        # the automatic cache rule on one machine state (DMENG_TEST_MEMSNAP: total, available, free level, wired, wire
        # limit) in dmeng, dmserve and the C API under the mac profile: each prints its numbers ("cache_auto"); the start
        # rule is the one the manifest declares (profiles.mac.cache_start, else the built-in headroom rule), and the
        # ceiling and start recomputed here from the printed inputs match (the same computation in all three)
        snap = (17.18, 9.0, 50, 3.0, 13.06)
        env = {"DMENG_TEST_MEMSNAP": ",".join(map(str, snap))}
        cs = (man["profiles"].get("mac") or {}).get("cache_start") or {"rule": "headroom", "headroom_gb": 3.0}
        want_rule = cs["rule"]
        ip = os.path.join(T, "cids.txt")
        open(ip, "w").write(",".join(str(LO + i) for i in range(8)))
        whys, bad = {}, []
        j = os.path.join(T, "cache.json")
        r = run([DMENG, "--model", M, "--profile", "mac", "--prompt-file", ip, "--n", "2", "--threads", "2", "--json", j], env=env)
        whys["dmeng"] = (json.load(open(j))["cache_auto"], json.load(open(j))["resolved"]) if r.returncode == 0 else (None, r.stderr[-300:])
        r = run([DMGATE, "--model", M, "--profile", "mac", "--prompt-ids", ip, "--n", "2", "--threads", "2", "--cache-gb", "auto", "--json", j], env=env)
        whys["capi"] = (json.load(open(j))["stats"]["cache_auto"], json.load(open(j))["stats"]["resolved"]) if r.returncode == 0 else (None, r.stderr[-300:])
        r = run([DMSERVE, "--model", M, "--profile", "mac", "--threads", "2"], env=env, inp="", timeout=120)
        ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
        whys["dmserve"] = (json.loads(ready[0][6:])["cache_auto"], json.loads(ready[0][6:])["resolved"]) if ready else (None, r.stderr[-300:])
        rows = []
        for fe, (w, res) in whys.items():
            if not isinstance(w, dict):
                bad.append(f"{fe}: no rule printed ({res})")
                continue
            to_come = w["other_gb"] - w["resident_now_gb"]
            other = w["resident_now_gb"] + w["kv_gb"] + w["ckpt_gb"] + w["scratch_gb"] + w["margin_gb"] + w["later_gb"]
            start = (snap[2] - w["floor_pct"]) / 100 * snap[0] - to_come if want_rule == "free_level" else snap[1] - to_come - w["headroom_gb"]
            ceil = snap[0] - w["reserve_gb"] - w["other_gb"]
            if w.get("start_rule") != want_rule:
                bad.append(f"{fe}: start rule {w.get('start_rule')} (manifest: {want_rule})")
            if abs(other - w["other_gb"]) > 0.03 or abs(start - w["start_unclamped_gb"]) > 0.03 or abs(ceil - w["unclamped_gb"]) > 0.03:
                bad.append(f"{fe}: printed numbers do not follow the rule ({w})")
            if abs(w["wire_room_gb"] - (snap[4] - snap[3] - 0.5)) > 0.02:
                bad.append(f"{fe}: wire room {w['wire_room_gb']}")
            rows.append(f"{fe} {w.get('start_rule')} start {w['start_unclamped_gb']:.2f} ceiling {w['unclamped_gb']:.2f} (plan: kv {w['kv_gb']:.3f}, checkpoints {w['ckpt_gb']:.3f}, later {w['later_gb']:.3f})")
        starts = {fe: (res or {}).get("cache_start") for fe, (w, res) in whys.items() if isinstance(res, dict)}
        if len(set(starts.values())) != 1:
            bad.append(f"resolved cache_start differs: {starts}")
        verdict("cache", "FAIL" if bad else "PASS", ("; ".join(bad) + " | " if bad else "") + "; ".join(rows))
    except Exception as e:
        verdict("cache", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- counters
STAGES = ["embed", "depth_attn", "message_gates_norms", "attn_proj", "attn_core", "attn_out", "moe_router", "experts", "run_ahead", "wait",
          "cache_mgmt", "prefetch_hooks", "post_mlp", "head"]
if wanted("counters"):
    try:
        bad, rows = [], []
        # (one compute thread: a tiny model's token is a fraction of a millisecond, and the clock reads between stages,
        # about 20 ns each, would otherwise be a visible share of it; on a real model they are 1e-4 of the token)
        for nm, fl in (("every expert held", ["--cache-gb", f"{ALL_GB:.6f}"]), ("few slots, names, run-ahead", ["--cache-gb", f"{FEW_GB:.6f}", *RT, "--run-ahead", str(K)])):
            js = dmeng("--prompt-file", os.path.join(T, "ids300.txt"), "--n", "200", "--threads", "1", *fl)
            st = js["by_stage_ms_per_token"]
            if list(st) != STAGES + ["sample"]:
                bad.append(f"{nm}: stage names {list(st)}")
            tot, ms = sum(st.values()), js["ms_per_token"]
            ident = js["compute_ms_per_token"] + js["wait_ms_per_token"] + st["run_ahead"] - ms
            rows.append(f"{nm}: stages {tot / ms:.4f} of the token, compute + wait + run-ahead - time {ident:+.1e} ms")
            if abs(tot - ms) > 0.01 * ms:
                bad.append(f"{nm}: stages sum to {tot:.4f} of {ms:.4f} ms")
            if abs(ident) > 1e-3 * ms:
                bad.append(f"{nm}: compute + wait + run-ahead != time")
        # the server's decode statistics: the same stage names, and MB per token equal to dmeng's on the same request (the
        # server with its prompt scratch kept, as dmeng: a prompt's scratch lent by the cache, PrefillOpts::lend, lowers
        # the cache for the prompt and the decode re-reads what it gave back)
        p = ids[300][:120]
        w = worker("cnt", [{"ids": p, "max_new": 60}], "--keep-scratch")
        dec = w[0]["E"]["json"].get("decode") or {}
        if list((dec.get("ms_per_step") or {}).keys()) != STAGES:
            bad.append(f"server stage names {list((dec.get('ms_per_step') or {}).keys())}")
        js = dmeng("--prompt-file", os.path.join(T, "ids120.txt"), "--n", "60", "--cache-gb", f"{FEW_GB:.6f}", "--prefill-chunk", "512")
        mb_d, mb_s = js["mb_read_per_token"], dec.get("mb_read_per_step", -1)
        rows.append(f"MB per token dmeng {mb_d:.4f}, server {mb_s:.4f}")
        if not (abs(mb_d - mb_s) <= 0.02 * max(mb_d, 1e-9)):
            bad.append(f"MB per token: dmeng {mb_d} vs server {mb_s}")
        verdict("counters", "FAIL" if bad else "PASS", ("; ".join(bad) + " | " if bad else "") + "; ".join(rows))
    except Exception as e:
        verdict("counters", "FAIL", str(e)[-400:])

json.dump({"family": FAM, "capabilities": caps, "checks": results}, open(os.path.join(OUT, "family_check.json"), "w"), indent=1)
open(os.path.join(OUT, "family_check.txt"), "w").write("\n".join(lines) + "\n")
sys.exit(1 if any(v["status"] == "FAIL" for v in results.values()) else 0)
