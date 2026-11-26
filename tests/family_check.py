"""The same checks on a tiny model of any family (tests/tiny_all.sh runs them on all three): nothing here names a family;
what a family supports is read from the engine itself (the "capabilities" of its JSON) and from the folder's manifest.

  python tests/family_check.py TINY_DIR OUT_DIR [--bin-dir build] [--ref-python PY]    (python: the standard library)

TINY_DIR holds model/ (the folder the engine runs, with moved-store.json) and release/ (the synthetic release the
reference reads). Checks, one verdict each (PASS, FAIL, or "not supported for this family" with the engine's refusal
checked):
  reference   exact decode against the independent reference: tests/<family>/tiny_check.py (its own python, --ref-python)
  invariance  exact mode: logits and greedy ids identical, bit for bit, with every expert held / few slots (evictions and
              re-reads), 1 / 4 / 5 compute threads, the folder's router naming, run-ahead, and an oracle replaying the
              recorded routing at leads 1 and one token
  prefill     the batched prefill equals single steps: dmserve --check (logits, fixed state, KV, greedy continuation; CPU,
              and the GPU's share on a Metal build), and dmeng's ids after a batched and a token-by-token prompt
  checkpoint  dmserve --check-ckpt (a long request, the same again, one diverging at 60%, each equal to a cold computation)
              and dmeng --state-check (state saved, restored into a fresh one, the same ids)
  server      the server worker's greedy and sampled ids equal dmeng's for the same prompts and seeds
  drafting    dmeng --verify-check (verify passes with rollbacks equal to single steps), --draft fixed / lookup equal to plain
              decoding (greedy and sampled), the server worker the same; without a verify pass: every front end refuses
  presets     the precedence (explicit > profile > manifest > built-in) in dmeng, the C API (dm_gate) and dmserve, each
              read back from its "resolved" JSON, and the three front ends printing the same resolved settings
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


def dmeng(*flags, name="r"):
    r = run([DMENG, "--model", M, *BASE, *flags])
    if r.returncode:
        raise RuntimeError(f"dmeng {' '.join(flags)}: {(r.stdout + r.stderr)[-600:]}")
    return last_json(r)


def logits(name, n, *flags):
    lp = os.path.join(T, name + ".f32")
    js = dmeng("--teacher", os.path.join(T, f"ids{n}.txt"), "--dump-logits", lp, *flags)
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
        bad, reads = [], []
        for nm, fl in variants:
            g, js = logits("v", 300, *fl)
            reads.append(js.get("reads", 0))
            if g != base:
                bad.append(nm)
        g0, _ = greedy("g0", "--cache-gb", f"{ALL_GB:.6f}")
        g1, _ = greedy("g1", *FEW, *RT, "--threads", "1")
        if g0 != g1:
            bad.append("greedy ids, few slots / names / 1 thread")
        verdict("invariance", "FAIL" if bad else "PASS",
                (f"differ: {', '.join(bad)}" if bad else f"{len(variants) + 1} runs identical (300 positions; reads per run {min(reads)}-{max(reads)})")
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
        ok = bool(js["passed"]) and all(sc)
        verdict("checkpoint", "PASS" if ok else "FAIL", f"dmserve --check-ckpt {'passed' if js['passed'] else 'FAILED'} ({rows}); dmeng --state-check same ids: {sc}")
    except Exception as e:
        verdict("checkpoint", "FAIL", str(e)[-400:])

# ---------------------------------------------------------------------------------------------- server
DRIVE = os.path.join(REPO, "bench", "drafts", "worker_drive.py")


def worker(name, reqs, *flags):
    rq, op = os.path.join(T, name + ".reqs.json"), os.path.join(T, name + ".json")
    json.dump(reqs, open(rq, "w"))
    r = run([sys.executable, DRIVE, op, rq, "--", DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{FEW_GB:.6f}", "--ctx", "2048",
             "--threads", "2", "--readers", "2", "--mem", "plain", "--res-mem", "plain", *flags])
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
if wanted("drafting"):
    if not caps.get("verify"):
        refusals = []
        for nm, cmd in (("dmeng --draft lookup", [DMENG, "--model", M, *BASE, "--prompt-ids", "11,12,13", "--n", "4", "--draft", "lookup", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmeng --drafts 1", [DMENG, "--model", M, *BASE, "--prompt-ids", "11,12,13", "--n", "4", "--drafts", "1", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmeng --verify-check", [DMENG, "--model", M, *BASE, "--prompt-ids", "11,12,13", "--n", "4", "--verify-check", "8", "--cache-gb", f"{ALL_GB:.6f}"]),
                        ("dmserve --draft lookup", [DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--draft", "lookup"]),
                        ("dmserve --drafts 1", [DMSERVE, "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--drafts", "1"])):
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
                if RT:  # the read-ahead: the drafts' next positions named before their pass (routers), dropped when the draft changes
                    frag2 = os.path.join(T, "frag_whole.txt")  # plain decoding's own output unchanged: long drafts kept, so named positions run
                    open(frag2, "w").write(" ".join([str(x) for x in ids[40][-6:]] + g))
                    out, js = greedy("da", "--cache-gb", f"{FEW_GB:.6f}", *S, *RT, "--draft", "fixed", "--draft-max", "7", "--draft-ahead", "8", "--skeleton", frag2, n=100, prompt=40)
                    ra = js.get("draft", {}).get("read_ahead", {})
                    (okd if out == plain and ra.get("positions", 0) > 0 and ra.get("names_spent", 0) > 0 else bad).append(
                        f"{sm} --draft-ahead 8 ({ra.get('positions', 0)} positions named ahead, {ra.get('positions_dropped', 0)} dropped)")
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
            out, i = [], 0
            while i < len(flags):
                f, v = flags[i], (flags[i + 1] if i + 1 < len(flags) else None)
                if f == "--gpu":
                    # one spelling in every front end (presets.h run_flag): dmeng is given the old alias, dmserve the name
                    out += [{"dmeng": "--gpu-frac", "dmserve": "--prefill-gpu-frac"}.get(front, "--gpu-frac"), v]; i += 2
                elif f == "--gpu-off":
                    out += ["--gpu-frac", "-1", "--gpu-experts", "-1"] if front == "capi" else ["--prefill-gpu", "off"]; i += 1
                elif f == "--ra-off":
                    out += ["--run-ahead", "-1" if front == "capi" else "0"]; i += 1
                else:
                    out += [f, v]; i += 2
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

        bad, ncell, cross = [], 0, []
        FRONT_SPECIFIC = {"prefill_chunk"}  # dmserve's prompt path always chunks (--chunk 512); the C API keeps the model's own
        for prof in man["profiles"]:
            r = run([DMENG, "--model", M, "--profile", prof, "--print-defaults"])
            P = json.loads(r.stdout)
            names = P["names"]
            exp_default = {"mode": P["engine_mode"], "eviction": P["eviction"], "prefill_gpu": P["prefill_gpu"],
                           "run_ahead": {k: P["run_ahead"][k] for k in ("picks", "depth", "budget", "max_tok", "sat_skip")},
                           "names": names if P["routers"] else None}
            if P["readers"] > 0:
                exp_default["readers"] = P["readers"]
            gpu = P["prefill_gpu"]
            cells = [("default", [], exp_default),
                     ("mode fast", ["--mode", "fast"], {"mode": json.loads(run([DMENG, "--model", M, "--profile", prof, "--mode", "fast", "--print-defaults"]).stdout)["engine_mode"]}),
                     ("top m 4", ["--m", "4"], {"names": {"rule": "top", "m": 4}} if P["routers"] else {"names": None}),
                     ("names off", ["--routers", "none"], {"routers": "", "names": None}),
                     ("cache 0.05", ["--cache-gb", "0.05"], {"cache_gb": 0.05, "cache_auto": False}),
                     ("evict freq", ["--evict", "freq"], {"eviction": "freq"}),
                     ("gpu 0.2", ["--gpu", "0.2"], {"prefill_gpu": {"projections": 0.2, "experts": gpu["experts"]}}),
                     ("gpu off", ["--gpu-off"], {"prefill_gpu": {"projections": 0, "experts": 0}}),
                     ("run-ahead off", ["--ra-off"], {"run_ahead": {"picks": 0}}),
                     ("run-ahead 4", ["--run-ahead", "4"], {"run_ahead": {"picks": 4, "budget": P["run_ahead"]["budget"]}}),
                     ("run-ahead limits", ["--run-ahead-budget", "0.5", "--run-ahead-max", "3", "--run-ahead-sat-skip", "1"],
                      {"run_ahead": {"budget": 0.5, "max_tok": 3, "sat_skip": True}}),
                     ("readers 2", ["--readers", "2"], {"readers": 2})]
            if routers:
                cells.append(("router file", ["--routers", routers[0]], {"routers": routers[0], "names": {"rule": "top", "m": 8}}))
            for nm, fl, exp in cells:
                got = {fe: resolved(fe, prof, spell(fe, fl)) for fe in ("dmeng", "capi", "dmserve")}
                ncell += 1
                for fe, g in got.items():
                    e = sub(exp, g)
                    if e:
                        bad.append(f"{prof} {nm} {fe}: {'; '.join(e)}")
                norm = {fe: json.dumps({k: v for k, v in g.items() if k not in FRONT_SPECIFIC and not (k == "cache_gb" and g.get("cache_auto"))}, sort_keys=True)
                        for fe, g in got.items()}
                if len(set(norm.values())) > 1:
                    keys = set().union(*[g.keys() for g in got.values()]) - FRONT_SPECIFIC
                    cross.append(f"{prof} {nm}: " + ", ".join(f"{k} {{{', '.join(f'{fe} {json.dumps(got[fe].get(k))}' for fe in got)}}}" for k in sorted(keys)
                                                             if len({json.dumps(got[fe].get(k)) for fe in got}) > 1))
        detail = f"{ncell} cells x 3 front ends ({', '.join(man['profiles'])})"
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
        # the server's decode statistics: the same stage names, and MB per token equal to dmeng's on the same request
        p = ids[300][:120]
        w = worker("cnt", [{"ids": p, "max_new": 60}])
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
