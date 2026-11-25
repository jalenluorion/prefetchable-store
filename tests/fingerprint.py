"""The tokens and logits of a tiny model under a fixed set of runs, as sha256 fingerprints: two builds (or a build before and
after a change that must not move a single bit) are compared by comparing these files.

  python tests/fingerprint.py MODEL_DIR OUT.json [--bin-dir build] [--against OLD.json]   (python: the standard library)

Every run is dmeng (teacher-forced logits, greedy ids) or dmserve's prefill check, with --profile none and every setting
explicit, so the fingerprint is of the computation, not of the defaults (tests/tiny_all.sh checks the resolved defaults
separately). Runs: exact mode at three precisions, with every expert held and with few slots (evictions, re-reads), one
and four threads, the folder's router naming, run-ahead, greedy decoding after a batched prefill, fast mode at few slots,
and dmserve --check (batched prefill vs single steps). With --against: exit 1 and list the runs whose fingerprints differ.
"""
import argparse, glob, hashlib, json, os, random, subprocess, sys

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--bin-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))
ap.add_argument("--against", default=None)
ap.add_argument("--only", default=None, help="comma-separated run names")
a = ap.parse_args()
M = os.path.abspath(a.model)
TMP = os.path.join(os.path.dirname(os.path.abspath(a.out)), "fp_" + os.path.basename(a.out).replace(".json", ""))
os.makedirs(TMP, exist_ok=True)
man = json.load(open(os.path.join(M, "moved-store.json")))
fam = man["family"]
cfg = json.load(open(os.path.join(M, "config.json")))
tc = cfg.get("text_config", cfg)
V = man["model"].get("vocab_unpadded") or man["model"]["vocab"]
K = man["experts"]["top_k"]
meta = json.load(open(os.path.join(M, "store", "meta.json")))
rb = meta["record_bytes"]
rb = rb if isinstance(rb, list) else [rb] * meta["layers"]
ALL_GB = 1.25 * sum(rb) * meta["experts"] / 1e9
FEW_GB = 0.4 * sum(rb) * meta["experts"] / 1e9
POOL = ["--pool-share", "0.625,0.375"] if fam == "inkling2_loop" else []  # two record sizes: two pools
router = sorted(glob.glob(os.path.join(M, ".dmeng", "router.*.bin")))
rnd = random.Random(7)
lo = 10 if fam == "qwen4_exp" else 0
ids = {n: [rnd.randrange(lo, V) for _ in range(n)] for n in (40, 300, 600)}
for n, p in ids.items():
    open(os.path.join(TMP, f"ids{n}.txt"), "w").write(",".join(map(str, p)))
BASE = ["--profile", "none", "--threads", "4", "--readers", "2", "--mem", "plain", "--res-mem", "plain", "--ctx", "2048"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def teacher(name, n, *flags):
    lp = os.path.join(TMP, name + ".f32")
    cmd = [os.path.join(a.bin_dir, "dmeng"), "--model", M, *BASE, "--teacher", os.path.join(TMP, f"ids{n}.txt"), "--dump-logits", lp, *flags]
    return cmd, [lp]


def greedy(name, *flags):
    op = os.path.join(TMP, name + ".ids")
    cmd = [os.path.join(a.bin_dir, "dmeng"), "--model", M, *BASE, "--prompt-file", os.path.join(TMP, "ids300.txt"), "--n", "32", "--out-ids", op, *flags]
    return cmd, [op]


ALL = ["--cache-gb", f"{ALL_GB:.6f}", *POOL]
FEW = ["--cache-gb", f"{FEW_GB:.6f}", *POOL]
RT = ["--routers", router[0], "--m", str(K)] if router else []
runs = {
    "exact_i16_all_40": teacher("exact_i16_all_40", 40, *ALL),
    "exact_f32_all_300": teacher("exact_f32_all_300", 300, "--prec", "f32", *ALL),
    "exact_f16_few_300": teacher("exact_f16_few_300", 300, "--prec", "f16", *FEW),
    "exact_i16_few_300": teacher("exact_i16_few_300", 300, *FEW),
    "exact_i16_few_300_t1": teacher("exact_i16_few_300_t1", 300, *FEW, "--threads", "1"),
    "exact_i16_few_300_routers": teacher("exact_i16_few_300_routers", 300, *FEW, *RT),
    "exact_i16_few_300_runahead": teacher("exact_i16_few_300_runahead", 300, *FEW, *RT, "--run-ahead", str(K)),
    "fast_i16_few_300": teacher("fast_i16_few_300", 300, *FEW, "--mode", "fast"),
    "greedy_all": greedy("greedy_all", *ALL, "--prefill-chunk", "64"),
    "greedy_few_routers": greedy("greedy_few_routers", *FEW, *RT, "--prefill-chunk", "1"),
}
res, info, bad = {}, {}, []
for name, (cmd, outs) in runs.items():
    if a.only and name not in a.only.split(","):
        continue
    if not router and "routers" in name:
        res[name] = "no router in the folder"
        continue
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        res[name] = {"error": (r.stdout + r.stderr)[-600:]}
    else:
        js = json.loads(r.stdout.strip().splitlines()[-1])
        res[name] = {"sha256": [sha(o) for o in outs]}
        info[name] = {"reads": js.get("reads")}
# dmserve: the batched prefill equals single steps (the bits of the comparison, not a fingerprint of values)
srv = [os.path.join(a.bin_dir, "dmserve"), "--model", M, "--profile", "none", "--cache-gb", f"{ALL_GB:.6f}", "--mem", "plain", "--ctx", "2048",
       "--chunk", "128", "--check", os.path.join(TMP, "ids600.txt"), "--gen", "16", "--threads", "4", "--readers", "2"]
if not a.only or "dmserve_check" in a.only:
    r = subprocess.run(srv, capture_output=True, text=True, env={**os.environ, "DMENG_QSA_PAR_MIN": "16"})
    try:
        js = json.loads(r.stdout.strip().splitlines()[-1])
        res["dmserve_check"] = {k: js.get(k) for k in ("logit_bits_differ", "fixed_state_bytes_differ", "kv_bytes_differ", "greedy_identical")}
    except Exception:
        res["dmserve_check"] = {"error": (r.stdout + r.stderr)[-600:]}
json.dump({"family": fam, "model": M, "runs": res, "info": info}, open(a.out, "w"), indent=1)
for k, v in res.items():
    print(f"{k:32s} {json.dumps(v)[:110]} {json.dumps(info.get(k, ''))}")
if a.against:
    old = json.load(open(a.against))["runs"]
    for k, v in res.items():
        if old.get(k) != v:
            bad.append(k)
            print(f"DIFFERS {k}: {json.dumps(old.get(k))[:200]}  ->  {json.dumps(v)[:200]}")
    print(f"FINGERPRINT {'DIFFERS: ' + ', '.join(bad) if bad else 'IDENTICAL'} ({len(res)} runs, {fam})")
    sys.exit(1 if bad else 0)
