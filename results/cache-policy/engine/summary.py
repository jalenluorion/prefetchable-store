"""Engine runs (runs/*.json) against the simulator on each run's own routing and router rows, and the ids against the
references: python3 summary.py > summary.md"""
import glob, json, os, re, subprocess

W = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(W, "runs")
DM = os.environ["DEEP_MEMORY"]  # the workspace checkout that holds the engine runs
PY = os.path.join(DM, "engine/.venv/bin/python")
ENG = os.path.join(DM, "engine")
REF = {"q": ENG + "/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids", "i": ENG + "/results/inkling_draft/runs/pitt_one_gb6_r1.ids"}
START = {"q": 9, "i": 10}
PJ = {"q": W + "/q36_d_evict3_policy.json", "i": W + "/ink_d_evict3_policy.json"}


def ids(p):
    return [int(x) for x in open(p).read().replace("\n", ",").split(",") if x.strip()]


rows = []
names = sorted(glob.glob(os.path.join(R, "[qi]_*.json")) + [os.path.join(R, "pitt_v3_3.3.json")],
               key=lambda p: (os.path.basename(p)[0] if not os.path.basename(p).startswith("pitt") else "q",
                              float(re.findall(r"_([\d.]+)\.json$", p)[0]), p))
first_rec = {}
for jp in names:
    n = os.path.basename(jp)[:-5]
    m = "q" if n.startswith(("q_", "pitt")) else "i"
    j = json.load(open(jp))
    rq = j.get("read_queue", {})
    pol = rq.get("evict_policy", "?")
    caps = rq.get("layer_caps", False)
    slots = ",".join(str(x) for x in j["slots"])
    simpol = {"hazard": "hazard", "v3_mlp": "v3"}.get(pol, "decay")
    args = [PY, os.path.join(W, "sim_trace.py"), os.path.join(R, n + ".rec"), os.path.join(R, n + ".trace"), str(START[m]), slots, simpol]
    if caps:
        args.append("caps")
    args += ["--policy-json", PJ[m], "--det"]
    s = json.loads(subprocess.run(args, capture_output=True, text=True).stdout.strip().splitlines()[-1])
    a, b = ids(os.path.join(R, n + ".ids")), ids(REF[m])
    k = min(len(a), len(b))
    d = next((i for i in range(k) if a[i] != b[i]), None)
    idv = f"identical ({k})" if d is None else f"differs at {d}"
    rec = open(os.path.join(R, n + ".rec"), "rb").read()
    first_rec.setdefault(m, rec)
    g = re.search(r"^GUARD (.*)$", open(os.path.join(R, n + ".log")).read(), re.M)
    gj = json.loads(g.group(1)) if g else {}
    rows.append((m, j["cache_gb"], n, slots, pol + (" + caps" if caps else ""), idv, "same" if rec == first_rec[m] else "DIFFERENT",
                 j["hit_rate"], s["hit_rate"], j["mb_read_per_token"], s["mb_per_token"], j.get("policy_ms_per_token", 0),
                 j.get("routers", {}).get("router_ms_per_token", 0), j["compute_ms_per_token"], j["wait_ms_per_token"], j["tok_s"],
                 gj.get("others_cpu_pct_mean"), gj.get("peak_footprint_gb_guard"), gj.get("before", {}).get("power", "?").replace(" Power", "")))
print("| model | GB | run | slots | policy | ids vs reference | routing vs first run | hit engine / sim | MB/token engine / sim | "
      "policy ms/tok | router ms/tok | compute ms/tok | wait ms/tok | tok/s | others' CPU % | footprint GB | power |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {'35B' if r[0] == 'q' else 'Inkling'} | {r[1]:g} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.4f} / {r[8]:.4f} | "
          f"{r[9]:.2f} / {r[10]:.2f} | {r[11]:.2f} | {r[12]:.2f} | {r[13]:.1f} | {r[14]:.1f} | {r[15]:.2f} | {r[16]} | {r[17]} | {r[18]} |")
