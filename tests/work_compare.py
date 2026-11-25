"""Two builds do the same work in a batched prefill: the work counters (include/dmeng/workcount.h) side by side.

  python3 tests/work_compare.py MODEL OUT --a BIN_A --b BIN_B [--n 1100] [--chunk 512] [--gpu 0.4]   (the standard library only)

Each build runs dmeng's batched prefill of the same random prompt (ids from 10 up, seed 11) with DMENG_WORKCOUNT=1 and
prints the prefill's counters (kernel calls and their (row, position) products by kernel and format, weight bytes the
packed kernels went through, the GPU's rows and sets, fork-joins, packed-input copies and quantisations, expert reads
issued and names asked, MoE waves). Settings: every expert slot available, --skip-reads (the reads land at once, so the
counts do not depend on the drive's timing) and the same with real reads (waves then follow the timing: run twice).
Prints every counter that differs and the ratio b / a. Exit 0 always (a report, not a verdict).
"""
import argparse, json, os, random, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--n", type=int, default=1100)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--gpu", type=float, default=0.4)
ap.add_argument("--cache-gb", default="2")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
V = json.load(open(os.path.join(a.model, "config.json")))
V = V.get("text_config", V)["vocab_size"]
rnd = random.Random(11)
ids = os.path.join(a.out, f"ids{a.n}.txt")
open(ids, "w").write(",".join(str(rnd.randrange(10, V)) for _ in range(a.n)))


def work(bin_dir, flags):
    r = subprocess.run([f"{bin_dir}/dmeng", "--model", a.model, "--profile", "none", "--prompt-file", ids, "--n", "1", "--threads", "4",
                        "--prefill-chunk", str(a.chunk), "--cache-gb", a.cache_gb, *flags], capture_output=True, text=True,
                       env={**os.environ, "DMENG_WORKCOUNT": "1"}, timeout=600)
    w = [l for l in r.stderr.splitlines() if l.startswith("WORK ")]
    if not w:
        raise SystemExit(f"{bin_dir}: no WORK line: {(r.stdout + r.stderr)[-500:]}")
    return json.loads(w[-1][5:])


cases = [("cpu, reads land at once", ["--skip-reads"]),
         (f"gpu share {a.gpu}/{a.gpu}, reads land at once", ["--skip-reads", "--prefill-gpu-frac", str(a.gpu), "--prefill-gpu-experts", str(a.gpu)]),
         ("cpu, real reads (1)", []), ("cpu, real reads (2)", []),
         (f"gpu share {a.gpu}/{a.gpu}, real reads", ["--prefill-gpu-frac", str(a.gpu), "--prefill-gpu-experts", str(a.gpu)])]
for name, fl in cases:
    wa, wb = work(a.a, fl), work(a.b, fl)
    print(f"== {name} (n {a.n}, sub-chunk {a.chunk})")
    for k in sorted(set(wa) | set(wb)):
        x, y = wa.get(k, 0), wb.get(k, 0)
        mark = "" if x == y else "  DIFF" + (f" b/a {y / x:.3f}" if x else "")
        print(f"  {k:28s} a {x:>14d}  b {y:>14d}{mark}")
