import json, re, sys
L = open(sys.argv[1]).read()
pb = json.loads(re.search(r'^\{"mode":"prefill_bench".*$', L, re.M).group(0).replace(':nan', ':0'))
n = pb["positions"]
rows = [l.split() for l in L.splitlines() if l.startswith("DMSTAGE ")]
tot = [0, 0]
print(f"{sys.argv[1].split('/')[-1]}: {n} positions, {pb['tok_s']:.2f} tok/s (indicative), others {re.search(r'others_cpu_pct_mean.: ([0-9.]+|null)', L).group(1)}")
print(f"{'stage':14s} {'M ins/tok':>10s} {'M cyc/tok':>10s}  (serial ins, cyc | pool ins, cyc)")
for r in rows:
    si, sc, pi, pc = map(float, r[2:6])
    if si + pi == 0: continue
    tot[0] += si + pi; tot[1] += sc + pc
    print(f"{r[1]:14s} {(si+pi)/n/1e6:10.2f} {(sc+pc)/n/1e6:10.2f}  ({si/n/1e6:.2f}, {sc/n/1e6:.2f} | {pi/n/1e6:.2f}, {pc/n/1e6:.2f})")
print(f"{'total':14s} {tot[0]/n/1e6:10.2f} {tot[1]/n/1e6:10.2f}")
