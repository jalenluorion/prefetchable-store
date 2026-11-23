"""(c) tables: sim_nv4.json (all records NVFP4, one pool) + the compute model with the 4-bit in-memory file (PREDICTION:
4-bit affine bytes = 0.529 of 8-bit (codes + f16 scale/bias per 64); A16 rates of the 35B's 4-bit resident shapes, P0 3.1:
a4qkv 34.0, a4out 35.7, a4head 37.7 GB/s, the screened head off (it needs an 8-bit head); experts all at 31.5 GB/s)."""
import json, collections
MB = {"experts": 48 * 8 * 4.489216, "attn_proj": 1024 * 0.529, "o_proj": 361 * 0.529, "head": 492 * 0.529, "msg_gates": 170, "router": 28}
RATE = {"experts": 31.5, "attn_proj": 34.0, "o_proj": 35.7, "head": 37.7, "msg_gates": 41.6, "router": 41.6}
mv = sum(MB[k] / RATE[k] for k in MB)
burst = mv + (50.0 - 45.1) * mv / 45.1
sust = burst / 0.62
rows = json.load(open("sim_nv4.json"))["rows"]
ex, fa = collections.defaultdict(list), {}
for r in rows:
    (ex[r["gb"]].append(r) if r["thr"] is None else fa.__setitem__(r["gb"], r))
print(f"compute ms/token (prediction): burst {burst:.0f}, sustained {sust:.0f}\n")
print("| slab GB | Exact MB/token (pitt / code / edit / agent; mean) | hit looped / outer | Fast MB/token (agent ratio on the mean) | weight left out (of 48) | Exact tok/s 0.7 / 1.1 GB/s | Fast tok/s 0.7 / 1.1 GB/s |")
print("|---|---|---|---|---|---|---|")
for gb in sorted(ex):
    xs = sorted(ex[gb], key=lambda r: ["pitt", "code", "edit", "agent"].index(r["text"]))
    m = sum(r["mb_per_token"] for r in xs) / 4
    ag = xs[3]["mb_per_token"]
    f = m * fa[gb]["mb_per_token"] / ag
    t = lambda mb, d, c: 1000 / (c + mb / d)
    print(f"| {gb} | {' / '.join(str(round(r['mb_per_token'])) for r in xs)}; **{m:.0f}** | {sum(r['hit_loop'] for r in xs)/4:.2f} / {sum(r['hit_out'] for r in xs)/4:.2f} | {f:.0f} | {fa[gb]['w_left_out_per_token']:.2f} |"
          f" {t(m,.7,burst):.2f} ({t(m,.7,sust):.2f}) / {t(m,1.1,burst):.2f} ({t(m,1.1,sust):.2f}) | {t(f,.7,burst):.2f} ({t(f,.7,sust):.2f}) / {t(f,1.1,burst):.2f} ({t(f,1.1,sust):.2f}) |")
