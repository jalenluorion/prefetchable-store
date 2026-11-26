"""Table of serve_ab.sh outputs: per run and request, prompt, computed, reused, seconds to first token, prompt positions
per second computed, MB read in the prompt, decode tok/s, draft tokens per step, others' CPU.
  python3 serve_table.py OUT"""
import glob, json, os, sys

O = sys.argv[1]
print("| run | req | prompt | computed | reused | s to first token | prompt tok/s | prompt MB read | decode tok/s | tokens/step | others CPU % |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(O, "*_r[0-9].json"))):
    n = os.path.basename(f)[:-5]
    g = [l for l in open(f[:-5] + ".log") if l.startswith("GUARD ")]
    oth = json.loads(g[-1][6:]).get("others_cpu_pct_mean") if g else None
    for k, rq in enumerate(json.load(open(f))):
        b = rq.get("B") or {}
        e = (rq.get("E") or {}).get("json", {})
        comp = b.get("computed", 0)
        pf = b.get("ms_prefill", 0) / 1e3
        pj = b.get("prefill") or {}
        dr = e.get("draft") or {}
        print(f"| {n} | {k + 1} | {rq['prompt_len']} | {comp} | {b.get('reused')} | {e.get('ms_first_token', 0) / 1e3:.2f} | "
              f"{comp / pf if pf else 0:.1f} | {pj.get('mb_read', '-')} | {e.get('decode_tok_s', 0):.2f} | {dr.get('tokens_per_step', '-')} | {oth} |")
