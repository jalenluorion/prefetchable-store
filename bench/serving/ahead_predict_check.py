#!/usr/bin/env python3
"""Offline gate for the idle-time prefill's prediction (python/serve/ahead.py), no model run: for every recorded main
request whose answer has no tool calls and that has a next main request, predict the next prompt's prefix from the
request + its recorded answer, and check that it is a prefix of the next request's real token ids. Also counts what the
next request would recompute without it (from where it leaves the live state: the prompt + the answer as generated).

  python bench/serving/ahead_predict_check.py --model-dir MODEL --format python/serve/qwen3_5_moe SESSION_DIR... > out.json"""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--format", required=True)
ap.add_argument("sessions", nargs="+")
a = ap.parse_args()
sys.path.insert(0, a.format)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "serve"))
import chat_format as CF  # noqa: E402
import ahead as AH  # noqa: E402

tok = CF.Tok(a.model_dir)
rows, ok_all = [], True
for d in a.sessions:
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "[0-9][0-9][0-9].json")))]
    mains = [r for r in recs if r["body"].get("tools")]
    for r, nxt in zip(mains, mains[1:]):
        ans = r.get("answer") or {}
        if ans.get("tool_calls"):
            continue
        b, bn = r["body"], nxt["body"]
        kw, _ = CF.template_kwargs(b)
        msgs = CF.to_template(b["messages"])
        ids, _ = tok.prompt_ids(msgs, b.get("tools"), **kw)
        asst = CF.to_message(ans.get("reasoning"), ans.get("content"), [])
        S, marks = AH.predict(tok, msgs, b.get("tools"), kw, asst)
        kwn, _ = CF.template_kwargs(bn)
        ids_n, _ = tok.prompt_ids(CF.to_template(bn["messages"]), bn.get("tools"), **kwn)
        # the live state after this request: prompt + the answer as the model wrote it (the turn rendered as the last one)
        is_prefix = ids_n[: len(S)] == S
        d_template, d_next = AH.lcp(ids, S), AH.lcp(S, ids_n)  # where the prediction leaves the prompt; where the next request leaves it
        # not a prefix: either the prediction is wrong (a failure) or the client changed the prompt before the point the
        # template changes it (e.g. a new OpenCode process reordering its skills list): nothing to predict there
        client_changed = not is_prefix and AH.lcp(ids, ids_n) < d_template
        ok_all &= is_prefix or client_changed
        rows.append({"session": os.path.basename(d.rstrip("/")), "n": r["n"], "next": nxt["n"], "prompt": len(ids), "next_prompt": len(ids_n),
                     "predicted": len(S), "predicted_is_prefix": is_prefix, "client_changed_earlier": client_changed, "prediction_leaves_prompt_at": d_template,
                     "wasted_if_wrong": 0 if is_prefix else len(S) - max(d_next, AH.lcp(ids, S)), "lcp_next_vs_prompt": AH.lcp(ids, ids_n),
                     "next_new_after_prediction": len(ids_n) - AH.lcp(S, ids_n),
                     "next_new_without": len(ids_n) - AH.lcp(ids, ids_n)})
print(json.dumps({"passed": ok_all and bool(rows), "rows": rows}, indent=1))
sys.exit(0 if ok_all and rows else 1)
