#!/usr/bin/env python3
"""Request lists for worker_drive.py (results/drafts/prompts/*_server_reqs.json), greedy (temperature 0).
  q36: the recorded OpenCode session's main requests (engine/results/agentic/replay/s2_clean, the first user turn's tool loop
       002-005 by default), tokenised as the front end does (the repo's qwen3_5_moe chat format, the
       request's own template switches), with the front end's tool-call fragments; max_new 128 each
  ink: Inkling's tool turns of make_prompts.py (toolrw, toolnew, toolshort) as one session, with their fragments
  python make_server_reqs.py ENGINE_DIR Q36_MODEL_DIR   (a Python with tokenizers + jinja2)"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ENG, MODEL = sys.argv[1], sys.argv[2]
PR = os.path.join(REPO, "results", "drafts", "prompts")
sys.path.insert(0, os.path.join(REPO, "python", "serve", "qwen3_5_moe"))
import chat_format as CF

tok = CF.Tok(MODEL)
reqs = []
for f in sorted(glob.glob(os.path.join(ENG, "results", "agentic", "replay", "s2_clean", "[0-9][0-9][0-9].json"))):
    r = json.load(open(f))
    b = r["body"]
    if not b.get("tools"):
        continue
    kw, _ = CF.template_kwargs(b)
    ids, marks = tok.prompt_ids(CF.to_template(b.get("messages", [])), b["tools"], **kw)
    if len(reqs) >= int(os.environ.get("Q36_REQS", 4)):  # the first user turn's tool loop (the gate keeps the runs short)
        break
    reqs.append({"ids": ids, "marks": marks, "max_new": 128, "temp": 0, "stop": [248046, 248044],
                 "skel": [tok.encode(t) for t in CF.tool_skeletons(b["tools"])], "n": r["n"]})
json.dump(reqs, open(os.path.join(PR, "q36_server_reqs.json"), "w"))
print("q36", len(reqs), [len(r["ids"]) for r in reqs])
ink = []
for case, mx in (("toolrw", 200), ("toolnew", 200), ("toolshort", 120)):
    ids = [int(x) for x in open(os.path.join(PR, f"ink_{case}_ids.txt")).read().split(",")]
    sk = [[int(x) for x in l.split()] for l in open(os.path.join(PR, f"ink_{case}_skel.txt")) if l.strip()]
    ink.append({"ids": ids, "max_new": mx, "temp": 0, "stop": [200006], "skel": sk})
json.dump(ink, open(os.path.join(PR, "ink_server_reqs.json"), "w"))
print("ink", len(ink), [len(r["ids"]) for r in ink])
