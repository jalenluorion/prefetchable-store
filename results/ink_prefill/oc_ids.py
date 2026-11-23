#!/usr/bin/env python3
"""The ids of a recorded OpenCode main request (engine/results/agentic/replay/s2_clean/NNN.json, OpenCode's own system
prompt and tool list) as Inkling's front end tokenises it (python/serve/generic/chat_format.py on the model folder), for
dmserve --check and the prefill byte counts.
  python oc_ids.py ENGINE_DIR INKLING_MODEL_DIR NNN OUT_IDS"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "serve", "generic"))
import chat_format as CF

eng, model, n, out = sys.argv[1:5]
b = json.load(open(os.path.join(eng, "results", "agentic", "replay", "s2_clean", f"{n}.json")))["body"]
tok = CF.Tok(model)
kw, _ = CF.template_kwargs(b)
ids, marks = tok.prompt_ids(CF.to_template(b.get("messages", [])), b.get("tools"), **kw)
open(out, "w").write(",".join(map(str, ids)))
print(json.dumps({"request": n, "messages": len(b.get("messages", [])), "tools": len(b.get("tools") or []), "ids": len(ids), "marks": marks}))
