#!/usr/bin/env python3
"""Prompts of the draft gates and timing (results/drafts/prompts/): token ids in each model's own chat template, and the
tool-call skeleton fragments (token ids, one fragment per line) the server's front end would send for the request's tools.

  python make_prompts.py q36 ENGINE_DIR Q36_MODEL_DIR     (a Python with tokenizers + jinja2; the repo's qwen3_5_moe format)
  python make_prompts.py ink ENGINE_DIR INK_MODEL_DIR     (Inkling's template through engine/ink/run_ink.py's render_chat)

Cases: prose (the Pittsburgh prompt), code (the coding chat prompt), edit (copy-heavy: change a file and print it whole),
toolnew (a tool call that writes a new file of ~600 tokens), toolrw (rewrite a file the conversation read with a tool),
toolshort (a short tool call). Tool cases have thinking off, so the answer is the call."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
which, ENG, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = os.path.join(REPO, "results", "drafts", "prompts")
os.makedirs(OUT, exist_ok=True)

FILE = open(os.path.join(HERE, "orders.py")).read()
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write the complete new contents of a text file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                                     "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the repository and return its output.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
NEW = ("Create a new file inventory.py: a small command-line inventory tracker. It keeps items (name, quantity, unit price) in a "
       "JSON file, with functions to load and save the file, add an item, remove an item, change a quantity, list the items sorted "
       "by name and print the total value, and a main() with argparse subcommands for each. Write the complete file with the "
       "write_file tool.")
RW = "In orders.py, make pending() also skip orders that have no items. Use the tools: read the file, then write the whole updated file."
SHORT = "Run the test suite with pytest -q and tell me if it passes."


def msgs_rw(args):
    return [{"role": "user", "content": RW},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "c1", "content": FILE}]


def save(name, ids, skel=None):
    open(os.path.join(OUT, f"{which}_{name}_ids.txt"), "w").write(",".join(map(str, ids)) + "\n")
    if skel is not None:
        open(os.path.join(OUT, f"{which}_{name}_skel.txt"), "w").write("".join(" ".join(map(str, f)) + "\n" for f in skel))
    print(which, name, len(ids), "skeleton fragments", len(skel or []))


def copy_ids(name, path):
    save(name, [int(x) for x in open(path).read().replace(",", " ").split()])


if which == "q36":
    sys.path.insert(0, os.path.join(REPO, "python", "serve", "qwen3_5_moe"))
    import chat_format as CF

    tok = CF.Tok(MODEL)
    skel = [tok.encode(t) for t in CF.tool_skeletons(TOOLS)]
    copy_ids("prose", os.path.join(ENG, "results", "stage2", "pitt_ids.txt"))
    copy_ids("code", os.path.join(ENG, "results", "stage3", "code_ids.txt"))
    copy_ids("edit", os.path.join(ENG, "results", "stage4", "copy_ids.txt"))
    off = {"enable_thinking": False}
    for name, msgs in (("toolnew", [{"role": "user", "content": NEW}]), ("toolrw", msgs_rw('{"path": "orders.py"}')),
                       ("toolshort", [{"role": "user", "content": SHORT}])):
        ids, _ = tok.prompt_ids(CF.to_template(msgs), TOOLS, **off)
        save(name, ids, skel)
else:
    src = open(os.path.join(ENG, "ink", "run_ink.py")).read().split("from tokenizers import Tokenizer")[0]
    sys.argv = [sys.argv[0], "--prompt-ids-only", "--model", MODEL]
    MY_MODEL, MY_ENG = os.path.abspath(MODEL), os.path.abspath(ENG)
    exec(src)  # argparse defaults + render_chat
    MODEL, ENG = MY_MODEL, MY_ENG
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    enc = lambda t: tk.encode(t, add_special_tokens=False).ids

    def ink_skel(tools):
        out = []
        for t in tools:
            f = t["function"]
            keys = list(f["parameters"]["properties"].keys())
            args = "{" + ",".join(f'"{k}":"X"' for k in keys) + "}"
            out.append(enc(f'<|message_model|>{f["name"]}<|content_invoke_tool_json|>{{"name":"{f["name"]}","args":{args}}}<|end_message|><|content_model_end_sampling|>'))
        return out

    skel = ink_skel(TOOLS)
    save("prose", [371, 151698, 149879, 149876, 152652, 149879, 220, 136558, 165300, 149899])
    copy_ids("code", os.path.join(ENG, "results", "inkling_approx", "texts", "code_prompt_ids.txt"))
    copy_ids("edit", os.path.join(ENG, "results", "inkling_draft", "edit_ids.txt"))
    for name, msgs in (("toolnew", [{"role": "user", "content": NEW}]), ("toolrw", msgs_rw({"path": "orders.py"})),
                       ("toolshort", [{"role": "user", "content": SHORT}])):
        save(name, enc(render_chat(MODEL, msgs, tools=TOOLS, reasoning_effort="none")), skel)
