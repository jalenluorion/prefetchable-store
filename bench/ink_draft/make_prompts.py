"""The draft head's extra timing prompt, in Inkling's chat template (results/ink_draft/prompts/):
  newfile_ids.txt  a tool-call turn that writes a NEW file of about 600 tokens (tools declared, nothing to copy from)
The rewrite-a-file-in-the-conversation case is the existing tool prompt (engine/results/inkling_draft/tool_ids.txt:
read_file result, then the model writes the whole updated file).
  INK_PY bench/ink_draft/make_prompts.py [--model MODEL]   (needs jinja2 and tokenizers)
"""
import argparse, json, os

from jinja2.ext import Extension
import jinja2
from tokenizers import Tokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", os.path.normpath(os.path.join(REPO, "..", "..", "..", "inkling", "model"))))
a = ap.parse_args()


def render_chat(model_dir, messages, **kw):  # as engine/ink/run_ink.py (mlx_run.py's): plain jinja2, no generation prompt
    class Generation(Extension):
        tags = {"generation"}

        def parse(self, parser):
            next(parser.stream)
            return parser.parse_statements(["name:endgeneration"], drop_needle=True)

    def raise_exception(msg):
        raise ValueError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, "jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, sort_keys=False, separators=None, **_: json.dumps(x, sort_keys=sort_keys, separators=separators, ensure_ascii=False)
    tpl = env.from_string(open(os.path.join(model_dir, "chat_template.jinja")).read())
    return tpl.render(messages=messages, **{k: v for k, v in kw.items() if v is not None})


tools = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write the complete new contents of a text file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                                     "required": ["path", "content"]}}},
]
ask = ("Create a new file inventory.py for our warehouse scripts. It should have a dataclass Item (sku, name, quantity, "
       "reorder_level, unit_cost), a function load_items(path) that reads items from a CSV file with a header row, "
       "low_stock(items) returning the items at or below their reorder level, restock_cost(items) returning the cost of "
       "bringing every low item back to twice its reorder level, and a main() that prints a short report. Add docstrings. "
       "Use the write_file tool to write the whole file.")
text = render_chat(a.model, [{"role": "user", "content": ask}], tools=tools, reasoning_effort="none")
tk = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
ids = tk.encode(text, add_special_tokens=False).ids
out = os.path.join(REPO, "results", "ink_draft", "prompts")
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "newfile_ids.txt"), "w").write(",".join(map(str, ids)) + "\n")
open(os.path.join(out, "newfile_prompt.txt"), "w").write(text)
print("newfile", len(ids))
