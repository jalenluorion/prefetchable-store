"""Qwen3.8-Flash-Next's text in and out (python/serve/qwen4_exp/chat_format.py) against the release's own rendering:
transformers' apply_chat_template on the same files (tokenizer.json, tokenizer_config.json, chat_template.jinja). Token ids
must be equal for every conversation (system prompt, tools, multi-turn tool calls and results, reasoning kept or not, each
reasoning effort and thinking off); a generated turn must parse back into its reasoning, content and tool calls, and a
rendered assistant turn must parse back into the message it came from. Needs only the small text files of the repo.

  python tests/qwen4_exp/chat_check.py FOLDER      (python: tokenizers, jinja2, transformers)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(REPO, "python", "serve", "qwen4_exp"))
import chat_format as CF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = CF.Tok(D)
CF.bind(tok)
hf = AutoTokenizer.from_pretrained(D)
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string", "description": "the file"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a command", "parameters": {"type": "object", "properties": {
             "command": {"type": "string"}}, "required": ["command"]}}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "/tmp/a.txt", "limit": 20})}}
CONVS = {
    "one user turn": [{"role": "user", "content": "Explain what a mixture of experts is."}],
    "system + user": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello, how are you?"}],
    "tools, a call and its result": [
        {"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Show me /tmp/a.txt"},
        {"role": "assistant", "content": "", "reasoning_content": "The user wants the file.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "line one\nline two"}],
    "two user turns, earlier reasoning": [
        {"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."},
        {"role": "user", "content": "And 3 + 3?"}],
}
fails = []


def say(ok, s):
    print(("PASS  " if ok else "FAIL  ") + s, flush=True)
    if not ok:
        fails.append(s)


for name, msgs in CONVS.items():
    tools = TOOLS if "tools" in name else None
    for eff in (None, "high", "medium", "low", "none"):
        kw = CF.effort_kwargs(eff)
        ours, _ = tok.prompt_ids(CF.to_template(msgs), tools, **kw)
        ref = hf.apply_chat_template(CF.to_template(msgs), tools=tools, add_generation_prompt=True, tokenize=True, **kw)
        ref = ref["input_ids"] if isinstance(ref, dict) or hasattr(ref, "keys") else ref
        say(list(ours) == list(ref), f"{name}, effort {eff}: {len(ours)} ids, {'equal' if list(ours) == list(ref) else 'DIFFER'} to apply_chat_template")
# a generated turn -> reasoning, content, calls (the prompt ends with the template's opened <think>)
p_ids, _ = tok.prompt_ids([{"role": "user", "content": "x"}], TOOLS)
thinking = tok.thinking(p_ids)
gen = ("Let me look at the file.\n</think>\n\nI will read it.\n\n<tool_call>\n<function=read>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n"
       "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call><|im_end|>")
r, c, calls = CF.parse_turn(gen, thinking, TOOLS)
args = json.loads(calls[0]["function"]["arguments"]) if calls else None
say(thinking and r == "Let me look at the file." and c == "I will read it." and len(calls) == 1 and calls[0]["function"]["name"] == "read"
    and args == {"path": "/tmp/a.txt", "limit": 20}, f"parse_turn: reasoning {r!r}, content {c!r}, calls {[x['function'] for x in calls]}")
# a rendered assistant turn parses back into its message
u = [{"role": "user", "content": "Show me /tmp/a.txt"}]
pre = tok.render(u, TOOLS, add_generation_prompt=True)
msg = {"role": "assistant", "content": "Reading.", "reasoning_content": "Need the file.", "tool_calls": [CALL]}
full = tok.render(CF.to_template(u + [msg]), TOOLS, add_generation_prompt=False, preserve_thinking=True)
r, c, calls = CF.parse_turn(full[len(pre):], thinking, TOOLS) if full.startswith(pre) else (None, None, [])
say(full.startswith(pre) and r == "Need the file." and c == "Reading." and len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"path": "/tmp/a.txt", "limit": 20},
    f"round trip of a rendered assistant turn: reasoning {r!r}, content {c!r}, {len(calls)} call(s)")
print("CHAT CHECK " + ("PASSED" if not fails else "FAILED"))
sys.exit(1 if fails else 0)
