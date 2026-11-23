"""The next prompt, predicted when a turn ends, for the worker's idle-time prefill (src/serving/ahead.h).

When an assistant turn ends without tool calls, the agent's next main request is the same conversation + this turn + a
new user message. Some chat templates render that conversation differently from the prompt just served (Qwen3.6's drops
the reasoning of every assistant turn before the last user message), so the next prompt leaves the cached state early
and the whole finished turn is prefilled again when the user answers. The part of the next prompt that does not depend
on the user's words is known now: render the conversation with the template the request path uses, with two different
placeholder user messages, and keep the common prefix up to the last message boundary before them. Nothing here
changes a prompt: a real request is always rendered from its own messages; a wrong prediction only costs idle compute."""


def lcp(a, b):
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


PLACEHOLDERS = ("Thanks. Next: please check the README.", "ok")


def predict(tok, messages, tools, kw, assistant_msg):
    """(ids, marks) of the predicted prefix of the next prompt; ([], []) if the template gives nothing stable.
    messages: the request's messages as the template takes them; assistant_msg: the turn just finished, as the client will
    send it back (to_message of the parsed turn)."""
    outs = []
    for ph in PLACEHOLDERS:
        ids, marks = tok.prompt_ids(list(messages) + [assistant_msg, {"role": "user", "content": ph}], tools, **kw)
        outs.append((ids, marks))
    (i1, m1), (i2, _) = outs
    common = lcp(i1, i2)
    cut = max([m for m in m1 if m <= common], default=0)
    return i1[:cut], [m for m in m1 if m <= cut]
