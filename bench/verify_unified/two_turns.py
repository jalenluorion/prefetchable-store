"""A two-message chat through the HTTP server (python/serve/dmserve.py): the first user message, the model's answer sent
back as the assistant turn (content and reasoning_content as the server returned them), then a second user message.
Prints seconds to the first streamed token of each request and the server's own request-log lines for them.
  two_turns.py PORT OUT.json "first message" "second message" [MAX_TOKENS]"""
import json, sys, time, urllib.request

port, out, m1, m2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 300
URL = f"http://127.0.0.1:{port}/v1/chat/completions"


def chat(msgs):
    body = json.dumps({"model": "m", "messages": msgs, "max_tokens": mx, "temperature": 0, "stream": True}).encode()
    t0, first, content, reasoning = time.time(), None, "", ""
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])["choices"][0].get("delta", {})
            if (d.get("content") or d.get("reasoning_content")) and first is None:
                first = time.time() - t0
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
    return {"ttft_s": first, "total_s": time.time() - t0, "content": content, "reasoning": reasoning}


for _ in range(600):  # wait for the server
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
        break
    except Exception:
        time.sleep(1)
msgs = [{"role": "user", "content": m1}]
a = chat(msgs)
msgs.append({"role": "assistant", "content": a["content"], "reasoning_content": a["reasoning"]})
msgs.append({"role": "user", "content": m2})
b = chat(msgs)
json.dump({"first": a, "second": b}, open(out, "w"))
print(json.dumps({"first_ttft_s": a["ttft_s"], "second_ttft_s": b["ttft_s"], "first_reasoning_chars": len(a["reasoning"]), "second_total_s": b["total_s"]}))
