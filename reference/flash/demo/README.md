# Qwen3.8-Flash-Next in stock OpenCode on a 16 GB Mac

```bash
./demo.sh          # extra flags go to mlx_run.py, e.g. ./demo.sh -v
```

`demo.sh` starts the store server if nothing answers on 127.0.0.1:8080, waits for it, then opens OpenCode in this
folder. The server is `mlx_run.py --dir model --serve 8080 --prompt-cache demo/cache --warm --reasoning-effort low`
(prefetch mode, 10 experts named per layer, keep 3, 24 slots per layer; log in `demo/server.log`). It keeps running
after OpenCode quits. Stop it with:

```bash
pkill -f "mlx_run.py --dir model"
```

## What is stock and what is not

- OpenCode (tested with 1.19.0) with its default agent, prompt and tools. No custom prompt, no rules, no
  instructions file, no tool switched off.
- `opencode.json` only defines the local provider: the model id, `reasoning: true`, a 32,768-token context, and
  `small_model: store/title`, which the server answers at once with the first words of the message, so OpenCode's
  per-thread title request never touches the model.
- `demo.sh` sets three OpenCode environment variables. `OPENCODE_DISABLE_EXTERNAL_SKILLS=1` and
  `OPENCODE_DISABLE_CLAUDE_CODE=1` keep the machine's personal skills and instruction files out of the prompt, so the
  prompt is what a clean install sends (about 7,900 tokens). `OPENCODE_DISABLE_AUTOUPDATE=1` keeps the installed
  version.
- Thinking is on at effort low (`--reasoning-effort low`). The model's own default is xhigh; `./demo.sh
  --reasoning-effort xhigh` restores it and re-prefills the cached prompt once.
- Sampling is the release's `generation_config.json` (temperature 1.0, top-p 0.95, top-k 20).

## The task in the video

`intervals.py` has a one-line bug and `check.py` fails on it. The prompt was: "check.py isnt working. find the issue
and fix it". To put the bug back afterwards:

```bash
git checkout intervals.py
```

## What the server does around the model

- Reasoning is streamed as `reasoning_content` until `</think>`, the answer as `content`. Tool calls in the
  template's XML format are returned as OpenAI `tool_calls`, with arguments typed from the tool schema.
- Prompts of 48 tokens or more are prefilled as one batch, with every needed expert read once.
- The decoded prefix is reused across turns. OpenCode's system prompt is snapshotted to `demo/cache` once (about
  340 MB) with the expert slots it held; a new thread restores it in 1 to 2 s and prefills only its first message.
  `--warm` replays OpenCode's last request at start-up and at midnight, because the prompt carries the date. On a new
  machine there is no last request yet, so the first message of the first thread pays the cold prefill.
- On macOS the model is GPU memory, and the pager ages pages by CPU references, which GPU reads never produce. Under
  pressure it compresses the model while it is being read every token. The runner wires what it can and touches one
  byte per 16 KB page from the CPU on a schedule.

## Measured on a base M4 MacBook Pro, 16 GB

| what | time |
|---|---|
| `./demo.sh` to "serving", today's snapshot on disk | 12 s |
| first message of a new thread, Enter to the first Thought line | 4.5 to 6.7 s |
| first message of the day with no snapshot (cold prefill of about 7,900 tokens) | 90 to 120 s on a quiet machine |
| decode at an 8k to 9k-token context | 4.3 to 5.4 tokens/s |
| tool result prefill: 77 / 202 / 1,416 tokens | 9 to 13 s / 15 s / about 31 s |
| the bug fix in the video, prompt to "all checks pass" | 3 min 25 s |

Before a demo, close what you can and keep the lid open on power. With about 9 GB for the server, macOS compresses
the model when the other apps want more than the remaining 7 GB, and the next tokens are slow until it is paged back.
Do not start OpenCode within a minute of quitting it (1.19.0 can stall at start-up).
