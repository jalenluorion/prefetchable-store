# prefetchable-store

Run mixture-of-experts language models that are larger than your memory, from the SSD, on a 16 GB MacBook and on an
iPhone. That includes a 180B-parameter model whose weights are eight times the laptop's memory.

| model | machine | memory used | speed |
|---|---|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/jalenluorion/Qwen3.8-Flash-Next-prefetchable-store) (180B parameters) | base M4 MacBook Pro, 16 GB | 8.6 GB | 4.3 to 4.4 tokens/s exact, 5.9 to 6.0 fast |
| [Qwen3.6-35B-A3B](https://huggingface.co/jalenluorion/Qwen3.6-35B-A3B-prefetchable-store) (35B parameters) | base M4 MacBook Pro, 16 GB | 9 to 11 GB | 26 to 29 tokens/s exact, 30 to 32 fast |
| [Inkling 2 Mini](https://huggingface.co/jalenluorion/Inkling-2-Mini-prefetchable-store) | base M4 MacBook Pro, 16 GB | 9 to 11 GB | 17 to 20 tokens/s exact, about 21 fast |
| Qwen3.6-35B-A3B | iPhone 15 Plus, 6 GB | 2.9 GB | 3.4 tokens/s exact, 4.1 fast |
| Inkling 2 Mini | iPhone 15 Plus, 6 GB | 2.9 GB | 1.5 tokens/s exact, 2.4 fast |
| Qwen3.6-35B-A3B | iPhone 17 Pro, 12 GB | | 11 tokens/s [mode to confirm] |
| Inkling 2 Mini | iPhone 17 Pro, 12 GB | | 4 tokens/s [mode to confirm] |

The engine makes released models **prefetchable**: the experts a layer will use are known before the layer runs,
early enough for the SSD to deliver them while the model computes something else. The released weights are frozen. The
only trained parts are small routers (62M parameters for the 35B, under 0.2% of the model) that name each layer's
experts two layers ahead (four layers ahead for the 180B model). The model's own router still makes every choice: in
`exact` mode the generated tokens are the same as with every expert in memory.

It is one C++ engine with a command-line tool, an OpenAI-compatible server for agent clients such as
[OpenCode](https://opencode.ai), and a C API that the iPhone app uses.

## Quick start

Apple Silicon with 16 GB or more and about 25 GB of free disk per model (about 127 GB for the 180B model):

```bash
git clone https://github.com/jalenluorion/prefetchable-store && cd prefetchable-store
cmake -S . -B build && cmake --build build -j
pip install tokenizers jinja2

hf download jalenluorion/Qwen3.6-35B-A3B-prefetchable-store --local-dir model
build/dmeng --model model --chat --prompt "Write a haiku about rivers." --n 200

python3 python/serve/dmserve.py --model-dir model --port 8090    # OpenAI-compatible server
```

For Inkling 2 Mini, download `jalenluorion/Inkling-2-Mini-prefetchable-store` instead; everything else is the same. A folder
downloaded from the Hub runs as it is. Each weight repo's card lists the files one device needs, so a Mac does not
download the phone's files.

**The 180B model.** Qwen3.8-Flash-Next runs on the Mac. Fetch its `mac` profile (about 127 GB: the experts, the
n-gram memory, the engine-ready in-memory weights and routers, the embedding table and the text files):

```bash
hf download jalenluorion/Qwen3.8-Flash-Next-prefetchable-store --local-dir flash \
  --include "store/*" "ngram/*" "derived/*" embeddings.safetensors moved-store.json config.json \
            generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja LICENSE
build/dmeng --model flash --chat --prompt "Explain what a mixture of experts is." --n 200
python3 python/serve/dmserve.py --model-dir flash --port 8090 --mode fast
```

The weight repo's card lists the files and the other way to start: from `resident.safetensors` and `routers_k4.npz`,
from which the engine makes the engine-ready files on its first run.

**On an iPhone.** [apps/ios](apps/ios/README.md) is PhoneStore, a chat app that runs Qwen3.6-35B-A3B or Inkling 2 Mini
on the phone: open `apps/ios/MovedStore.xcodeproj` in Xcode, set your own signing team and bundle identifier, and run it
on an iPhone with an A16 chip or newer. The app downloads the model from the Hub in the profile that fits the phone
(15.5 to 21.3 GB) and checks every file.

`dmeng --help` lists the everyday options (`--mode exact|fast`, `--cache-gb`, `--n`, `--temp`); `dmeng --help-advanced`
lists the rest ([docs/ADVANCED.md](docs/ADVANCED.md)). On a 16 GB machine close other large apps first: the expert cache
is sized from the memory that is free at start.

## Prefetchable models

A mixture-of-experts model uses a small part of itself per token: 8 of 256 experts in each of the 35B's 40 layers. Most
of the model could therefore stay on a drive. What stops that is timing. A released model picks a layer's experts from
the hidden state at that layer, so an expert is needed the moment it is chosen: either every expert sits in memory in
case it is picked, or each token stops and waits for the drive.

A model is prefetchable when that timing is turned around:

- **The choice is known ahead of use.** Each layer's experts are named before the layer runs, from what the model has
  already computed.
- **The lead covers the read.** The names arrive early enough, and a token reads little enough, for the drive to
  deliver the experts while other layers compute.
- **The model stays in charge.** Names only schedule reads. Which experts are used is still the model's own decision,
  so nothing about its output has to change.

When these hold, the drive stops being a wait and becomes background work, and the memory a model needs is set by what
it computes with per token, not by its size.

The more a model outgrows the machine's memory, the more this matters. The 180B model here has weights eight times
the size of the laptop's memory, and it runs on that laptop.

Released models are not built this way. This repository retrofits it onto frozen weights; building it in from the
first training step is the subject of the paper.

## How it works

Here the experts live on the SSD, one contiguous record each, and a cache in memory holds the ones in use. Three things
keep the drive ahead of the computation:

- **Early names.** A small router per layer reads the token's state **two layers earlier** (four for
  Qwen3.8-Flash-Next) and names the experts that layer will want. Their records are read while the layers in between
  compute. The routers are trained once per model by imitating the model's own routers, with the model frozen.
- **Rows named by the token ids.** Qwen3.8-Flash-Next also has an n-gram memory, 51.2B parameters on the SSD. A token
  uses 16 of its rows, whose addresses follow from the previous token ids alone, so they are read as soon as a token is
  chosen.
- **Run-ahead.** When a layer has to wait for a record, the idle cores run the next layer on what is already in memory
  and name its experts too, so one wait covers the next layer's reads. The waiting layer is then computed in full.
- **A cache that keeps what will be used.** Records are evicted by a score learned from how experts recur, and reads
  that are needed now go ahead of reads that were only named.

Names and run-ahead only schedule reads. What the model computes is decided by its own router:

- **`exact` (default on the Mac).** Every chosen expert is used; one that is not in memory yet is waited for. The
  tokens never depend on the cache, the routers or timing.
- **`fast` (default in the iPhone app).** A chosen expert that is not in memory is waited for only if its routing weight is above a threshold;
  otherwise the layer goes on without it. Faster on small caches and slow drives, with a small measured change in
  output (see Quality). For Inkling 2 Mini and Qwen3.8-Flash-Next (whose router picks 10 experts per layer) the
  missing picks are read heaviest first until at most 0.13 of the routing weight would be left out, and the rest are
  left out; the 35B keeps the threshold, measured faster on the iPhone.

Prompts run through a batched prefill that reads each expert once per span and leaves the same state, bit for bit, as
token-by-token steps; on the Mac the GPU computes a share of it beside the CPU with the same bits. The server keeps a
prefix cache with checkpoints, so a follow-up turn computes only what is new, and it starts on the likely next turn
while idle.

| part | where it lives |
|---|---|
| routed experts (4-bit; Inkling's as released) | SSD, one record per expert; a cache of 5 to 9 GB in memory (3 GB for the 180B model) |
| n-gram memory (Qwen3.8-Flash-Next, 8-bit rows) | SSD, 16 rows per token |
| attention layers, shared experts, norms, output head | memory, 8-bit |
| early routers | memory |
| input embedding | SSD, one row per token |

On an iPhone the same engine runs with a smaller cache and 4-bit in-memory weights fitted on calibration text.

## Speed and memory

Base M4 MacBook Pro, 16 GB, internal SSD, plugged in, 8 threads, greedy decoding, 400 generated tokens, range over two
runs. "On demand" is the like-for-like comparison: the same engine and cache size with no early names and no run-ahead,
every expert read when its layer asks for it.

**Qwen3.6-35B-A3B** (tokens/s)

| expert cache | text | on demand | exact | fast | memory |
|---|---|---|---|---|---|
| 5 GB | prose | 18.0-18.4 | 22.3-23.1 | 26.3-28.8 | 7.3 GB |
| 5 GB | code | 16.0-16.1 | 18.8-18.9 | 22.6-23.2 | 7.8 GB |
| 6.6 GB | prose | 22.1-22.2 | 25.7-26.9 | 30.0-31.3 | 8.9 GB |
| 6.6 GB | code | 19.3-19.5 | 22.1-22.2 | 27.5-27.8 | 9.4 GB |
| 8 GB | prose | | 27.7-29.2 | 31.3-32.0 | 10.3 GB |
| 8 GB | code | | 22.3-23.7 | 26.8-29.9 | 10.8 GB |

**Inkling 2 Mini** (tokens/s)

| expert cache | text | on demand | exact | fast | memory |
|---|---|---|---|---|---|
| 6 GB | prose | 12.7-12.8 | 14.1-14.2 | 18.0-18.8 | 7.5 GB |
| 6 GB | code | 9.1 | 9.8-10.0 | 13.0-13.4 | 8.2 GB |
| 7.8 GB | prose | 15.5-16.6 | 17.2-17.7 | 20.8 | 9.3 GB |
| 7.8 GB | code | 11.7-12.8 | 13.2-13.3 | 15.2-16.8 | 10.0 GB |
| 9 GB | prose | | 18.3-19.6 | 21.6 | 10.5 GB |
| 9 GB | code | | 14.7-14.8 | 19.1 | 11.2 GB |

**Qwen3.8-Flash-Next, 180B** (160 generated tokens, two alternating runs, range; tokens/s)

| expert cache | exact | fast | read per token, exact / fast | memory |
|---|---|---|---|---|
| 3.0 GB | 4.3-4.4 | 5.9-6.0 | 692 MB / 481-485 MB | 8.6 GB |
| 3.6 GB | | 6.5 (one pair of runs) | | |

For comparison, stock llama.cpp with a 4-bit GGUF of the same model, memory-mapped, CPU only, on the same Mac: 0.45
and 0.39 tokens/s. That uses a different 4-bit file, so it shows what the stock tool gives on this machine, not a
controlled comparison (logs in [results/flash_next/llama_cpp](results/flash_next/llama_cpp)).

**Agent turns** (recorded OpenCode requests: prefill, then generation to the end of the turn; tokens/s)

| model | turn | expert cache | exact | fast |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | file rewrite by tool call | 6.6 GB | 17.5 | 19.3 |
| Qwen3.6-35B-A3B | code answer | 6.6 GB | 22.7 | 26.8 |
| Inkling 2 Mini | file rewrite by tool call | 7.8 GB | 12.4 | 16.6 |
| Inkling 2 Mini | code answer | 7.8 GB | 13.5 | 16.8 |
| Qwen3.8-Flash-Next | decode only, after each tool result of a session at 8,000 to 10,000 tokens of context | 3.0 GB | | 4.1-4.9 |

In stock OpenCode, a bug-fix task ([bench/serving/demo_task.sh](bench/serving/demo_task.sh)) took Qwen3.8-Flash-Next
(fast, 3.0 GB cache) 2 min 36 s and 3 min 01 s from the prompt to "all checks pass", in two runs.

**Long prompts through the server** (the recorded first request of an OpenCode session; seconds to first token)

| model | cold request | the same context again | 2,000 more tokens |
|---|---|---|---|
| Qwen3.6-35B-A3B (7,572 tokens) | 68.6 | 15.3 | 24.0 |
| Inkling 2 Mini (8,365 tokens) | 162.4 | 6.7 | 46.6 |
| Qwen3.8-Flash-Next (7,598 tokens), fast, 3.0 GB cache | 81.6 | | |

Qwen3.8-Flash-Next's cold request reads each expert it needs once, about 68 GB. Later requests of the same session
(fast, 3.0 GB cache; seconds to first token):

| request | seconds |
|---|---|
| a tool result of 20 to 59 new tokens | 3.1-6.2 |
| a tool result of 209 / 314 new tokens | 12.9 / 13.2 |
| the first message of a new thread, from the saved system prompt | 3.3 |

Through the server the process used 9.0 GB at its peak, during the long prompt, and 8.5 GB between requests.

**iPhone 15 Plus** (6 GB phone; the app uses 2.9 GB; measured on the device)

| | Qwen3.6-35B-A3B, exact | Qwen3.6-35B-A3B, fast | Inkling 2 Mini, exact | Inkling 2 Mini, fast |
|---|---|---|---|---|
| decode in a chat, tokens/s | 3.4 | 4.1 | | |
| decode, long prose / code, tokens/s | 2.9 / 2.1 | 3.7 / 2.9 | 1.5 / - | 2.4 / - |
| first token of a new chat, sent at once / typed ahead | 4.8 s / 3.7 s | 4.8 s / 3.9 s | | |
| first token of a follow-up | 4.0 s | 4.2 s | | |
| first token after relaunching the app | 4.2 s | 4.3 s | | |
| launch to ready | 2.7-3.2 s | 2.7-2.9 s | | |

The phone's drive sets its decode speed (about 0.85 GB/s). The 35B reads 180 to 240 MB per token; Inkling 2 Mini reads
300 MB in fast and 590 MB in exact, and its exact tokens equal the Mac's.

**iPhone 17 Pro** (12 GB; measured by a user on the published app, out of the box)

On a phone with more memory the expert cache grows with it: the engine sizes it from what iOS makes available to the
app.

| | mode | decode, tokens/s |
|---|---|---|
| Qwen3.6-35B-A3B | [mode to confirm] | 11 |
| Inkling 2 Mini | [mode to confirm] | 4 |

The run records behind the tables of Qwen3.6-35B-A3B and Inkling 2 Mini on the Mac and the iPhone 15 Plus are in
[results/headline](results/headline).

## Quality

`exact` changes nothing: at any cache size the generated tokens equal those of the same weights held fully in memory.

`fast` leaves out a few low-weight experts. Share of positions where its top choice equals `exact`'s, on held-out
text with the true previous tokens:

| model, machine | prose | code or agent text |
|---|---|---|
| Qwen3.6-35B-A3B, Mac (3.3 to 5 GB cache) | 0.96 | 0.98 |
| Inkling 2 Mini, Mac (6 to 7.8 GB cache) | 0.965-0.973 | 0.966-0.967 |
| Qwen3.8-Flash-Next, Mac (3.0 GB cache) | 0.964 | 0.992 |

For Qwen3.8-Flash-Next the two texts are `exact`'s own 600-token answers to an explanation prompt and to a code-review
agent prompt, fed back; `fast` changes the loss against `exact` by +0.016 and +0.008 nats.

Qwen3.8-Flash-Next in `exact` gives the same token ids at two cache sizes. Against the released bf16 model on its
reference prompt, the first 26 generated tokens are equal, and with the released model's tokens fed in, 62 of 64 top
choices are equal.

On the iPhone the in-memory weights are 4-bit as well. Against the Mac's 8-bit `exact` engine, the phone's weights
agree on 0.913 of prose and 0.884 of agent-text positions in `exact`, and on 0.891 and 0.881 in `fast`, measured on
the phone.

The stored weights themselves (4-bit experts for the 35B and Qwen3.8-Flash-Next, Inkling's experts as released,
Qwen3.8-Flash-Next's 8-bit n-gram rows, 8-bit in-memory weights) are compared with each release on the weight repos'
model cards.

## Supported models

| model | weights | notes |
|---|---|---|
| Qwen3.6-35B-A3B | [jalenluorion/Qwen3.6-35B-A3B-prefetchable-store](https://huggingface.co/jalenluorion/Qwen3.6-35B-A3B-prefetchable-store) | Mac and iPhone |
| Inkling 2 Mini | [jalenluorion/Inkling-2-Mini-prefetchable-store](https://huggingface.co/jalenluorion/Inkling-2-Mini-prefetchable-store) | Mac and iPhone |
| Qwen3.8-Flash-Next (180B) | [jalenluorion/Qwen3.8-Flash-Next-prefetchable-store](https://huggingface.co/jalenluorion/Qwen3.8-Flash-Next-prefetchable-store) | Mac |

A model family is a short description of its layers, state and expert block in `src/models/<family>/`; the store, the
step, the routers and the serving are shared. The weight repos share one layout and one manifest
([docs/MANIFEST-PRESETS.md](docs/MANIFEST-PRESETS.md)).

## What is in the repo

| path | what it is |
|---|---|
| `include/dmeng/`, `src/core/` | the engine: expert store and readers, the expert step, early routers, run-ahead, kernels |
| `src/models/` | one folder per model family (`qwen3_5_moe`, `inkling2_loop`, `qwen4_exp`); `common/` the Gated DeltaNet both Qwen families share |
| `src/serving/` | prefix cache, idle-time prefill, memory watcher |
| `tools/` | `dmeng` (command line), `dmserve` (model worker), `ms_derive` (engine-ready files from a weight repo) |
| `python/` | text in and out, the OpenAI-compatible front end, conversion from released weights |
| `bindings/c/`, `apps/ios/` | the C API and the iPhone chat app ([apps/ios/README.md](apps/ios/README.md)) |
| `tests/` | small random models of each family checked against independent references; exactness checks for prefill |
| `bench/` | measurement scripts: benchmarks, serving and agent sessions |
| `results/` | the run records behind the tables above; `results/flash_next/` the 180B model's conversion checks, router evaluation and the llama.cpp logs |

The default build targets Apple M4 (`-DDM_ARCH=...` to change); `-DDM_METAL=OFF` builds without the GPU prefill path.

## Limits

- These are retrofits of models that were not built to be read from a drive. A token reads 40 to 270 MB depending on
  the model, the text and the cache (the 180B model at a 3.0 GB cache: 481 to 485 MB in `fast`, 692 MB in `exact`), so
  the cache size and the drive set the speed.
- The 35B's experts are stored at 4-bit and the in-memory weights at 8-bit (4-bit on small iPhones); Qwen3.8-Flash-Next's
  experts at 4-bit, its n-gram rows and in-memory weights at 8-bit. The numbers above are for those weights, not for the
  released bf16 weights.
- Qwen3.8-Flash-Next runs on the Mac only.
- One request at a time. CPU kernels for Apple Silicon, with a GPU share of prompt prefill on the Mac. Text only.
- Timings are from one machine. A machine under memory pressure is slower: the cache is sized from free memory.

## Paper

Paper soon. This repository retrofits released models. Models built for this from the first training step are what we
call prefetchable, and they are the subject of the paper.

## License

Code: MIT (`LICENSE`). Each model is under its upstream licence (Qwen3.6-35B-A3B: Apache-2.0; Qwen3.8-Flash-Next: the
Qwen Community License 1.0; Inkling 2 Mini: Apache-2.0).
