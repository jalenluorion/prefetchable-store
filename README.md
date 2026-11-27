# prefetchable-store

Run mixture-of-experts language models that are larger than your memory, from the SSD, on a 16 GB MacBook and on an
iPhone. That includes a 180B-parameter model whose weights are eight times the laptop's memory.

| model | machine | weights (experts / in memory) | memory used (GB) | speed (tokens/s) |
|---|---|---|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/jalenluorion/Qwen3.8-Flash-Next-prefetchable-store) (180B parameters) | base M4 MacBook Pro, 16 GB | 4-bit / 8-bit | 8.6 | 4.3-4.4 `exact`, 5.9-6.0 `fast` |
| [Qwen3.6-35B-A3B](https://huggingface.co/jalenluorion/Qwen3.6-35B-A3B-prefetchable-store) (35B parameters) | base M4 MacBook Pro, 16 GB | 4-bit / 8-bit | 9-11 | 25-27 `exact`, 30-32 `fast` |
| [Inkling 2 Mini](https://huggingface.co/jalenluorion/Inkling-2-Mini-prefetchable-store) | base M4 MacBook Pro, 16 GB | 4-bit and 8-bit (as released) / 8-bit | 9-11 | 17-19 `exact`, 20-22 `fast` |
| Qwen3.6-35B-A3B | iPhone 15 Plus, 6 GB | 4-bit / 4-bit | 2.9 | 3.0-3.7 `exact`, 5.0-6.4 `fast` |
| Inkling 2 Mini | iPhone 15 Plus, 6 GB | 4-bit / 4-bit | 2.9 | 1.4-1.5 `exact`, 4.5-5.3 `fast` |
| Qwen3.6-35B-A3B | iPhone 17 Pro, 12 GB | 4-bit / 8-bit | about 6 | 11-13 `exact`, 12-14 `fast` (estimated, see [Speed and memory](#speed-and-memory)) |
| Inkling 2 Mini | iPhone 17 Pro, 12 GB | 4-bit and 8-bit (as released) / 8-bit | about 6 | about 6 `exact`, 7-8 `fast` (estimated, see [Speed and memory](#speed-and-memory)) |

Inkling 2 Mini runs on the Mac and on the 12 GB iPhone with its weights as released, not quantized again to fit: the
experts, 16 of its 18 GB, are the release's own bytes (4-bit in layers 0-19, 8-bit in layers 20-23), and the engine computes the attention layers and output head from
an 8-bit copy of the released values that matches them at the noise floor. The other rows run our 4-bit and 8-bit
copies of the released weights.

The engine makes released models **prefetchable**: the experts a layer will use are known before the layer runs,
early enough for the SSD to deliver them while the model computes something else. The released weights are frozen. The
only trained parts are small routers (62M parameters for the 35B, under 0.2% of the model) that name each layer's
experts two layers ahead (four layers ahead for the 180B model). The model's own router still makes every choice: in
`exact` mode the generated tokens are the same as with every expert in memory.

It is one C++ engine with a command-line tool, an OpenAI-compatible server for agent clients such as
[OpenCode](https://opencode.ai), and a C API that the iPhone app uses.

## Quick start

Apple Silicon with 16 GB or more, and free disk for the model: 22 GB for Qwen3.6-35B-A3B, 18 GB for Inkling 2 Mini,
127 GB for Qwen3.8-Flash-Next.

```bash
git clone https://github.com/jalenluorion/prefetchable-store && cd prefetchable-store
cmake -S . -B build && cmake --build build -j
pip install huggingface_hub tokenizers jinja2

python3 python/fetch.py 35b model          # or: inkling, 180b
build/dmeng --model model --chat --prompt "Write a haiku about rivers." --n 200
python3 python/serve/dmserve.py --model-dir model --port 8090    # OpenAI-compatible server
```

The three models run the same way. `python/fetch.py` downloads the files the model's manifest lists for this machine
(the `mac` profile of its `moved-store.json`), so a Mac does not download the phone's files; a whole weight repo
downloaded from the Hub runs as it is, too. `--mode fast` on either command switches from `exact` to `fast`.

`dmeng --help` lists the everyday options (`--mode exact|fast`, `--cache-gb`, `--n`, `--temp`); `dmeng --help-advanced`
lists the rest ([docs/ADVANCED.md](docs/ADVANCED.md)). On a 16 GB machine close other large apps first: the expert cache
is sized from the memory that is free at start.

**On an iPhone.** [apps/ios](apps/ios/README.md) is PhoneStore, a chat app that runs Qwen3.6-35B-A3B or Inkling 2 Mini
on the phone: open `apps/ios/MovedStore.xcodeproj` in Xcode, set your own signing team and bundle identifier, and run it
on an iPhone with an A16 chip or newer. The app downloads the model from the Hub in the profile that fits the phone
(15.5 to 21.3 GB) and checks every file.

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
it computes with per token, not by its size. The more a model outgrows the machine's memory, the more this matters, as
for the 180B model above.

Released models are not built this way. This repository retrofits it onto their frozen weights, as described next.

## How it works

The experts live on the SSD, one contiguous record each, and a cache in memory holds the ones in use. These keep the
drive ahead of the computation:

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

Names and run-ahead only schedule reads. What the model computes is decided by its own router, in one of two modes:

- **`exact` (default on the Mac).** Every chosen expert is used; one that is not in memory yet is waited for. The
  tokens never depend on the cache, the routers or timing.
- **`fast` (default on the iPhone).** A chosen expert that is not in memory is waited for only if its routing weight
  is above a threshold (0.15 for the 35B and Inkling 2 Mini, 0.10 for the 35B on 6 GB iPhones) or if its read is already in flight; otherwise the layer
  goes on without it. Qwen3.8-Flash-Next, whose router picks 10 experts per layer, reads its missing picks heaviest
  first until at most 0.13 of the routing weight would be left out, leaves out the rest, and does not wait for reads in
  flight. `fast` is faster on small caches and slow drives, with a small measured change in output (see
  [Quality](#quality)).

Prompts run through a batched prefill that reads each expert once per span and leaves the same state, bit for bit, as
token-by-token steps; on the Mac the GPU computes a share of it beside the CPU with the same bits. The server keeps a
prefix cache with checkpoints, so a follow-up turn computes only what is new, and it starts on the likely next turn
while idle.

Where each part of a model lives:

| part | where it lives |
|---|---|
| routed experts (4-bit; Inkling 2 Mini's as released) | SSD, one record per expert; a cache of 5-9 GB in memory (3 GB for the 180B model) |
| n-gram memory (Qwen3.8-Flash-Next, 8-bit rows) | SSD, 16 rows per token |
| attention layers, shared experts, norms, output head | memory, 8-bit |
| early routers | memory |
| input embedding | SSD, one row per token |

On an iPhone the same engine runs with a smaller cache and 4-bit in-memory weights fitted on calibration text. On a
phone with 11 GB or more it also uses spare memory, in both modes (see [iPhone 17 Pro](#iphone-17-pro-estimated)).

## Speed and memory

The run records behind the Mac and iPhone 15 Plus tables of Qwen3.6-35B-A3B and Inkling 2 Mini are in
[results/headline](results/headline).

### Mac

Base M4 MacBook Pro, 16 GB, internal SSD, plugged in, 8 threads, greedy decoding, 400 generated tokens, range over two
runs. "On demand" is the like-for-like comparison: the same engine and cache size with no early names and no run-ahead, every expert read
when its layer asks for it.

**Qwen3.6-35B-A3B**, 4-bit experts and 8-bit in-memory weights, decode (tokens/s)

| expert cache | text | on demand | `exact` | `fast` | memory (GB) |
|---|---|---|---|---|---|
| 5 GB | prose | 20.1-20.7 | 22.2-22.4 | 26.8-28.1 | 7.3 |
| 5 GB | code | 17.7-17.8 | 18.7-19.1 | 26.2-27.3 | 7.5 |
| 6.6 GB | prose | 23.5-24.1 | 25.4-26.6 | 30.3-32.1 | 8.9 |
| 6.6 GB | code | 20.0-21.8 | 20.6-21.8 | 29.1-29.7 | 9.1 |
| 8 GB | prose | 27.1-27.3 | 26.5-27.1 | 29.5-31.3 | 10.3 |
| 8 GB | code | 22.1-22.2 | 23.5-24.6 | 30.6-31.2 | 10.6 |

**Inkling 2 Mini**, 4-bit and 8-bit experts (as released) and 8-bit in-memory weights, decode (tokens/s)

| expert cache | text | on demand | `exact` | `fast` | memory (GB) |
|---|---|---|---|---|---|
| 6 GB | prose | 12.9-13.1 | 13.9-14.0 | 21.2-21.7 | 7.5 |
| 6 GB | code | 9.4-9.6 | 9.7 | 18.9-19.4 | 7.9 |
| 7.8 GB | prose | 16.2-16.6 | 17.3 | 20.2-21.9 | 9.3 |
| 7.8 GB | code | 12.9 | 13.1 | 20.0-20.2 | 9.7 |
| 9 GB | prose | 18.4-18.6 | 17.9-19.0 | 20.2-21.5 | 10.5 |
| 9 GB | code | 13.5-14.1 | 15.2-15.3 | 20.2-20.3 | 10.9 |

**Qwen3.8-Flash-Next, 180B**, 4-bit experts and 8-bit in-memory weights, decode (160 generated tokens, two alternating runs, range; tokens/s)

| expert cache | `exact` | `fast` | read per token, `exact` / `fast` (MB) | memory (GB) |
|---|---|---|---|---|
| 3.0 GB | 4.3-4.4 | 5.9-6.0 | 692 / 481-485 | 8.6 |
| 3.6 GB | | 6.5 (one pair of runs) | | |

For comparison, stock llama.cpp with a 4-bit GGUF of the same model, memory-mapped, CPU only, on the same Mac: 0.45
and 0.39 tokens per second. That uses a different 4-bit file, so it shows what the stock tool gives on this machine,
not a controlled comparison (logs in [results/flash_next/llama_cpp](results/flash_next/llama_cpp)).

**Agent turns** (recorded OpenCode requests: prefill, then generation to the end of the turn; tokens/s, range over two
runs)

| model | turn | expert cache | `exact` | `fast` |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | file rewrite by tool call | 6.6 GB | 18.0 | 26.7-27.8 |
| Qwen3.6-35B-A3B | code answer | 6.6 GB | 21.8-22.0 | 30.0-30.3 |
| Inkling 2 Mini | file rewrite by tool call | 7.8 GB | 11.9 | 18.6-19.4 |
| Inkling 2 Mini | code answer | 7.8 GB | 13.0-13.2 | 19.4-19.8 |
| Qwen3.8-Flash-Next | decode only, after each tool result of a session at 8,000-10,000 tokens of context | 3.0 GB | | 4.1-4.9 |

In stock OpenCode, a bug-fix task ([bench/serving/demo_task.sh](bench/serving/demo_task.sh)) took Qwen3.6-35B-A3B
2 min 08 s and 6 min 02 s, Inkling 2 Mini 46 s and 1 min 09 s, and Qwen3.8-Flash-Next 2 min 36 s and 3 min 01 s from the
prompt to "all checks pass" (`fast`, two runs each). Once the system prompt is saved, a new thread's first token comes
after 1.7 to 1.8 s for the 35B and Inkling 2 Mini.

**Long prompts through the server** (the recorded first request of an OpenCode session; seconds to first token, range
over two runs for the 35B and Inkling 2 Mini)

| model | cold request | the same context again | 2,000 more tokens |
|---|---|---|---|
| Qwen3.6-35B-A3B (7,572 tokens) | 68.9-69.2 | 15.1 | 22.9-23.4 |
| Inkling 2 Mini (8,365 tokens) | 124.4-124.5 | 4.8 | 35.1 |
| Qwen3.8-Flash-Next (7,598 tokens), `fast`, 3.0 GB cache | 81.6 | | |

Qwen3.8-Flash-Next's cold request reads each expert it needs once, about 68 GB. Its later requests in the same session
(`fast`, 3.0 GB cache; seconds to first token):

| request | seconds |
|---|---|
| a tool result of 20-59 new tokens | 3.1-6.2 |
| a tool result of 209 / 314 new tokens | 12.9 / 13.2 |
| the first message of a new thread, from the saved system prompt | 3.3 |

Through these requests the process peaked at 9.9 GB for the 35B (6.6 GB cache), 9.2 GB for Inkling 2 Mini (6 GB cache)
and 9.0 GB for Qwen3.8-Flash-Next.

### iPhone 15 Plus

6 GB phone; the app uses 2.9 GB; 4-bit experts and 4-bit in-memory weights; measured on the device.

| | Qwen3.6-35B-A3B, `exact` | Qwen3.6-35B-A3B, `fast` | Inkling 2 Mini, `exact` | Inkling 2 Mini, `fast` |
|---|---|---|---|---|
| decode in a chat (tokens/s) | 3.4-3.7 | 5.8-6.4 | 1.3-1.4 | 5.1-5.3 |
| decode, long prose / code (tokens/s) | 3.0-3.4 / 2.4-2.7 | 5.0-5.9 / 3.5-4.2 | 1.4-1.5 / 1.0 | 4.5-5.1 / 4.9 |
| first token of a new chat, sent at once / typed ahead (s) | 4.6-5.4 / 3.2-3.3 | 3.2-3.3 / 2.3 | 10.5-11.1 / 5.2 | 2.1-2.6 / 1.1 |
| first token of a follow-up (s) | 4.0-4.3 | 2.6 | 6.1-6.2 | 1.4-1.6 |
| first token after relaunching the app (s) | 4.4-4.6 | 2.7 | - | 3.1 |
| launch to ready (s) | 1.9-2.4 | 1.9-2.4 | 1.4-1.8 | 1.4-1.8 |

The phone's drive sets its decode speed (about 0.85 GB/s): the 35B reads 210 to 310 MB per token in `exact` and 100
to 175 in `fast`, Inkling 2 Mini 570 and 110. A dash is a cell without a settled reading.

### iPhone 17 Pro (estimated)

On a phone with more memory the expert cache is larger, the system's spare memory holds more of the model's files, and
Inkling 2 Mini runs with its weights as released. The figures are estimates: readings from an iPhone 17
Pro (6.7 tokens per second for Inkling 2 Mini in `fast`, 11 for the 35B) carried to the current defaults with a stand-in,
the engine on the Mac under the `phone` profile at that phone's drive rate and cache size. Most of a token on that phone
is compute, so `exact` and `fast` are close.

| | weights (experts / in memory) | `exact` (tokens/s) | `fast` (tokens/s) |
|---|---|---|---|
| Qwen3.6-35B-A3B | 4-bit / 8-bit | 11-13 | 12-14 |
| Inkling 2 Mini | 4-bit and 8-bit (as released) / 8-bit | about 6 | 7-8 |

## Quality

`exact` changes nothing: at any cache size the generated tokens equal those of the same weights held fully in memory.

`fast` leaves out a few low-weight experts. On the Mac, the share of positions where its top choice equals `exact`'s,
and its change in loss, on `exact`'s own continuation of a prose prompt and of a code or agent prompt:

| model, machine | prose | code or agent text | loss change, prose / code or agent text (nats) |
|---|---|---|---|
| Qwen3.6-35B-A3B, Mac (6.6 GB cache, 1,100 positions) | 0.987 | 0.976 | +0.005 / +0.006 |
| Inkling 2 Mini, Mac (7.8 GB cache, 1,100 positions) | 0.985 | 0.975 | +0.005 / +0.007 |
| Qwen3.8-Flash-Next, Mac (3.0 GB cache, 600 positions) | 0.964 | 0.992 | +0.016 / +0.008 |

On the iPhone 15 Plus the in-memory weights are 4-bit and the cache is 1.4 GB. Against the Mac's 8-bit `exact` engine,
the 35B there agrees on 0.913 of prose and 0.884 of agent-text positions in `exact`, and on 0.860 and 0.868 in `fast`.
Most of that difference is the quantization: `exact` changes nothing but the weights, and it already differs on 0.087
and 0.116 of positions. The rest, 0.053 and 0.016, is `fast`: with so small a cache more of a token's picks are missing,
so it leaves out more than on the Mac.

The stored weights themselves (4-bit experts for the 35B and Qwen3.8-Flash-Next, Inkling 2 Mini's experts as released,
Qwen3.8-Flash-Next's 8-bit n-gram rows, 8-bit in-memory weights) are compared with each release on the weight repos'
model cards.

## Supported models

| model | weights | `fetch.py` name | download for a Mac (GB) | runs on |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | [jalenluorion/Qwen3.6-35B-A3B-prefetchable-store](https://huggingface.co/jalenluorion/Qwen3.6-35B-A3B-prefetchable-store) | `35b` | 21.3 | Mac and iPhone |
| Inkling 2 Mini | [jalenluorion/Inkling-2-Mini-prefetchable-store](https://huggingface.co/jalenluorion/Inkling-2-Mini-prefetchable-store) | `inkling` | 18.0 | Mac and iPhone |
| Qwen3.8-Flash-Next (180B) | [jalenluorion/Qwen3.8-Flash-Next-prefetchable-store](https://huggingface.co/jalenluorion/Qwen3.8-Flash-Next-prefetchable-store) | `180b` | 126.5 | Mac |

A model family is a short description of its layers, state and expert block in `src/models/<family>/`; the store, the
step, the routers and the serving are shared. The weight repos share one layout and one manifest
([docs/MANIFEST-PRESETS.md](docs/MANIFEST-PRESETS.md)).

## What is in the repo

| path | what it is |
|---|---|
| `include/dmeng/`, `src/core/` | the engine: expert store and readers, the expert step, early routers, run-ahead, kernels |
| `src/models/` | one folder per model family (`qwen3_5_moe`, `inkling2_loop`, `qwen4_exp`); `common/` the Gated DeltaNet both Qwen families share |
| `src/serving/` | prefix cache, idle-time prefill, memory watcher |
| `tools/` | `dmeng` (command line), `dmserve` (model worker), `ms_derive` (engine-ready files from a weight repo), `manifest_dump` (checks a download against its manifest) |
| `python/` | `fetch.py` (download a model), text in and out, the OpenAI-compatible front end, conversion from released weights |
| `bindings/c/`, `apps/ios/` | the C API and the iPhone chat app ([apps/ios/README.md](apps/ios/README.md)) |
| `tests/` | small random models of each family checked against independent references; exactness checks for prefill |
| `bench/` | measurement scripts: benchmarks, serving and agent sessions |
| `results/` | the run records behind the tables above; `results/flash_next/` the 180B model's conversion checks, router evaluation and the llama.cpp logs |

The default build targets Apple M4 (`-DDM_ARCH=-mcpu=apple-m3`, for example, to change); `-DDM_METAL=OFF` builds
without the GPU prefill path.

## Limits

- These are retrofits of models that were not built to be read from a drive. A token reads 20 to 270 MB on
  the Mac (480 to 690 MB for the 180B model) and 100 to 570 MB on the iPhone 15 Plus, so the cache size and the drive
  set the speed.
- The 35B's experts are stored at 4-bit, Inkling 2 Mini's as released, and the in-memory weights of both at 8-bit
  (4-bit on small iPhones); Qwen3.8-Flash-Next's experts at 4-bit, its n-gram rows and in-memory weights at 8-bit. The
  numbers above are for those weights, not for the released bf16 weights.
- Qwen3.8-Flash-Next runs on the Mac only.
- One request at a time. CPU kernels for Apple Silicon, with a GPU share of prompt prefill on the Mac. Text only.
- Timings are from one Mac and one iPhone 15 Plus. A machine under memory pressure is slower: the cache is sized from
  free memory.

## Paper

Paper soon. This repository makes released models prefetchable after the fact, with their weights frozen; the paper is
about models built to be prefetchable from the first training step.

## License

Code: MIT (`LICENSE`). Each model is under its upstream license (Qwen3.6-35B-A3B: Apache-2.0; Qwen3.8-Flash-Next: the
Qwen Community License 1.0; Inkling 2 Mini: Apache-2.0).
