# Tests

## Running the gates

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j     # every check uses build/ by default
tests/prefill_check.sh MODEL_DIR                     # batched prefill == single steps (logits, state, KV, greedy ids)
build/dmeng --model MODEL_DIR --prompt-file bench/prompts/qwen3_5_moe/pitt.ids --n 64 --verify-check 32   # verify passes == single steps (exit 3 if not)
build/dm_gate --model MODEL_DIR --prompt-ids bench/prompts/qwen3_5_moe/pitt.ids --n 400 --out capi.ids   # then compare with dmeng --out-ids
build/manifest_dump MODEL_DIR --check --sha          # a downloaded folder against its moved-store.json (--profile mac: a one-profile download)
TORCH_PY=PY_TORCH MLX_PY=PY_MLX tests/tiny_all.sh   # tiny models of all three families, the same checks on each (below)
```

The scripts print a verdict line and exit 0 only when they pass; `dmeng --verify-check` reports in its JSON line and
exits 3 on a difference; `dm_gate` writes ids for a comparison. A real model folder is a weight repository downloaded
from the Hub (the README's quick start).

## One command for the tiny models: `tests/tiny_all.sh [OUT]`

Builds a tiny random model of each family with its generator here (`tests/<family>/make_tiny.py`: a synthetic release
in the real release's layout, run through the repo's converter, a random router set as the weight repo publishes it, and
the manifest written by `tests/tiny_manifest.py` from the family's published manifest in `python/convert/<family>/`, so
the tiny folder resolves its defaults as the real one does), derives the engine-ready files with `ms_derive`, lists them
with their sha256 in the manifest, and runs `tests/family_check.py` on each: the same checks for every family, read from
the engine itself (its JSON's `"capabilities"`) and the manifest, never from the family's name. It ends with a family x
check matrix; a check a family cannot run is "not supported for this family", and the engine's refusal is checked.
`TORCH_PY` is a python with torch, transformers, safetensors, numpy and tokenizers (the 35B's and Flash-Next's
generators and references), `MLX_PY` one with mlx and mlx-lm with the Inkling 2 model (mlx-lm PR 2086).

| check | what it compares |
|---|---|
| reference | exact decode against an independent reference: `tests/<family>/tiny_check.py` (transformers' model for the 35B and Flash-Next, the PR's MLX model for Inkling, each given exactly the engine's weights): the fp32 reference path within 1e-4 of max \|logit\| where the family has one, the speed path's agreement |
| invariance | exact mode: logits (300 positions) and greedy ids bit-identical with every expert held / few slots, 1 / 4 / 5 threads, router names, run-ahead, oracle replays; names + run-ahead with the names' reads held in flight (`DMENG_TEST_SPEC_READ_US`: a miss whose free slots are all still being read waits for one); the same with spare memory on (`--spare-mem 1`, probation budgets small and whole, an arm promoting every reuse), each run checked to have served reuses from the file cache |
| prefill | batched prefill == single steps (dmserve --check: logits, fixed state, KV, greedy continuation; CPU, sub-chunks, GPU share, few slots) and dmeng's ids batched == token by token |
| checkpoint | dmserve --check-ckpt (long / identical / diverging requests, each equal to a cold computation) and dmeng --state-check |
| server | the server worker's greedy and sampled ids equal dmeng's |
| drafting | --verify-check (verify passes with rollbacks == single steps), --draft fixed / lookup == plain decoding (greedy, sampled), the server worker the same; the C API's `dm_step_n` == `dm_step` + `dm_sample` with drafts off and on (greedy, sampled, penalties), tokens per step above 1 with fragments and on a regenerated reply, the draft counters reported, the iOS slab's reserve; a family without a verify pass: dmeng, dmserve and the C API must refuse `--draft`, `--drafts 1`, `--verify-check`, also a declared default |
| presets | the precedence (explicit > profile > manifest > built-in) for every declared profile in dmeng, the C API and dmserve, read back from each one's `"resolved"` JSON, and the three printing the same settings |
| spare | spare memory resolved per mode in dmeng, the C API and dmserve (the phone profile's rule: on from 11 GB of physical memory in exact mode, off in fast, by `DMENG_TEST_PHYS_MEM` 12 / 8 GB; the switch `--spare-mem` / `run:spare-mem` forcing it in either mode; no spare keys on mac and phone-small), and a mode change through the C API in one process (`dm_gate --arms rule/mode=exact,rule/mode=fast,...`: each `dm_open` resolves it for its mode, `dm_close` leaves none of its records in the file cache, the exact ids unchanged) |
| files | every engine-ready file in the cache folder is listed in the manifest with its sha256, the manifest declares its 8-bit rule (`resident.q8_rule`), and a copy of the in-memory file with one byte changed is refused at load |
| caps | per-layer caps of 2K + 4 slots, the routers naming K per step: `--layer-caps uniform:fit` refuses no name the layer can hold (the fixed reserve refuses most), ids identical with caps, fit caps and one pool |
| background | background reads (`--bg-ranks`) at 2K + 4 slots per layer without caps, with caps and with fit caps: exact ids identical to no background reads; the reads issued and used per token reported (a capped layer gives a background read a slot from its own records, under the same reserve as a name) |
| lock | `--mem mlock --cache-start-gb` under `ulimit -l` below the start: the lock the OS refuses stops the budget there, `"cache_lock"` reports the bytes locked (= the budget) and the refusals, ids identical to an unlimited run |
| cache | the automatic cache rule under one machine state (`DMENG_TEST_MEMSNAP`) in dmeng, dmserve and the C API: the start rule the manifest declares (`profiles.mac.cache_start`, else the built-in headroom rule), the printed ceiling and start recomputed from the printed inputs |
| counters | the stages sum to the token's time within 1%, compute = time - wait - run-ahead, and the server's MB per token equals dmeng's |

`tests/resolve_compare.py TINY_MODEL PUBLISHED_MANIFEST OUT --a BUILD_A --b BUILD_B`: two builds (e.g. before and after a
change) resolve a family's published manifest, the same manifest without `profiles`, and a folder without a manifest to
the same settings in dmeng, the C API (also as on an iPhone: `DMENG_TEST_AVAIL_OPEN`) and dmserve, for every profile and
mode; one line per difference.

`tests/fingerprint.py MODEL OUT.json [--against OLD.json]`: sha256 of the logits and ids of eleven fixed runs (every
setting explicit), to prove a change moves no bit (two builds, or before and after a refactor).

The binaries come from `build/` (configure and build the repo first); `DMENG_BIN` / `DMSERVE_BIN` point a check at
another build. The checks print a verdict line and exit 0 only when they pass; the two drivers write ids or decision
lines for a comparison made outside them.

## Checks

| check | what it compares | needs |
|---|---|---|
| `qwen3_5_moe/tiny_check.py` | dmeng on the tiny random Qwen3.6 against transformers' Qwen3_5MoeForCausalLM (float32, the engine's 8-bit codes, the store's experts): the fp32 reference path within 1e-4 of max \|logit\| where it exists; the speed path's agreement | `TINY` (make_tiny.py's OUT), numpy, torch, transformers |
| `qwen4_exp/tiny_check.py` | dmeng on the tiny random Qwen3.8-Flash-Next against transformers' Qwen4ExpForCausalLM (float32, the engine's 8-bit codes, the store's experts and n-gram rows): the reference path within 1e-4 of max \|logit\|; exact mode bit-identical across budgets, threads, routers (lag 4, from the npz), run-ahead and oracle replays; fast mode; batched prefill == single steps (CPU and GPU share); the prefix cache's checkpoints and a restored state (its verify pass and drafting: family_check.py's drafting check) | `TINY` (make_tiny.py's OUT), numpy, torch, transformers |
| `qwen4_exp/server_check.py` | the OpenAI-compatible server on the tiny Flash-Next: /v1/models, a completion with tools, the same streamed, a follow-up from the prefix cache | `TINY`, tokenizers, jinja2 |
| `qwen4_exp/chat_check.py FOLDER` | Flash-Next's chat format (python/serve/qwen4_exp) against transformers' apply_chat_template on the release's text files: token ids of conversations with tools, tool results and every reasoning effort; parsing a generated turn | the release's tokenizer and template files, transformers |
| `inkling2_loop/tiny_check.py` | dmeng on the tiny random Inkling against the MLX model of the release (the fp32 path within 1e-4 with the bf16 and the engine's 8-bit resident), exact across budgets and oracle replays | `TINY`, numpy, mlx, mlx-lm (PR 2086) |
| `prefill_check.sh MODEL_DIR` | a batched prefill leaves byte-identical logits, state and KV to single decode steps, and the same greedy continuation | a real model folder |
| `c_api/dm_gate.c` (driver, built as `dm_gate`) | ids through the C API only, greedy or sampled with the app's sampler, through `dm_step` or `dm_step_n` (drafting: `--draft`, `--skeleton`, `--repeat`; arms in one process: `--arms`), optional mid-decode budget change and cancel test, to compare with dmeng's | a real model folder |
| `evict_check/evict_check.cpp` (driver) | the cache's eviction decisions on a recorded routing, one line per eviction, to compare with the offline simulator | a recorded routing |
| `spare_tier/spare_tier_check.cpp` (built as `spare_tier_check`; `tiny_all.sh` runs it) | spare memory's mechanics on a small store of known bytes: a read lands in probation (the mapping's bytes, resident, no page-in when read; the page-in counter's control), a reuse inside a token does not promote and one at a later token does (the slab gets the file's bytes, the file pages go), a pinned record waits, the budget drops the oldest's pages, a reclaimed record is a miss (unpinned) or re-read in place (pinned), a promotion of reclaimed pages is dropped, names land in probation, a pause drains it, two pools, the tier off | nothing (a scratch folder) |
| `spare_tier/spare_tier_bench.cpp` (built as `spare_tier_bench`; a measurement, not a gate) | spare memory's cost to the compute on a synthetic store of real-size records: a Core-like step (slot_for_use, misses, the next step's names read while this one computes, tier_step), a bandwidth-bound pass over every byte on 4 threads; compute ms, the compute's faults and page-ins (the readers' own taken out), spare memory off and on | a scratch folder (~0.6 GB) |

## Optional: the real models against an independent MLX runner

`qwen3_5_moe/compare_mlx.py` and `inkling2_loop/compare_mlx.py` teacher-force the real model through dmeng and through a
separate Python/MLX runner (an `mlx_run.py` with an exact mode, not part of this repo) over the same 60 tokens and compare
the logits position by position. They load the full model twice, so they are not part of a routine run.

```bash
python tests/inkling2_loop/compare_mlx.py --model MODEL --mlx-run PATH/mlx_run.py --mlx-python PATH/python
python tests/qwen3_5_moe/compare_mlx.py --model MODEL --mlx-run PATH/mlx_run.py --mlx-python PATH/python
```

Steps can run one at a time (`prepare`, `mlx`, `compare`); `WRAP` prefixes every process that loads the model (a guard
that waits for a quiet machine, for example). Without the model folder or the MLX runner they print `skipped: ...` and
exit 0.
