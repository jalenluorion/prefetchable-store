# Tests

## Running the gates

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j     # every check uses build/ by default
tests/prefill_check.sh MODEL_DIR                     # batched prefill == single steps (logits, state, KV, greedy ids)
build/dmeng --model MODEL_DIR --prompt-file bench/prompts/qwen3_5_moe/pitt.ids --n 64 --verify-check 32   # verify passes == single steps (exit 3 if not)
build/dm_gate --model MODEL_DIR --prompt-ids bench/prompts/qwen3_5_moe/pitt.ids --n 400 --out capi.ids   # then compare with dmeng --out-ids
build/manifest_dump MODEL_DIR --check --sha          # a downloaded folder against its moved-store.json (--profile mac: a one-profile download)
tests/draft_check.sh Q36_TINY INK_TINY INK_RES0 INK_RES8      # needs the tiny models (below)
```

The scripts print a verdict line and exit 0 only when they pass; `dmeng --verify-check` reports in its JSON line and
exits 3 on a difference; `dm_gate` writes ids for a comparison. A real model folder is a weight repository downloaded
from the Hub (the README's quick start). The tiny random models the `tiny_check.py` and `draft_check.sh` checks use are
made by generator scripts that are not in this repository yet; those checks need them passed in.

The binaries come from `build/` (configure and build the repo first); `DMENG_BIN` / `DMSERVE_BIN` point a check at
another build. The checks print a verdict line and exit 0 only when they pass; the two drivers write ids or decision
lines for a comparison made outside them.

## Checks

| check | what it compares | needs |
|---|---|---|
| `qwen3_5_moe/tiny_check.py` | dmeng on a tiny random Qwen3.6-geometry model against transformers / MLX references, every precision and several expert budgets | `Q36_TINY` (the tiny model and its references), numpy, torch |
| `inkling2_loop/tiny_check.py` | dmeng on the tiny random Inkling against the MLX model of the release, exact across budgets and oracle replays | `INK_TINY`, `INK_RES0`, `INK_RES8`, numpy, mlx |
| `draft_check.sh Q36_TINY INK_TINY INK_RES0 INK_RES8` | draft and verify: kept logits and the final state bit-identical through verify passes with rollbacks (both tiny models, several precisions, split-K attention); ids with drafting on equal plain decoding in dmeng and in the server worker | the tiny models |
| `prefill_check.sh MODEL_DIR` | a batched prefill leaves byte-identical logits, state and KV to single decode steps, and the same greedy continuation | a real model folder |
| `c_api/dm_gate.c` (driver, built as `dm_gate`) | greedy ids through the C API only (optional mid-decode budget change and cancel test), to compare with dmeng's | a real model folder |
| `evict_check/evict_check.cpp` (driver) | the cache's eviction decisions on a recorded routing, one line per eviction, to compare with the offline simulator | a recorded routing |

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
