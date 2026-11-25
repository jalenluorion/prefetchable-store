# Manifest keys for modes, profiles and defaults

`moved-store.json` carries three keys that describe the user-facing surface: `modes`, `profiles` and `defaults`.
The engine (`dmeng`, `dmserve`, the C API) reads them when they are present and falls back to built-in values when
they are absent. The built-in values are one table for every model (`src/core/presets.cpp` BUILTIN): what differs
between models is declared in each one's manifest, never chosen in the engine by the model's name. What the table keeps
is per profile, a property of the device class: `mac`: the prefill's GPU share 0.4 / 0.4, run-ahead at depth 1;
`phone`, `phone-small`: 8 readers, run-ahead at depth 1 limited to 0.2 of the token's compute, off while the readers are
saturated, in exact mode only; the run-ahead names the manifest's `experts.top_k` picks unless a profile gives
`picks`. A profile must be declared in the manifest (`profiles.<name>`), and a folder without a manifest has no profile
(`--profile none`). Unknown keys are ignored, so a manifest may carry more than this file lists (notes, download
lists). The weight repos' manifests as this repository writes them are `python/convert/<family>/moved-store.json`.

Also declared per model: `resident.q8_rule` (required to derive the in-memory file: MLX's affine quantiser, 8 bits,
groups of 64, the release's weights cast to `f16` or `bf16` first: the engine takes the cast from here), and
`drafting.cost_prior` (the drafting controller's prior, the cost of a pass of 0..8 positions over one step, used when
it states where it comes from, `measured` or `estimated`; else the engine's neutral prior, a pass of n positions costing
n steps; either is replaced by the pass costs the engine has measured itself, `DIR/.dmeng/draft_cost.txt`).

## `modes`

```json
"modes": {
  "exact": {"definition": "Generated tokens never depend on the cache, the routers or timing.", "engine": "lossless"},
  "fast":  {"definition": "The model's router still picks; a picked expert that is not in memory is waited for if its routing weight is over 0.1 or its read is already in flight, otherwise it is left out.", "engine": "approx:drop"}
}
```

The engine reads only `engine` (the internal mode string: `lossless` or `approx:PRESET[:k=v...]`). `definition` is for
people. Built-in: `exact` = `lossless`, `fast` = `approx:drop` (Flash-Next: `approx:drop:w=0.13:mass=0.13:late=-1`).
The optional `:k=v` pairs set the preset's rule: `w` (a missing pick heavier than this is read), `mass` (then the
heaviest missing picks are read until those left weigh at most this; left out per step <= `mass`), `late` (ms a light
pick in flight is waited for; < 0 never, >= 1000 always) and `bg` (0 | 1: background fetch of left-out picks).

## `profiles`

```json
"profiles": {
  "mac": {
    "when": "macOS",
    "mode": "exact",
    "resident": "resident_q8",
    "precision": "8-bit",
    "cache": "auto",
    "parts": [{"path": "store/layer_00.bin", "bytes": 123, "sha256": "..."}],
    "bytes": 21309261277
  },
  "phone": {
    "when": "iOS with at least 5.0 GB available at open",
    "min_available_gb": 5.0,
    "mode": "exact", "resident": "resident_q8", "precision": "8-bit", "cache": "slab",
    "parts": [], "bytes": 0
  },
  "phone-small": {
    "when": "iOS with less than 5.0 GB available at open",
    "mode": "fast", "resident": "resident_q4", "precision": "4-bit", "cache": "slab",
    "names": {"rule": "confidence", "tau": 0.5, "candidates": 16},
    "parts": [], "bytes": 0
  }
}
```

Read by the engine, per profile:

| key | type | meaning | built-in |
|---|---|---|---|
| `mode` | `"exact"` or `"fast"` | the profile's default mode | mac `exact`, phone `exact`, phone-small `fast` |
| `resident` | derived id | the in-memory weights: `resident_q8` or `resident_q4` | `resident_q8`; phone-small on the 35B: `resident_q4` when that file is present, else `resident_q8` |
| `cache` | `"auto"` or `"slab"` | expert cache sizing: `auto` = from free memory at start, then follow memory pressure (Mac); `slab` = one locked slab sized at open from the memory the app has (iOS) | mac `auto`, phones `slab` |
| `prefill_gpu` | `{"projections": F, "experts": F}` | the batched prefill's exact CPU + GPU split (Metal builds): the shares of the attention projections' and the experts' rows computed on the GPU, the same bits as the CPU (results/ios/p3-gpu/GPU-PREFILL.md); `dmeng --prefill-gpu off` / `dmserve --prefill-gpu off` / a negative C API share turn it off | the 35B on `mac`: 0.4 / 0.4 (the recorded 7,572-token OpenCode prompt 80.0 -> 67.4 s); Inkling on `mac`: projections 0.4, experts 0.4 (its NVFP4 experts on the GPU, layers 0-19; the MXFP8 layers stay on the CPU; the recorded 8,365-token OpenCode request, applications alternated in one process: 21.5 s per 1,000 positions off, 20.5 projections only, 17.5 with the experts' share); every other profile: CPU only |
| `run_ahead` | `{"picks": N, "depth": D, "budget": X, "max_tok": M, "sat_skip": B}` | while a step waits on reads, run ahead `D` layers and name the `N` experts that layer would pick; limits: its speculative ms per token at most `X` x the token's compute, at most `M` run-aheads per token, none while every reader is busy (`B`; built-in: no limits, no skip); an optional `"modes": ["exact"]` keeps it on only in those modes (exact: names only, tokens unchanged; results/parity/PARITY.md); `--run-ahead 0` / a negative C API `run_ahead` turn it off | `mac`, both models: 8 at depth 1 (one process, alternating arms: 35B prose +11.6%, code +2.5%; Inkling prose +6.4%, code -0.3%); `phone` and `phone-small`, both models: 8 at depth 1, budget 0.2, sat_skip on, in exact mode only (off in fast; same-load device measurements) |
| `readers` | number | reader threads | `phone` and `phone-small`: 8; `mac`: the front end's default (4) |
| `min_available_gb` | number, on `phone` only | the automatic choice on iOS: `phone` at or above it, `phone-small` below | 5.0 |
| any key of `defaults` | as below | overrides that default for this profile only | the 35B on `phone-small`: `names` = the confidence rule (tau 0.5, 16 candidates) |

Not read by the engine (for the download tool and people): `when`, `precision`, `parts` (each part: `path`,
`bytes`, `sha256`), `bytes` (the total to download), `note`, `files`. Profiles other than these three (for example
`minimal`) are allowed and ignored by the automatic choice; `--profile NAME` can still select one if it carries the
keys above.

## `defaults`

```json
"defaults": {
  "routers": ["linear_k2"],
  "names": {"rule": "top", "m": 8},
  "cross_token": null,
  "eviction": "auto",
  "drafts": false
}
```

Inkling 2 Mini:

```json
"defaults": {
  "routers": ["linear_k2"],
  "names": {"rule": "confidence", "tau": 0.5, "candidates": 16},
  "cross_token": {"routers": ["xtok_s47_h512"], "source": 47, "tau": 0.5, "priority_reads": true},
  "eviction": "auto",
  "drafts": false
}
```

| key | type | meaning |
|---|---|---|
| `routers` | list of ids from the manifest's `routers` | the early routers that name experts; the first id that resolves to a file is used; `[]` = none (cache only) |
| `names` | object | how many experts the routers name per step: `{"rule": "top", "m": 8}` (a fixed top m) or `{"rule": "confidence", "tau": 0.5, "candidates": 16}` (the top candidates whose calibrated probability is at least tau) |
| `cross_token` | object or `null` | routers that name the next token's first steps from a late step of this one: `routers` (ids from `cross_token_routers`), `source` (the step they read, informational; the file carries it), `tau` (threshold), `priority_reads` (speculative reads by priority instead of first in, first out) |
| `eviction` | string | `auto`, `hazard` or `freq` |
| `layer_caps` | string | per-layer slot caps: `auto` (the engine's rule), `uniform` or `none` (one pool), optionally with `:fit` (in a layer at its cap a name leaves min(2K + the loop holds, cap - names per step - the layer's own holds) slots free instead of 2K + the holds, so a layer whose cap holds its names admits them; the same wherever the cap is at least 2K + holds + names + the layer's holds); absent = `auto` |
| `drafts` | bool | draft and verify on by default (`false` in v1; `--drafts 1` turns it on) |

Inkling's `linear_k2` (one linear layer, lag 2; engine file 49 MB, source `routers/linear_k2.safetensors`) replaced the
MLP `mlp_k2_h512` as the default: in one process with alternating segments it was 2-4% faster beside run-ahead and
reads about a fifth of the router bytes per token (44 against 222 MB). The MLP stays available by explicit option
(`--routers FILE`, `routers_id = "mlp_k2_h512"`).

Qwen3.8-Flash-Next (`qwen4_exp`; one profile, `mac`: exact, 8-bit in-memory weights, cache auto, names a top 4, the
GPU's share of the prefill 0.4 / 0.4, run-ahead naming 10 at depth 1):

```json
"defaults": {
  "routers": ["mlp_k4"],
  "names": {"rule": "top", "m": 4},
  "cross_token": null,
  "eviction": "auto",
  "layer_caps": "none",
  "drafts": false
}
```

Measured on the base M4 (16 GB; 120 tokens of the reference prompt, counters from one process each, timing from
alternating runs) at 2.0 GB (15 records per layer) and 3.1 GB (24):

- Fast mode: with 10 picks the routing weights are about 0.1 each, so the 35B's rule (wait for a pick over 0.1, leave
  out the rest) left out 0.32 of each step's weight. The rule here reads the heaviest missing picks until the rest
  weighs at most 0.13: 0.087 of the weight is left out on average (0.09 at the 90th percentile of tokens), at 465 MB
  read per token at 3.1 GB and 604 MB at 2.0 GB. A cap of 0.10 left out 0.066 at 650 MB (2.0 GB), 0.20 left out 0.15 at
  486 MB, 0.25 left out 0.19 at 419 MB.
- Names: a top 4 against a top 6, 10 and the confidence rule (tau 0.5 of 16): in fast mode 4 and 6 run at the same
  speed in alternating runs and 4 reads 6-9% less; 10 and the confidence rule read 785-918 MB at 2.0 GB, a third of
  their names wasted. In exact mode a top 4 in one pool ran 8% faster than a top 10 with per-layer caps in two
  alternating pairs (hit rate 0.58 against 0.49).
- One pool: below 2 x 10 records per layer the per-layer caps of `auto` refuse almost every name (a name must leave 20
  unpinned records in its layer: 241 of 352 names per token refused at 2.0 GB), so the cache runs on misses alone.

Its router entry `mlp_k4` points at `routers_k4.npz`, the lag-4 set as the trainer exported it (float32): the engine reads
the npz in place and derives its 8-bit file from it (step 4 below), the same bytes as the Python converter's.

The built-in `defaults`, used for any key the manifest leaves out, are the same for every model: no routers, a top 8,
no cross-token routers, eviction `auto`, no drafting. The names rule is a per-profile value each manifest declares:
the 35B names a top 8 on `mac` and `phone` and uses the confidence rule on `phone-small`; Inkling uses the confidence
rule on every profile.

## How an id becomes a file

For a router id `X` (or a cross-token id), the engine looks, in order, for:

1. the `path` of the manifest's `derived` entry `router.X.q8`, relative to the model folder (a folder downloaded from
   the Hub: `derived/router.X.q8.dmr1.<hash>.bin`);
2. `derived/router.X.q8.dmr1.<hash>.bin` in the model folder (the same name without a `path` key);
3. the same name as in 2 in the engine's own cache folder, `DIR/.dmeng/`;
4. otherwise it derives the file from `routers/X.safetensors` (or the router entry's `path`: an `.npz` is read in place)
   into `DIR/.dmeng/` (routers only; cross-token files are not derived by the engine, and the default is skipped with a
   note if none is found).

`<hash>` is the first 12 hex digits of the sha256 of the source part (`from[0]`). Every engine-ready file found this
way (and `embeddings.safetensors`, under `derived` or `parts`) is checked against the manifest's `bytes` and `sha256`
before it is used, once per version of the file (the result is kept in `DIR/.dmeng/checked/`); a router that does not
match is left out with a note, an in-memory file that does not match stops the run. The in-memory weights follow
steps 1 to 3 the same way; `resident_q8` is derived from `resident.safetensors` when missing, `resident_q4` is not
(`ms_derive resident DIR --bits 4` makes it).

A default that does not resolve (no file and nothing to derive it from) is dropped with one line on stderr; the run
continues without it. Tokens in `exact` do not depend on any of these files.
