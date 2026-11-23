# Manifest keys for modes, profiles and defaults

`moved-store.json` carries three keys that describe the user-facing surface: `modes`, `profiles` and `defaults`.
The engine (`dmeng`, `dmserve`, the C API) reads them when they are present and falls back to built-in values equal
to the tables below when they are absent. Unknown keys are ignored, so a manifest may carry more than this file
lists (notes, download lists).

## `modes`

```json
"modes": {
  "exact": {"definition": "Generated tokens never depend on the cache, the routers or timing.", "engine": "lossless"},
  "fast":  {"definition": "The model's router still picks; a picked expert that is not in memory is waited for if its routing weight is over 0.1 or its read is already in flight, otherwise it is left out.", "engine": "approx:drop"}
}
```

The engine reads only `engine` (the internal mode string: `lossless` or `approx:PRESET`). `definition` is for
people. Built-in: `exact` = `lossless`, `fast` = `approx:drop`.

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
  "routers": ["mlp_k2_h512"],
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
| `drafts` | bool | draft and verify on by default (`false` in v1; `--drafts 1` turns it on) |

Built-in values, used when the manifest has no `defaults` (and for any key it leaves out), are the two objects above
by family (`qwen3_5_moe`, `inkling2_loop`). The names rule is a per-profile default: the 35B names a top 8 on `mac`
and `phone` and uses the confidence rule on `phone-small` (its built-in profile carries `names`); Inkling uses the
confidence rule on every profile.

## How an id becomes a file

For a router id `X` (or a cross-token id), the engine looks, in order, for:

1. the `path` of the manifest's `derived` entry `router.X.q8`, relative to the model folder (a folder downloaded from
   the Hub: `derived/router.X.q8.dmr1.<hash>.bin`);
2. `derived/router.X.q8.dmr1.<hash>.bin` in the model folder (the same name without a `path` key);
3. a few built-in locations of research folders (for example `routers/lin/q36_router_lin_k2.dmr`), which hold the
   files the defaults were measured with;
4. the same name as in 2 in the engine's own cache folder, `DIR/.dmeng/`;
5. otherwise it derives the file from `routers/X.safetensors` into `DIR/.dmeng/` (routers only; cross-token files are
   not derived by the engine, and the default is skipped with a note if none is found).

`<hash>` is the first 12 hex digits of the sha256 of the source part (`from[0]`). The in-memory weights follow
steps 1, 2 and 4 the same way; `resident_q8` is derived from `resident.safetensors` when missing, `resident_q4` is not
(`ms_derive resident DIR --bits 4` makes it).

A default that does not resolve (no file and nothing to derive it from) is dropped with one line on stderr; the run
continues without it. Tokens in `exact` do not depend on any of these files.
