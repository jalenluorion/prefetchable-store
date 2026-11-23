# Advanced options

Everyday use needs only `--model`, a prompt, and perhaps `--mode` or `--profile` (see the README). Everything else is
here, grouped; the full list with every value is printed by `dmeng --help-advanced` and `dmserve --help-advanced`
(the C++ worker), and `python python/serve/dmserve.py --help-advanced` (the HTTP front end).

## How defaults and flags combine

The profile (`--profile auto`, the default: chosen from the device) fills every option the command line leaves unset,
from the manifest's `profiles` / `defaults` keys with built-in fallbacks ([MANIFEST-PRESETS.md](MANIFEST-PRESETS.md)).
A flag you give overrides that one default. The early names are treated as one group: giving any of `--prefetch`,
`--routers`, `--xtok`, `--oracle` or `--plan-routers` (dmeng), or `--routers` / `--xtok` (dmserve), makes them entirely
the command line's. `--routers none` / `--xtok none` turn them off in the server.

`--profile none` turns the profile off and gives the engine's own values: no names, a 5 GB cache, single-step prompts
in `dmeng`, no disk prefix and no hot set in the server. Scripts written before profiles existed, which relied on
leaving a flag out to mean "off", should add `--profile none`. `--print-defaults` prints what a folder and a profile
resolve to, as JSON, without loading the model.

## Groups

- **Profiles and modes**: `--profile auto|mac|phone|phone-small|none`, `--mode exact|fast` (also the internal names
  `lossless` and `approx[:PRESET]`, with the measurement presets `bounded`, `drop` and `never`), `--drafts 0|1`,
  `--print-defaults`.
- **Model and memory**: `--resident F` (another in-memory weights file, such as the 4-bit `resident_q4`),
  `--embeddings F`, `--cache-gb G|auto`, `--reserve-gb G` (memory left to the system by `auto`, default 5.5),
  `--mem plain|mlock|purgeable|file` and its `--file-*` options, `--res-mem`, `--threads`, `--readers`, `--split`,
  `--prec`, `--ctx`, `--pool-share`, `--half-life`.
- **Early names** (they only schedule reads; in `exact` the tokens do not depend on them): `--prefetch
  none|router|oracle|transplant|plan`, `--routers F --m M`, the confidence rule `--rt-tau T --rt-cand C`,
  cross-token names `--xtok F --xtok-tau T` and `--xtok-bg F`, the read planner `--plan-routers ...`, `--router-stats`,
  `--router-ecache`, `--router-heavy`.
- **Reads and eviction**: `--evict auto|hazard|freq|pred`, `--layer-caps`, `--evict-policy F`, `--rd-prio`,
  `--rd-excl`, `--spec-max`, `--cancel`; the approximate rule set by hand: `--late`, `--read-w`, `--read-mass`,
  `--bg`, `--drop`.
- **Prompt path**: `--prefill-chunk N`, `--prefill-gpu-frac F` (dmeng); `--chunk`, `--span`, `--stream`,
  `--gpu-frac`, `--keep-scratch`, the prefix cache (`--ckpt-gb`, `--park-gb`, `--ckpt-every`, `--disk DIR|none`,
  `--disk-gb`, `--disk-every`, `--preload`), idle-time prefill (`--ahead-span`, `--ahead-park`; the front end's
  `--ahead 0|1`), the saved hot set (`--hotset F|none`), the memory watcher (`--mem-watch`, `--mem-low-pct`,
  `--mem-high-pct`, `--mem-grow-after-s`, `--mem-retry-s`, `--cache-min-gb`, `--headroom-gb`) (dmserve).
- **Draft and verify** (greedy only; the tokens are plain decoding's): `--draft lookup|fixed|head|both`,
  `--draft-max`, `--draft-thr`, `--head-backoff`, `--early`, `--verify-names`, `--skeleton`, `--probe-every`.
- **Outputs and checks**: `--json`, `--out-ids`, `--steps-out`, `--record`, `--dump-names`, teacher forcing
  (`--teacher`, `--dump-logits`, `--nll-out`, `--ref-nll`, `--qual-out`, `--ref-qual`, `--taps-out`), `--state-check`,
  `--verify-check`, `--draft-check`, `--skip-reads`, `--force-routing`; the worker's `--check`, `--prefill-bench`,
  `--check-ahead`.
- **Server front end**: `--model-id`, `--worker`, `--log`, `--log-ids`, `--log-bodies`, `--max-new-cap`, `--seed`,
  `--side-grace`, `--idle-s`, `--allow-force`, `--format`; everything after `--` goes to the worker.

## C API

`dm_open_opts` (bindings/c/dmeng_c.h) has the same `profile` and `mode` fields; every option left at zero or NULL
takes the profile's default. `profile = "none"` gives the engine's own values.

## Checking a download

`manifest_dump DIR --check --profile NAME [--sha]` checks that the parts a profile lists are present with their sizes
(and sha256), for a folder that holds only one profile's files.
