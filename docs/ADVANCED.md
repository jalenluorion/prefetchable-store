# Advanced options

Everyday use needs only `--model`, a prompt, and perhaps `--mode` or `--profile` (see the README). Everything else is
here, grouped; the full list with every value is printed by `dmeng --help-advanced` and `dmserve --help-advanced`
(the C++ worker), and `python3 python/serve/dmserve.py --help-advanced` (the HTTP front end).

## How defaults and flags combine

The profile (`--profile auto`, the default: chosen from the device) fills every option the command line leaves unset,
from the manifest's `profiles` / `defaults` keys with built-in fallbacks ([MANIFEST-PRESETS.md](MANIFEST-PRESETS.md)).
A flag you give overrides that one default. The early names are treated as one group: giving any of `--prefetch`,
`--routers`, `--xtok`, `--oracle` or `--plan-routers` (dmeng), or `--routers` / `--xtok` (dmserve), makes them entirely
the command line's. `--routers none` / `--xtok none` turn them off in the server.

`--profile none` turns the profile off and gives the engine's own values: no names, a 5 GB cache, single-step prompts
in `dmeng`, no disk prefix and no hot set in the server. A script that relies on leaving a flag out to mean "off"
should pass `--profile none`. `--print-defaults` prints what a folder and a profile resolve to, as JSON, without
loading the model.

## Groups

- **Profiles and modes**: `--profile auto|mac|phone|phone-small|none`, `--mode exact|fast` (also the internal names
  `lossless` and `approx:drop[:w=W]`, fast mode's leave-out rule with its weight), `--drafts 0|1`,
  `--print-defaults`.
- **Model and memory**: `--resident F` (another in-memory weights file, such as the 4-bit `resident_q4`),
  `--embeddings F`, `--cache-gb G|auto`, `--reserve-gb G` (memory left to the system by `auto`, default 5.5),
  `--mem plain|mlock|purgeable|file` and its `--file-*` options, `--spare-mem 0|1` (spare memory, the OS file cache
  under the locked slab: the built-in `phone` profile turns it on from 11 GB of physical memory, in both modes),
  `--res-mem`, `--threads`, `--readers`, `--split`, `--prec`, `--ctx`, `--pool-share`, `--half-life`.
- **Early names** (they only schedule reads; in `exact` the tokens do not depend on them): `--prefetch
  none|router|oracle|plan`, `--routers F --m M`, the confidence rule `--rt-tau T --rt-cand C`,
  cross-token names `--xtok F --xtok-tau T` and `--xtok-bg F`, the read planner `--plan-routers ...`, `--router-stats`,
  `--router-ecache`, `--router-heavy`.
- **Reads and eviction**: `--evict auto|hazard|freq|pred`, `--layer-caps`, `--evict-policy F`, `--rd-prio`,
  `--rd-excl`, `--spec-max`, `--cancel`, `--name-gate 0|1` (no new named read while urgent reads wait for a reader;
  default 0).
- **Prompt path**: in dmeng `--prefill-chunk N`, `--prefill-gpu-frac F`; in dmserve `--chunk`, `--span`, `--stream`,
  `--gpu-frac`, `--keep-scratch`, the prefix cache (`--ckpt-gb`, `--park-gb`, `--ckpt-every`, `--disk DIR|none`,
  `--disk-gb`, `--disk-every`, `--preload`), idle-time prefill (`--ahead-span`, `--ahead-park`; the front end's
  `--ahead 0|1`), the saved hot set (`--hotset F|none`) and the memory watcher (`--mem-watch`, `--mem-low-pct`,
  `--mem-high-pct`, `--mem-grow-after-s`, `--mem-retry-s`, `--cache-min-gb`, `--headroom-gb`).
  A request's marks (dmserve) or dm_sync's (the C API) are the positions where its messages end. The first one at or
  past `disk_min` is taken as the system prompt's boundary, and its checkpoint is pinned in memory (never evicted)
  and, with a disk folder, written to disk. One checkpoint is pinned at a time: a request whose first mark moves (a
  typing position, say) pins its own and releases the previous one, which then counts against `--ckpt-gb` like any
  other. The session's statistics (dmserve's request log `session`, dm_stats_json's `prefix_cache`) report
  `pinned_checkpoints`, `pinned_mb`, `pins_released`, `pins_over_budget` and `checkpoint_mb` against
  `checkpoint_budget_mb`.
- **Draft and verify** (greedy and sampled; the tokens are plain decoding's; see Drafting below): `--draft lookup|fixed|head|both`,
  `--draft-max`, `--draft-thr`, `--head-backoff`, `--early`, `--verify-names`, `--skeleton`, `--probe-every`,
  `--copy-free`, `--draft-cost`.
- **Outputs and checks**: `--json`, `--out-ids`, `--steps-out`, `--record`, `--dump-names`, teacher forcing
  (`--teacher`, `--dump-logits`, `--nll-out`, `--ref-nll`, `--qual-out`, `--ref-qual`, `--taps-out`), `--state-check`,
  `--verify-check`, `--draft-check`, `--skip-reads`, `--force-routing`; the worker's `--check`, `--prefill-bench`,
  `--check-ahead`.
- **Server front end**: `--model-id`, `--worker`, `--log`, `--log-ids`, `--log-bodies`, `--max-new-cap`, `--seed`,
  `--side-grace`, `--idle-s`, `--allow-force`, `--format`; everything after `--` goes to the worker.

## Drafting

Draft and verify: a cheap source proposes the next tokens, one batched pass of the model verifies them, and the kept
tokens are exactly the ones plain decoding produces (greedy: the same ids; sampled: the same tokens for the same seed,
each token drawn with its own draw). The three front ends share it: one controller (`include/dmeng/draft.h`: the
sources, the choice of how many drafts a pass carries from measured pass costs, the probes, the statistics), one
resolved setting, one set of counters.

- **Sources.** `lookup`: copy drafts, the longest earlier match of the sequence's last 2 to 16 tokens in the prompt, the
  reply so far and the session's earlier replies, plus given fragments (a request's tool-call format, a file being
  rewritten). `head`: the model's own draft head (Inkling 2 Mini's `mtp.safetensors`). `both`: the two. `fixed` runs the whole
  lookup draft every step (a test of the verify pass, not a policy).
- **Resolved once** for dmeng, dmserve and the C API: an explicit value (`--draft MODE`, `--drafts 0|1`; the C API's
  model options `run:draft=MODE`, `run:drafts=0|1`) > the profile's declared `drafts` {`lookup`, `head`} > off. The pass
  width: `--draft-max N` (`run:draft-max=N`) > the profile's `drafts.max` > 7 drafts (8 positions, the verify pass's
  limit). A family without a multi-position verify pass, or a draft head the folder does not have, fails at open in all
  three, a declared default too. `resolved.draft` and, while drafting is on, `resolved.draft_max` report it.
- **dmeng**: `--draft`, `--draft-max`, `--skeleton F` (fragments), `--early`, `--draft-cost`, the head's `--draft-thr`
  and `--head-backoff`. **dmserve**: the same flags; a request's fragments come with it (the `R` line's `n_skel`
  lines); pass costs in `DIR/.dmeng/draft_cost.txt` under a profile. **The C API** (`bindings/c/dmeng_c.h`):
  `dm_step_n(e, id, sampler, history, n_history, stop, n_stop, max_out, out, &n_out, logits)` feeds `id` and returns
  1 to 8 tokens, each the token `dm_sample` gives over the caller's history at that place (without drafting it is
  `dm_step` then `dm_sample`); `dm_set_draft_fragments` gives the fragments; `run:draft-cost=FILE|none` (default under
  a profile: the profile's cache folder, as dmserve). `dm_step` stays plain decoding.
- **Memory.** A verify pass keeps buffers for 8 positions (the 35B: a second copy of the recurrent state for the
  rollback, and each position's logits; Inkling 2 Mini less; Qwen3.8-Flash-Next has no verify pass), allocated at the first pass and not shrunk by
  a smaller `max`; the head adds its weights and state (Inkling 2 Mini: 0.64 GB of weights, results/ink_draft/DRAFT-HEAD.md). With
  drafting off none of it is allocated. On iOS the C API's slab rule gives drafting's memory out of the expert slab
  (a second fixed state and 8 positions' logits, plus the head's file: `policy.slab_rule.draft_reserve_gb`), so the
  footprint the rule aims at holds; `draft.memory` in `dm_stats_json` reports the first pass's footprint rise and the
  head's memory, to check the reserve on the device.
- **Statistics**: `draft` in dmeng's JSON, in dmserve's `E` line and in `dm_stats_json`: steps, tokens, tokens per
  step, verifies, drafted, accepted, offered, declined, probes, by source, the cost model's widths; the C API adds
  `calls` and `tokens_per_call` (dm_step_n) and `memory`.

## C API

`dm_open_opts` (bindings/c/dmeng_c.h) has the same `profile` and `mode` fields; every option left at zero or NULL
takes the profile's default. `profile = "none"` gives the engine's own values. Run settings the struct has no field for
go through `model_options` as `run:FLAG=VALUE` (drafting: `run:draft=lookup`, `run:draft-max=N`; see Drafting above).

The output head: `dm_open` builds the screened head (exact; dmserve builds the same one) unless the model cannot split
its head (a mapped resident, a head that is not 8-bit: `resolved.screened_head` false, `policy.screened_head_why`).
`dm_step_n` called without a logits buffer then computes only the output rows its sampler can draw, where that is
provable from `dm_sample`'s order: greedy (temperature 0, no penalties: the argmax) and top-k sampling with
`1 <= top_k <= 64` and no penalties (the top k logits bit for bit; top-p, min-p and temperature read only those). Any
penalty, `top_k` 0 or above 64, or a logits buffer passed take the full head; `dm_sync` and `dm_step` always compute
every logit. The tokens are the full head's in every case (tests/family_check.py, check `head`). `policy.head_calls`
counts the decode calls by head. For measurement, `run:fast-head=0` (or `DMENG_CAPI_FAST_HEAD=0` in the environment)
leaves it off, and `DMENG_CAPI_PRIO=presets` replaces the C API's priority-read rule with dmeng's and dmserve's
(`priority_reads_on`; `policy.priority_rule`); neither changes a token in exact mode.

## Checking a download

`manifest_dump DIR --check --profile NAME [--sha]` checks that the parts a profile lists are present with their sizes
(and sha256), for a folder that holds only one profile's files.
