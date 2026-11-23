# port-ios: changes to shared files (for the merge into unified-wip)

Branch `port-ios`, rebased on unified-wip 1764761 (2026-11-22). Every change below keeps the defaults of unified-wip:
nothing changes for a caller that does not set the new fields, and lossless token ids do not depend on any of them.
Eviction code is untouched except for the slot range its loops scan (`S.n` -> `S.cap`, equal unless a run-time budget
lowered it), so the eviction-policy work in `moved-store-port-ios-evict` merges with at most that one-word conflict.

## Headers

| file | change | default |
|---|---|---|
| `include/dmeng/weights.h` | `Mem::FILE = 3` (expert slots only); `mem_of("file")` | unchanged modes |
| `include/dmeng/store.h` | `Readers::Job::view` (FILE: the piece's address in the mapping; `dst = nullptr` reads into the reader's scratch); `Readers::file_skip_resident`, `file_prefault`, `qos`, counters `file_resident_skips`, `file_bytes_skipped` | skip 1, prefault 1, qos -1 = user-initiated (as before) |
| | `SlotPool::cap`, `min_cap` (slots in use; a lowered run-time budget retires the slots above `cap`) | `cap = n` |
| | `CacheStats::file_checks`, `file_refaults`, `bytes_refault` | 0 |
| | `Cache::fmap`, `fmap_bytes`, `file_check`, `file_evict_hint`, `file_hit_check()`, `file_resident_bytes()`, `held_bytes()` | check 1 (first + last page), hint off |
| | `Cache::budget_req`, `set_budget()`, `apply_budget()`, `share_`, `budget_bytes()`; private `slot_memory()`, `file_compact()` (FILE shrink: the policy's lowest-value entries go first, survivors packed below the new cap) | no request |
| | `Cache::FileAdapt fa` (adaptive FILE budget: rotating resident sample, refault EMA, shrink / grow rule, min / max, trace) and `token_end_file()` | `fa.on = false` |
| | locked core: `core_slab`, `core_n`, `core_tab`, `core_of`, `core_every`, `core_swaps`, `core_init()`, `core_held()`, private `core_refresh()`; `rec()` serves a core record from the slab | no core |
| | `Cache::rec()`: FILE returns `fmap[L] + e * rec` (the view), else the slot as before | |
| `include/dmeng/core.h` | `CoreOpts::qos`, `reader_qos` (qos_class_t values) | -1 = user-interactive workers / user-initiated readers, as before |
| `include/dmeng/pool.h` | `Pool(int nthreads, int spin_us = 2000, int qos = -1)`; member `qos_` | -1 = user-interactive |
| `include/dmeng/lm.h` | `using PrefillProgress = std::function<void(int done, int total, double mb_read)>`; `PrefillStats::progress` (input); `LM::on_progress` (public member, ModelLM fills in the MB read); cancel's comment: within one layer for a layer-major model | empty |
| `include/dmeng/model.h` | `resolve_model(..., const std::string& derived_dir = "", const std::string& emb_file = "")`: derived dir; embedding rows from a standalone file (relative to the model dir or absolute); with none given and the manifest's embedding file absent, `<dir>/embeddings.safetensors` is used if present | <dir>/.dmeng and the manifest's file as before (the fallback only applies when that file is missing) |
| `include/dmeng/address.h` | `Routers::mem` (how the router file is held; set before `load`) | `Mem::MLOCK`, as before |
| `src/format/manifest.h` | `Manifest::cache_override` (replaces the cache folder) | empty |

## Sources

| file | change |
|---|---|
| `src/core/store.cpp` | FILE mode: layer files mapped PROT_READ / MAP_SHARED at init, no F_NOCACHE (F_RDAHEAD 0 kept), no slab; reader: mincore skip of resident pieces, buffered pread into a thread-local scratch, prefault one byte per page of the view; `issue()` FILE jobs; `drop()` optional MADV_DONTNEED; `file_hit_check()` (mincore of first + last page, re-read as an urgent miss into the same slot when gone); `file_resident_bytes()`, `held_bytes()`; run-time budget (`apply_budget`, `slot_memory`: munlock + MADV_FREE_REUSABLE / purgeable EMPTY / FILE nothing; grow re-commits); victim / pick_victim / steal scan `[0, cap)`; cancel paths return a slot to the free list only below `cap`; reader QoS from `Readers::qos` |
| `src/core/moe.cpp` | `Pool(o.threads, 2000, o.qos)`, `cache.rd.qos = o.reader_qos`; `token_start` applies a pending budget; `token_end` runs `cache.token_end_file` in FILE mode when the adaptive budget or the core is on; lossless: a FILE hit is `file_hit_check`ed before it counts (a refault counts as late) |
| `src/core/pool.cpp` | worker QoS from the constructor |
| `src/core/model_lm.cpp` | default `Model::prefill` reports progress per position; `ModelLM::prefill` wraps `on_progress` with the MB issued |
| `src/core/address.cpp` | `Routers::load` uses `mem` |
| `src/core/weights.cpp` | `mem_of("file")` |
| `src/models/qwen3_5_moe/prefill.cpp` | `forward_chunk` returns bool; per layer: pending budget applied, progress (positions done = span start + n x L / NL), cancel checked: on cancel the next layer's names are unpinned and the fixed state saved at the span's start is loaded back (pos unchanged), so `prefill` returns false within one layer with a consistent state; FILE hits checked as in decode; `Qwen35::prefill` passes cancel / progress and stops at a cancelled span |
| `src/models/registry.cpp`, `src/format/manifest.cpp` | the derived-dir override; the embedding-file override and fallback; `join()` keeps an absolute path |
| `src/models/qwen3_5_moe/qwen3_5_moe.cpp`, `src/models/inkling2_loop/inkling2_loop.cpp` | the embedding row file may be an absolute path |
| `tools/dmeng.cpp` | `--embeddings F`, `--mem file`, `--file-check`, `--file-hint`, `--file-skip-resident`, `--file-prefault`, `--file-adapt`, `--file-min-gb`, `--file-core-gb`; JSON: `held_gb`, `mb_preads_per_token`, `file{...}` (refaults, resident skips, file-cache resident GB at load / end by mincore) |
| `CMakeLists.txt` | `dmeng_c` gets `src/format` as a private include dir; `dm_gate` (C, tests/c_api) |

## New files (no conflict)

- `bindings/c/dmeng_c.h`, `dmeng_c.cpp`: the C API (dm_open / dm_close / dm_info / dm_sync / dm_step / dm_cancel /
  dm_persist / dm_sample / dm_argmax / dm_set_cache_bytes / dm_counters / dm_stats_json / dm_bench), `bindings/c/public/`
  (a link to the header, SwiftPM's public-headers folder).
- `tests/c_api/dm_gate.c`: the gate through the C API only.
- `Package.swift` (repo root) + `apps/ios/DmengKit/Dmeng.swift` + `src/format/public/dmeng_format.h` (placeholder for
  SwiftPM): the engine as a Swift package for the app (A16 flags, no SME file).
- `apps/ios/README.md`, `results/ios/p1/` (P1.md, gate scripts, runs, the Mac file-cache probe).

## Semantics to keep in mind when merging

- `Cache::rec()` in FILE mode reads `S.s[slot].e`: a slot must hold its expert id from `issue()` until it is dropped
  (true today: `issue` sets `layer` / `e` before submitting reads; `drop` and the cancel paths reset them after).
- New loops over slots that choose a slot to take must scan `[0, cap)`, not `[0, n)`; loops that only read (held,
  held_experts, statistics) may keep `n`.
- A new place that takes a hit in lossless mode should call `file_hit_check` in FILE mode (decode and the batched
  prefill do; the approximate mode does not: a dropped page there is a page fault, still the right bytes).

## phone-stage (P2, 2026-11-22): changes on top of unified-wip f89d451

Branch `phone-stage` (worktree `moved-store-phone`). Defaults unchanged: a caller that leaves the new fields zero gets
unified-wip's behaviour. Results: `results/ios/p2/P2.md`.

| file | change | default |
|---|---|---|
| `bindings/c/dmeng_c.h` | `dm_open_opts` gains (at the end) `evict` ("freq" / "hazard"), `layer_caps` (1 = uniform), `plan` (1 / 2: the read planner over the given router, marginal-cost rule / byte bucket, instead of a fixed top `router_m`), `plan_u`, `reader_qos` (DM_QOS_* + 1); `res_memory_mode` accepts `DM_MEM_FILE` | 0 / NULL: decayed count, one pool, fixed top-m routers, user-initiated readers |
| `bindings/c/dmeng_c.cpp` | the fields above applied after `Core::init` (`enable_hazard`, `uniform_layer_caps`, `Planner` with `rd.prio` and `cancel_stale` as `dmeng --prefetch plan`); `dm_stats_json` adds `policy{...}` and the hook's own stats (`hook`) | |
| `include/dmeng/weights.h`, `src/core/weights.cpp` | `ResFile` with `Mem::FILE`: the whole file mapped PROT_READ / MAP_SHARED, `base = map + hdr`, no copy (the resident and the routers when `res_memory_mode` is FILE); members `map`, `map_size`; the destructor unmaps | the other modes load as before |
| `apps/ios/DmengKit/Dmeng.swift` | `DmengOptions.evict`, `uniformLayerCaps`, `plan`, `planU`, `readerQoS` | |

Note: with `res_memory_mode` FILE the resident's data pointer is `map + hdr` (not page aligned unless the header is);
the CPU kernels take any pointer, the Mac's Metal GEMM (`gpu_gemm.mm`, `newBufferWithBytesNoCopy`) does not and must
not be combined with it.

### phone-stage, later (P2 evening, on rc)

| file | change | default |
|---|---|---|
| `include/dmeng/core.h`, `src/core/moe.cpp` | approx preset `drop` (missing picks of released weight <= 0.1 left out, no substitute, no floor of experts in memory); `Approx::w_tok` per-token weight left out | presets unchanged; `drop` only when asked |
| `src/format/derive.{h,cpp}`, `tools/ms_derive.cpp` | `derive_resident(..., bits)`: `resident_q4` (the 8-bit recipe's matrices at 4 bits, `.q4` suffix, cfg `resident_bits 4`) | bits 8 |
| `bindings/c/dmeng_c.{h,cpp}` | `resident_bits`, `router_tau`, `router_cand`, `split`, `mode`, `prefill_threads` (a second pool swapped in during dm_sync), `disk_min` (iOS 16), `preload_experts`; `dm_last_prefill_json`; approx statistics in `dm_stats_json`; iOS defaults: locked slab sized from os_proc_available_memory, locked resident | macOS unchanged |
| `include/dmeng/store.h`, `src/core/store.cpp` | `caps_suspend()` / `caps_resume()`: per-layer caps off for a batched prefill pass (names never borrow past a cap, so the next layer's early names were refused and the reads ran in series with compute); `caps_auto_apply` defers new caps while suspended | caps as before outside a batched pass |
| `src/models/qwen3_5_moe/prefill.cpp` | the batched prefill suspends the caps for its pass | |
| `tools/dmeng.cpp` | `--drop` overrides the preset only when given | |
| `tests/c_api/dm_gate.c` | `--policy-probe`, `--resident`, `--mode` | |
