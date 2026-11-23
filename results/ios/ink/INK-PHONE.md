# Inkling 2 Mini on the iPhone 15 Plus (6 GB, A16): sizing and first-run design (FINISH.md I4), 2026-11-23

What this is: a sizing note for dmeng (branch `phone-ink`, from `v2` 2d39abf). It uses the engine's code, the existing Mac and
phone measurements, and a cache simulation on Inkling's recorded routing. Nothing ran on the phone or the model, and no port
was built. Every tok/s figure below is a **prediction**, with its assumptions stated next to it.

Files in this folder: `ink_phone_sim.py` (the simulator: engine pin semantics, two pools or one slab, loop hold on, soft
or off, Fast's drop rule), `run_phone_budgets.py` (the three phone slab sizes), `ink_phone_tables.py` (tables and the
compute model), `sim.json` / `sim_exact.json` / `sim.log` (raw output), `tables.md` (every row).

**Simulator check.** I replayed the engine's own Pittsburgh runs (POLICY.md, dmeng, 160 tokens). Decayed count gives
281.7 / 139.7 / 74.8 MB/token against the engine's 283.4 / 140.3 / 74.8 at 4 / 6 / 7.8 GB. Hazard gives 274.15 against 274.15.

## 1. Memory budget on the 15 Plus (Exact, 8-bit in-memory weights)

| part | GB | how |
|---|---:|---|
| in-memory weights, 8-bit (`resident_q8.dmres1`) | 1.267 | 1,266,515,968 B: attention projections, o, head, routers, message gates |
| screened head table | 0.007 | 201,024 x 36 B |
| early router (`mlp_k2_h512` q8) | 0 (0.231) | **off for the first run** (see section 4: reads take 6-12x the compute) |
| cross-token router (`xtok_s47`) | 0 (0.021) | off |
| sliding rings: 40 applications x (1,024 + 7) x 8 KV heads x 128 x K,V x 2 B | 0.169 | `Inkling::new_state` |
| convolution states: 48 x 4 | 0.004 | |
| global KV: 8 global applications x 32,768 B/position | 0.134 at 4K / 0.268 at 8K / 0.537 at 16K | DmengKit default `ctx = 16384`; pages are touched only as the chat grows. **Set ctx 4096** |
| prefill scratch: chunk 64 (~0.32 MB/position) + span arrays (74 KB/position) at span 512 | 0.06 | the defaults (chunk 256, span 8192) take 0.08 GB + 74 KB x prompt length |
| app, Swift, runtime | 0.24 | derived from the 35B phone-small: 2.96 footprint - 1.135 resident - 1.42 slab - 0.068 router - ~0.10 state |
| prefix-cache checkpoint in memory (one fixed state) | (0.17) | iOS `ckpt_gb 0.2` / `park_gb 0.3` each hold one; keep them on disk for the first run |
| **everything but the slab** | **1.88** | no router, ctx 4K, no in-memory checkpoint |

| target | slab | NVFP4 + MXFP8 slots (share 0.2) | one-slab units of 4.49 MB |
|---|---:|---|---:|
| footprint ~3.0 GB (equal to the old app) | **1.12-1.15** (simulated at 1.15) | 204 + 27 | 256 |
| C API rule at ~4.2 GB available: 4.2 - 1.27 - 0 - 0.3 - 0.9 | **1.70** | 302 + 41 | 378 |
| iOS limit with zero margin: 4.2 - 1.88 | 2.32 (simulated at 2.35) | 418 + 57 | 523 |

On the 35B, memory warnings started at a footprint of about 3.5-3.8 GB (P2). So 1.7 GB is the largest slab that is safe to
use, and 2.35 GB is a ceiling, not a setting. One problem with the rule: dm_open's slab rule (`bindings/c/dmeng_c.cpp:254`)
charges a fixed 0.3 GB for state. Inkling's state is 0.30 GB at 4K and 0.71 GB at 16K, so the rule has to use the model's
own `fixed_bytes + pos_bytes x ctx`.

Option only, not approved: a calibrated 4-bit in-memory copy would be about 0.69 GB (the 35B's 4-bit/8-bit ratio, 0.54).
That frees about 0.58 GB, so the footprint-equal slab would go from 1.15 to about 1.73 GB. Exact would then read about 750
instead of 945 MB/token (+25% tok/s). Tokens would differ from the 8-bit ones.

## 2. The floor

### What forces ~3.6 GB today (all in `src/core`)

- **`Cache::init` sets `min_cap`** (store.cpp:318) to min(slots, 2 x 128 + 2 x 8 + 2 = 274) per pool.
  - NVFP4: 274 x 4,489,216 B = 1.230 GB. MXFP8: 274 x 8,224,768 B = 2.254 GB. Together **3.48 GB** of slots. SERVE2.md
    rounds this to "~3.6 GB in force".
  - It binds whenever the pools were opened at least that large (the Mac's auto cache), because `apply_budget` and
    `lock_start` never go below `min_cap`.
  - Why 274: the span prefill (`po_span 8192`, `po_chunk 256`) keeps an application's union pinned across the span's
    sub-chunks (`span_hold`, moe_chunk.cpp:442; up to 128). It also names the next application's union (`span_name`,
    up to 128).
  - `Core::moe_chunk` has a fallback (moe_chunk.cpp ~300) that releases loop holds and names but **not** `span_hold`. When
    the union does not fit, it aborts at moe_chunk.cpp:312.
  - Measured unions over the recorded prompts (code / edit / agent), largest per application, NVFP4 / MXFP8:

    | chunk (positions) | 1 | 4 | 16 | 64 | 256 |
    |---|---|---|---|---|---|
    | union | 8 / 8 | 31 / 31 | 74 / 71 | 118 / 96 | 128 / 121 |

- **The loop hold** (moe.cpp:119-121, `Core::finish_step` moe.cpp:454) pins a looped layer's picks until its third pass.
  - The worst case is 12 layers x 2 passes x 8 = 192 held, plus the running step's 8: **200 NVFP4 slots = 0.898 GB**. At
    share 0.2 that means a budget of at least **1.12 GB**. Otherwise `Core::take_slot` (moe.cpp:433) aborts, because the
    decode path has no fallback. The largest pin count seen in the simulation was 180.
  - The same 192 goes into `extra_reserve`. A name in layers 0-19 needs more than 2K + 192 = 208 unpinned slots, so no
    early reads happen there below about 2 GB.
  - The hold buys nothing measurable. On the Mac, hold off gave the same MB/token (RESULTS-INKLING.md (b)). Here, hazard
    with the hold soft vs off differs by under 2% at 1.0 GB and not at all from 2 GB up.
- **`Cache::init`** (store.cpp:290) needs 2K + 2 = 18 slots per pool. For MXFP8 that is 148 MB, so a budget of at least
  0.74 GB at share 0.2.
- **Read-ahead:** names for the next application: 8 (top-m) or up to 16 candidates (confidence rule).
- **On a phone slab** (opened small), `min_cap` equals the whole pool. A memory warning therefore cannot shrink it.

**Today on a 1.15 GB slab:** decode is correct, but only just (204 NVFP4 slots against a worst case of 200). Prefill
aborts as soon as a chunk's MXFP8 union exceeds 27 slots: the default span always does, and chunk 4 can (31 measured).
It runs only with `prefill_chunk=2;prefill_span=2;prefill_stream=0`, and that re-reads the union every 2 positions, so it
is far too slow.

### The smallest cache that is correct in Exact after the change

| what must fit | two pools | one slab (MXFP8 = 2 units) |
|---|---:|---:|
| pins only: one application's 8 records per pool | 8 x 4.49 + 8 x 8.22 MB = **0.10 GB** | 0.07 GB |
| pins + one step of read-ahead (16 per pool; with the 2K + 2 open rule) | **0.20 GB** (0.23) | 0.14 GB |

Both are below the C API's 0.6 GB clamp. After the change the slab size is a speed choice, not a correctness limit.

### Changes for the first run, by file and function

1. **`src/core/moe_chunk.cpp`, `Core::moe_chunk`: compute the union in waves.**
   - Pin at most cap - 2K records.
   - Compute their (position, pick) rows into `C.eout`, unpin, then take the next wave.
   - The combination after the last wave stays unchanged: it reads `eout` rows in pick order, so the result is bit-identical.
   - The fallback also releases `span_hold`.
   - Prefill then needs one wave, and spans can stay long. Long spans are what keep prefill reads to about one read of each
     application's union per span.
2. **`src/core/moe.cpp`, `Core::take_slot`: release the loop holds** before aborting, the same fallback as moe_chunk (pins
   only, so outputs cannot change). Alternatively, the phone profiles set `CoreOpts::loop_hold = false`.
   - In `Core::init`, `extra_reserve` should count the holds actually held, not the worst case of 192.
3. **`src/core/store.cpp`, `Cache::init`:** set `min_cap = 2K + 2` once (1) is in, so the warning handler can shrink the
   slab.
4. **`bindings/c/dmeng_c.cpp`, `dm_open`:** the slab rule's state term comes from the model (section 1).
5. **No change** to the pool split:
   - Hazard is within 1-4% across share 0.1 / 0.2 / 0.3 and one slab (one slab best, -4% at 0.8 GB, -2% at 1.0 GB), so the single shared pool is not worth building now.
   - Reading MXFP8 on demand (4 layers x 8 picks = 263 MB/token) would add 376 ms/token at 0.7 GB/s, so it is not worth it.

## 3. Bytes per token at phone sizes (simulation, decode only, prompt replayed first and not counted)

Texts: pitt (10 + 159), code (462 + 399), edit (543 + 639), agent (839 + 450). Policy is hazard (the engine's `auto`), two
pools at share 0.2, loop hold off. Hazard with per-layer caps (what `auto` does below 768 slots) is within 1% of these
rows; all rows are in `tables.md`.

| slab GB | pitt | code | edit | agent | **mean MB/token** | hit looped / outer | decayed count (old default), mean |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.8 | 1,064 | 1,148 | 1,149 | 1,206 | 1,142 | 0.49 / 0.12 | 1,469 |
| 1.0 | 935 | 1,024 | 1,025 | 1,079 | 1,016 | 0.56 / 0.16 | 1,362 |
| **1.15** | 869 | 950 | 953 | 1,008 | **945** | 0.61 / 0.19 | - |
| 1.4 | 755 | 850 | 855 | 909 | 842 | 0.67 / 0.23 | 1,092 |
| **1.7** | 655 | 764 | 763 | 814 | **749** | 0.72 / 0.28 | - |
| 2.0 | 576 | 684 | 681 | 728 | 667 | 0.76 / 0.34 | 834 |
| **2.35** | 498 | 606 | 600 | 641 | **586** | 0.80 / 0.39 | - |
| 2.5 | 470 | 579 | 571 | 610 | 557 | 0.81 / 0.41 | 677 |
| 3.0 | 389 | 501 | 487 | 516 | 473 | 0.85 / 0.47 | 559 |

For scale: one token traverses 1,839 MB of experts, and the 35B on this phone reads about 250 MB/token. With the decayed
count, the hold matters (soft vs off: 1,062 vs 1,362 at 1.0 GB). With hazard, it does not.

**Fast** (a missing pick with weight fraction <= 0.1 is left out). Only the agent trace (`cache-policy-work/runs/ag_ink.trace`)
has routing weights; the four `.rec` recordings do not. The trace's picks match `ag_ink.rec` 100%.

| slab GB | Exact MB/token | Fast MB/token | weight left out per token (of 48) | picks dropped per token (of 384) |
|---:|---:|---:|---:|---:|
| 0.8 | 1,206 | 616 | 8.2 | 110 |
| 1.15 | 1,008 | 512 | 7.4 | 100 |
| 1.7 | 814 | 399 | 6.3 | 86 |
| 2.35 | 641 | 307 | 5.1 | 71 |
| 3.0 | 516 | 240 | 4.3 | 60 |
| 6.0 (Mac-size reference) | - | 83 | 1.9 | 28 |

Fast halves the reads, but it leaves out **15% of the routing weight** at 1.15 GB. The 35B's phone Fast leaves out 6% (2.54
of 40). The Inkling Mac bounded mode replaced 1-2%. The quality of this has not been measured, so Fast stays off on this
phone until V2's agreement check passes at these sizes.

## 4. Predicted speed

**Compute (prediction).** I divided Inkling's bytes per class by the A16's measured rate for the nearest class of 35B
kernels (P0.md 3.1, best cells):

- NVFP4 experts at the 4-bit expert shapes' 31.5 GB/s: 50.2 ms
- MXFP8 at the 8-bit 43.4 GB/s: 6.1 ms
- attention projections: 23.6 ms
- o: 8.3 ms
- head: 10.8 ms
- message gates: 4.1 ms
- router: 0.7 ms
- the M4's non-matvec remainder, scaled by the same ratio: 11.3 ms

Total: **115 ms burst**, and **185 ms sustained** at the A16's 0.62 power factor. As a cross-check, scaling the Mac's 50 ms
by the 35B's measured phone/M4 ratios gives 112 / 182 ms.

The weak assumption: NVFP4 and MXFP8 have never been timed on the A16.

**Bench cells to replace it.** Use the P0 `-kernelBench` (`apps/ios/p0bench`, `bench_cpu.cpp`), at 2 / 4 / 6 threads, ui
QoS, 3 s per cell:

- NVFP4 gate_up 2304 x 2304 (interleaved, I16) and down 2304 x 1152 (`mv_nvfp4`)
- MXFP8 at the same two shapes (`mv_mxfp8`, fp16 activations)
- 8-bit affine g64: 3072 x 2304 (q, gate), 1024 x 2304 (k, v), 384 x 2304 (r), 2304 x 3072 (o), the head slice 32,768 x 2304
  plus the screened head
- bf16 router 128 x 2304 and a message-gate down projection
- a 60 s sustained NVFP4 cell
- for prefill: batched NVFP4 at N 8-256

tok/s = 1 / (compute + MB / drive). Each cell gives burst / sustained.

| mode | slab GB | MB/token | drive 0.7 GB/s | drive 1.1 GB/s |
|---|---:|---:|---|---|
| Exact | **1.15** (footprint-equal) | 945 | **0.68** / 0.65 | 1.03 / 0.96 |
| Exact | **1.7** (largest that fits) | 749 | **0.84** / 0.80 | 1.26 / 1.15 |
| Exact | 2.35 (zero margin) | 586 | 1.05 / 0.98 | 1.54 / 1.39 |
| Fast | 1.15 | 480 | 1.25 / 1.15 | 1.81 / 1.61 |
| Fast | 1.7 | 367 | 1.56 / 1.41 | 2.23 / 1.93 |

The Fast MB/token is the agent text's Fast/Exact ratio applied to the four-text mean.

On the drive: at 1.15 GB a token is about 1.35 s of reads against 0.115 s of compute. Early reads (names) can hide at most
the compute, which is under 9% of that and not worth the router's 0.23 GB of slab. That is why the router is off for the
first run.

Prefill: a long prompt reads each application's union once per span, about 44 x ~110 x 4.49 MB + 4 x ~115 x 8.22 MB ≈
26 GB. That is about **37 s at 0.7 GB/s** for any prompt up to the span, because it is drive-bound. A 16-token message
reads about 10 GB, about 14 s.

## 5. Storage

Phone profile parts, with no bf16 resident (the same scheme as the 35B's phone-small: a derived resident plus an
embedding-table file):

| part | bytes |
|---|---:|
| `store/layer_00..19.bin` (NVFP4) | 20 x 574,619,648 = 11,492,392,960 |
| `store/layer_20..23.bin` (MXFP8) | 4 x 1,052,770,304 = 4,211,081,216 |
| `derived/resident_q8.dmres1.15a39fa8aa48.bin` | 1,266,515,968 |
| `embeddings.safetensors` (`model.llm.embed.weight` bf16 [201,024, 2,304]; **to be made**, not on the Hub) | 926,318,592 + header |
| tokenizer.json, tokenizer_config, special_tokens_map, chat_template.jinja, config, generation_config, store/meta.json, moved-store.json | ~0.026 GB |
| **total (no router)** | **~17.92 GB** |
| optional `derived/router.mlp_k2_h512.q8` (+ `xtok_s47` 20.7 MB) | 231,473,152 -> 18.15 GB |

Left out: `resident.safetensors` 3.22 GB, which the Mac reads embedding rows from and which `dmeng_c` would otherwise
derive from; `mtp.safetensors` 0.64 GB (draft head); the five router safetensors (2.07 GB).

The 18.17 GB hub figure matches store + derived resident + embedding table + mlp_k2 router + small files (18.15 GB by my
count), so a profile listing those parts is right. If it lists `resident.safetensors` instead of the embedding table, it
is about 20.5 GB. Check the Hub manifest's phone profile before downloading.

The two models share no parts: the tokenizers, stores and residents all differ. The phone holds the 35B (20.35 GB) with
about 1 GB free, so it can hold **one model at a time**. Switching means:

1. delete the current model's profile parts (`Documents/dmeng-model/<model>/`; the prefix-cache disk copies are per model too);
2. download the other model's profile, resumable and SHA-checked as today. By cable at the measured 32.3 MB/s this is about
   9 min.

## 6. What the app needs beyond the 35B path

- **Tokenizer (Swift).**
  - What it is: byte-level BPE, 199,998-entry vocabulary plus added special tokens, a `Split` regex pre-tokenizer, 200,058 used ids, special tokens
    `<|message_user|>` / `<|message_model|>` / `<|message_system|>` / `<|message_tool|>` / `<|content_text|>` /
    `<|end_message|>`..., eos 200006.
  - Approach: if the app's Swift BPE reads `tokenizer.json` generically, Inkling needs only its own regex and added tokens.
  - Gate: id-equality against `local/inkling/model` (HF tokenizer) on the four texts.
  - Estimate: ~1 day.
- **Chat template and output parsing.**
  - Template: `local/inkling/model/chat_template.jinja`. It has role tokens, a "Thinking effort level" system message
    (default 0.9), and tool specs rendered in the system turn.
  - Parser reference: the engine's `python/serve/generic/chat_format.py`, which renders that template and splits the output
    into reasoning / text / tool-call segments, then maps them to OpenAI tool calls.
  - Port: the fixed render for the app's roles, plus the segment parser, in Swift.
  - Estimate: ~1.5-2 days with fixtures generated by the Python front end.
- **Settings.** A model picker with the on-device state of each model (absent / downloading / ready) and size. Selecting the
  other model starts the delete-then-download flow from section 5. Chats are per model: prefix-cache copies and token ids
  do not carry over. Estimate: ~1 day.
- **C API.** The same `dm_open` with the Inkling folder (the family comes from the manifest). Options to set:

  | option | value |
  |---|---|
  | profile | `phone-small` |
  | mode | `exact` (Inkling's built-in phone-small default is `fast`, see section 3) |
  | `routers_id` | `none` |
  | `xtok` | `none` |
  | drafts | off (the default; the Inkling draft head is not in the phone set) |
  | `ctx` | 4096 |
  | `embeddings` | `embeddings.safetensors` |
  | `model_options` | `prefill_release=1;prefill_chunk=64;prefill_span=512`, after change 1 in section 2 |
  | `ckpt_gb` / `park_gb` | small enough that a checkpoint goes to disk |

  `resident_bits` stays 8. Prefill GPU stays off: Inkling's experts have no GPU kernel, and the projections' share has not
  been checked on the A16. No new C entry points.

## 7. Verdict

- Inkling fits a 6 GB phone in Exact at a ~1.15 GB slab (footprint ~3.0 GB). Decode is correct there today, only just; prefill
  is not, until the waves change. After the changes in section 2 the correct minimum is 0.10 GB of pins (0.20 GB with
  read-ahead).
- **Predicted: 0.65-0.68 tok/s Exact at 0.7 GB/s** (1.0 at 1.1 GB/s) at the footprint-equal slab, and 0.80-0.84 at 1.7 GB.
  Fast is about 2x faster but leaves out 15% of the routing weight. The phone is drive-bound: ~945 MB/token against 115 ms
  of compute.
- **First thing to build:** waves in `Core::moe_chunk` with the hold fallback in `Core::take_slot`, then one Exact phone run
  at 1.15 GB with ctx 4096 and no routers, alongside the NVFP4 / MXFP8 kernel bench cells.

## 8. phone-small with 4-bit in-memory weights and NVFP4 records for layers 20-23 (added 2026-11-23)

This section covers phone-small only. Mac and larger phones stay as released.

**What changes on the device:**
- In-memory weights: 4-bit file (`ms_derive resident DIR --bits 4`, plain affine g64, the same matrices as the 8-bit
  file), 0.713 GB. It frees 0.554 GB of slab at equal footprint.
- Experts: layers 20-23 rewritten as NVFP4 (`python/derive/ink_mx_to_nv.py`), so every record is 4,489,216 B and the cache
  is one pool. The block scale is chosen by least squares; relative squared error is 0.0073 against 0.0090 for the
  max-based scale.
- The screened head stays off, because it needs an 8-bit head.

The simulator (`run_nv4.py`, `nv4_tables.py`) uses hazard, one pool and loop hold off. The slab at the same footprint
is 1.15 + 0.55 ≈ 1.70 GB; a 1.8 GB slab was also run.

Compute is a **prediction**:
- 4-bit affine bytes are 0.529 of the 8-bit ones;
- rates are the 35B's 4-bit resident shapes on the A16 (P0 3.1);
- all experts run at the 4-bit expert rate;
- result: 97 ms burst, 157 ms sustained.

tok/s = 1 / (compute + MB / drive). Burst first, sustained in brackets.

| slab GB | Exact MB/token (pitt / code / edit / agent; mean) | hit looped / outer | Fast MB/token (agent ratio on the mean) | weight left out (of 48) | Exact tok/s 0.7 / 1.1 GB/s | Fast tok/s 0.7 / 1.1 GB/s |
|---|---|---|---|---|---|---|
| 1.15 | 740 / 795 / 819 / 868; **805** | 0.66 / 0.16 | 419 | 7.05 | 0.80 (0.76) / 1.21 (1.12) | 1.44 (1.32) / 2.09 (1.86) |
| 1.7 | 551 / 634 / 643 / 690; **629** | 0.75 / 0.28 | 316 | 5.88 | 1.00 (0.95) / 1.49 (1.37) | 1.82 (1.64) / 2.60 (2.25) |
| 1.8 | 525 / 608 / 617 / 662; **603** | 0.77 / 0.30 | 302 | 5.69 | 1.04 (0.98) / 1.55 (1.42) | 1.89 (1.70) / 2.69 (2.32) |
| 2.25 | 428 / 516 / 519 / 558; **505** | 0.82 / 0.38 | 246 | 4.92 | 1.22 (1.14) / 1.80 (1.62) | 2.23 (1.97) / 3.11 (2.63) |
| 2.9 | 329 / 420 / 409 / 435; **398** | 0.87 / 0.48 | 188 | 4.04 | 1.50 (1.38) / 2.18 (1.93) | 2.73 (2.35) / 3.73 (3.05) |

At the same footprint (1.70 GB) against today's 1.15 GB with 8-bit weights:

| | 1.15 GB, 8-bit weights | 1.70 GB, 4-bit + NVFP4 |
|---|---|---|
| Exact MB/token | 945 | 629 |
| Exact tok/s at 0.7 GB/s | 0.68 | 1.00 (predicted) |
| Exact tok/s at 1.1 GB/s | 1.03 | 1.49 (predicted) |
| Fast tok/s at 0.7 GB/s | 1.25 | 1.82 (predicted) |
| Fast weight left out per token | 7.4 of 48 | 5.9 of 48 |

Fast's agreement with Exact has not been measured. The engine session measures it before Fast becomes the default.

**At the phone's drive rate now** (0.82-0.85 GB/s raw with 8 readers; taken as 0.8 GB/s), predicted tok/s, burst
(sustained):

| slab | Exact MB/token | Exact tok/s | Fast MB/token | Fast tok/s |
|---|---:|---|---:|---|
| 1.7 GB | 629 | 1.13 (1.06) | 316 | 2.03 (1.81) |
| 1.8 GB | 603 | 1.18 (1.10) | 302 | 2.11 (1.87) |

**Simulator against the engine session's measurement.** The engine measured Fast at 1.7 GB with both 4-bit steps:
260 MB/token on prose and 368 on agent.

- **Agent:** the simulator gives 347 on the same agent trace, 6% under.
- **Prose:** no prose trace with weights exists here. Applying the agent text's Fast/Exact ratio to Pittsburgh's
  Exact 551 gives an estimate of 277, 7% over.

So the simulator is within about ±7% of the measured reads. It models no early names (reads by the routers), no
background fetches and no names' evictions; the measured run had the profile's routers.

**Phone set:** 36 files, 15,708,540,348 B (`phone_set/phone_set.tsv`, made by `phone_set/make_phone_set.py`). The copy
script (phone-store `ios/PhoneStoreApp/Inkling/copy_inkling_to_phone.sh`, not run) leaves out `LICENSE` (11,343 B, not on
the Mac).
