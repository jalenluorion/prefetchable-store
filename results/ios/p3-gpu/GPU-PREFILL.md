# P3: the prefill's CPU + GPU split on iOS (branch phone-gpu, 2026-11-23)

Goal: the phone's cold first token on the 472-token input (20.1-24 s on the 15 Plus, 4-bit resident, 4 CPU threads)
down to 17 s or less, exactly. This branch ports the Mac's exact Metal split of the batched prefill to iOS, extends it
to the 4-bit resident and to the experts, and adds a faster exact GPU kernel. Nothing here ran on the phone (another
builder holds it): every number below is the Mac's or a projection, labelled as such.

## What changed (commits on phone-gpu, from phone-stage 2bacd5c)

| file | change |
|---|---|
| `src/core/gpu_gemm.{h,mm}` | rewritten: one Metal dispatch per *set* of products (a tile table maps threadgroups to jobs); 8-bit **and 4-bit** affine codes; inputs read directly or **through a position map** (each expert's positions gathered on the GPU from the layer's MoE input, no per-expert copy); several wrapped regions (resident file region, expert slab; overlapping windows if a region exceeds `maxBufferLength`); weights wrapped with `newBufferWithBytesNoCopy` (page-aligned, no copies); a set the GPU does not complete (e.g. an iOS app moved to the background) is **computed on the CPU** instead, so results never depend on the GPU's availability. Two kernels, the same bits: `gd` (int32 group dots, the Mac's original) and `gm` (default, below). |
| `src/models/qwen3_5_moe/prefill.cpp` | `run_split` replaces `run_set`: the attention projections (`prefill_gpu_frac`: qkv, z, o; q, k, v, o) and **the experts' products** (`prefill_gpu_experts`: routed experts held in the slab, gate / up / down, and the shared expert) split by rows: the GPU takes the first `frac` of each matrix's rows (whole 64-row blocks), the CPU the rest, at the same time. Expert waves smaller than `prefill_gpu_min_wave` (128 assignments) stay on the CPU. The prefill JSON gains `detail.gpu` (per part: GMAC each side, wait for the GPU after the CPU's rows, staging ms, sets, failed sets, kernel). |
| `src/core/gpu_check.{h,cpp}` | `gpu_gemm_selftest`: both GPU kernels against the CPU kernels bit for bit on random data, then rates (GPU alone, CPU alone on N threads, both together, the balanced share). |
| `bindings/c/dmeng_c.{h,cpp}` | `dm_open_opts`: `prefill_gpu_frac`, `prefill_gpu_experts`, `prefill_gpu_min` (0 = 256 positions), `prefill_gpu_int_kernel`, `prefill_gpu_min_wave` (0 = 128); **all 0 = off, the default everywhere**. `dm_gpu_check(threads, seconds)`: the self-test on the device. |
| `apps/ios/DmengKit/Dmeng.swift` | `DmengOptions.prefillGpuFrac / prefillGpuExperts / prefillGpuMin / prefillGpuIntKernel / prefillGpuMinWave`; `DmengModel.gpuCheck(threads:seconds:)`. |
| `Package.swift` | `gpu_gemm.mm` no longer excluded: built with `DM_METAL`, Metal + Foundation linked, on iOS and macOS. |
| `CMakeLists.txt`, `tools/dmserve.cpp`, `tools/dmeng.cpp`, `tests/c_api/dm_gate.c`, `tests/kernels/gpu_gemm_check.cpp` | Mac build of the check; `--gpu-experts`, `--gpu-min`, `--gpu-kernel`, `--gpu-min-wave` (dmserve), `--prefill-gpu-experts` (dmeng), `--gpu-frac / --gpu-experts / --gpu-min / --gpu-int-kernel / --gpu-min-wave / --prefill-threads` (dm_gate); dm_gate's JSON now carries the prefill account. |

### Why the GPU's rows are the CPU's bits

Per (row, position) and 64-input group, the CPU kernels (`gemm_aff8_i16` / `gemm_aff4_i16`) form an exact int32 group
dot of the codes with the 15-bit activations (hi * 256 + lo planes), then update lane g mod 4 with
`fma(dot, scale * d)` and `fma(bias, gsum)` and sum the lanes pairwise. Both GPU kernels do the same:

- `gd`: the int32 dot directly (the Mac's kernel), 4-bit codes read as nibbles (element 2j low, 2j+1 high).
- `gm` (default): the two plane sums `sum code * hi` and `sum code * lo` on the fp32 matrix units (simdgroup 8 x 8
  multiply-accumulate on half inputs). Every product and partial sum is an integer of magnitude at most
  64 x 255 x 128 < 2^24, so every fp32 operation on them is exact in any order; `dot = int(hi_sum) * 256 + int(lo_sum)`
  is the CPU's `(sum_hi << 8) + sum_lo`.
- The epilogue is the CPU's operation for operation, compiled with `MTLMathModeSafe` (no reassociation, contraction
  or approximation).

Each output is computed once, by one side. This is exact: no step is approximate.

## Gates (Mac M4, Metal; shared machine)

| gate | what | result |
|---|---|---|
| `gpu_gemm_check` / `dm_gpu_check` (no model) | both kernels vs the CPU kernels: 8- and 4-bit, direct and gathered inputs, n 1-300, K 256-4096 | **4,737,280 outputs, 0 differ** |
| `tests/prefill_check.sh` 8-bit resident, 473-token wiki prompt, split 0.5 / 0.5, `--gpu-min 64` | batched prefill with the split vs token-by-token decode | **logit bits 0, state bytes 0, KV bytes 0, greedy identical** (GPU 302.5 + 267.8 GMAC, 0 failed sets) |
| same, **4-bit resident** (resident_q4 a603e898ddee) | | **logit bits 0, state bytes 0, KV bytes 0, greedy identical** |
| `dm_gate` (C API) 8-bit, Pittsburgh, 400 tokens, split 0.5 / 0.5, `--gpu-min 8` (the prompt is 9 tokens) | ids vs `r_r8_pitt_gb3.3_r1.ids` | **identical, 400 of 400** (GPU used: 5.8 + 5.1 GMAC) |
| `dm_gate` 8-bit, coding prompt (357 tokens), 400 tokens, split 0.5 / 0.5 | ids vs `dc_off_code.ids` | **identical, 400 of 400** |
| `dm_gate` **4-bit**, wiki prompt (473 tokens), 400 tokens, split 0.6 / 0.6 vs split off | ids | **identical, 400 of 400** (GPU 358 + 279 GMAC, 0 failed sets) |

Run them again: `results/ios/p3-gpu/mac/run_gates.sh check8|check4|capi8|capi4|timing` (every model run through
the guard). Logs, ids and JSON are next to the script.

## Mac timing (orientation only: the M4 is not the A16, and the Mac was shared and loaded)

473-token wiki prompt, 4-bit resident, CPU threads limited as listed, 1.8 GB locked slab (the phone's size),
`dmserve --prefill-bench`, second prefill of the process (the first in parentheses).

| cell | CPU threads | prefill ms (first) | tok/s | read wait ms | attn proj + out ms | attn core ms | experts ms | GPU wait attn / experts ms | others' CPU % |
|---|---|---|---|---|---|---|---|---|---|
| off | 4 | 7122 (6586) | 66.4 | 28 | 2617 | 1231 | 2817 | - | 303 |
| attn 0.5 | 4 | 6427 (5305) | 73.6 | 28 | 1513 | 1295 | 3014 | 25 / 0 | 146 |
| experts 0.5 | 4 | 6612 (6413) | 71.5 | 19 | 2875 | 1280 | 1973 | 0 / 47 | 147 |
| both 0.5 | 4 | 4583 (4469) | 103.2 | 40 | 1299 | 1118 | 1702 | 7 / 52 | 154 |
| both 0.35 | 4 | 5015 (5007) | 94.3 | 39 | 1577 | 1017 | 1997 | 0 / 3 | 87 |
| both 0.65 | 4 | 4927 (5036) | 96.0 | 29 | 1419 | 1072 | 2005 | 448 / 630 | 143 |
| both 0.5, integer kernel | 4 | 5876 (5907) | 80.5 | 24 | 1909 | 1105 | 2442 | 640 / 825 | 211 |
| both 0.65 | 2 | 6144 (6286) | 77.0 | 10 | 1552 | 1844 | 2065 | 8 / 18 | 156 |
| off | 2 | 10814 (10862) | 43.7 | 0 | 3913 | 1826 | 4386 | - | 108 |

Readings (Mac, not the phone): with 4 CPU threads both parts at 0.5 cut the prefill from 6.6-7.1 s to 4.5-4.6 s
(-35 %); each part alone gives about half of that. On the M4 the GPU and 4 CPU threads are about equal (self-test:
matrix kernel 291 GMAC/s, CPU 145, together 301 + 144), so 0.65 overloads the GPU (the CPU waits 0.45 + 0.63 s for
it) and 0.35 underloads it. The integer kernel at 0.5 is slower (5.9 s, waits 0.64 + 0.83 s): the matrix kernel is
the one to use. With 2 CPU threads the split at 0.65 cuts 10.8 s to 6.1-6.3 s (-43 %): the fewer the CPU's cores,
the larger the GPU's share should be. At 8 threads (the C API gate) 0.6 was slower than off (4.84 vs 4.47 s): the
share must follow the CPU / GPU ratio of the device, which is why the A16's starting point below comes from the
A16's rates. Read waits were 0-40 ms here (the Mac's drive is fast), unlike the phone.

## What to expect on the A16, and why

Phone baseline (P2, 5 cold runs of the 472-token row, 4 threads, 4-bit resident, 1.72 GB slab): 20.5-24.0 s;
compute by stage: attention projections + outputs 5.5-5.9 s (604 GMAC: ~105 GMAC/s), attention core 2.75-2.9 s,
experts 7.4-7.9 s (536 GMAC: ~70 GMAC/s with gather / activation / combine), router + names + cache + embed ~0.9 s;
read wait 2.7-6.2 s; **13.67 GB read**.

Rates (A16 bench, P2, N 256): GPU integer kernel 87 GMAC/s alone, 73 with the CPU busy; CPU 4-bit 124 alone on 6
threads, 77 with the GPU busy (the two share power and memory: x 0.84 and x 0.62). The matrix kernel is not yet
measured on the A16; on the M4 it is 1.8x the integer kernel on projections and 1.56x on expert sets. If that holds:
~130 GMAC/s (projections) and ~100 (expert sets) with the CPU busy.

Balanced share f = G / (G + C) with the CPU's own rate times 0.62:

| part | GPU kernel | GPU (busy) | CPU (busy) | balanced f | compute now -> projected |
|---|---|---|---|---|---|
| attention projections | matrix | ~130 | ~65 | **0.65-0.7** | 5.7 s -> ~3.1 s |
| experts | matrix | ~100 | ~43 | **0.65-0.7** | 7.6 s -> ~4.7 s (products 3.7 + the CPU's gather / activation / combine) |
| attention projections | integer | 73 | ~65 | 0.5-0.55 | 5.7 s -> ~4.4 s |

Projected compute: ~16.6 s -> **~11.5 s** (matrix kernel), ~13.5 s (integer). **But the row then meets the drive**:
13.67 GB at the locked slab's 0.74 GB/s is 18.5 s of reading, and today's runs (20.5-24 s) already sit only 2-5.5 s
above that. Expected row with the split: **~18.5-19.5 s at 0.74 GB/s** (a clear win over today's 20.1-21.4 s and the
old app's 21 s); **17 s needs the drive at >= 0.81 GB/s through the prefill** (P2 saw 0.44-0.9 GB/s between runs).
After the split the row's limit is bytes over drive rate, not compute: to reach 17 s reliably, the next levers are
the drive rate (readers, `split` pieces per record, reader QoS) or fewer bytes. Check the drive's share first:
`mb_read / (ms - first layer)` against 0.74-0.9 GB/s.

## Merge and measure (for the phone builder)

1. On branch phone-stage: `git merge phone-gpu` (four commits on top of
   2bacd5c; touches `Package.swift`, `bindings/c/dmeng_c.{h,cpp}`, `apps/ios/DmengKit/Dmeng.swift`,
   `src/core/gpu_gemm.*`, `src/core/gpu_check.*`, `src/models/qwen3_5_moe/{prefill.cpp,qwen3_5_moe.{h,cpp}}`, tools,
   tests, CMake). The new `dm_open_opts` fields are appended at the end of the struct and default to 0 (off).
2. App side (not applied here): `results/ios/p3-gpu/app-launch-args.swift.txt` adds launch arguments
   `-dmengGpuFrac F -dmengGpuExperts F -dmengGpuMin N -dmengGpuKernel integer|matrix` to DmengChatEngine's options.
   Build as usual (the app's `native/dmeng` link points at that checkout; the package now compiles
   `gpu_gemm.mm` and links Metal).
3. On the phone, first the self-test (no model, ~16 s at 2 s per rate): `DmengModel.gpuCheck(threads: 4, seconds: 2)`
   (e.g. from Probe behind a launch argument). It must print `GPU CHECK PASSED` (`"exact":true`: the A16's GPU family
   differs from the M4's, so this is the device-side proof) and gives the A16 rates of both kernels and the balanced
   share for 4 threads (`split_frac`).
4. Exactness on the phone with the split on: the gate with prompt ids, `-dmengGate p2/wiki_ids.txt -dmengGateN 400`
   (copy `results/ios/p3-gpu/mac/wiki_ids.txt` to the phone's p2 folder) with `-dmengGpuFrac 0.65 -dmengGpuExperts
   0.65`: the ids must equal `results/ios/p3-gpu/mac/capi_wiki4_off.ids` (the Mac's 4-bit ids; the phone's 4-bit
   lossless ids equal the Mac's, P2 0e). (Pittsburgh is 9 tokens, under `prefill_gpu_min`, so it does not exercise
   the split unless `-dmengGpuMin 8`.)
5. The 472-token row (`e7`-style `-chatOnce "$WIKI"` cold, as in `phone_dmeng_1122.sh`), one cell each, `cool`
   between cells, in this order:
   - split off (the same build: the baseline)
   (Experts' gate / up and the shared expert have 512 rows: their GPU share moves in steps of 1/8 (0.65 gives
   0.625, 0.6 gives 0.5); the attention matrices and the down projections have 2048+ rows: steps of 1/32 or finer.)
   - `-dmengGpuFrac 0.65 -dmengGpuExperts 0.65` (matrix kernel)
   - `0.5 / 0.5` and `0.75 / 0.75`
   - `0.5 / 0.5 -dmengGpuKernel integer` (if the self-test shows the matrix kernel is not faster on the A16)
   - with the best fractions: `-dmengPrefillThreads 3` and `6` (the GPU takes most products, so the CPU's remaining
     work, attention core and expert gather / activations, may prefer a different count; 6 was slower without the
     split)
   Read from the `DMENG PREFILL` line: `ms`, `ms_wait`, `mb_read`, and `detail.gpu.attn|experts.wait_ms`. Tune each
   part separately: a large `wait_ms` (the CPU waiting for the GPU's rows) means that part's fraction is too high;
   ~0 means it can go up by 0.05-0.1. Watch `failed` (must be 0 in the foreground) and thermal state (CPU + GPU draw
   more power; the row is ~20 s, P2 saw fair at ~40 s).

## What can block it on iOS

- Not run on the phone yet: compiled and linked for arm64 iOS 17 (`swift build --triple arm64-apple-ios17.0` of this
  package; a test executable calling `dm_open` with the split on and `dm_gpu_check` links against Metal). The
  CMake iOS build stops at `kernels_sme.cpp` (SME; the package excludes it, as before).
- Memory: the GPU keeps ~35-50 MB of shared staging / output buffers between prefills at 472 positions (released with
  `prefill_release`), plus the Metal device and pipelines. The phone ran at 0.86 GB available with two memory
  warnings during this prefill (P2), so check the footprint with the split on. The wrapped resident and slab are
  the same pages (no copies); confirm on the device that wrapping does not count them twice.
- `maxBufferLength`: if the A16's is below the 1.72 GB slab, the slab is wrapped in overlapping windows (handled; the
  open log line says `prefill GPU split on (... GB wrapped)`). A region that cannot be wrapped leaves those matrices on
  the CPU, still exact.
- Background: a set the GPU refuses (app in the background) is redone on the CPU (logged once; `failed` counts it).
- Kernel compile at open: the Metal source is compiled at `dm_open` when an option is on (tens of ms on the Mac).

## Mac

### The 35B's cold prefill of the recorded OpenCode request (7,571 tokens)

As `engine/unify/work/chain4.sh` runs it: `dmserve --prefill-bench engine/results/agentic/replay/s2_clean/ids/002.txt
--cache-gb 5 --mem mlock --ctx 8192 --chunk 512 --span 8192 --stream 1 --gen 4 --rep 2` (8-bit resident, the Mac's
defaults), this branch's build, one process per cell (its two prefills are the two runs; the second starts with the
slab as the first left it), every run through the guard. "Integer kernel, projections 0.38" is today's setting (the
old kernel and share; this branch's grouped dispatch). Load: others' mean CPU during the cell and swap at its start.
Script: `mac35/run_oc.sh THREADS CELLS...`, table: `mac35/oc_table.py`.

| threads | cell | prefill s, run 1 / run 2 | tok/s (best) | peak footprint GB | GPU wait attn / experts s (run 2) | others' CPU % (mean) | swap MB start |
|---|---|---|---|---|---|---|---|
| 8 | split off | 83.4 / 92.6 | 90.8 | 7.97 | - | 117 | 12175 |
| 8 | split off (again, later) | 80.0 / 80.2 | 94.6 | 7.99 | - | 45 | 9634 |
| 8 | integer kernel, projections 0.38 (old setting) | 86.4 / 102.3 | 87.7 | 8.05 | 6.0 / 0.0 | 159 | 12274 |
| 8 | matrix, projections 0.5 | 76.0 / 77.1 | 99.6 | 8.04 | 6.6 / 0.0 | 59 | 8508 |
| 8 | matrix, projections 0.65 | 97.0 / 94.3 | 80.3 | 8.04 | 13.1 / 0.0 | 199 | 12847 |
| 8 | matrix, 0.5 / experts 0.5 | 70.9 / 75.1 | 106.9 | 8.08 | 3.4 / 6.5 | 103 | 8363 |
| 8 | matrix, 0.65 / experts 0.65 | 78.1 / 78.5 | 97.0 | 8.06 | 12.0 / 14.9 | 61 | 8459 |
| 8 | matrix, 0.65 / experts 0.5 | 74.0 / 74.4 | 102.3 | 8.04 | 11.7 / 7.0 | 64 | 8379 |
| 8 | matrix, projections 0.4 | 76.2 / 77.7 | 99.3 | 8.04 | 1.4 / 0.0 | 65 | 9155 |
| 8 | matrix, 0.4 / experts 0.4 | 67.4 / 68.1 | 112.3 | 8.05 | 0.4 / 0.6 | 71 | 8230 |
| 8 | matrix, 0.45 / experts 0.45 | 66.0 / 74.1 | 114.7 | 8.04 | 0.7 / 1.1 | 100 | 8509 |
| 8 | matrix, 0.4 / experts 0.3 | 68.2 / 75.4 | 111.0 | 8.05 | 0.4 / 0.0 | 91 | 8804 |
| 10 | split off | 88.3 / 89.5 | 85.7 | 7.98 | - | 109 | 10346 |
| 10 | matrix, 0.4 / experts 0.4 | 74.9 / 106.6 | 101.0 | 8.06 | 0.9 / 1.7 | 159 | 9806 |

Readings. The load moved between cells (others' CPU 45-199 %), so compare cells at similar load, and the repeat of
split off (80.0 / 80.2 s at 45 %) is the fairer baseline than the first (83.4 / 92.6 s at 117 %).
- **Best: matrix kernel, projections 0.4 and experts 0.4, 8 threads: 67.4 / 68.1 s (112 tok/s), against 80.0 / 80.2
  s split off: -16 %.** 0.45 / 0.45 matches it in its first run (66.0 s) and lost its second to load (74.1 s, 100 %);
  0.4 / 0.3 is between them. Exactness at 0.4 / 0.4 is below.
- On the M4 the balanced share is ~0.4, not the phone's ~0.65: on this prompt 8 CPU threads run the 8-bit
  projections at ~370 GMAC/s and the GPU at ~275, so at 0.65 the CPU waits 12-13 s for the GPU's rows (0.65 alone:
  no gain; 0.65 / 0.65: 78 s). The "GPU wait" column is the signal: 0.4-1.5 s at the best cells.
- The experts' share is half the gain: projections alone at 0.4-0.5 give 76-77 s; adding the experts at the same
  share gives 67-71 s.
- Today's setting (integer kernel, projections 0.38) gave nothing measurable here (86.4 / 102.3 s, at 159 % load;
  its CPU waited 6 s for the integer kernel's rows, so its share was already too high for that kernel).
- 10 threads are slower than 8 on the shared Mac (off 88.3 / 89.5 s; 0.4 / 0.4 74.9 / 106.6 s, the second at 159 %
  load). The 10-thread column was cut to these two cells, and the 0.75 cells were dropped, for the queue.
- Peak footprint is unchanged by the split: 7.97-7.99 GB off, 8.04-8.08 GB on (the GPU's buffers, ~60-90 MB).

### Exactness on that prompt

`mac35/check_oc.sh a40e40 --gpu-frac 0.4 --gpu-experts 0.4`: `dmserve --check` with a batched reference with the
split off (`--ref-no-gpu`, new: single steps over 7,572 positions would take too long; the batched path without the
split equals single steps by `tests/prefill_check.sh`) against the split at 0.4 / 0.4, same process, same flags as
the table: **logit bits differ 0, fixed-state bytes differ 0, KV bytes differ 0, 16 greedy ids identical** (7,572
positions; reference 80.0 s, split 71.8 s; GPU 3,781 + 3,215 GMAC, 0 failed sets).

### (a) What Inkling's prefill would need

- **Projections: yes, as is.** Inkling's in-memory projections are 8-bit affine (AFF8) and its batched chunk already
  packs inputs into XQB (`xqb_set_from`) and runs `gemm_aff8_i16` row jobs (`run_rows_n` over packed inputs from 6
  positions). The GPU kernel computes exactly those bits for AFF8, so each application's set of projections can go
  through `run_split` unchanged: wrap the 8-bit resident, replace the `run_rows_n` calls of the packed path by sets.
  What is per position (depth attention, message gates, norms, convolutions in position order, KV writes, attention)
  stays on the CPU, as the 35B's attention core does.
- **Looped layers: no obstacle.** Each pass of a looped layer is its own application with the same weights: one set per
  application (the weights are wrapped once and read by every pass). The passes are serial (pass k + 1 reads pass k),
  so a set cannot span passes. Inkling's span runs application-major with sub-chunks of 256, so sets are as large as
  the 35B's. Expect the same kind of share as the projections' row above, on its smaller share of the work.
- **Experts: new kernels, in the core.** Inkling's experts go through `core.moe_chunk`, not the model, and are NVFP4
  (gate / up) and MXFP8 (down). NVFP4 is exact the same way (`gemm_nvfp4_i16`: E2M1 codes as small integers, an exact
  integer sub-block dot per 16 inputs, then a float epilogue with the E4M3 scale and the global multiplier): a GPU
  version needs that epilogue's exact operation order and its own bit-for-bit test, and `moe_chunk` would need the
  row split and the position maps of this branch's expert path. MXFP8 is harder: the CPU kernel multiplies fp16
  activations by the fp8 values with FMLAL into fp32 in a fixed order. That sum is not exact, so the GPU must repeat
  the same sequence of fp32 operations per output (a per-thread serial loop, not the matrix units), which is
  slower. Until then Inkling's experts stay on the CPU, and only its projections can take the GPU share.

### (b) A share of a multi-position verify pass (drafting, 2-8 positions)

Not at 2-8 positions. A verify pass is one serial chain of sets: per layer at least two (attention in, attention out)
plus the experts' sets, about 120-160 per pass for the 35B, each waited for before the next layer can start. At 0.15 ms
per round trip on the M4 that is 18-24 ms of waiting per pass, and on the A16 at 0.42 ms 50-67 ms, against a whole
pass of tens of ms. What a set saves is the GPU's share of the CPU's time for that set: a layer's 8-bit attention
products are ~34 MB of weights, and at N <= 8 the CPU is bandwidth-bound on them (about 0.3-0.4 ms per layer on the
M4). Half of that (~0.17 ms) is about one round trip, and the GPU reads the same unified memory, so it adds little
bandwidth. Also the matrix kernel's tiles are 16-32 positions wide, so N = 4 uses 1/8 to 1/4 of each tile. The split
starts to pay once a set's CPU time is several round trips and the CPU is compute-bound: from about N = 32 on the M4
and N = 64 on the A16 (where the A16 bench measured the GPU's batched dot at 50 GMAC/s at N 64, rising to 87 at N 256),
which is prefill, not verify. Keep verify on the CPU. A verify of 32+ positions (long drafts) could use
`prefill_gpu_min` as its threshold.
