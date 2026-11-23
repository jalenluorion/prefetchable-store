# Q4C: the calibrated 4-bit resident for Qwen3.6-35B-A3B (2026-11-23)

The calibrated in-memory 4-bit weights (E6, `jobs/babel/q36_q4c_export.py`, run q36_r2_q4c: affine 4-bit, groups of 64
inputs, fp16 scale and bias per group fitted to a calibration-weighted output error) in the engine's 4-bit resident
container, under the plain file's name, as a drop-in for `resident_q4` on memory-limited phones.

## The file

| | |
|---|---|
| container | `deep-memory/local/q36/q4c/derived/resident_q4.dmres1.a603e898ddee.bin` |
| bytes | 1,135,001,600 (= the plain resident_q4) |
| sha256 | `9bcb7685b1913e0beb251016abd6ee04c688323cce4c0f25c9a6182d992b06b9` |
| source export | `deep-memory/local/q36/q4c/q36_resident_q4c_g64.safetensors`, 1,076,622,352 B, sha256 `1b2552a1c92b81a26246be56fe65eb331e846ff8a8f6338826cbfa9363bea783` |
| plain file it replaces | same name, 1,135,001,600 B, sha256 `07d18ccfc459038a393cff498c4821c5133fc0be7d4ed17a91336cb4961c8ba6` |
| Mac reference ids (Pittsburgh, 400) | `results/ios/p3-q4c/mac/q4c_pitt_r1.ids` (= `q4c_pitt_r2.ids`) |

The name keeps the plain file's hash (`a603e898ddee` = resident.safetensors, the source of every non-4-bit tensor), so
the loader, the app's completeness check and the Hub layout take it unchanged; only the manifest's sha256 differs.

## How it was made

`ms_derive resident DIR --bits 4 --from-calibrated FILE --out FILE [--ref PLAIN]` (this branch, `src/format/derive.cpp`
`put_q4_cal`): the same recipe as `resident --bits 4` (same tensors, same order, same writer), except that each of the 251
4-bit matrices takes `BASE.codes` / `.scales` / `.biases` from the export as stored (BASE = the release key without
`.weight`; the export's codes are already two per byte, element 2j in the low nibble, exactly the container's `.q4`
layout; no repacking, no re-quantisation). Shapes are checked against the release's [R, C] and the export's `BASE.shape`;
every export matrix must be used once and nothing else may be in it. `--out` is required (never the model's cache path).

    ./build/ms_derive resident deep-memory/local/q36/model --bits 4 \
      --from-calibrated deep-memory/local/q36/q4c/q36_resident_q4c_g64.safetensors \
      --out deep-memory/local/q36/q4c/derived/resident_q4.dmres1.a603e898ddee.bin \
      --ref <plain resident_q4.dmres1.a603e898ddee.bin>
    # 251 matrices, 1114 tensors, 1.135 GB, 6.6 s; --ref: 361 identical, 753 differ (the 251 x .q4/.s/.b), 0 missing

## Checks without a model run (`python/derive/q4c_verify.py OUT CAL PLAIN`, output in `verify.txt`)

- mapping: the export's metadata `engine_name`, the name derived from its release key, and the container's `.q4`
  stems are the same 251 names; none missing, none extra; dtypes and shapes match ([R, C/2] u8, [R, C/64] f16).
- payloads: all 753 `.q4` / `.s` / `.b` tensors byte-equal to the export's codes / scales / biases.
- dequantised f32 (`code * s + b`, the same numpy expression from both files): bitwise equal for 21 matrices
  (GDN 0.qkv, 0.z, 0.o, 1.qkv, 20.qkv, 21.o; attention 3.q/k/v/o, 23.q, 23.o, 39.q, 39.o; shared expert 0.sg/su/sd,
  17.sg, 38.su, 39.sd; head), 667,942,912 weights.
- every other tensor (361: norms, conv, A_log, dt_bias in f32; in_proj_a / in_proj_b, router, shared-expert gate in
  bf16) bitwise equal to the plain file's; the header (34 cfg lines incl. `resident_bits 4`, 1,114 tensor lines) is
  byte-identical to the plain file's; size equal. The engine's own reader (`DmFile::open`, via `--ref`) parses it.

## Engine check on the Mac (C API gate, `build/dm_gate`, through engine/bench/guard.py)

Exact (lossless), no router, `--mem file --cache-gb 3.3 --threads 8 --readers 4`, Pittsburgh prompt
(`engine/results/stage2/pitt_ids.txt`), 400 tokens, `--resident <file>`, model folder `deep-memory/local/q36/model`
(read only; nothing written there). AC power; all runs rc 0, no guard kill, no swap growth.

| run | file | tok/s (Mac, shared, others' CPU %) | ids |
|---|---|---|---|
| `mac/q4c_pitt_r1` | calibrated | 18.9 (94) | reference for the phone |
| `mac/q4c_pitt_r2` | calibrated | 22.1 (63) | **identical to r1, 400 of 400 (deterministic)**; ids sha256 `902fe132...e1ea0` |
| `mac/plain_pitt` | plain resident_q4 | 5.4 (507: a busy Mac) | identical to P2's `mac/q4_pitt_r1.ids` (this build reproduces the plain gate) |

Rough sanity figure against the 8-bit reference (`engine/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids`), position by
position (meaningless after the first divergence; the engine session measures agreement properly):
calibrated: first divergence at position 1, 3 of 400 equal (0.007); plain: first divergence at position 2, 5 of 400
(0.013). Calibrated vs plain: first divergence at 1, 11 of 400. Both read as fluent, on-topic text (the calibrated one:
"... the construction of Fort Duquesne by the French in 1754 ..."): the calibrated weights are a different 4-bit model,
not a broken one.

## Swap onto the phone (same file name)

The phone keeps the plain 4-bit file under the engine's derived name (in the app's model folder: `derived/` or
`.dmeng/`, wherever `-p2VerifyManifest` / `DmengLaunch.completeness` found it). Replace it in place:

1. Locate the plain file on the phone (the path the 4-bit gate runs used) and confirm it is the plain one: `-p2Hash`
   on it gives `07d18ccf...8ba6`.
2. Delete it (the app's delete path, or `xcrun devicectl device files delete` for a folder devicectl made; folders
   devicectl creates are read-only to the app). Free space needed for the copy: 1.135 GB + the 1.5 GB spare rule.
3. Copy `deep-memory/local/q36/q4c/derived/resident_q4.dmres1.a603e898ddee.bin` to the same path under the same name
   (`xcrun devicectl device copy to ... --source <that file> --destination <same path>`, or the resumable
   `ios/copy_dmeng_to_phone.sh` with this one file); check the size: 1,135,001,600 B.
4. Re-hash on the phone (`-p2Hash`): must be `9bcb7685b1913e0beb251016abd6ee04c688323cce4c0f25c9a6182d992b06b9`.
   If the app's verified-files record holds the old sha256 for this path, it re-verifies (the record keys by path with
   size, mtime and sha256); for the Hub, `moved-store.json`'s phone-small part `derived/resident_q4.dmres1.a603e898ddee.bin`
   carries the new sha256 and the same bytes.
5. Exact gate, as the 4-bit gate of section 0e of P2.md: lossless (Exact), no router, Pittsburgh prompt
   (`engine/results/stage2/pitt_ids.txt`), 400 tokens, the locked slab; compare against
   `results/ios/p3-q4c/mac/q4c_pitt_r1.ids`: pass = 400 of 400 identical. (The old reference
   `results/ios/p2/mac/q4_pitt_r1.ids` is the plain file's and no longer applies.)
