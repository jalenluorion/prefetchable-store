## Compute model calibration (Mac M4, the same bench)

| class | Mac GB/s (threads) | ms/token at aff8 resident |
|---|---|---|
| a8qkv | 94.5 (8) | |
| a8out | 94.8 (8) | |
| a8rt | 84.8 (8) | |
| a8head | 95.2 (8) | |
| a4gu | 80.7 (8) | |
| a4dn | 74.9 (8) | |
| attention_proj | | 14.4 |
| shared_expert | | 1.4 |
| router | | 0.3 |
| head | | 5.7 |
| experts | | 7.2 |
| matvec_total | | 28.9 |

Mac predicted 32.5 ms/token (matvec 28.9 + other 3.6) vs measured 31.5-33.6: calibration factor 1.00

bytes/token with aff8 resident: resident 2.06 GB + experts 0.57 GB
bytes/token with aff4 resident: resident 1.10 GB + experts 0.57 GB

## Phone compute (A16), CPU

| class | phone GB/s (threads) | Mac GB/s | ratio |
|---|---|---|---|
| a8qkv | 42.7 (4) | 94.5 | 0.45 |
| a8out | 43.5 (4) | 94.8 | 0.46 |
| a8rt | 41.6 (4) | 84.8 | 0.49 |
| a8head | 45.6 (4) | 95.2 | 0.48 |
| a4qkv | 34.0 (6) | 88.2 | 0.38 |
| a4out | 35.7 (6) | 89.2 | 0.40 |
| a4head | 37.7 (6) | 89.1 | 0.42 |
| a4gu | 32.1 (6) | 80.7 | 0.40 |
| a4dn | 30.9 (6) | 74.9 | 0.41 |

aff4 resident: attention_proj 20.8, shared_expert 2.0, router 0.5, head 7.6, experts 17.9, matvec_total 48.8; other (scaled) 6.1; **CPU compute 54.9 ms/token raw, 55.0 calibrated**
  Metal (exact kernels, wall incl. dispatch): attention_proj 26.9, shared_expert 2.4, router 0.9, head 8.7, experts 21.7, matvec_total 60.7; with other 66.7 ms/token

aff8 resident: attention_proj 31.6, shared_expert 3.1, router 0.5, head 11.8, experts 17.9, matvec_total 65.0; other (scaled) 8.1; **CPU compute 73.1 ms/token raw, 73.2 calibrated**
  Metal (exact kernels, wall incl. dispatch): attention_proj 31.6, shared_expert 3.3, router 0.9, head 12.2, experts 21.7, matvec_total 69.7; with other 77.8 ms/token

CPU+GPU on the experts' kernel: GPU alone 26.1, CPU alone 22.5, together 14.0 + 14.5 = 28.5 GB/s

## Tier table (lossless decode; reads of 1,769,472-byte records; policy = the engine's decayed count / Belady)

| tier | resident | cache GB (slots) | text | policy | hit | misses/token | MB/token | drive GB/s | read ms | compute ms | tok/s cache-only (sum) | tok/s named early (max) | limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 GB today | aff4 1.10 GB | 1.27 (720) | pitt | ours | 0.533 | 150 | 265 | 0.8 | 331 | 55 | 2.59 | 3.02 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | pitt | ours | 0.533 | 150 | 265 | 1.1 | 241 | 55 | 3.38 | 4.16 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | pitt | belady | 0.738 | 84 | 148 | 0.8 | 185 | 55 | 4.16 | 5.39 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | pitt | belady | 0.738 | 84 | 148 | 1.1 | 135 | 55 | 5.27 | 7.42 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | coding | ours | 0.438 | 180 | 318 | 0.8 | 398 | 55 | 2.21 | 2.51 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | coding | ours | 0.438 | 180 | 318 | 1.1 | 289 | 55 | 2.90 | 3.46 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | coding | belady | 0.691 | 99 | 175 | 0.8 | 218 | 55 | 3.66 | 4.58 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | coding | belady | 0.691 | 99 | 175 | 1.1 | 159 | 55 | 4.68 | 6.29 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | agent_text | ours | 0.384 | 197 | 349 | 0.8 | 436 | 55 | 2.04 | 2.29 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | agent_text | ours | 0.384 | 197 | 349 | 1.1 | 317 | 55 | 2.69 | 3.15 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | agent_text | belady | 0.652 | 111 | 197 | 0.8 | 246 | 55 | 3.32 | 4.06 | drive (hardware rate x implementation's miss count) |
| 6 GB today | aff4 1.10 GB | 1.27 (720) | agent_text | belady | 0.652 | 111 | 197 | 1.1 | 179 | 55 | 4.27 | 5.58 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | pitt | ours | 0.662 | 108 | 192 | 0.8 | 239 | 55 | 3.40 | 4.18 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | pitt | ours | 0.662 | 108 | 192 | 1.1 | 174 | 55 | 4.37 | 5.74 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | pitt | belady | 0.810 | 61 | 107 | 0.8 | 134 | 55 | 5.29 | 7.46 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | pitt | belady | 0.810 | 61 | 107 | 1.1 | 98 | 55 | 6.56 | 10.25 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | coding | ours | 0.592 | 131 | 231 | 0.8 | 289 | 55 | 2.91 | 3.46 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | coding | ours | 0.592 | 131 | 231 | 1.1 | 210 | 55 | 3.77 | 4.76 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | coding | belady | 0.777 | 71 | 126 | 0.8 | 158 | 55 | 4.70 | 6.34 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | coding | belady | 0.777 | 71 | 126 | 1.1 | 115 | 55 | 5.89 | 8.72 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | agent_text | ours | 0.538 | 148 | 262 | 0.8 | 327 | 55 | 2.62 | 3.06 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | agent_text | ours | 0.538 | 148 | 262 | 1.1 | 238 | 55 | 3.42 | 4.21 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | agent_text | belady | 0.736 | 84 | 149 | 0.8 | 186 | 55 | 4.14 | 5.36 | drive (hardware rate x implementation's miss count) |
| 6 GB file cache (2.10 GB measured; row 2.0) | aff4 1.10 GB | 2.0 (1130) | agent_text | belady | 0.736 | 84 | 149 | 1.1 | 136 | 55 | 5.25 | 7.37 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | pitt | ours | 0.764 | 76 | 134 | 1.1 | 122 | 55 | 5.67 | 8.23 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | pitt | ours | 0.764 | 76 | 134 | 2.0 | 67 | 55 | 8.21 | 14.96 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | pitt | belady | 0.886 | 36 | 64 | 1.1 | 59 | 55 | 8.80 | 17.05 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | pitt | belady | 0.886 | 36 | 64 | 2.0 | 32 | 55 | 11.46 | 18.19 | compute |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | coding | ours | 0.723 | 89 | 157 | 1.1 | 143 | 55 | 5.06 | 7.02 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | coding | ours | 0.723 | 89 | 157 | 2.0 | 78 | 55 | 7.50 | 12.76 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | coding | belady | 0.858 | 45 | 80 | 1.1 | 73 | 55 | 7.81 | 13.70 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | coding | belady | 0.858 | 45 | 80 | 2.0 | 40 | 55 | 10.51 | 18.19 | compute |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | agent_text | ours | 0.669 | 106 | 188 | 1.1 | 171 | 55 | 4.43 | 5.86 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | agent_text | ours | 0.669 | 106 | 188 | 2.0 | 94 | 55 | 6.72 | 10.66 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | agent_text | belady | 0.826 | 56 | 99 | 1.1 | 90 | 55 | 6.92 | 11.16 | drive (hardware rate x implementation's miss count) |
| 8 GB (assumed +2 GB cache) | aff4 1.10 GB | 3.3 (1864) | agent_text | belady | 0.826 | 56 | 99 | 2.0 | 49 | 55 | 9.59 | 18.19 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | pitt | ours | 0.915 | 27 | 48 | 1.1 | 44 | 73 | 8.54 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | pitt | ours | 0.915 | 27 | 48 | 2.0 | 24 | 73 | 10.28 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | pitt | belady | 0.952 | 16 | 28 | 1.1 | 25 | 73 | 10.19 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | pitt | belady | 0.952 | 16 | 28 | 2.0 | 14 | 73 | 11.50 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | coding | ours | 0.877 | 39 | 70 | 1.1 | 63 | 73 | 7.33 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | coding | ours | 0.877 | 39 | 70 | 2.0 | 35 | 73 | 9.26 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | coding | belady | 0.947 | 17 | 30 | 1.1 | 27 | 73 | 9.94 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | coding | belady | 0.947 | 17 | 30 | 2.0 | 15 | 73 | 11.33 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | agent_text | ours | 0.839 | 51 | 91 | 1.1 | 83 | 73 | 6.42 | 12.10 | drive (hardware rate x implementation's miss count) |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | agent_text | ours | 0.839 | 51 | 91 | 2.0 | 45 | 73 | 8.43 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | agent_text | belady | 0.925 | 24 | 42 | 1.1 | 38 | 73 | 8.96 | 13.67 | compute |
| 12 GB (assumed +6 GB cache, row 6.6) | aff8 2.09 GB | 6.6 (3729) | agent_text | belady | 0.925 | 24 | 42 | 2.0 | 21 | 73 | 10.60 | 13.67 | compute |
