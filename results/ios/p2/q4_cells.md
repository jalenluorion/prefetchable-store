| cell | tok/s | compute ms | wait ms | MB/tok | hit | weight left out / token (mean, p90) | steps with a drop | picks left out / tok | read+waited / tok | footprint | thermal | warnings | vs 4-bit lossless |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_pitt_bounded | 3.44 | 75 | 216 | 167 | 0.540 | 4.389, 5.960 | 0.740 | 56.8 | 58.7 | 3.26 | nominal -> fair | 0 | first divergence 2, 0.007 equal |
| q4_pitt_bounded_k2tau | 3.43 | 79 | 213 | 190 | 0.628 | 2.441, 3.332 | 0.576 | 33.2 | 64.2 | 3.28 | nominal -> serious | 0 | first divergence 7, 0.043 equal |
| q4_pitt_drop | 4.83 | 81 | 127 | 99 | 0.525 | 5.026, 6.346 | 0.798 | 66.2 | 56.1 | 3.26 | nominal -> serious | 0 | first divergence 2, 0.025 equal |
| q4_pitt_drop_k2tau | 4.16 | 74 | 167 | 181 | 0.632 | 2.535, 3.470 | 0.596 | 34.3 | 63.0 | 3.26 | nominal -> fair | 0 | first divergence 45, 0.182 equal |
| q4_pitt_drop_k4tau | 4.16 | 71 | 169 | 181 | 0.637 | 2.965, 4.098 | 0.632 | 39.6 | 55.6 | 3.26 | nominal -> fair | 0 | first divergence 127, 0.338 equal |
| q4_pitt_lossless | 3.02 | 80 | 251 | 199 | 0.649 | -, - | - | - | - | 3.26 | nominal -> serious | 0 |  |
| q4_pitt_lossless_k2tau | 3.08 | 78 | 247 | 238 | 0.733 | -, - | - | - | - | 3.26 | nominal -> serious | 0 | first divergence none, 1.000 equal |
| q4_pitt_lossless_k4tau | 2.92 | 77 | 265 | 241 | 0.762 | -, - | - | - | - | 3.27 | nominal -> serious | 0 | first divergence none, 1.000 equal |
| q4_code_bounded | 1.94 | 80 | 436 | 327 | 0.267 | 0.059, 0.251 | 0.007 | 0.8 | 184.6 | 2.79 | nominal -> serious | 1 | first divergence none, 1.000 equal |
| q4_code_drop | 3.10 | 79 | 244 | 185 | 0.308 | 6.764, 8.228 | 0.820 | 84.3 | 104.7 | 2.79 | nominal -> serious | 1 | first divergence 11, 0.055 equal |
| q4_code_drop_k2tau | 3.19 | 74 | 239 | 190 | 0.314 | 6.595, 8.072 | 0.811 | 82.1 | 104.4 | 2.86 | nominal -> fair | 1 | first divergence 11, 0.060 equal |
| q4_code_lossless | 1.77 | 79 | 485 | 328 | 0.421 | -, - | - | - | - | 2.79 | fair -> serious | 1 |  |
| q4_code_lossless_k2tau | 1.93 | 87 | 432 | 329 | 0.425 | -, - | - | - | - | 2.85 | fair -> serious | 1 | first divergence none, 1.000 equal |
