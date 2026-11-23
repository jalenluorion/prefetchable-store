compute ms/token (prediction): burst 115, sustained 185; per class {'experts_nvfp4': 50.2, 'experts_mxfp8': 6.1, 'attn_proj': 23.6, 'o_proj': 8.3, 'head': 10.8, 'msg_gates': 4.1, 'router': 0.7}, other 11.3

## Exact: MB read per token (decode), per text; mean; hit looped / outer (mean)

| policy | pools | hold | GB | pitt | code | edit | agent | mean | hit looped / outer |
|---|---|---|---|---|---|---|---|---|---|
| decay | one | off | 0.8 | 1410 | 1249 | 1355 | 1432 | 1362 | 0.373 / 0.000 |
| decay | one | off | 1.0 | 1214 | 1155 | 1225 | 1311 | 1226 | 0.477 / 0.000 |
| decay | one | off | 1.4 | 948 | 981 | 1037 | 1098 | 1016 | 0.638 / 0.005 |
| decay | one | off | 2.0 | 696 | 800 | 810 | 868 | 794 | 0.794 / 0.037 |
| decay | one | off | 2.5 | 561 | 725 | 699 | 745 | 682 | 0.866 / 0.066 |
| decay | one | off | 3.0 | 473 | 603 | 592 | 653 | 580 | 0.879 / 0.227 |
| decay | share0.2 | off | 0.8 | 1488 | 1357 | 1495 | 1536 | 1469 | 0.289 / 0.000 |
| decay | share0.2 | off | 1.0 | 1409 | 1248 | 1369 | 1421 | 1362 | 0.372 / 0.000 |
| decay | share0.2 | off | 1.4 | 995 | 1062 | 1124 | 1188 | 1092 | 0.535 / 0.077 |
| decay | share0.2 | off | 2.0 | 721 | 832 | 854 | 928 | 834 | 0.700 / 0.141 |
| decay | share0.2 | off | 2.5 | 557 | 700 | 696 | 755 | 677 | 0.797 / 0.187 |
| decay | share0.2 | off | 3.0 | 459 | 584 | 568 | 626 | 559 | 0.857 / 0.259 |
| decay | share0.2 | on | 4.0 | 282 | - | - | - | 282 | 0.919 / 0.651 |
| decay | share0.2 | on | 6.0 | 140 | - | - | - | 140 | 0.960 / 0.833 |
| decay | share0.2 | on | 7.8 | 75 | - | - | - | 75 | 0.978 / 0.915 |
| decay | share0.2 | soft | 0.8 | 1067 | 1149 | 1172 | 1218 | 1152 | 0.535 / 0.000 |
| decay | share0.2 | soft | 1.0 | 982 | 1054 | 1080 | 1133 | 1062 | 0.604 / 0.000 |
| decay | share0.2 | soft | 1.4 | 774 | 899 | 904 | 961 | 884 | 0.697 / 0.073 |
| decay | share0.2 | soft | 2.0 | 626 | 745 | 742 | 790 | 726 | 0.786 / 0.132 |
| decay | share0.2 | soft | 2.5 | 520 | 658 | 647 | 688 | 628 | 0.837 / 0.178 |
| decay | share0.2 | soft | 3.0 | 457 | 574 | 556 | 605 | 548 | 0.867 / 0.255 |
| hazard | one | off | 0.8 | 1046 | 1089 | 1099 | 1162 | 1099 | 0.538 / 0.092 |
| hazard | one | off | 1.0 | 932 | 983 | 1000 | 1055 | 993 | 0.607 / 0.123 |
| hazard | one | off | 1.4 | 762 | 849 | 865 | 913 | 847 | 0.691 / 0.188 |
| hazard | one | off | 2.0 | 578 | 684 | 685 | 726 | 668 | 0.773 / 0.317 |
| hazard | one | off | 2.5 | 477 | 586 | 582 | 619 | 566 | 0.820 / 0.389 |
| hazard | one | off | 3.0 | 397 | 511 | 492 | 522 | 481 | 0.855 / 0.455 |
| hazard | share0.1 | off | 0.8 | 1064 | 1128 | 1134 | 1191 | 1129 | 0.527 / 0.072 |
| hazard | share0.1 | off | 1.0 | 953 | 1005 | 1015 | 1071 | 1011 | 0.601 / 0.110 |
| hazard | share0.1 | off | 1.4 | 771 | 849 | 858 | 909 | 847 | 0.696 / 0.183 |
| hazard | share0.1 | off | 2.0 | 594 | 683 | 684 | 724 | 671 | 0.784 / 0.297 |
| hazard | share0.1 | off | 2.5 | 488 | 588 | 575 | 615 | 567 | 0.837 / 0.365 |
| hazard | share0.1 | off | 3.0 | 407 | 515 | 494 | 523 | 485 | 0.873 / 0.431 |
| hazard | share0.2 | off | 0.8 | 1064 | 1148 | 1149 | 1206 | 1142 | 0.487 / 0.116 |
| hazard | share0.2 | off | 1.0 | 935 | 1024 | 1025 | 1079 | 1016 | 0.563 / 0.160 |
| hazard | share0.2 | off | 1.15 | 869 | 950 | 953 | 1008 | 945 | 0.607 / 0.185 |
| hazard | share0.2 | off | 1.4 | 755 | 850 | 855 | 909 | 842 | 0.665 / 0.232 |
| hazard | share0.2 | off | 1.7 | 655 | 764 | 763 | 814 | 749 | 0.715 / 0.282 |
| hazard | share0.2 | off | 2.0 | 576 | 684 | 681 | 728 | 667 | 0.756 / 0.336 |
| hazard | share0.2 | off | 2.35 | 498 | 606 | 600 | 641 | 586 | 0.795 / 0.392 |
| hazard | share0.2 | off | 2.5 | 470 | 579 | 571 | 610 | 557 | 0.809 / 0.411 |
| hazard | share0.2 | off | 3.0 | 389 | 501 | 487 | 516 | 473 | 0.850 / 0.468 |
| hazard | share0.2 | on | 4.0 | 274 | - | - | - | 274 | 0.921 / 0.665 |
| hazard | share0.2 | on | 6.0 | 135 | - | - | - | 135 | 0.959 / 0.845 |
| hazard | share0.2 | on | 7.8 | 72 | - | - | - | 72 | 0.977 / 0.923 |
| hazard | share0.2 | soft | 0.8 | 1042 | 1113 | 1125 | 1170 | 1112 | 0.522 / 0.078 |
| hazard | share0.2 | soft | 1.0 | 925 | 991 | 1003 | 1054 | 993 | 0.584 / 0.151 |
| hazard | share0.2 | soft | 1.4 | 754 | 843 | 850 | 897 | 836 | 0.669 / 0.232 |
| hazard | share0.2 | soft | 2.0 | 576 | 684 | 681 | 726 | 666 | 0.757 / 0.336 |
| hazard | share0.2 | soft | 2.5 | 470 | 579 | 571 | 610 | 557 | 0.809 / 0.411 |
| hazard | share0.2 | soft | 3.0 | 389 | 501 | 487 | 516 | 473 | 0.850 / 0.468 |
| hazard | share0.3 | off | 0.8 | 1089 | 1185 | 1177 | 1232 | 1171 | 0.443 / 0.147 |
| hazard | share0.3 | off | 1.0 | 959 | 1066 | 1059 | 1121 | 1051 | 0.517 / 0.187 |
| hazard | share0.3 | off | 1.4 | 781 | 878 | 877 | 932 | 867 | 0.627 / 0.255 |
| hazard | share0.3 | off | 2.0 | 595 | 708 | 708 | 754 | 691 | 0.722 / 0.346 |
| hazard | share0.3 | off | 2.5 | 490 | 593 | 594 | 633 | 578 | 0.778 / 0.424 |
| hazard | share0.3 | off | 3.0 | 410 | 507 | 504 | 540 | 490 | 0.821 / 0.482 |
| hazard_caps | share0.2 | off | 0.8 | 1046 | 1152 | 1148 | 1179 | 1131 | 0.453 / 0.240 |
| hazard_caps | share0.2 | off | 1.0 | 914 | 1050 | 1037 | 1073 | 1019 | 0.504 / 0.326 |
| hazard_caps | share0.2 | off | 1.4 | 739 | 875 | 864 | 902 | 845 | 0.594 / 0.429 |
| hazard_caps | share0.2 | off | 2.0 | 560 | 695 | 686 | 725 | 667 | 0.695 / 0.513 |
| hazard_caps | share0.2 | off | 2.5 | 463 | 594 | 585 | 622 | 566 | 0.746 / 0.575 |
| hazard_caps | share0.2 | off | 3.0 | 389 | 512 | 501 | 532 | 484 | 0.789 / 0.624 |

## Fast (drop, weight fraction <= 0.1 left out when missing), agent text only (the one trace with weights)

| policy | pools | GB | MB/token | Exact MB/token | weight left out per token (of 48) | dropped picks per token |
|---|---|---|---|---|---|---|
| decay | one | 0.8 | 796 | 1432 | 8.54 | 114.4 |
| decay | one | 1.0 | 712 | 1311 | 8.00 | 107.8 |
| decay | one | 1.4 | 582 | 1098 | 6.88 | 94.0 |
| decay | one | 2.0 | 438 | 868 | 5.63 | 78.7 |
| decay | one | 2.5 | 350 | 745 | 4.85 | 68.3 |
| decay | one | 3.0 | 283 | 653 | 4.21 | 59.5 |
| decay | one | 4.0 | 178 | - | 3.20 | 45.6 |
| decay | one | 6.0 | 89 | - | 2.01 | 29.5 |
| decay | share0.2 | 0.8 | 841 | 1536 | 9.10 | 120.8 |
| decay | share0.2 | 1.0 | 749 | 1421 | 8.47 | 113.1 |
| decay | share0.2 | 1.4 | 626 | 1188 | 7.41 | 99.8 |
| decay | share0.2 | 2.0 | 470 | 928 | 6.07 | 83.0 |
| decay | share0.2 | 2.5 | 369 | 755 | 5.18 | 71.8 |
| decay | share0.2 | 3.0 | 287 | 626 | 4.43 | 61.7 |
| decay | share0.2 | 4.0 | 180 | - | 3.26 | 46.0 |
| decay | share0.2 | 6.0 | 88 | - | 2.00 | 29.2 |
| hazard | one | 0.8 | 603 | 1162 | 7.97 | 107.7 |
| hazard | one | 1.0 | 544 | 1055 | 7.49 | 101.5 |
| hazard | one | 1.4 | 452 | 913 | 6.63 | 90.7 |
| hazard | one | 2.0 | 349 | 726 | 5.53 | 76.6 |
| hazard | one | 2.5 | 288 | 619 | 4.81 | 67.2 |
| hazard | one | 3.0 | 236 | 522 | 4.18 | 59.0 |
| hazard | one | 4.0 | 163 | - | 3.25 | 46.4 |
| hazard | one | 6.0 | 84 | - | 1.96 | 28.9 |
| hazard | share0.2 | 0.8 | 616 | 1206 | 8.18 | 109.8 |
| hazard | share0.2 | 1.0 | 550 | 1079 | 7.67 | 103.4 |
| hazard | share0.2 | 1.15 | 512 | 1008 | 7.35 | 99.6 |
| hazard | share0.2 | 1.4 | 454 | 909 | 6.83 | 93.0 |
| hazard | share0.2 | 1.7 | 399 | 814 | 6.25 | 85.7 |
| hazard | share0.2 | 2.0 | 352 | 728 | 5.69 | 78.4 |
| hazard | share0.2 | 2.35 | 307 | 641 | 5.12 | 70.8 |
| hazard | share0.2 | 2.5 | 291 | 610 | 4.91 | 68.1 |
| hazard | share0.2 | 3.0 | 240 | 516 | 4.29 | 60.1 |
| hazard | share0.2 | 4.0 | 165 | - | 3.28 | 46.4 |
| hazard | share0.2 | 6.0 | 83 | - | 1.93 | 28.3 |

## Predicted tok/s = 1 / (compute + MB / drive), burst / sustained compute

| mode | policy | pools | GB | MB/token (mean) | 0.7 GB/s | 1.1 GB/s |
|---|---|---|---|---|---|---|
| Exact | decay | one | 0.8 | 1362 | 0.49 / 0.47 | 0.74 / 0.70 |
| Exact | decay | one | 1.0 | 1226 | 0.54 / 0.52 | 0.81 / 0.77 |
| Exact | decay | one | 1.4 | 1016 | 0.64 / 0.61 | 0.96 / 0.90 |
| Exact | decay | one | 2.0 | 794 | 0.80 / 0.76 | 1.20 / 1.10 |
| Exact | decay | one | 2.5 | 682 | 0.92 / 0.86 | 1.36 / 1.24 |
| Exact | decay | one | 3.0 | 580 | 1.06 / 0.99 | 1.56 / 1.40 |
| Fast (agent ratio) | decay | one | 0.8 | 756 | 0.84 / 0.79 | 1.25 / 1.15 |
| Fast (agent ratio) | decay | one | 1.0 | 666 | 0.94 / 0.88 | 1.39 / 1.26 |
| Fast (agent ratio) | decay | one | 1.4 | 539 | 1.13 / 1.05 | 1.65 / 1.48 |
| Fast (agent ratio) | decay | one | 2.0 | 400 | 1.46 / 1.32 | 2.09 / 1.82 |
| Fast (agent ratio) | decay | one | 2.5 | 321 | 1.75 / 1.55 | 2.46 / 2.10 |
| Fast (agent ratio) | decay | one | 3.0 | 252 | 2.11 / 1.83 | 2.91 / 2.41 |
| Exact | decay | share0.2 | 0.8 | 1469 | 0.45 / 0.44 | 0.69 / 0.66 |
| Exact | decay | share0.2 | 1.0 | 1362 | 0.49 / 0.47 | 0.74 / 0.70 |
| Exact | decay | share0.2 | 1.4 | 1092 | 0.60 / 0.57 | 0.90 / 0.85 |
| Exact | decay | share0.2 | 2.0 | 834 | 0.77 / 0.73 | 1.15 / 1.06 |
| Exact | decay | share0.2 | 2.5 | 677 | 0.92 / 0.87 | 1.37 / 1.25 |
| Exact | decay | share0.2 | 3.0 | 559 | 1.09 / 1.02 | 1.60 / 1.44 |
| Fast (agent ratio) | decay | share0.2 | 0.8 | 804 | 0.79 / 0.75 | 1.18 / 1.09 |
| Fast (agent ratio) | decay | share0.2 | 1.0 | 718 | 0.88 / 0.83 | 1.30 / 1.19 |
| Fast (agent ratio) | decay | share0.2 | 1.4 | 575 | 1.07 / 0.99 | 1.57 / 1.41 |
| Fast (agent ratio) | decay | share0.2 | 2.0 | 422 | 1.39 / 1.27 | 2.00 / 1.76 |
| Fast (agent ratio) | decay | share0.2 | 2.5 | 331 | 1.70 / 1.52 | 2.40 / 2.06 |
| Fast (agent ratio) | decay | share0.2 | 3.0 | 256 | 2.08 / 1.81 | 2.88 / 2.39 |
| Exact | hazard | one | 0.8 | 1099 | 0.59 / 0.57 | 0.90 / 0.84 |
| Exact | hazard | one | 1.0 | 993 | 0.65 / 0.62 | 0.98 / 0.92 |
| Exact | hazard | one | 1.4 | 847 | 0.75 / 0.72 | 1.13 / 1.05 |
| Exact | hazard | one | 2.0 | 668 | 0.93 / 0.88 | 1.38 / 1.26 |
| Exact | hazard | one | 2.5 | 566 | 1.08 / 1.01 | 1.59 / 1.43 |
| Exact | hazard | one | 3.0 | 481 | 1.25 / 1.15 | 1.81 / 1.61 |
| Fast (agent ratio) | hazard | one | 0.8 | 570 | 1.08 / 1.00 | 1.58 / 1.42 |
| Fast (agent ratio) | hazard | one | 1.0 | 512 | 1.18 / 1.09 | 1.72 / 1.54 |
| Fast (agent ratio) | hazard | one | 1.4 | 419 | 1.40 / 1.28 | 2.02 / 1.77 |
| Fast (agent ratio) | hazard | one | 2.0 | 321 | 1.74 / 1.55 | 2.46 / 2.10 |
| Fast (agent ratio) | hazard | one | 2.5 | 264 | 2.03 / 1.78 | 2.82 / 2.35 |
| Fast (agent ratio) | hazard | one | 3.0 | 217 | 2.35 / 2.02 | 3.20 / 2.61 |
| Exact | hazard | share0.1 | 0.8 | 1129 | 0.58 / 0.56 | 0.88 / 0.83 |
| Exact | hazard | share0.1 | 1.0 | 1011 | 0.64 / 0.61 | 0.97 / 0.91 |
| Exact | hazard | share0.1 | 1.4 | 847 | 0.75 / 0.72 | 1.13 / 1.05 |
| Exact | hazard | share0.1 | 2.0 | 671 | 0.93 / 0.87 | 1.38 / 1.26 |
| Exact | hazard | share0.1 | 2.5 | 567 | 1.08 / 1.01 | 1.59 / 1.43 |
| Exact | hazard | share0.1 | 3.0 | 485 | 1.24 / 1.14 | 1.80 / 1.60 |
| Exact | hazard | share0.2 | 0.8 | 1142 | 0.57 / 0.55 | 0.87 / 0.82 |
| Exact | hazard | share0.2 | 1.0 | 1016 | 0.64 / 0.61 | 0.96 / 0.90 |
| Exact | hazard | share0.2 | 1.15 | 945 | 0.68 / 0.65 | 1.03 / 0.96 |
| Exact | hazard | share0.2 | 1.4 | 842 | 0.76 / 0.72 | 1.14 / 1.05 |
| Exact | hazard | share0.2 | 1.7 | 749 | 0.84 / 0.80 | 1.26 / 1.15 |
| Exact | hazard | share0.2 | 2.0 | 667 | 0.94 / 0.88 | 1.39 / 1.26 |
| Exact | hazard | share0.2 | 2.35 | 586 | 1.05 / 0.98 | 1.54 / 1.39 |
| Exact | hazard | share0.2 | 2.5 | 557 | 1.10 / 1.02 | 1.61 / 1.44 |
| Exact | hazard | share0.2 | 3.0 | 473 | 1.26 / 1.16 | 1.83 / 1.62 |
| Fast (agent ratio) | hazard | share0.2 | 0.8 | 583 | 1.05 / 0.98 | 1.55 / 1.40 |
| Fast (agent ratio) | hazard | share0.2 | 1.0 | 518 | 1.17 / 1.08 | 1.71 / 1.52 |
| Fast (agent ratio) | hazard | share0.2 | 1.15 | 480 | 1.25 / 1.15 | 1.81 / 1.61 |
| Fast (agent ratio) | hazard | share0.2 | 1.4 | 421 | 1.40 / 1.27 | 2.01 / 1.76 |
| Fast (agent ratio) | hazard | share0.2 | 1.7 | 367 | 1.56 / 1.41 | 2.23 / 1.93 |
| Fast (agent ratio) | hazard | share0.2 | 2.0 | 323 | 1.74 / 1.55 | 2.45 / 2.09 |
| Fast (agent ratio) | hazard | share0.2 | 2.35 | 281 | 1.94 / 1.71 | 2.70 / 2.27 |
| Fast (agent ratio) | hazard | share0.2 | 2.5 | 265 | 2.02 / 1.77 | 2.81 / 2.34 |
| Fast (agent ratio) | hazard | share0.2 | 3.0 | 220 | 2.33 / 2.00 | 3.18 / 2.60 |
| Exact | hazard | share0.3 | 0.8 | 1171 | 0.56 / 0.54 | 0.85 / 0.80 |
| Exact | hazard | share0.3 | 1.0 | 1051 | 0.62 / 0.59 | 0.93 / 0.88 |
| Exact | hazard | share0.3 | 1.4 | 867 | 0.74 / 0.70 | 1.11 / 1.03 |
| Exact | hazard | share0.3 | 2.0 | 691 | 0.91 / 0.85 | 1.35 / 1.23 |
| Exact | hazard | share0.3 | 2.5 | 578 | 1.06 / 0.99 | 1.56 / 1.41 |
| Exact | hazard | share0.3 | 3.0 | 490 | 1.23 / 1.13 | 1.78 / 1.59 |
| Exact | hazard_caps | share0.2 | 0.8 | 1131 | 0.58 / 0.56 | 0.87 / 0.82 |
| Exact | hazard_caps | share0.2 | 1.0 | 1019 | 0.64 / 0.61 | 0.96 / 0.90 |
| Exact | hazard_caps | share0.2 | 1.4 | 845 | 0.76 / 0.72 | 1.13 / 1.05 |
| Exact | hazard_caps | share0.2 | 2.0 | 667 | 0.94 / 0.88 | 1.39 / 1.26 |
| Exact | hazard_caps | share0.2 | 2.5 | 566 | 1.08 / 1.01 | 1.59 / 1.43 |
| Exact | hazard_caps | share0.2 | 3.0 | 484 | 1.24 / 1.14 | 1.80 / 1.60 |
