# Mono baselines vs ours (ALL systems)

Generated: (no timestamp; deterministic overwrite)

| System | Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| acmeair | Bunch_MEM_Structural | 0.3851 | 47.83 | 28.00 | 0.3074 | -0.0311 | 0.6364 | 3 | 4 | -1 | 1.0 | with_u | 0.05 |
| acmeair | COGCN_SimpleFusion | 0.3929 | 21.74 | 31.00 | 0.3074 | -0.0288 | 0.7045 | 3 | 4 | -1 | 0.5 | no_u | 0.05 |
| acmeair | Mono2Micro_Semantic | 0.3421 | 34.78 | 31.00 | 0.1627 | -0.0141 | 0.7045 | 3 | 4 | -1 | 0.0 | with_u | 0.05 |
| acmeair | Ours_CAC_noU | 0.3938 | 39.13 | 26.00 | 0.4032 | 0.0121 | 0.5909 | 3 | 4 | -1 | - | no_u | 0.05 |
| acmeair | Ours_CAC_withU | 0.3923 | 39.13 | 25.00 | 0.5033 | -0.0356 | 0.5682 | 3 | 4 | -1 | - | with_u | 0.05 |
| daytrader | Bunch_MEM_Structural | 0.4418 | 34.38 | 46.00 | 0.3953 | 0.0224 | 0.7797 | 4 | 5 | -1 | 1.0 | with_u | 0.18 |
| daytrader | COGCN_SimpleFusion | 0.4778 | 37.50 | 49.00 | 0.2932 | -0.0056 | 0.8305 | 4 | 5 | -1 | 0.5 | no_u | 0.18 |
| daytrader | Mono2Micro_Semantic | 0.4934 | 34.38 | 50.00 | 0.4593 | 0.0010 | 0.8475 | 5 | 5 | 0 | 0.0 | with_u | 0.18 |
| daytrader | Ours_CAC_noU | 0.4267 | 31.25 | 47.00 | 0.6435 | -0.0180 | 0.7966 | 4 | 5 | -1 | 0.3 | no_u | 0.18 |
| daytrader | Ours_CAC_withU | 0.5139 | 50.00 | 27.00 | 1.3199 | -0.0061 | 0.4576 | 6 | 5 | 1 | 0.3 | with_u | 0.18 |
| jpetstore | Bunch_MEM_Structural | 0.2953 | 27.40 | 67.00 | 0.1913 | 0.0116 | 0.7283 | 4 | 4 | 0 | 1.0 | with_u | 0.14 |
| jpetstore | COGCN_SimpleFusion | 0.3421 | 32.88 | 42.00 | 0.2473 | 0.0159 | 0.4565 | 3 | 4 | -1 | 0.5 | no_u | 0.14 |
| jpetstore | Mono2Micro_Semantic | 0.3381 | 39.73 | 49.00 | 0.1356 | 0.0156 | 0.5326 | 3 | 4 | -1 | 0.0 | with_u | 0.14 |
| jpetstore | Ours_CAC_noU | 0.4077 | 46.58 | 40.00 | 0.3889 | 0.0215 | 0.4348 | 3 | 4 | -1 | 0.1 | no_u | 0.14 |
| jpetstore | Ours_CAC_withU | 0.4318 | 46.58 | 52.00 | 0.2183 | 0.0013 | 0.5652 | 3 | 4 | -1 | 0.1 | with_u | 0.14 |
| plants | Bunch_MEM_Structural | 0.6012 | 61.29 | 49.00 | 0.2540 | -0.0726 | 1.0000 | 3 | 4 | -1 | 1.0 | with_u | 0.22 |
| plants | COGCN_SimpleFusion | 0.7140 | 67.74 | 35.00 | 0.6387 | -0.0264 | 0.7143 | 3 | 4 | -1 | 0.5 | no_u | 0.22 |
| plants | Mono2Micro_Semantic | 0.5833 | 54.84 | 31.00 | 0.5080 | -0.0094 | 0.6327 | 3 | 4 | -1 | 0.0 | with_u | 0.22 |
| plants | Ours_CAC_noU | 0.4094 | 45.16 | 31.00 | 0.3193 | 0.0290 | 0.6327 | 3 | 4 | -1 | - | no_u | 0.22 |
| plants | Ours_CAC_withU | 0.4094 | 45.16 | 31.00 | 0.3193 | 0.0290 | 0.6327 | 3 | 4 | -1 | - | with_u | 0.22 |

## All-Systems Summary (mean ± std)

| Method | BCubedF1 | IFN | ICP | K | K-Diff |
|---|---:|---:|---:|---:|---:|
| Bunch_MEM_Structural | 0.4308 ± 0.1286 | 47.50 ± 15.97 | 0.7861 ± 0.1544 | 4 ± 0.58 | -1 ± 0.50 |
| COGCN_SimpleFusion | 0.4817 ± 0.1647 | 39.25 ± 7.93 | 0.6765 ± 0.1574 | 3 ± 0.50 | -1 ± 0.00 |
| Mono2Micro_Semantic | 0.4392 ± 0.1202 | 40.25 ± 10.69 | 0.6793 ± 0.1324 | 4 ± 1.00 | -1 ± 0.50 |
| Ours_CAC_noU | 0.4094 ± 0.0135 | 36.00 ± 9.35 | 0.6137 ± 0.1487 | 3 ± 0.50 | -1 ± 0.00 |
| Ours_CAC_withU | 0.4368 ± 0.0539 | 33.75 ± 12.42 | 0.5559 ± 0.0725 | 4 ± 1.50 | 0 ± 1.00 |

**Note**: This summary supports reviewer-facing comparisons of overall advantage and stability (mean±std).
We recommend reporting: per-system results under K-lock + all-systems mean±std.
