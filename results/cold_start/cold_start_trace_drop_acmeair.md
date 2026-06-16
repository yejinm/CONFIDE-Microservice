# Cold-start sweep (trace drop) - acmeair

This table evaluates end-to-end performance under sparse runtime traces by randomly dropping traceIds.

| drop_rate | seed | traces_kept | traces_dropped | method | BCubedF1 | MoJoSim | K | GT_K | run_dir |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0.00 | 1337 | 580 | 0 | COGCN_SimpleFusion | 0.3929 | 21.74 | 3 | 4 | `results/cold_start/run_rate_0.00` |
| 0.00 | 1337 | 580 | 0 | Ours_CAC_withU | 0.4132 | 47.83 | 3 | 4 | `results/cold_start/run_rate_0.00` |
| 0.20 | 1337 | 464 | 116 | COGCN_SimpleFusion | 0.3929 | 30.43 | 3 | 4 | `results/cold_start/run_rate_0.20` |
| 0.20 | 1337 | 464 | 116 | Ours_CAC_withU | 0.4344 | 47.83 | 3 | 4 | `results/cold_start/run_rate_0.20` |
| 0.40 | 1337 | 348 | 232 | COGCN_SimpleFusion | 0.3929 | 43.48 | 3 | 4 | `results/cold_start/run_rate_0.40` |
| 0.40 | 1337 | 348 | 232 | Ours_CAC_withU | 0.3923 | 39.13 | 3 | 4 | `results/cold_start/run_rate_0.40` |
| 0.60 | 1337 | 232 | 348 | COGCN_SimpleFusion | 0.3929 | 26.09 | 3 | 4 | `results/cold_start/run_rate_0.60` |
| 0.60 | 1337 | 232 | 348 | Ours_CAC_withU | 0.4344 | 47.83 | 3 | 4 | `results/cold_start/run_rate_0.60` |
| 0.80 | 1337 | 116 | 464 | COGCN_SimpleFusion | 0.3929 | 34.78 | 3 | 4 | `results/cold_start/run_rate_0.80` |
| 0.80 | 1337 | 116 | 464 | Ours_CAC_withU | 0.4053 | 43.48 | 3 | 4 | `results/cold_start/run_rate_0.80` |
| 0.90 | 1337 | 58 | 522 | COGCN_SimpleFusion | 0.3929 | 30.43 | 3 | 4 | `results/cold_start/run_rate_0.90` |
| 0.90 | 1337 | 58 | 522 | Ours_CAC_withU | 0.3671 | 39.13 | 3 | 4 | `results/cold_start/run_rate_0.90` |
| 0.95 | 1337 | 29 | 551 | COGCN_SimpleFusion | 0.3929 | 21.74 | 3 | 4 | `results/cold_start/run_rate_0.95` |
| 0.95 | 1337 | 29 | 551 | Ours_CAC_withU | 0.3671 | 39.13 | 3 | 4 | `results/cold_start/run_rate_0.95` |
