# Cold-start sweep (trace drop) - plants

This table evaluates end-to-end performance under sparse runtime traces by randomly dropping traceIds.

| drop_rate | seed | traces_kept | traces_dropped | method | BCubedF1 | MoJoSim | K | GT_K | run_dir |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0.00 | 1337 | 1860 | 0 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.00` |
| 0.00 | 1337 | 1860 | 0 | Ours_CAC_withU | 0.4541 | 51.61 | 5 | 4 | `results/cold_start/run_rate_0.00` |
| 0.20 | 1337 | 1488 | 372 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.20` |
| 0.20 | 1337 | 1488 | 372 | Ours_CAC_withU | 0.4564 | 41.94 | 5 | 4 | `results/cold_start/run_rate_0.20` |
| 0.40 | 1337 | 1116 | 744 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.40` |
| 0.40 | 1337 | 1116 | 744 | Ours_CAC_withU | 0.4022 | 38.71 | 5 | 4 | `results/cold_start/run_rate_0.40` |
| 0.60 | 1337 | 744 | 1116 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.60` |
| 0.60 | 1337 | 744 | 1116 | Ours_CAC_withU | 0.4005 | 38.71 | 5 | 4 | `results/cold_start/run_rate_0.60` |
| 0.80 | 1337 | 372 | 1488 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.80` |
| 0.80 | 1337 | 372 | 1488 | Ours_CAC_withU | 0.4963 | 51.61 | 5 | 4 | `results/cold_start/run_rate_0.80` |
| 0.90 | 1337 | 186 | 1674 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.90` |
| 0.90 | 1337 | 186 | 1674 | Ours_CAC_withU | 0.5292 | 48.39 | 4 | 4 | `results/cold_start/run_rate_0.90` |
| 0.95 | 1337 | 93 | 1767 | COGCN_SimpleFusion | 0.7140 | 67.74 | 3 | 4 | `results/cold_start/run_rate_0.95` |
| 0.95 | 1337 | 93 | 1767 | Ours_CAC_withU | 0.5292 | 48.39 | 4 | 4 | `results/cold_start/run_rate_0.95` |
