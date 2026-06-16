# Cold-start sweep (trace drop) - jpetstore

This table evaluates end-to-end performance under sparse runtime traces by randomly dropping traceIds.

| drop_rate | seed | traces_kept | traces_dropped | method | BCubedF1 | MoJoSim | K | GT_K | run_dir |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0.00 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 38.36 | 3 | 4 | `results/cold_start/run_rate_0.00` |
| 0.00 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.00` |
| 0.20 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 32.88 | 3 | 4 | `results/cold_start/run_rate_0.20` |
| 0.20 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.20` |
| 0.40 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 32.88 | 3 | 4 | `results/cold_start/run_rate_0.40` |
| 0.40 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.40` |
| 0.60 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 32.88 | 3 | 4 | `results/cold_start/run_rate_0.60` |
| 0.60 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.60` |
| 0.80 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 32.88 | 3 | 4 | `results/cold_start/run_rate_0.80` |
| 0.80 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.80` |
| 0.90 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 32.88 | 3 | 4 | `results/cold_start/run_rate_0.90` |
| 0.90 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.90` |
| 0.95 | 1337 | 0 | 0 | COGCN_SimpleFusion | 0.3421 | 38.36 | 3 | 4 | `results/cold_start/run_rate_0.95` |
| 0.95 | 1337 | 0 | 0 | Ours_CAC_withU | 0.4318 | 46.58 | 3 | 4 | `results/cold_start/run_rate_0.95` |
