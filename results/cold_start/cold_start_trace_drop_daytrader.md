# Cold-start sweep (trace drop) - daytrader

This table evaluates end-to-end performance under sparse runtime traces by randomly dropping traceIds.

| drop_rate | seed | traces_kept | traces_dropped | method | BCubedF1 | MoJoSim | K | GT_K | run_dir |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0.00 | 1337 | 1600 | 0 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.00` |
| 0.00 | 1337 | 1600 | 0 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.00` |
| 0.20 | 1337 | 1280 | 320 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.20` |
| 0.20 | 1337 | 1280 | 320 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.20` |
| 0.40 | 1337 | 960 | 640 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.40` |
| 0.40 | 1337 | 960 | 640 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.40` |
| 0.60 | 1337 | 640 | 960 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.60` |
| 0.60 | 1337 | 640 | 960 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.60` |
| 0.80 | 1337 | 320 | 1280 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.80` |
| 0.80 | 1337 | 320 | 1280 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.80` |
| 0.90 | 1337 | 160 | 1440 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.90` |
| 0.90 | 1337 | 160 | 1440 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.90` |
| 0.95 | 1337 | 80 | 1520 | COGCN_SimpleFusion | 0.4778 | 37.50 | 4 | 5 | `results/cold_start/run_rate_0.95` |
| 0.95 | 1337 | 80 | 1520 | Ours_CAC_withU | 0.5139 | 50.00 | 6 | 5 | `results/cold_start/run_rate_0.95` |
