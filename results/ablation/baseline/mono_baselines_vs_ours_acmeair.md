# Mono baselines vs ours (acmeair)

Generated: (no timestamp; deterministic overwrite)

| Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | pred_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Mono2Micro_Semantic | 0.3421 | 34.78 | 31.00 | 0.1627 | -0.0141 | 0.7045 | 3 | 4 | -1 | 0.0 | with_u | data/processed/fusion/acmeair_pred_Mono2Micro_Semantic.json |
| Bunch_MEM_Structural | 0.3851 | 47.83 | 28.00 | 0.3074 | -0.0311 | 0.6364 | 3 | 4 | -1 | 1.0 | with_u | data/processed/fusion/acmeair_pred_Bunch_MEM_Structural.json |
| COGCN_SimpleFusion | 0.3929 | 21.74 | 31.00 | 0.3074 | -0.0288 | 0.7045 | 3 | 4 | -1 | 0.5 | no_u | data/processed/fusion/acmeair_pred_COGCN_SimpleFusion.json |
| Ours_CAC_noU | 0.3938 | 39.13 | 26.00 | 0.4032 | 0.0121 | 0.5909 | 3 | 4 | -1 | - | no_u | data/processed/fusion/acmeair_pred_Ours_CAC_noU.json |
| Ours_CAC_withU | 0.3923 | 39.13 | 25.00 | 0.5033 | -0.0356 | 0.5682 | 3 | 4 | -1 | - | with_u | data/processed/fusion/acmeair_pred_Ours_CAC_withU.json |

## Notes
- Mono2Micro_Semantic/Bunch_MEM_Structural/COGCN_SimpleFusion are *equivalent reproductions* under our evidence space by switching matrix inputs (mu) and uncertainty (U).
- Ours_CAC_noU keeps the CAC pipeline identical to Ours_CAC_withU except forcing U≡0 (strict uncertainty ablation under the same mu/cap/K-lock).
- K-lock uses --target_from_gt to keep service granularity comparable.
- IFN/NED/SM/ICP are reported only when dependency matrix exists in data/processed/dependency/<system>_dependency_matrix.json.
- pred_path records the exact prediction JSON used for evaluation (artifact reproducibility).
