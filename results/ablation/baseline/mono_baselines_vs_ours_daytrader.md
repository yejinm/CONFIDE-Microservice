# Mono baselines vs ours (daytrader)

Generated: (no timestamp; deterministic overwrite)

| Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | pred_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Mono2Micro_Semantic | 0.4457 | 46.88 | 44.00 | 0.2339 | 0.0184 | 0.7458 | 4 | 5 | -1 | 0.0 | with_u | data/processed/fusion/daytrader_pred_Mono2Micro_Semantic.json |
| Bunch_MEM_Structural | 0.4418 | 34.38 | 46.00 | 0.3953 | 0.0224 | 0.7797 | 4 | 5 | -1 | 1.0 | with_u | data/processed/fusion/daytrader_pred_Bunch_MEM_Structural.json |
| COGCN_SimpleFusion | 0.4778 | 37.50 | 49.00 | 0.2932 | -0.0056 | 0.8305 | 4 | 5 | -1 | 0.5 | no_u | data/processed/fusion/daytrader_pred_COGCN_SimpleFusion.json |
| Ours_CAC_noU | 0.4267 | 31.25 | 47.00 | 0.6435 | -0.0180 | 0.7966 | 4 | 5 | -1 | 0.3 | no_u | data/processed/fusion/daytrader_pred_Ours_CAC_noU.json |
| Ours_CAC_withU | 0.5139 | 50.00 | 27.00 | 1.3199 | -0.0061 | 0.4576 | 6 | 5 | +1 | 0.3 | with_u | data/processed/fusion/daytrader_pred_Ours_CAC_withU.json |
| Ours_noDADE_withU | 0.4341 | 40.62 | 44.00 | 1.0440 | 0.0462 | 0.7458 | 5 | 5 | +0 | 0.3 | with_u | data/processed/fusion/daytrader_pred_Ours_noDADE_withU.json |

## Notes
- Mono2Micro_Semantic/Bunch_MEM_Structural/COGCN_SimpleFusion are *equivalent reproductions* under our evidence space by switching matrix inputs (mu) and uncertainty (U).
- Ours_CAC_noU keeps the CAC pipeline identical to Ours_CAC_withU except forcing U≡0 (strict uncertainty ablation under the same mu/cap/K-lock).
- K-lock uses --target_from_gt to keep service granularity comparable.
- IFN/NED/SM/ICP are reported only when dependency matrix exists in data/processed/dependency/<system>_dependency_matrix.json.
- pred_path records the exact prediction JSON used for evaluation (artifact reproducibility).
