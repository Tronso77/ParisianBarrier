# Summary (no repricing)

## Files discovered
- **oof_raw**: `models/gbm_parisian_pricer_cv/oof_predictions.csv`
- **oof_mono**: `models/gbm_parisian_pricer_cv/eval_mono/test_predictions_mono.csv`
- **cv_preds**: `models/gbm_parisian_pricer_cv/preds_cv_mono.csv`
- **sweepD**: `results/fig_sweep_D_cv_mono.csv`
- **slice_B**: `results/slice_B.csv`
- **slice_K**: `results/slice_K.csv`
- **heatmap_binned_oof**: `results/heatmap_B_D_binned_oof.csv`
- **cjy_map**: `results/cjy_map_pilot.csv`
- **mc_convergence**: `results/mc_convergence_paths.csv`
- **mc_varred**: `results/mc_varred.csv`
- **mc_bias_steps**: `results/mc_bias_vs_steps.csv`

## ML_OOF_raw
- rows: 3000
- MAE: 7.96886815644756
- RMSE: 22.23206264284545
- RelMAE_vsVanilla: 0.015326077315813317
- Monotone_viols_Brel: 421
- Monotone_checks_Brel: 500
- Monotone_rate_Brel: 0.842
- Monotone_viols_Dfrac: 510
- Monotone_checks_Dfrac: 600
- Monotone_rate_Dfrac: 0.85
- Monotone_viols_Krel: 3
- Monotone_checks_Krel: 600
- Monotone_rate_Krel: 0.005

## ML_OOF_mono
- rows: 3000
- MAE: 7.921623282437132
- RMSE: 22.083741784100233
- RelMAE_vsVanilla: 0.01465349657429672
- Monotone_viols_Brel: 0
- Monotone_checks_Brel: 500
- Monotone_rate_Brel: 0.0
- Monotone_viols_Dfrac: 0
- Monotone_checks_Dfrac: 600
- Monotone_rate_Dfrac: 0.0
- Monotone_viols_Krel: 0
- Monotone_checks_Krel: 600
- Monotone_rate_Krel: 0.0

## ML_CV_dataset
- rows: 3000
- MAE: 6.306275448777966
- RMSE: 17.713443260720965
- RelMAE_vsVanilla: 0.01167771777830182
- pred_col: Pred_OUT
- Monotone_viols_Brel: 0
- Monotone_checks_Brel: 500
- Monotone_rate_Brel: 0.0
- Monotone_viols_Dfrac: 0
- Monotone_checks_Dfrac: 600
- Monotone_rate_Dfrac: 0.0
- Monotone_viols_Krel: 0
- Monotone_checks_Krel: 600
- Monotone_rate_Krel: 0.0

## Sweep_D
- rows: 16
- MAE_bps: 17.474289514880503
- MaxAbs_bps: 30.231211018021398

## CJY_vs_Lattice
- rows: 120
- relerr_median: 0.5005699604660556
- relerr_mean: 0.6609594625781978
- relerr_p95: 0.9999999999425281
- aw_steh_median: 1712254.572089858
- aw_steh_mean: 786842176.7607839

## MC_convergence
- rows: 5
- lastN: 50000
- MAE_bps: 12.0926082603458

## MC_varred
- best_method: + CV(vanilla)
- best_SE: 0.0003868131395063

## MC_bias_vs_steps
- min_abs_bias_bps: 11.160017133891
- config: grid / BB off
