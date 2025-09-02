# Thesis snapshot (no repricing)

## ML model metrics
- CSV: `results/table_model_metrics.csv`
- Fig: `results/fig_model_mae_rmse.png`
- Fig: `results/fig_monotone_rates.png`

## CJY filtered summary
- Kept 5 of 120 points (keep ratio ≈ 0.04).
- RelErr median: 0.499, mean: 0.31, p95: 0.5
- CSV (filtered points): `results/cjy_filtered_points.csv`
- Binned CSV: `results/cjy_filtered_relerr_by_bins.csv`

## Monte Carlo
- Summary CSV: `results/table_mc_summary.csv`
- Fig: `results/fig_mc_convergence.png`
- Fig: `results/fig_mc_varred.png`
- Fig: `results/fig_mc_bias_vs_steps.png`

## New artifacts (fix)
- Table: `results/table_runtime.tex`
- Table: `results/table_model_metrics.tex`

## New artifacts (CJY & MC)
- New: `results/cjy_filtered_scatter.png`
- New: `results/fig_mc_bb_vs_noBB.png`
- New: `results/fig_mc_mlmc_quick.png`
- New: `results/cjy_filtered_relerr_by_bins.csv`
- New: `results/mc_bb_vs_noBB.csv`
- New: `results/mc_mlmc_quick.csv`
