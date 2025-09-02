# src/ML/eval_parisian_pricer.py
from __future__ import annotations
from pathlib import Path
import sys, json, math
import numpy as np
import pandas as pd

# --------- locate project root & IO defaults ----------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "results").exists():
            return p
    return here.parents[2]

ROOT = find_project_root()
DEFAULT_TEST_PREDS = ROOT / "models" / "gbm_parisian_pricer" / "test_predictions.csv"
DEFAULT_META       = ROOT / "models" / "gbm_parisian_pricer" / "meta.json"
OUT_DIR            = ROOT / "models" / "gbm_parisian_pricer" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CLI override for the test preds
TEST_PREDS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_TEST_PREDS
META_PATH  = DEFAULT_META if DEFAULT_META.exists() else None

# --------- helpers ----------
def fmt_pct(x):
    return f"{100.0*float(x):.2f}%"

def must_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

def monotonic_violations(df: pd.DataFrame, group_keys: list[str], x_key: str, y_key: str, sign: int, tol=1e-9):
    """
    Count local monotonicity violations within groups.
    sign = +1 expects y increasing with x; sign = -1 expects y decreasing with x.
    Returns (violations, comparisons).
    """
    v = 0
    c = 0
    for _, g in df.sort_values(group_keys + [x_key]).groupby(group_keys, dropna=False):
        if len(g) < 2: 
            continue
        x = g[x_key].to_numpy()
        y = g[y_key].to_numpy()
        dx = np.diff(x)
        dy = np.diff(y)
        # only compare where x actually increases
        mask = dx > 0
        dv = dy[mask] * sign  # should be >= -tol
        v += int(np.sum(dv < -tol))
        c += int(np.sum(mask))
    return v, c

# --------- load ----------
if not TEST_PREDS.exists():
    raise FileNotFoundError(f"Could not find test predictions at:\n  {TEST_PREDS}")

df = pd.read_csv(TEST_PREDS)

req = ["Vanilla","Pred_OUT","LAT_OUT","T","sigma","K","S0","B","D"]
must_cols(df, req)

# reconstruct engineered features if present in your training CSV
df["K_rel"]  = df["K"] / df["S0"]
df["B_rel"]  = df["B"] / df["S0"]
df["D_frac"] = (df["D"] / df["T"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

# --------- bounds & core errors ----------
df["Pred_IN"]  = df["Vanilla"] - df["Pred_OUT"]
df["AbsErr"]   = np.abs(df["Pred_OUT"] - df["LAT_OUT"])
df["RelErr_vsVanilla"] = df["AbsErr"] / np.maximum(df["Vanilla"], 1e-12)

bounds_flag = (df["Pred_OUT"] < -1e-12) | (df["Pred_OUT"] - df["Vanilla"] > 1e-12) | (df["Pred_IN"] < -1e-12)
bounds_rate = bounds_flag.mean()

mae  = float(df["AbsErr"].mean())
rmse = float(np.sqrt(np.mean((df["Pred_OUT"] - df["LAT_OUT"])**2)))
rel_mae = float(df["RelErr_vsVanilla"].mean())

# --------- MC z-scores (optional) ----------
z_stats = None
if {"MC_OUT","SE_OUT"}.issubset(df.columns):
    mask = (df["SE_OUT"] > 0)
    if mask.any():
        z = (df.loc[mask,"MC_OUT"] - df.loc[mask,"Pred_OUT"]) / df.loc[mask,"SE_OUT"]
        z_stats = dict(
            count=int(mask.sum()),
            mean=float(z.mean()),
            std=float(z.std(ddof=1)),
            frac_gt2=float((z.abs() > 2).mean()),
            frac_gt3=float((z.abs() > 3).mean()),
        )
        df.loc[mask, "Z_MC_vs_Pred"] = z
    else:
        z_stats = dict(count=0, mean=np.nan, std=np.nan, frac_gt2=np.nan, frac_gt3=np.nan)

# --------- monotonicity checks (logic sanity) ----------
# Expectations for OUT (down, cumulative, call):
#  B_rel: higher barrier -> OUT ↓ (non-increasing)  => sign = -1
#  D_frac: larger window (harder to KO) -> OUT ↑    => sign = +1
#  sigma: higher vol -> OUT ↓                       => sign = -1
#  K_rel: higher strike -> OUT ↓ (call)             => sign = -1

group_keys = ["T","sigma","K_rel"]  # hold these fixed when checking vs B_rel and D_frac
vB, cB = monotonic_violations(df, group_keys, "B_rel",  "Pred_OUT", sign=-1)
vD, cD = monotonic_violations(df, group_keys, "D_frac", "Pred_OUT", sign=+1)

# --------- sliced error heatmaps ----------
# B_rel × D_frac heatmap of mean relative error
B_bins = [0.0, 0.7, 0.85, 0.9, 0.95, 0.98, 1.2]
D_bins = [0.0, 0.02, 0.05, 0.1, 0.2, 1.0]
df["B_rel_bin"] = pd.cut(df["B_rel"].clip(B_bins[0]+1e-6, B_bins[-1]-1e-6), bins=B_bins)
df["D_frac_bin"] = pd.cut(df["D_frac"].clip(D_bins[0]+1e-6, D_bins[-1]-1e-6), bins=D_bins)
heat_BD = df.pivot_table(index="B_rel_bin", columns="D_frac_bin", values="RelErr_vsVanilla", aggfunc="mean")
heat_BD.to_csv(OUT_DIR / "heatmap_relerr_Brel_by_Dfrac.csv")

# sigma × T heatmap
T_bins = sorted(df["T"].unique().tolist())
S_bins = sorted(df["sigma"].unique().tolist())
heat_ST = df.pivot_table(index="sigma", columns="T", values="RelErr_vsVanilla", aggfunc="mean")
heat_ST.to_csv(OUT_DIR / "heatmap_relerr_sigma_by_T.csv")

# --------- worst cases dump ----------
worst = df.sort_values("RelErr_vsVanilla", ascending=False).head(50)
worst.to_csv(OUT_DIR / "worst50.csv", index=False)

# --------- summary JSON ----------
summary = dict(
    n_rows=int(len(df)),
    mae=mae,
    rmse=rmse,
    rel_mae=rel_mae,
    bounds_rate=float(bounds_rate),
    monotonic_checks=dict(
        B_rel=dict(violations=int(vB), comparisons=int(cB), rate=float(vB/max(cB,1))),
        D_frac=dict(violations=int(vD), comparisons=int(cD), rate=float(vD/max(cD,1))),
    ),
    z_stats=z_stats,
    files=dict(
        test_preds=str(TEST_PREDS),
        summary=str(OUT_DIR / "summary.json"),
        worst50=str(OUT_DIR / "worst50.csv"),
        heatmap_Brel_Dfrac=str(OUT_DIR / "heatmap_relerr_Brel_by_Dfrac.csv"),
        heatmap_sigma_T=str(OUT_DIR / "heatmap_relerr_sigma_by_T.csv"),
    ),
)
(Path(OUT_DIR) / "summary.json").write_text(json.dumps(summary, indent=2))

# --------- pretty print ----------
print("\n=== Parisian OUT ML Pricer — Evaluation ===")
print(f"Rows: {summary['n_rows']}")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"RelMAE vs Vanilla: {fmt_pct(rel_mae)}")
print(f"Bound violations (should be 0): {fmt_pct(bounds_rate)}")

print("\n[Monotonic checks on Pred_OUT]")
print(f"  vs B_rel (should be non-increasing): violations {vB}/{cB}  ({fmt_pct(vB/max(cB,1))})")
print(f"  vs D_frac (should be non-decreasing): violations {vD}/{cD}  ({fmt_pct(vD/max(cD,1))})")

if z_stats is not None:
    print("\n[MC z-score vs Pred_OUT]")
    print(f"  count={z_stats['count']}  mean={z_stats['mean']:.3f}  std={z_stats['std']:.3f}  "
          f"|z|>2={fmt_pct(z_stats['frac_gt2'])}  |z|>3={fmt_pct(z_stats['frac_gt3'])}")

print("\nArtifacts:")
for k, v in summary["files"].items():
    print(f"  {k}: {v}")
print()
