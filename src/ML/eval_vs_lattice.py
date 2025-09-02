from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

def find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p/"models").exists(): return p
    return here.parents[2]
ROOT = find_root()

# default input = model test predictions saved by trainer
IN_CSV = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else (ROOT/"models/gbm_parisian_pricer/test_predictions.csv")
OUT_DIR = IN_CSV.parent / "eval_lattice"
OUT_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(IN_CSV)
if "LAT_OUT" not in df.columns:
    raise SystemExit("test_predictions.csv must include LAT_OUT. Re-train using the lattice dataset and save df_te rows.")

# metrics vs lattice
abs_err  = np.abs(df["Pred_OUT"] - df["LAT_OUT"])
mae      = float(abs_err.mean())
rmse     = float(np.sqrt(((df["Pred_OUT"] - df["LAT_OUT"])**2).mean()))
rel_mae  = float((abs_err / np.maximum(df["LAT_OUT"], 1e-12)).mean())

# conditional monotonicity checks (slice by T, sigma, K_rel)
def monotone_violations(group: pd.DataFrame, x_col: str, should="noninc", tol=1e-5) -> int:
    # average predictions at identical x to avoid dx=0
    g = (group[[x_col, "Pred_OUT"]]
         .groupby(x_col, as_index=False)["Pred_OUT"].mean()
         .sort_values(x_col))
    y = g["Pred_OUT"].to_numpy()
    if len(y) < 2:
        return 0
    dy = np.diff(y)
    tol = 1e-10
    if should == "noninc":
        return int(np.sum(dy > tol))      # any increase violates non-increasing
    else:  # "nondec"
        return int(np.sum(dy < -tol))     # any decrease violates non-decreasing

# prepare bins for reporting
df["K_rel"] = df["K"]/df["S0"]
df["B_rel"] = df["B"]/df["S0"]
df["D_frac"]= (df["D"]/df["T"]).clip(0,1)

viol_B = viol_D = 0
n_B = n_D = 0
# Check B non-increasing at fixed D
for (_, g1) in df.groupby(["T","sigma","K_rel","D_frac"]):
    if g1["B_rel"].nunique() >= 3:
        viol_B += monotone_violations(g1, "B_rel", "noninc"); n_B += 1

# Check D non-decreasing at fixed B
for (_, g2) in df.groupby(["T","sigma","K_rel","B_rel"]):
        if g2["D_frac"].nunique() >= 3:
            viol_D += monotone_violations(g2, "D_frac", "nondec"); n_D += 1

summary = dict(
    rows=int(len(df)),
    MAE_vs_Lattice=mae,
    RMSE_vs_Lattice=rmse,
    RelMAE_vs_Lattice=rel_mae,
    Monotone_violations_Brel=viol_B,
    Monotone_checked_groups_Brel=n_B,
    Monotone_violations_Dfrac=viol_D,
    Monotone_checked_groups_Dfrac=n_D,
)
(OUT_DIR/"summary.json").write_text(json.dumps(summary, indent=2))
df.assign(AbsErr=abs_err, RelErr=np.where(df["LAT_OUT"]>0, abs_err/df["LAT_OUT"], 0.0)).sort_values("RelErr", ascending=False)\
  .head(100).to_csv(OUT_DIR/"worst100.csv", index=False)

print("\n=== ML Pricer vs Lattice (test set) ===")
for k,v in summary.items():
    print(f"{k}: {v}")
print(f"\nArtifacts:\n  summary: {OUT_DIR/'summary.json'}\n  worst100: {OUT_DIR/'worst100.csv'}")
