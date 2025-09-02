# src/ML/posthoc_monotone.py
from __future__ import annotations
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

def monotone_pass(df: pd.DataFrame, group_keys: list[str], x_col: str,
                  increasing: bool, pred_in: str, pred_out: str) -> pd.DataFrame:
    out = df.copy()
    out[pred_out] = out[pred_in]
    for _, g in out.groupby(group_keys, sort=False):
        if g.shape[0] < 2:
            continue
        g = g.sort_values(x_col)
        x = g[x_col].to_numpy()
        y = g[pred_in].to_numpy()
        if np.allclose(x, x[0]):
            continue
        y_mono = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit_transform(x, y)
        out.loc[g.index, pred_out] = y_mono
    return out

def monotone_violations(group: pd.DataFrame, x_col: str, pred_col: str, should: str) -> int:
    gg = (group[[x_col, pred_col]]
          .groupby(x_col, as_index=False)[pred_col].mean()
          .sort_values(x_col))
    if len(gg) < 2:
        return 0
    dy = np.diff(gg[pred_col].to_numpy())
    tol = 1e-10
    if should == "noninc":  # should NOT increase
        return int(np.sum(dy > tol))
    else:                   # should NOT decrease
        return int(np.sum(dy < -tol))

def eval_stats(df: pd.DataFrame, col_pred: str):
    abs_err = np.abs(df[col_pred] - df["LAT_OUT"])
    mae     = float(abs_err.mean())
    rmse    = float(np.sqrt(((df[col_pred] - df["LAT_OUT"])**2).mean()))
    rel_mae = float((abs_err / np.maximum(df["LAT_OUT"], 1e-12)).mean())
    # violations per slices
    viol_B = viol_D = viol_K = n_B = n_D = n_K = 0
    for _, g in df.groupby(["T","sigma"]):
        # check along B_rel for each (K_rel, D_frac)
        for _, g2 in g.groupby(["K_rel","D_frac"]):
            if g2["B_rel"].nunique() >= 3:
                viol_B += monotone_violations(g2, "B_rel", col_pred, "noninc"); n_B += 1
        # check along D_frac for each (K_rel, B_rel)
        for _, g2 in g.groupby(["K_rel","B_rel"]):
            if g2["D_frac"].nunique() >= 3:
                viol_D += monotone_violations(g2, "D_frac", col_pred, "nondec"); n_D += 1
        # check along K_rel for each (B_rel, D_frac)
        for _, g2 in g.groupby(["B_rel","D_frac"]):
            if g2["K_rel"].nunique() >= 3:
                viol_K += monotone_violations(g2, "K_rel", col_pred, "noninc"); n_K += 1
    return dict(
        MAE_vs_Lattice=mae, RMSE_vs_Lattice=rmse, RelMAE_vs_Lattice=rel_mae,
        Monotone_violations_Brel=viol_B, Monotone_checked_groups_Brel=n_B,
        Monotone_violations_Dfrac=viol_D, Monotone_checked_groups_Dfrac=n_D,
        Monotone_violations_Krel=viol_K, Monotone_checked_groups_Krel=n_K,
    )

def main():
    here = Path(__file__).resolve()
    ROOT = next((p for p in [here.parent] + list(here.parents) if (p/"models").exists()), here.parents[2])
    in_csv = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else (ROOT/"models/gbm_parisian_pricer/test_predictions.csv")
    out_dir = in_csv.parent / "eval_mono"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    for col in ["S0","K","B","T","D","sigma","LAT_OUT","Vanilla","Pred_OUT"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column '{col}' in {in_csv}")

    # derived
    df["K_rel"]  = df["K"]/df["S0"]
    df["B_rel"]  = df["B"]/df["S0"]
    df["D_frac"] = (df["D"]/df["T"]).clip(0, 1)
    df["B_rel_r"] = df["B_rel"].round(6)
    df["D_frac_r"] = df["D_frac"].round(6)
    df["K_rel_r"] = df["K_rel"].round(6)

    # iterative alternating projections across axes
    cur = df.copy()
    cur["Pred_cur"] = np.clip(cur["Pred_OUT"], 0.0, cur["Vanilla"])
    best = eval_stats(cur, "Pred_cur")
    best_col = "Pred_cur"

    max_iter = 8
    for it in range(1, max_iter+1):
        # B pass (non-increasing in B_rel, fixing T,sigma,K_rel,D_frac)
        cur = monotone_pass(cur, ["T","sigma","K_rel","D_frac_r"], "B_rel", False, "Pred_cur", "Pred_cur")
        # D pass (non-decreasing in D_frac, fixing T,sigma,K_rel,B_rel)
        cur = monotone_pass(cur, ["T","sigma","K_rel","B_rel_r"], "D_frac", True, "Pred_cur", "Pred_cur")
        # K pass (non-increasing in K_rel, fixing T,sigma,B_rel,D_frac)
        cur = monotone_pass(cur, ["T","sigma","B_rel_r","D_frac_r"], "K_rel", False, "Pred_cur", "Pred_cur")

        # clip to bounds
        cur["Pred_cur"] = np.clip(cur["Pred_cur"], 0.0, cur["Vanilla"])

        stats = eval_stats(cur, "Pred_cur")
        improved = (
            (stats["Monotone_violations_Brel"] < best["Monotone_violations_Brel"]) or
            (stats["Monotone_violations_Dfrac"] < best["Monotone_violations_Dfrac"]) or
            (stats["Monotone_violations_Krel"] < best["Monotone_violations_Krel"])
        )
        print(f"[iter {it}] {stats}")
        best = stats
        best_col = "Pred_cur"
        # simple early stop if no violations remain along all three axes
        if (best["Monotone_violations_Brel"] == 0 and
            best["Monotone_violations_Dfrac"] == 0 and
            best["Monotone_violations_Krel"] == 0):
            break

    # write outputs
    out_csv = out_dir / "test_predictions_mono.csv"
    out = df.copy()
    out["Pred_OUT_mono"] = cur[best_col].to_numpy()
    out.to_csv(out_csv, index=False)

    summary = dict(rows=int(len(df)), **best, source=str(in_csv), output=str(out_csv))
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Post-hoc Monotone Smoothing (B↓, D↑, K↓ • iterative) ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"\nArtifacts:\n  preds_mono: {out_csv}\n  summary: {out_dir/'summary.json'}")

if __name__ == "__main__":
    main()
