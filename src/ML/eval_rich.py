from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

def bucketers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["K_rel"]  = out["K"]/out["S0"]
    out["B_rel"]  = out["B"]/out["S0"]
    out["D_frac"] = (df["D"]/df["T"]).clip(0, 1)
    out["mny_bin"] = pd.cut(out["S0"]/out["K"], bins=[0,0.8,0.95,1.05,1.2,10],
                            labels=["deep_OTM","OTM","ATM","ITM","deep_ITM"], include_lowest=True)
    out["T_bin"] = pd.cut(out["T"], bins=[0,0.25,0.5,1,2,5,100],
                          labels=["<3m","3-6m","6-12m","1-2y","2-5y",">5y"], include_lowest=True)
    out["bar_q"] = pd.qcut((out["S0"]-out["B"]).abs(), q=5, duplicates="drop")
    return out

def price_metrics(df: pd.DataFrame, col_pred="Pred_OUT"):
    err = df[col_pred] - df["LAT_OUT"]
    abs_err = err.abs().to_numpy()
    mae  = float(abs_err.mean())
    rmse = float(np.sqrt((err.to_numpy()**2).mean()))
    # Two baselines for RelMAE
    rel_lat = float((abs_err/np.maximum(df["LAT_OUT"].to_numpy(),1e-12)).mean())
    rel_van = float((abs_err/np.maximum(df["Vanilla"].to_numpy(),1e-12)).mean())
    return dict(MAE=mae, RMSE=rmse, RelMAE_vsLattice=rel_lat, RelMAE_vsVanilla=rel_van)

def _mono_viol_counts(df, xcol, col_pred, should):
    g = (df[[xcol, col_pred]].groupby(xcol, as_index=False)[col_pred]
         .mean().sort_values(xcol))
    y = g[col_pred].to_numpy()
    if len(y) < 2:
        return 0, 0  # violations, checks
    dy = np.diff(y); tol = 1e-10
    checks = len(y) - 1
    if should == "noninc":
        viol = int(np.sum(dy > tol))
    else:
        viol = int(np.sum(dy < -tol))
    return viol, checks

def audit_monotone(df: pd.DataFrame, col_pred="Pred_OUT"):
    vB=vD=vK=0
    cB=cD=cK=0
    groups = 0
    for (_, g) in df.groupby(["T","sigma"], observed=True):
        for _, gg in g.groupby(["K_rel","D_frac"], observed=True):
            if gg["B_rel"].nunique() >= 3:
                vb, cb = _mono_viol_counts(gg, "B_rel", col_pred, "noninc"); vB += vb; cB += cb; groups += 1
        for _, gg in g.groupby(["K_rel","B_rel"], observed=True):
            if gg["D_frac"].nunique() >= 3:
                vd, cd = _mono_viol_counts(gg, "D_frac", col_pred, "nondec"); vD += vd; cD += cd; groups += 1
        for _, gg in g.groupby(["B_rel","D_frac"], observed=True):
            if gg["K_rel"].nunique() >= 3:
                vk, ck = _mono_viol_counts(gg, "K_rel", col_pred, "noninc"); vK += vk; cK += ck; groups += 1
    def rate(v, c):
        return (float(v)/float(c)) if c>0 else 0.0
    return dict(
        Monotone_viols_Brel=vB, Monotone_checks_Brel=cB, Monotone_rate_Brel=rate(vB,cB),
        Monotone_viols_Dfrac=vD, Monotone_checks_Dfrac=cD, Monotone_rate_Dfrac=rate(vD,cD),
        Monotone_viols_Krel=vK, Monotone_checks_Krel=cK, Monotone_rate_Krel=rate(vK,cK),
    )

def _safe_group_metrics(df: pd.DataFrame, key: str, outdir: Path):
    g = df.groupby(key, observed=True)
    # Pandas >=2.2 supports include_groups; older versions don't.
    try:
        res = g.apply(lambda gg: pd.Series(price_metrics(gg)), include_groups=False).reset_index()
    except TypeError:
        res = g.apply(lambda gg: pd.Series(price_metrics(gg))).reset_index()
    res.to_csv(outdir/f"metrics_by_{key}.csv", index=False)
    return res

def make_plots(df: pd.DataFrame, outdir: Path, col_pred="Pred_OUT"):
    if not HAS_PLT:
        print("[warn] matplotlib not available; skipping plots")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    # parity
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(df["LAT_OUT"], df[col_pred], s=6, alpha=0.6)
    lim = [0, float(max(df["LAT_OUT"].max(), df[col_pred].max()))]
    plt.plot(lim, lim)
    plt.xlabel("Lattice (LAT_OUT)"); plt.ylabel("ML Pred")
    plt.title("Parity")
    plt.savefig(outdir/"parity.png", bbox_inches="tight"); plt.close()
    # error vs features
    err = (df[col_pred]-df["LAT_OUT"]).abs()
    for f in ["T","sigma","K_rel","B_rel","D_frac"]:
        plt.figure()
        plt.scatter(df[f], err, s=6, alpha=0.6)
        plt.xlabel(f); plt.ylabel("|Error|")
        plt.title(f"Error vs {f}")
        plt.savefig(outdir/f"error_vs_{f}.png", bbox_inches="tight"); plt.close()

def main():
    here = Path(__file__).resolve()
    ROOT = next((p for p in [here.parent] + list(here.parents) if (p/"models").exists()), here.parents[2])
    in_csv = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else (ROOT/"models/gbm_parisian_pricer/test_predictions.csv")
    out_dir = in_csv.parent / "eval_rich"
    out_dir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(in_csv)
    # prefer smoothed predictions if present
    if "Pred_OUT_mono" in df0.columns:
        df0["Pred_OUT"] = df0["Pred_OUT_mono"]
    need = ["S0","K","B","T","D","sigma","LAT_OUT","Vanilla","Pred_OUT"]
    miss = [c for c in need if c not in df0.columns]
    if miss: raise SystemExit(f"Missing column(s) in {in_csv}: {miss}")

    df = bucketers(df0)
    top = price_metrics(df)
    mono = audit_monotone(df)

    # bucketed tables (no empty-category warnings)
    _safe_group_metrics(df, "mny_bin", out_dir)
    _safe_group_metrics(df, "T_bin", out_dir)
    _safe_group_metrics(df, "bar_q", out_dir)

    make_plots(df, out_dir)

    summary = dict(rows=int(len(df)), **top, **mono,
                   outputs=dict(dir=str(out_dir),
                                parity_png=str(out_dir/"parity.png")))
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Rich Eval ===")
    for k,v in summary.items(): print(f"{k}: {v}")
    print(f"\nArtifacts in: {out_dir}")

if __name__ == "__main__":
    main()
