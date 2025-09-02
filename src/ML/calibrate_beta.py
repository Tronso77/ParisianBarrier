# src/ML/calibrate_beta.py
from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

def bucketers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["K_rel"]  = out["K"]/out["S0"]
    out["B_rel"]  = out["B"]/out["S0"]
    out["D_frac"] = (out["D"]/out["T"]).clip(0,1)
    out["mny_bin"] = pd.cut(out["S0"]/out["K"], bins=[0,0.8,0.95,1.05,1.2,10],
                            labels=["deep_OTM","OTM","ATM","ITM","deep_ITM"], include_lowest=True)
    out["T_bin"] = pd.cut(out["T"], bins=[0,0.25,0.5,1,2,5,100],
                          labels=["<3m","3-6m","6-12m","1-2y","2-5y",">5y"], include_lowest=True)
    # barrier proximity bucket (optional; can tighten later)
    out["bar_q"] = pd.qcut((out["S0"]-out["B"]).abs(), q=5, duplicates="drop")
    return out

def fit_betas(df: pd.DataFrame, pred_col="Pred_OUT", y_col="LAT_OUT", by=("T_bin","mny_bin")):
    dfb = bucketers(df)
    params = {}
    for key, g in dfb.groupby(list(by), observed=True):
        x = g[pred_col].to_numpy().astype(float)
        y = g[y_col].to_numpy().astype(float)
        # beta* x ~ y -> LS solution (x⋅y)/(x⋅x), constrained to [0,1] to preserve upper bound.
        den = float(np.dot(x, x))
        if den <= 0 or len(g)==0:
            beta = 1.0
        else:
            beta = float(np.dot(x, y)/den)
            beta = max(0.0, min(1.0, beta))
        params[str(tuple(key if isinstance(key, tuple) else (key,)))] = beta
    return params, list(by)

def apply_betas(df: pd.DataFrame, params: dict, keys: list[str], pred_col="Pred_OUT"):
    dfb = bucketers(df)
    # Map betas to rows
    betas = []
    for _, row in dfb.iterrows():
        key = tuple(row[k] for k in keys)
        key_s = str(tuple(key))
        b = params.get(key_s, 1.0)
        betas.append(float(b))
    betas = np.array(betas, dtype=float)
    out = df.copy()
    out["Pred_OUT_orig"] = out[pred_col].to_numpy()
    out[pred_col] = np.clip(out["Pred_OUT_orig"] * betas, 0.0, np.maximum(out.get("Vanilla", out["Pred_OUT_orig"]), 0.0))
    return out

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def load_json(path: Path):
    return json.loads(Path(path).read_text())

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python src/ML/calibrate_beta.py fit <in_csv> [out_dir]")
        print("  python src/ML/calibrate_beta.py apply <in_csv> <params_json> [out_csv]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "fit":
        in_csv = Path(sys.argv[2]).resolve()
        out_dir = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else in_csv.parent / "calib"
        df = pd.read_csv(in_csv)
        need = ["Pred_OUT","LAT_OUT","Vanilla","S0","K","B","T","D","sigma"]
        missing = [c for c in need if c not in df.columns]
        if missing: raise SystemExit(f"Missing columns in {in_csv}: {missing}")
        params, keys = fit_betas(df, pred_col="Pred_OUT", y_col="LAT_OUT", by=("T_bin","mny_bin"))
        meta = dict(keys=keys, params=params, source=str(in_csv))
        save_json(meta, out_dir/"beta_params.json")
        print(json.dumps({"saved": str(out_dir/"beta_params.json"), "n_buckets": len(params)}, indent=2))

    elif mode == "apply":
        in_csv = Path(sys.argv[2]).resolve()
        params_json = Path(sys.argv[3]).resolve()
        out_csv = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else in_csv.with_name(in_csv.stem + "_cal.csv")
        df = pd.read_csv(in_csv)
        meta = load_json(params_json)
        out = apply_betas(df, params=meta["params"], keys=meta["keys"], pred_col="Pred_OUT")
        out.to_csv(out_csv, index=False)
        # quick print of MAE shift
        err0 = (df["Pred_OUT"] - df["LAT_OUT"]).abs()
        err1 = (out["Pred_OUT"] - out["LAT_OUT"]).abs()
        s = dict(rows=int(len(out)),
                 MAE_before=float(err0.mean()), MAE_after=float(err1.mean()),
                 RMSE_before=float(np.sqrt(((df["Pred_OUT"]-df["LAT_OUT"])**2).mean())),
                 RMSE_after=float(np.sqrt(((out["Pred_OUT"]-out["LAT_OUT"])**2).mean())),
                 output=str(out_csv))
        print(json.dumps(s, indent=2))
    else:
        raise SystemExit("mode must be 'fit' or 'apply'")

if __name__ == "__main__":
    main()
