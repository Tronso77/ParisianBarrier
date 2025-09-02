# src/ML/train_parisian_pricer_cv.py
from __future__ import annotations
from pathlib import Path
import sys, json, math
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ---------- root & IO ----------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p/"results").exists(): return p
    return here.parents[2]
ROOT = find_project_root()

if len(sys.argv) > 1 and sys.argv[1].strip():
    IN_CSV = Path(sys.argv[1]).expanduser().resolve()
else:
    IN_CSV = ROOT / "results" / "dataset_parisian_gbm_pricing.csv"

OUT_DIR = ROOT / "models" / "gbm_parisian_pricer_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NFOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EPS_RATIO = 1e-6
TARGET_COL = "LAT_OUT"

# ---------- features (mirrors your trainer) ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = ["Vanilla","S0","K","B","T","D","sigma","r","q"]
    miss = [c for c in req if c not in out.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")
    out["K_rel"] = out.get("moneyness", pd.Series(out["K"]/out["S0"]))
    out["B_rel"] = out.get("barrier_rel", pd.Series(out["B"]/out["S0"]))
    out["D_frac"] = (out["D"]/out["T"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["carry"] = out["r"] - out["q"]
    out["log_moneyness"]   = np.log(np.maximum(out["S0"],1e-12)/np.maximum(out["K"],1e-12))
    out["log_barrier_gap"] = np.log(np.maximum(out["S0"],1e-12)/np.maximum(out["B"],1e-12))
    out["sigma2T"] = (out["sigma"]**2) * out["T"]
    out["T_sigma"] = out["T"] * out["sigma"]
    out["sqrtT"]   = np.sqrt(out["T"])
    out["logD"]    = np.log(out["D_frac"] + 1e-6)
    out["gap_rel"] = (out["S0"] - out["B"]) / np.maximum(out["S0"], 1e-12)
    out["gap_rel_sqrtT"] = out["gap_rel"] * out["sqrtT"]
    return out

FEAT_COLS = [
    "T","sigma","K_rel","B_rel","D_frac",
    "log_moneyness","log_barrier_gap","carry","sigma2T","T_sigma",
    "sqrtT","logD","gap_rel","gap_rel_sqrtT"
]
# monotone: K_rel↓, B_rel↓, D_frac↑
MONO = [(-1 if c=="K_rel" else +1 if c=="D_frac" else -1 if c=="B_rel" else 0) for c in FEAT_COLS]

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS_RATIO, 1 - EPS_RATIO)
    return np.log(p/(1-p))
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0/(1.0 + np.exp(-z))

def price_metrics(y_true_price, y_pred_price, vanilla):
    mae  = mean_absolute_error(y_true_price, y_pred_price)
    rmse = mean_squared_error(y_true_price, y_pred_price, squared=False)
    rel  = (y_pred_price - y_true_price) / np.maximum(vanilla, 1e-12)
    mape = float(np.mean(np.abs(rel)))
    return dict(MAE=float(mae), RMSE=float(rmse), RelMAE=float(mape))

# ---------- helpers ----------
def stratify_bins(df: pd.DataFrame) -> np.ndarray:
    mny = (df["S0"]/df["K"]).clip(1e-9, None)
    mny_bin = pd.cut(mny, bins=[0,0.8,0.95,1.05,1.2,10], labels=False, include_lowest=True)
    T_bin = pd.cut(df["T"], bins=[0,0.25,0.5,1,2,5,100], labels=False, include_lowest=True)
    sig_bin = pd.cut(df["sigma"], bins=[0,0.1,0.2,0.3,0.5,1.0,5.0], labels=False, include_lowest=True)
    key = (mny_bin.fillna(2).astype(int)*100) + (T_bin.fillna(2).astype(int)*10) + sig_bin.fillna(2).astype(int)
    return key.to_numpy()

# ---------- train CV ----------
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(IN_CSV)
    df = pd.read_csv(IN_CSV)
    df = add_features(df)

    vanilla = df["Vanilla"].to_numpy()
    y_px    = df[TARGET_COL].to_numpy()
    ratio   = np.divide(y_px, np.maximum(vanilla, 1e-12))
    z_tgt   = logit(ratio)
    X       = df[FEAT_COLS].to_numpy()

    # stratified folds by coarse T/moneyness/sigma
    y_strat = stratify_bins(df)
    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(df), dtype=float)
    folds_meta = []
    models = []

    for i, (tr, va) in enumerate(skf.split(X, y_strat), start=1):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=20_000,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=200,
            colsample_bytree=0.90,
            subsample=0.90,
            subsample_freq=1,
            reg_lambda=1e-3,
            reg_alpha=0.0,
            monotone_constraints=MONO,
            random_state=SEED+i,
            n_jobs=-1,
            metric="l2",
        )
        model.fit(
            X[tr], z_tgt[tr],
            eval_set=[(X[va], z_tgt[va])],
            callbacks=[early_stopping(stopping_rounds=400, first_metric_only=True),
                       log_evaluation(period=200)]
        )
        best_it = getattr(model, "best_iteration_", None)
        z_hat = model.predict(X[va], num_iteration=best_it)
        p_hat = np.clip(sigmoid(z_hat) * vanilla[va], 0.0, vanilla[va])
        oof[va] = p_hat

        m = price_metrics(y_px[va], p_hat, vanilla[va])
        folds_meta.append(dict(fold=i, best_iteration=int(best_it) if best_it else None, **m))
        joblib.dump(model, OUT_DIR / f"fold_{i}.joblib")
        models.append(model)
        print(f"[fold {i}] {m} (best_iter={best_it})")

    # OOF metrics + save
    m_oof = price_metrics(y_px, oof, vanilla)
    print("\n=== OOF metrics (CV) ===")
    print(m_oof)

    # save artifacts
    (OUT_DIR/"features.json").write_text(json.dumps({"feature_names": FEAT_COLS}, indent=2))
    meta = dict(
        library="lightgbm",
        model_class="LGBMRegressor",
        seed=SEED,
        target_kind="ratio_logit",
        target_desc="logit(LAT_OUT/Vanilla)",
        features=FEAT_COLS,
        monotone_constraints=MONO,
        eps_ratio=EPS_RATIO,
        folds=folds_meta,
        oof_metrics=m_oof,
        nfolds=NFOLDS,
        source=str(IN_CSV),
        train_domain=dict(
            T=[float(df["T"].min()), float(df["T"].max())],
            sigma=[float(df["sigma"].min()), float(df["sigma"].max())],
            K_rel=[float(df["K_rel"].min()), float(df["K_rel"].max())],
            B_rel=[float(df["B_rel"].min()), float(df["B_rel"].max())],
            D_frac=[float(df["D_frac"].min()), float(df["D_frac"].max())],
        ),
    )
    (OUT_DIR/"meta_cv.json").write_text(json.dumps(meta, indent=2))

    oof_df = df.copy()
    oof_df["Pred_OUT"] = oof
    oof_df.to_csv(OUT_DIR/"oof_predictions.csv", index=False)
    print(f"\nSaved -> {OUT_DIR/'fold_*.joblib'}")
    print(f"Saved -> {OUT_DIR/'meta_cv.json'}")
    print(f"Saved -> {OUT_DIR/'features.json'}")
    print(f"Saved -> {OUT_DIR/'oof_predictions.csv'}")

if __name__ == "__main__":
    main()
