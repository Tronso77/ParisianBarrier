from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ---------- robust project root & IO ----------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "results").exists():
            return p
    return here.parents[2]

ROOT = find_project_root()

# CLI: python src/ML/train_parisian_pricer.py results/dataset_parisian_gbm_pricing.csv
if len(sys.argv) > 1 and sys.argv[1].strip():
    IN_CSV = Path(sys.argv[1]).expanduser().resolve()
else:
    IN_CSV = ROOT / "results" / "dataset_parisian_gbm_pricing.csv"

OUT_DIR = ROOT / "models" / "gbm_parisian_pricer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Using project root: {ROOT}")
print(f"[INFO] Reading dataset:   {IN_CSV}")
print(f"[INFO] Model out dir:     {OUT_DIR}")

# ---------- config ----------
SEED       = 42
TEST_FRAC  = 0.15
VAL_FRAC   = 0.12
EPS_RATIO  = 1e-6
TARGET_COL = "LAT_OUT"

# ---------- features ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Vanilla","S0","K","B","T","D","sigma","r","q"]:
        if col not in out.columns:
            raise ValueError(f"Missing column '{col}' in input dataset.")
    out["K_rel"] = out["moneyness"] if "moneyness" in out.columns else out["K"]/out["S0"]
    out["B_rel"] = out["barrier_rel"] if "barrier_rel" in out.columns else out["B"]/out["S0"]
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

# Monotone constraints: K_rel ↓ (-1), B_rel ↓ (-1), D_frac ↑ (+1)
MONO = [(-1 if c=="K_rel" else +1 if c=="D_frac" else -1 if c=="B_rel" else 0) for c in FEAT_COLS]

# ---------- transforms & metrics ----------
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

# ---------- main ----------
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {IN_CSV}")

    df0 = pd.read_csv(IN_CSV)
    df  = add_features(df0)

    vanilla = df["Vanilla"].to_numpy()
    out_px  = df[TARGET_COL].to_numpy()
    ratio   = np.divide(out_px, np.maximum(vanilla, 1e-12))
    z_tgt   = logit(ratio)

    X_all = df[FEAT_COLS].to_numpy()

    X_tr, X_te, z_tr, z_te, v_tr, v_te, df_tr, df_te = train_test_split(
        X_all, z_tgt, vanilla, df, test_size=TEST_FRAC, random_state=SEED
    )
    X_trn, X_val, z_trn, z_val, v_trn, v_val = train_test_split(
        X_tr, z_tr, v_tr, test_size=VAL_FRAC, random_state=SEED
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=10_000,
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
        random_state=SEED,
        n_jobs=-1,
        metric="l2",
    )

    model.fit(
        X_trn, z_trn,
        eval_set=[(X_val, z_val)],
        callbacks=[
            early_stopping(stopping_rounds=300, first_metric_only=True),
            log_evaluation(period=100),
        ],
    )

    def predict_price(X, v):
        z_hat = model.predict(X, num_iteration=getattr(model, "best_iteration_", None))
        y_hat = sigmoid(z_hat)     # OUT/Van ratio
        p_hat = y_hat * v
        return np.clip(p_hat, 0.0, v)

    y_tr_hat = predict_price(X_tr, v_tr)
    y_te_hat = predict_price(X_te, v_te)

    y_te_true = df_te[TARGET_COL].to_numpy()
    m_tr = price_metrics(df_tr[TARGET_COL].to_numpy(), y_tr_hat, v_tr)
    m_te = price_metrics(y_te_true, y_te_hat, v_te)
    best_it = getattr(model, "best_iteration_", None)

    # save artifacts
    joblib.dump(model, OUT_DIR / "model.joblib")
    # store with a consistent key that predictor reads
    (OUT_DIR / "features.json").write_text(json.dumps({"feature_names": FEAT_COLS}, indent=2))

    meta = dict(
        library="lightgbm",
        model_class="LGBMRegressor",
        seed=SEED,
        target_kind="ratio_logit",        # <-- important for predictor
        target_desc="logit(LAT_OUT/Vanilla)",
        features=FEAT_COLS,
        monotone_constraints=MONO,
        eps_ratio=EPS_RATIO,
        train_metrics=m_tr,
        test_metrics=m_te,
        train_size=int(len(z_tr)),
        test_size=int(len(z_te)),
        best_iteration=int(best_it) if best_it is not None else None,
        source=str(IN_CSV),
        train_domain=dict(T=[float(df["T"].min()), float(df["T"].max())],
                          sigma=[float(df["sigma"].min()), float(df["sigma"].max())],
                          K_rel=[float(df["K_rel"].min()), float(df["K_rel"].max())],
                          B_rel=[float(df["B_rel"].min()), float(df["B_rel"].max())],
                          D_frac=[float(df["D_frac"].min()), float(df["D_frac"].max())]),
    )
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    te = df_te.copy()
    te["Pred_OUT"] = y_te_hat
    te["AbsErr"] = np.abs(te["Pred_OUT"] - te[TARGET_COL])
    te["RelErr_vsVanilla"] = te["AbsErr"] / np.maximum(te["Vanilla"], 1e-12)
    te.sort_values("RelErr_vsVanilla", ascending=False).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    print("\n=== Parisian OUT ML Pricer (GBM • LightGBM, monotone K/B/D + logit ratio target) ===")
    print(f"Train metrics: {m_tr}")
    print(f"Test  metrics: {m_te}")
    if best_it is not None:
        print(f"Best iteration: {best_it}")
    print(f"Saved -> {OUT_DIR/'model.joblib'}")
    print(f"Saved -> {OUT_DIR/'meta.json'}")
    print(f"Saved -> {OUT_DIR/'features.json'}")
    print(f"Saved -> {OUT_DIR/'test_predictions.csv'}")

if __name__ == "__main__":
    main()
