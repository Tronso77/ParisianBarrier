# src/ML/predict_parisian_cv.py
from __future__ import annotations
from pathlib import Path
import sys, json, math, glob, os
import numpy as np
import pandas as pd
import joblib

# ---------- project root ----------
def find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "models").exists():
            return p
    return here.parents[2]

ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # allow "src.*" imports

# ---------- pricing deps ----------
try:
    from src.pricing.parisian_lattice import LatticeParams, price_parisian_binomial
    from src.pricing.vanilla import bs_price
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = find_root()
    for _p in (str(_ROOT), str(_ROOT/_Path("src"))):
        if _p not in _sys.path:
            _sys.path.append(_p)
    try:
        from src.pricing.parisian_lattice import LatticeParams, price_parisian_binomial
        from src.pricing.vanilla import bs_price
    except ModuleNotFoundError:
        from pricing.parisian_lattice import LatticeParams, price_parisian_binomial
        from pricing.vanilla import bs_price

def lattice_out_price(S0,K,T,r,q,sigma,B,D,steps=800) -> float:
    p = LatticeParams(S0=S0,K=K,T=T,r=r,q=q,sigma=sigma,
                      B=B,D=D,steps=steps,option_type="call",
                      inout="out",direction="down",style="cumulative",
                      in_via_parity=True)
    return float(price_parisian_binomial(p))

# ---------- feature utils ----------
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0/(1.0+np.exp(-x))

def build_all_features(S0,K,T,r,q,sigma,B,D) -> dict:
    K_rel  = K/max(S0,1e-12)
    B_rel  = B/max(S0,1e-12)
    D_frac = (D/max(T,1e-12))
    carry  = r - q
    log_m  = math.log(max(S0,1e-12)/max(K,1e-12))
    log_bg = math.log(max(S0,1e-12)/max(B,1e-12))
    sqrtT  = math.sqrt(max(T,1e-12))
    sigma2T= (sigma**2)*T
    T_sigma= T*sigma
    logD   = math.log(max(D_frac, 0.0) + 1e-6)
    gap_rel= (S0 - B)/max(S0,1e-12)
    gap_rel_sqrtT = gap_rel * sqrtT
    return dict(
        T=T, sigma=sigma, K_rel=K_rel, B_rel=B_rel, D_frac=D_frac,
        log_moneyness=log_m, log_barrier_gap=log_bg, carry=carry,
        sigma2T=sigma2T, T_sigma=T_sigma, sqrtT=sqrtT, logD=logD,
        gap_rel=gap_rel, gap_rel_sqrtT=gap_rel_sqrtT
    )

def load_cv_dir(model_dir: Path):
    feats_json = json.loads((model_dir/"features.json").read_text())
    feat_names = feats_json.get("feature_names") or feats_json.get("features")
    meta = json.loads((model_dir/"meta_cv.json").read_text())
    fold_paths = sorted(Path(p) for p in glob.glob(str(model_dir/"fold_*.joblib")))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.joblib found in {model_dir}")
    models = [joblib.load(fp) for fp in fold_paths]
    return dict(models=models, meta=meta, feature_names=feat_names, fold_paths=[str(p) for p in fold_paths])

# ---------- optional β calibration (same bins as calibrate_beta.py) ----------
MNY_EDGES = [0, 0.8, 0.95, 1.05, 1.2, 10]
MNY_LABELS= ["deep_OTM","OTM","ATM","ITM","deep_ITM"]
T_EDGES   = [0, 0.25, 0.5, 1, 2, 5, 100]
T_LABELS  = ["<3m","3-6m","6-12m","1-2y","2-5y",">5y"]

def _label_from_edges(x: float, edges, labels):
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]:
            return labels[i]
    return labels[-1]

def beta_lookup(beta_meta: dict | None, S0: float, K: float, T: float) -> float:
    if not beta_meta:
        return 1.0
    mny = (S0/max(K,1e-12))
    mny_lab = _label_from_edges(mny, MNY_EDGES, MNY_LABELS)
    t_lab   = _label_from_edges(T, T_EDGES, T_LABELS)
    key = str((t_lab, mny_lab))
    return float(beta_meta["params"].get(key, 1.0))

# ---------- batch monotone smoother (B↓, D↑, K↓) ----------
from sklearn.isotonic import IsotonicRegression

def _mono_pass(df: pd.DataFrame, group_keys, x_col, increasing, pred_in, pred_out):
    out = df.copy()
    out[pred_out] = out[pred_in]
    for _, g in out.groupby(group_keys, sort=False, observed=True):
        if g.shape[0] < 2: 
            continue
        g = g.sort_values(x_col)
        x = g[x_col].to_numpy()
        if np.allclose(x, x[0]):
            continue
        y = g[pred_in].to_numpy()
        y_mono = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit_transform(x, y)
        out.loc[g.index, pred_out] = y_mono
    return out

def _batch_monotone(df: pd.DataFrame, pred_col="Pred_OUT", max_iter=6):
    cur = df.copy()
    # derived axes
    cur["K_rel"]  = cur["K"]/cur["S0"]
    cur["B_rel"]  = cur["B"]/cur["S0"]
    cur["D_frac"] = (cur["D"]/cur["T"]).clip(0,1)
    cur["B_rel_r"] = cur["B_rel"].round(6)
    cur["D_frac_r"] = cur["D_frac"].round(6)
    cur["K_rel_r"] = cur["K_rel"].round(6)
    cur[pred_col] = np.clip(cur[pred_col], 0.0, cur["Vanilla"]) if "Vanilla" in cur else np.maximum(cur[pred_col], 0.0)

    for _ in range(max_iter):
        cur = _mono_pass(cur, ["T","sigma","K_rel","D_frac_r"], "B_rel", False, pred_col, pred_col)  # B non-increasing
        cur = _mono_pass(cur, ["T","sigma","K_rel","B_rel_r"], "D_frac", True,  pred_col, pred_col)  # D non-decreasing
        cur = _mono_pass(cur, ["T","sigma","B_rel_r","D_frac_r"], "K_rel", False, pred_col, pred_col) # K non-increasing
        # keep bounds if Vanilla present
        if "Vanilla" in cur:
            cur[pred_col] = np.clip(cur[pred_col], 0.0, cur["Vanilla"])
        else:
            cur[pred_col] = np.maximum(cur[pred_col], 0.0)
    return cur

# ---------- core predict ----------
def predict_out_price_cv(
    S0: float, K: float, T: float, r: float, q: float, sigma: float, B: float, D: float,
    model_dir: str | Path | None = None, lattice_steps: int = 800, blend: bool = False,
    beta_meta: dict | None = None
):
    model_dir = Path(model_dir).resolve() if model_dir else (ROOT/"models/gbm_parisian_pricer_cv")
    pack = load_cv_dir(model_dir)
    vanilla = bs_price(S0,K,T,r,q,sigma,"call")

    feats_all = build_all_features(S0,K,T,r,q,sigma,B,D)
    x = np.array([[feats_all[k] for k in pack["feature_names"]]], float)

    z_hats = [m.predict(x, num_iteration=getattr(m, "best_iteration_", None)) for m in pack["models"]]
    z_hat  = np.mean(z_hats, axis=0)
    ratio  = float(sigmoid(float(np.asarray(z_hat).ravel()[0])))
    out_ml = float(np.clip(ratio * vanilla, 0.0, vanilla))

    # optional β calibration (safe)
    if beta_meta:
        beta = beta_lookup(beta_meta, S0, K, T)
        out_ml = float(np.clip(out_ml * beta, 0.0, vanilla))

    # out-of-domain guard
    domain = pack["meta"].get("train_domain")
    if domain:
        dom_ok = (
            domain["T"][0]      <= feats_all["T"]      <= domain["T"][1]      and
            domain["sigma"][0]  <= feats_all["sigma"]  <= domain["sigma"][1]  and
            domain["K_rel"][0]  <= feats_all["K_rel"]  <= domain["K_rel"][1]  and
            domain["B_rel"][0]  <= feats_all["B_rel"]  <= domain["B_rel"][1]  and
            domain["D_frac"][0] <= feats_all["D_frac"] <= domain["D_frac"][1]
        )
    else:
        dom_ok = True

    if not dom_ok:
        out_lat = lattice_out_price(S0,K,T,r,q,sigma,B,D,lattice_steps)
        out_ml = 0.7*out_ml + 0.3*out_lat if blend else out_lat

    return out_ml, (vanilla - out_ml)

# ---------- CLI ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    # single-point mode
    ap.add_argument("--S0", type=float)
    ap.add_argument("--K", type=float)
    ap.add_argument("--T", type=float)
    ap.add_argument("--r", type=float)
    ap.add_argument("--q", type=float)
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--B", type=float)
    ap.add_argument("--D", type=float)
    # batch mode
    ap.add_argument("--in_csv", type=str, help="CSV with columns: S0,K,T,r,q,sigma,B,D")
    ap.add_argument("--out_csv", type=str, help="Output CSV path (defaults to <in_csv basename>_preds.csv)")
    ap.add_argument("--nrows", type=int, default=None, help="Limit number of rows to read for quick runs")
    ap.add_argument("--fallback", action="store_true", help="Enable lattice fallback for out-of-domain rows (slow)")
    ap.add_argument("--monotone_batch", action="store_true", help="Enforce B↓,D↑,K↓ on the batch output (isotonic passes)")
    # model & behavior
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--blend", action="store_true", help="Blend ML with lattice when out-of-domain")
    ap.add_argument("--beta_json", type=str, help="Optional path to beta_params.json (per-bucket calibration)")
    args = ap.parse_args()

    beta_meta = None
    if args.beta_json:
        beta_meta = json.loads(Path(args.beta_json).read_text())

    # ---- batch mode ----
    if args.in_csv:
        df = pd.read_csv(args.in_csv, nrows=args.nrows)
        need = ["S0","K","T","r","q","sigma","B","D"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in {args.in_csv}: {missing}")

        # ---- load models / features once ----
        model_dir = Path(args.model_dir).resolve() if args.model_dir else (ROOT/"models/gbm_parisian_pricer_cv")
        pack = load_cv_dir(model_dir)
        feat_names = pack["feature_names"]

        # ---- build features (vectorized) ----
        S0 = df["S0"].to_numpy(float); K = df["K"].to_numpy(float); T = df["T"].to_numpy(float)
        r  = df["r"].to_numpy(float);   q = df["q"].to_numpy(float); sigma = df["sigma"].to_numpy(float)
        B  = df["B"].to_numpy(float);   D = df["D"].to_numpy(float)

        K_rel  = K/np.maximum(S0,1e-12)
        B_rel  = B/np.maximum(S0,1e-12)
        D_frac = D/np.maximum(T,1e-12)
        carry  = r - q
        log_m  = np.log(np.maximum(S0,1e-12)/np.maximum(K,1e-12))
        log_bg = np.log(np.maximum(S0,1e-12)/np.maximum(B,1e-12))
        sqrtT  = np.sqrt(np.maximum(T,1e-12))
        sigma2T= (sigma**2)*T
        T_sigma= T*sigma
        logD   = np.log(np.maximum(D_frac, 0.0) + 1e-6)
        gap_rel= (S0 - B)/np.maximum(S0,1e-12)
        gap_rel_sqrtT = gap_rel * sqrtT

        feat_df = pd.DataFrame({
            "T":T, "sigma":sigma, "K_rel":K_rel, "B_rel":B_rel, "D_frac":D_frac,
            "log_moneyness":log_m, "log_barrier_gap":log_bg, "carry":carry,
            "sigma2T":sigma2T, "T_sigma":T_sigma, "sqrtT":sqrtT, "logD":logD,
            "gap_rel":gap_rel, "gap_rel_sqrtT":gap_rel_sqrtT
        })
        X = feat_df[feat_names].to_numpy()

        # ---- vanilla (reuse if present) ----
        if "Vanilla" in df.columns:
            vanilla = df["Vanilla"].to_numpy(float)
        else:
            # fallback: compute per-row (slower)
            vanilla = np.array([bs_price(float(S0[i]), float(K[i]), float(T[i]), float(r[i]), float(q[i]), float(sigma[i]), "call")
                                for i in range(len(df))], dtype=float)

        # ---- CV ensemble predict in one shot ----
        z_list = []
        for m in pack["models"]:
            best_it = getattr(m, "best_iteration_", None)
            z_list.append(m.predict(X, num_iteration=best_it))
        z_hat = np.mean(np.vstack(z_list), axis=0)               # shape (n,)
        ratio = 1.0/(1.0 + np.exp(-np.clip(z_hat, -50, 50)))     # sigmoid
        pred  = np.clip(ratio * vanilla, 0.0, vanilla)

        # ---- optional β calibration (vectorized) ----
        beta_meta = None
        if getattr(args, "beta_json", None):
            beta_meta = json.loads(Path(args.beta_json).read_text())
        if beta_meta:
            mny = S0/np.maximum(K,1e-12)
            mny_lab = pd.cut(mny, bins=MNY_EDGES, labels=MNY_LABELS, include_lowest=True).astype(str)
            t_lab   = pd.cut(T,   bins=T_EDGES,   labels=T_LABELS,   include_lowest=True).astype(str)
            keys = [str((t_lab.iloc[i], mny_lab.iloc[i])) for i in range(len(df))]
            betas = np.array([beta_meta["params"].get(k, 1.0) for k in keys], dtype=float)
            pred  = np.clip(pred * betas, 0.0, vanilla)

        # ---- optional lattice fallback for out-of-domain rows (slow) ----
        if args.fallback:
            dom = pack["meta"].get("train_domain", {})
            if dom:
                mask = (
                    (T     >= dom["T"][0])     & (T     <= dom["T"][1]) &
                    (sigma >= dom["sigma"][0]) & (sigma <= dom["sigma"][1]) &
                    (K_rel >= dom["K_rel"][0]) & (K_rel <= dom["K_rel"][1]) &
                    (B_rel >= dom["B_rel"][0]) & (B_rel <= dom["B_rel"][1]) &
                    (D_frac>= dom["D_frac"][0])& (D_frac<= dom["D_frac"][1])
                )
                idx_ood = np.where(~mask)[0]
                if len(idx_ood) > 0:
                    for i in idx_ood:
                        pred[i] = lattice_out_price(float(S0[i]), float(K[i]), float(T[i]), float(r[i]), float(q[i]),
                                                    float(sigma[i]), float(B[i]), float(D[i]), steps=800)

        out = df.copy()
        out["Pred_OUT"] = pred
        out["Pred_IN"]  = vanilla - pred

        # ---- optional monotone smoothing (vectorized isotonic passes) ----
        if args.monotone_batch:
            out = _batch_monotone(out, pred_col="Pred_OUT")

        out_path = args.out_csv or os.path.splitext(args.in_csv)[0] + "_preds.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(json.dumps({"rows": int(len(out)), "output": str(out_path)}, indent=2))
        return

    # ---- single-point mode ----
    required = ["S0","K","T","r","q","sigma","B","D"]
    if any(getattr(args,k) is None for k in required):
        raise SystemExit("Provide either --in_csv or all of --S0 --K --T --r --q --sigma --B --D")
    vals = dict(S0=args.S0, K=args.K, T=args.T, r=args.r, q=args.q, sigma=args.sigma, B=args.B, D=args.D)
    v_out, v_in = predict_out_price_cv(**vals, model_dir=args.model_dir, blend=args.blend, beta_meta=beta_meta)
    print(json.dumps({"OUT": v_out, "IN": v_in}, indent=2))

if __name__ == "__main__":
    main()
