from __future__ import annotations
from pathlib import Path
import sys, json, math
import numpy as np
import pandas as pd
import joblib

# ---------- project root ----------
def find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "results").exists():
            return p
    return here.parents[2]

ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # allow "src.*" imports

# ---------- lattice fallback ----------
from src.pricing.parisian_lattice import LatticeParams, price_parisian_binomial
from src.pricing.vanilla import bs_price

def lattice_out_price(S0,K,T,r,q,sigma,B,D,steps=800) -> float:
    p = LatticeParams(S0=S0,K=K,T=T,r=r,q=q,sigma=sigma,
                      B=B,D=D,steps=steps,option_type="call",
                      inout="out",direction="down",style="cumulative",
                      in_via_parity=True)
    return float(price_parisian_binomial(p))

# ---------- helpers ----------
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

def load_model_dir(model_dir: Path):
    meta = json.loads((model_dir/"meta.json").read_text())
    model = joblib.load(model_dir/"model.joblib")
    feats_json = json.loads((model_dir/"features.json").read_text())
    feat_names = feats_json.get("feature_names") or feats_json.get("features")
    if not isinstance(feat_names, list):
        raise ValueError("features.json must contain a list under 'feature_names' (or 'features').")
    return dict(model=model, meta=meta, feature_names=feat_names)

def predict_out_price(
    S0: float, K: float, T: float, r: float, q: float, sigma: float, B: float, D: float,
    model_dir: str | Path | None = None, lattice_steps: int = 800, blend: bool = False
):
    model_dir = Path(model_dir).resolve() if model_dir else (ROOT/"models/gbm_parisian_pricer")
    pack = load_model_dir(model_dir)
    vanilla = bs_price(S0,K,T,r,q,sigma,"call")

    feats_all = build_all_features(S0,K,T,r,q,sigma,B,D)
    x = np.array([[feats_all[k] for k in pack["feature_names"]]], float)

    target_kind = pack["meta"].get("target_kind", "price")  # default to price if missing
    pred_arr = pack["model"].predict(x, num_iteration=pack["meta"].get("best_iteration"))
    pred = float(np.asarray(pred_arr).ravel()[0])
    if target_kind == "ratio_logit":
        ratio = float(sigmoid(pred))
        out_ml = float(np.clip(ratio * vanilla, 0.0, vanilla))
    else:
        out_ml = float(np.clip(pred, 0.0, vanilla))

    # out-of-domain guard (use training box if present)
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
    import argparse, os
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
    # model & behavior
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--blend", action="store_true", help="Blend ML with lattice when out-of-domain")
    args = ap.parse_args()

    # ---- batch mode ----
    if args.in_csv:
        import pandas as pd
        df = pd.read_csv(args.in_csv)
        need = ["S0","K","T","r","q","sigma","B","D"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in {args.in_csv}: {missing}")
        outs, ins = [], []
        for row in df.itertuples(index=False):
            vals = dict(S0=float(getattr(row,'S0')),
                        K=float(getattr(row,'K')),
                        T=float(getattr(row,'T')),
                        r=float(getattr(row,'r')),
                        q=float(getattr(row,'q')),
                        sigma=float(getattr(row,'sigma')),
                        B=float(getattr(row,'B')),
                        D=float(getattr(row,'D')),
                        model_dir=args.model_dir,
                        blend=args.blend)
            v_out, v_in = predict_out_price(**vals)
            outs.append(v_out); ins.append(v_in)
        out = df.copy()
        out["Pred_OUT"] = outs
        out["Pred_IN"]  = ins
        out_path = args.out_csv or os.path.splitext(args.in_csv)[0] + "_preds.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        import json
        print(json.dumps({"rows": int(len(out)), "output": str(out_path)}, indent=2))
        return

    # ---- single-point mode ----
    required = ["S0","K","T","r","q","sigma","B","D"]
    if any(getattr(args,k) is None for k in required):
        raise SystemExit("Provide either --in_csv or all of --S0 --K --T --r --q --sigma --B --D")
    vals = dict(S0=args.S0, K=args.K, T=args.T, r=args.r, q=args.q,
                sigma=args.sigma, B=args.B, D=args.D,
                model_dir=args.model_dir, blend=args.blend)
    v_out, v_in = predict_out_price(**vals)
    import json
    print(json.dumps({"OUT": v_out, "IN": v_in}, indent=2))

if __name__ == "__main__":
    main()
