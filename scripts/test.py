# scripts/11_healthcheck.py
from __future__ import annotations
from pathlib import Path
import sys, time, math, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# imports from your project
from src.pricing.vanilla import bs_price
from src.pricing.parisian_cjy import CJYParams, StepGrid, price_parisian_cjy
from src.pricing.parisian_lattice import LatticeParams, price_parisian_binomial
from src.models.monte_carlo import MonteCarloEngine

def nearest_q_from_forwards(vol_df: pd.DataFrame, S0: float, r: float, T: float) -> float:
    F_by_T = vol_df.groupby("T")["F"].first()
    Ts = F_by_T.index.to_numpy()
    j = int(np.argmin(np.abs(Ts - T)))
    F_T = float(F_by_T.iloc[j])
    return r - math.log(F_T / S0) / T

def default_cases(S0: float) -> list[dict]:
    # a few representative cases; add more if you like
    return [
        dict(T=1/12, K=S0, B=0.95*S0, D=1/240, sigma=0.10),
        dict(T=1/12, K=S0, B=0.95*S0, D=1/240, sigma=0.15),
        dict(T=1/12, K=S0, B=0.95*S0, D=1/240, sigma=0.20),
        dict(T=1/12, K=S0, B=0.95*S0, D=1/60,  sigma=0.15),   # shorter window
        dict(T=0.17,  K=S0, B=0.95*S0, D=1/240, sigma=0.15),
    ]

def main():
    print("=== Parisian Project Health Check ===")
    # 1) environment quick info
    import platform, numpy, scipy
    print(f"Python: {platform.python_version()}  numpy: {numpy.__version__}  scipy: {scipy.__version__}")

    # 2) market snapshot (for S0 and q(T))
    spot = pd.read_csv(ROOT/"data"/"sx5e_daily.csv", parse_dates=["Date"])
    vol  = pd.read_csv(ROOT/"data"/"sx5e_volsurface_tidy.csv", parse_dates=["exp_date"])
    S0 = float(spot["SX5E_Close"].iloc[-1])
    r  = 0.03

    # 3) test set: prefer results/pde_prices_fast.csv (for K,B,D,sigma,T), else defaults
    cases = []
    pde_path = ROOT/"results"/"pde_prices_fast.csv"
    if pde_path.exists():
        pde_df = pd.read_csv(pde_path)
        for _, rr in pde_df.iterrows():
            cases.append(dict(T=float(rr["T"]), K=float(rr["K"]), B=float(rr["B"]),
                              D=float(rr["D"]), sigma=float(rr["sigma"])))
        print(f"Loaded {len(cases)} cases from {pde_path}")
    else:
        cases = default_cases(S0)
        print(f"Using default cases ({len(cases)}).")

    # 4) knobs (recommended baselines)
    CJY_Nx = 900      # step-PDE space grid
    CJY_Nt = None     # auto time grid
    CJY_M  = 96       # Abate–Whitt Euler terms

    LAT_STEPS = 600   # binomial layers

    MC_STEPS  = 500   # time steps
    MC_PATHS  = 50_000
    MC_ANTITH = True
    MC_SEED   = 42

    rows = []
    flags = []

    for c in cases:
        T, K, B, D, sigma = c["T"], c["K"], c["B"], c["D"], c["sigma"]
        q = nearest_q_from_forwards(vol, S0, r, T)
        vanilla = bs_price(S0, K, T, r, q, sigma, "call")

        # CJY (OUT via Laplace inversion of step option)
        Smin = min(S0, B)/20.0
        Smax = max(S0, B)*10.0
        grid_step = StepGrid(Nx=CJY_Nx, S_left=Smin, S_right=Smax, Nt=None)
        t0 = time.perf_counter()
        cjy_out = price_parisian_cjy(CJYParams(S0, K, T, r, q, sigma, B, D, "call", "down", "out"),
                                     grid=grid_step, M=CJY_M)
        t1 = time.perf_counter()
        cjy_time = t1 - t0
        cjy_in = vanilla - cjy_out

        # Lattice (OUT; IN via parity)
        lat_p = LatticeParams(S0, K, T, r, q, sigma, B, D,
                              steps=LAT_STEPS, option_type="call", inout="out",
                              direction="down", style="cumulative", in_via_parity=True)
        t0 = time.perf_counter()
        lat_out = price_parisian_binomial(lat_p)
        t1 = time.perf_counter()
        lat_time = t1 - t0
        lat_in = vanilla - lat_out

        # Monte Carlo (OUT; IN via parity)
        eng = MonteCarloEngine(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                               model="GBM", n_paths=MC_PATHS, n_steps=MC_STEPS,
                               seed=MC_SEED, antithetic=MC_ANTITH,
                               brownian_bridge=False, control_variate=False, verbose=False)
        mc_out, se_out = eng.price_parisian(K=K, barrier=B, window=D,
                                            option_type="call", inout="out",
                                            direction="down", style="cumulative")
        mc_in  = vanilla - mc_out

        # diagnostics
        parity_cjy = abs((cjy_in + cjy_out) - vanilla)
        parity_lat = abs((lat_in + lat_out) - vanilla)
        bounds_ok = (0.0 <= cjy_out <= vanilla + 1e-6) and (0.0 <= lat_out <= vanilla + 1e-6)
        z_mc_vs_cjy = (mc_out - cjy_out) / max(se_out, 1e-12)

        row = dict(
            T=T, D=D, K=K, B=B, sigma=sigma, S0=S0, r=r, q=q, Vanilla=vanilla,
            CJY_OUT=cjy_out, CJY_IN=cjy_in, CJY_time_s=cjy_time,
            LAT_OUT=lat_out, LAT_IN=lat_in, LAT_time_s=lat_time, LAT_STEPS=LAT_STEPS,
            MC_OUT=mc_out, SE_OUT=se_out, MC_IN=mc_in, MC_time_s=None,
            Z_MC_vs_CJY=z_mc_vs_cjy, MC_STEPS=MC_STEPS, MC_PATHS=MC_PATHS,
            CJY_Nx=CJY_Nx, CJY_M=CJY_M,
            ParityErr_CJY=parity_cjy, ParityErr_LAT=parity_lat, BoundsOK=bounds_ok
        )
        rows.append(row)

        # flag suspicious
        issues = []
        if not bounds_ok:
            issues.append("bounds")
        if parity_cjy > 1e-3 * max(1.0, vanilla):
            issues.append(f"parity_cjy={parity_cjy:.3e}")
        if parity_lat > 1e-3 * max(1.0, vanilla):
            issues.append(f"parity_lat={parity_lat:.3e}")
        # large z with tiny absolute diff is often benign; still flag for visibility
        if abs(z_mc_vs_cjy) > 3.0:
            issues.append(f"z={z_mc_vs_cjy:.2f}")
        if issues:
            flags.append((c, issues))

    df = pd.DataFrame(rows)
    # nice print
    cols = ["T","D","K","B","sigma","S0","r","q","Vanilla",
            "CJY_OUT","LAT_OUT","MC_OUT","SE_OUT","Z_MC_vs_CJY",
            "CJY_IN","LAT_IN","MC_IN","ParityErr_CJY","ParityErr_LAT","BoundsOK",
            "LAT_STEPS","MC_STEPS","MC_PATHS","CJY_Nx","CJY_M","CJY_time_s","LAT_time_s"]
    print(df[cols].to_string(index=False))

    # flagged rows
    if flags:
        print("\n⚠️  Flags:")
        for case, issues in flags:
            print(f"  T={case['T']:.6f} sigma={case['sigma']:.3f} B={case['B']:.2f} D={case['D']:.6f} -> {', '.join(issues)}")
    else:
        print("\n✅ No issues flagged.")

    # save artifacts
    out = ROOT/"results"/"healthcheck_summary.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    with open(ROOT/"results"/"healthcheck_config.json","w") as fh:
        json.dump(dict(CJY_Nx=CJY_Nx, CJY_M=CJY_M, LAT_STEPS=LAT_STEPS,
                       MC_STEPS=MC_STEPS, MC_PATHS=MC_PATHS), fh, indent=2)
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
