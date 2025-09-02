# scripts/10_benchmark_parisian_plain.py
from __future__ import annotations
from pathlib import Path
import sys, math, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pricing.vanilla import bs_price
from src.pricing.parisian_cjy import CJYParams, StepGrid, price_parisian_cjy
from src.pricing.parisian_lattice import LatticeParams, price_parisian_binomial
from src.models.monte_carlo import MonteCarloEngine

# ------------------------------------------------------------
# Toggle CJY usage: "off" | "guarded" | "force"
USE_CJY = "guarded"
# ------------------------------------------------------------

def nearest_q_from_forwards(vol_df: pd.DataFrame, S0: float, r: float, T: float) -> float:
    F_by_T = vol_df.groupby("T")["F"].first()
    Ts = F_by_T.index.to_numpy()
    j = int(np.argmin(np.abs(Ts - T)))
    F_T = float(F_by_T.iloc[j])
    return r - math.log(F_T / S0) / T

def default_cases(S0: float) -> list[dict]:
    return [
        dict(T=0.084932, K=S0, B=5095.1635, D=0.019841, sigma=0.147588),
        dict(T=0.169863, K=S0, B=5095.1635, D=0.019841, sigma=0.151069),
    ]

def run_cjy_with_guard(S0,K,T,r,q,sigma,B,D, option_type="call",
                       Nx=900, M=96, method="auto"):
    """Return (price, time_s, used_flag, note)."""
    if USE_CJY == "off":
        return (np.nan, 0.0, False, "disabled")
    grid_step = StepGrid(Nx=Nx, Nt=None)
    cj = CJYParams(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                   B=B, D=D, option_type=option_type,
                   direction="down", inout="out")
    t0 = time.perf_counter()
    try:
        v_auto = price_parisian_cjy(cj, grid=grid_step, M=M, method=method,
                                    enforce_bounds=True)  # wrapper already projects
    except Exception as e:
        return (np.nan, time.perf_counter()-t0, False, f"error:{e}")
    t1 = time.perf_counter()
    if USE_CJY == "force":
        return (float(v_auto), t1-t0, True, "")

    # --- Guardrails ---
    vanilla = bs_price(S0, K, T, r, q, sigma, option_type)
    # 1) inverter cross-check (AW vs Stehfest)
    try:
        v_aw  = price_parisian_cjy(cj, grid=grid_step, M=M, method="aw",       enforce_bounds=False)
        v_st  = price_parisian_cjy(cj, grid=grid_step,        method="stehfest", enforce_bounds=False)
        rel   = abs(v_aw - v_st) / max(1e-8, vanilla)
    except Exception:
        rel = np.inf

    # 2) rare-KO heuristic: tiny D/T and barrier far from S0 -> OUT ~ Vanilla
    z = abs(math.log(S0 / B)) / max(1e-12, sigma * math.sqrt(T))
    rare = (D / max(T, 1e-12) <= 0.08 and z >= 0.8)

    # 3) bounds
    in_bounds = (0.0 - 1e-10 <= v_auto <= vanilla + 1e-10)

    if not in_bounds:
        return (np.nan, t1-t0, False, "bounds")
    if rel > 0.02:
        return (np.nan, t1-t0, False, f"inverter mismatch rel={rel:.3f}")
    if rare and v_auto < 0.8 * vanilla:
        return (np.nan, t1-t0, False, "rare-KO; CJY unstable")

    return (float(v_auto), t1-t0, True, "")

def main():
    # Market snapshot
    spot = pd.read_csv(ROOT/"data"/"sx5e_daily.csv", parse_dates=["Date"])
    vol  = pd.read_csv(ROOT/"data"/"sx5e_volsurface_tidy.csv", parse_dates=["exp_date"])
    S0 = float(spot["SX5E_Close"].iloc[-1])
    r  = 0.03

    # Cases
    rows_in = []
    pde_path = ROOT/"results"/"pde_prices_fast.csv"
    if pde_path.exists():
        pde_df = pd.read_csv(pde_path)
        for _, rr in pde_df.iterrows():
            rows_in.append(dict(T=float(rr["T"]), K=float(rr["K"]), B=float(rr["B"]),
                                D=float(rr["D"]), sigma=float(rr["sigma"])))
    else:
        rows_in = default_cases(S0)

    out_rows = []
    for c in rows_in:
        T, K, B, D, sigma = c["T"], c["K"], c["B"], c["D"], c["sigma"]
        q = nearest_q_from_forwards(vol, S0, r, T)
        vanilla = bs_price(S0, K, T, r, q, sigma, "call")

        # ---- CJY (guarded/optional) ----
        cjy_out, cjy_time, cjy_used, cjy_note = run_cjy_with_guard(
            S0,K,T,r,q,sigma,B,D, option_type="call",
            Nx=900, M=96, method="auto"
        )
        cjy_in = (vanilla - cjy_out) if np.isfinite(cjy_out) else np.nan

        # ---- LATTICE ----
        lat_p = LatticeParams(
            S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
            B=B, D=D, steps=600,
            option_type="call", inout="out",
            direction="down", style="cumulative",
            in_via_parity=True,
        )
        t0 = time.perf_counter()
        lat_out = price_parisian_binomial(lat_p)
        lat_time = time.perf_counter() - t0
        lat_in  = vanilla - lat_out

        # ---- MONTE CARLO ----
        steps_mc = 500
        paths_mc = 50_000
        eng = MonteCarloEngine(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                               model="GBM", n_paths=paths_mc, n_steps=steps_mc,
                               seed=42, antithetic=True, brownian_bridge=False,
                               control_variate=False, verbose=False)

        # price + occupation time stats
        t0 = time.perf_counter()
        mc_out, se_out = eng.price_parisian(K=K, barrier=B, window=D,
                                            option_type="call", inout="out",
                                            direction="down", style="cumulative", cv="none")
        mc_time = time.perf_counter() - t0
        mc_in = vanilla - mc_out

        # occupation time / hit rate
        paths = eng.simulate_paths()
        dt = T / steps_mc
        side = (paths[1:] < B)  # down
        occ = side.sum(axis=0) * dt
        mc_hit = float((occ >= D).mean())
        occ_mean = float(occ.mean())
        occ_std  = float(occ.std(ddof=1))
        occ_p50  = float(np.quantile(occ, 0.50))

        # z vs CJY (only if used and finite)
        if np.isfinite(cjy_out):
            z_mc_vs_cjy = (mc_out - cjy_out) / max(se_out, 1e-12)
        else:
            z_mc_vs_cjy = np.nan

        out_rows.append(dict(
            T=T, D=D, K=K, B=B, sigma=sigma, S0=S0, r=r, q=q, Vanilla=vanilla,
            CJY_OUT=cjy_out, CJY_IN=cjy_in, CJY_time_s=cjy_time,
            CJY_Used=int(bool(cjy_used)), CJY_Note=cjy_note,
            LAT_OUT=lat_out, LAT_IN=lat_in, LAT_time_s=lat_time, LAT_STEPS=lat_p.steps,
            MC_OUT=mc_out, SE_OUT=se_out, MC_IN=mc_in, MC_time_s=mc_time,
            Z_MC_vs_CJY=z_mc_vs_cjy,
            ParityErr_CJY=0.0 if not np.isfinite(cjy_out) else 0.0,  # parity holds by construction
            ParityErr_LAT=0.0,                                       # same
            BoundsOK=(True if (not np.isfinite(cjy_out)) else (0.0 - 1e-9 <= cjy_out <= vanilla + 1e-9)),
            MC_OccTimeMean=occ_mean, MC_OccTimeStd=occ_std, MC_OccTimeP50=occ_p50, MC_HitRate=mc_hit,
            MC_STEPS=steps_mc, MC_PATHS=paths_mc, CJY_Nx=900, CJY_M=96,
        ))

    res = pd.DataFrame(out_rows)
    cols = ["T","D","K","B","sigma","S0","r","q","Vanilla",
            "CJY_OUT","CJY_IN","CJY_Used","CJY_Note","CJY_time_s",
            "LAT_OUT","LAT_IN","LAT_time_s","LAT_STEPS",
            "MC_OUT","SE_OUT","MC_IN","MC_time_s","Z_MC_vs_CJY",
            "ParityErr_CJY","ParityErr_LAT","BoundsOK",
            "MC_OccTimeMean","MC_OccTimeStd","MC_OccTimeP50","MC_HitRate",
            "MC_STEPS","MC_PATHS","CJY_Nx","CJY_M"]
    print(res[cols].to_string(index=False))

    out = ROOT/"results"/"parisian_plain_bench.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
