# scripts/13_hedge_compare.py
from __future__ import annotations
from pathlib import Path
import sys, math, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.monte_carlo import MonteCarloEngine
from src.pricing.payoff_parisian import parisian_indicator
from src.pricing.vol_surface import sigma_from_KT
from src.pricing.vanilla import bs_price, bs_delta_call

def nearest_q_from_forwards(vol_df: pd.DataFrame, S0: float, r: float, T: float) -> float:
    F_by_T = vol_df.groupby("T")["F"].first()
    Ts = F_by_T.index.to_numpy()
    j = int(np.argmin(np.abs(Ts - T)))
    F_T = float(F_by_T.iloc[j])
    return r - math.log(F_T / S0) / max(T, 1e-12)

def summarize_pnl(pnl: np.ndarray) -> dict:
    return dict(
        mean=float(np.mean(pnl)),
        std=float(np.std(pnl, ddof=1)),
        p5=float(np.percentile(pnl, 5)),
        p50=float(np.percentile(pnl, 50)),
        p95=float(np.percentile(pnl, 95)),
    )

def main():
    # market snapshot
    spot = pd.read_csv(ROOT/"data"/"sx5e_daily.csv", parse_dates=["Date"])
    vol  = pd.read_csv(ROOT/"data"/"sx5e_volsurface_tidy.csv", parse_dates=["exp_date"])
    S0 = float(spot["SX5E_Close"].iloc[-1])
    r  = 0.03

    # pick 1–2 representative contracts (or load from results/pde_prices_fast.csv)
    T = 0.084932
    K = S0
    B = 5095.1635
    D = 0.019841
    sigma = 0.147588
    q = nearest_q_from_forwards(vol, S0, r, T)

    # premium: use CJY OUT price as the "fair" premium
    from src.pricing.parisian_cjy import CJYParams, StepGrid, price_parisian_cjy
    cj = CJYParams(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, B=B, D=D,
                   option_type="call", direction="down", inout="out")
    cjy_grid = StepGrid(Nx=700, Nt=None)
    premium = price_parisian_cjy(cj, grid=cjy_grid, M=64)

    # simulation (under Q)
    MC_STEPS = 500
    MC_PATHS = 50_000
    eng = MonteCarloEngine(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                           model="GBM", n_paths=MC_PATHS, n_steps=MC_STEPS,
                           seed=123, antithetic=True, brownian_bridge=False,
                           control_variate=False, verbose=False)
    paths = eng.simulate_paths()              # (n_steps+1, n_paths)
    S = paths
    dt = T / MC_STEPS
    times = np.linspace(0.0, T, MC_STEPS + 1)

    # Parisian OUT indicator and alive-by-time
    alive_final = ~parisian_indicator(paths, barrier=B, dt=dt, window=D,
                                      direction="down", style="cumulative")
    side = (S[1:] < B)
    occ = np.cumsum(side, axis=0) * dt
    alive_t = (occ < D)                       # shape (n_steps, n_paths)

    # transaction costs & policy knobs
    half_spread_bps = 1.0     # 1 bp half-spread
    thresh = 0.005            # 50 bps delta-threshold
    rehedge_k = 1             # 1=every step; 5=every 5 steps, etc.

    # --- policy A: NO HEDGE (control) ---
    payoff_out = np.exp(-r*T) * np.maximum(S[-1] - K, 0.0) * alive_final.astype(float)
    pnl_nohedge = premium - payoff_out
    summ_no = summarize_pnl(pnl_nohedge)

    # --- policy B: Vanilla-Delta while alive ---
    cash = np.full(MC_PATHS, premium, dtype=float)
    shares = np.zeros(MC_PATHS, dtype=float)

    # initial hedge
    tau_0 = T
    sigma_0 = sigma_from_KT(vol, K, tau_0)
    delta = bs_delta_call(S[0], K, tau_0, r, q, sigma_0) * alive_t[0]
    shares += delta
    cash -= delta * S[0]

    for t in range(1, MC_STEPS):
        if t % rehedge_k != 0:
            # just carry cash forward one step
            cash *= math.exp(r * dt)
            continue

        cash *= math.exp(r * dt)

        tau = max(T - times[t], 1e-12)
        sigma_t = sigma_from_KT(vol, K, tau)
        delta_new = bs_delta_call(S[t], K, tau, r, q, sigma_t) * alive_t[t]

        # threshold & costs
        d_shares = delta_new - shares
        need_trade = np.abs(d_shares) > thresh
        traded = d_shares * need_trade
        # execution price ~ mid; pay half-spread
        cost = np.abs(traded) * S[t] * (half_spread_bps * 1e-4)
        cash -= traded * S[t] + cost
        shares += traded

    # final step accrual + close hedge
    cash *= math.exp(r * dt)
    pnl_vanilla = cash + shares * S[-1] - payoff_out
    summ_van = summarize_pnl(pnl_vanilla)

    # report
    print("\n=== Hedging comparison (Parisian OUT, cumulative, down) ===")
    print(f"Contract: T={T:.4f}, K={K:.2f}, B={B:.2f}, D={D:.5f}, r={r:.2%}, q≈{q:.2%}, sigma={sigma:.2%}")
    print(f"Simulation: steps={MC_STEPS}, paths={MC_PATHS}, dt={dt:.6f}, prem(CJY)={premium:.6f}")
    print(f"KO hit rate: {1.0 - float(np.mean(alive_final)):.2%}")
    print("\nNo-hedge     :", summ_no)
    print("Vanilla-Delta:", summ_van)

    # save distribution
    out = pd.DataFrame({
        "pnl_nohedge": pnl_nohedge,
        "pnl_vanilla": pnl_vanilla,
        "alive_final": alive_final.astype(int),
        "ST": S[-1],
    })
    outp = ROOT/"results"/"hedge_compare.csv"
    out.to_csv(outp, index=False)
    print(f"\nSaved -> {outp}")

if __name__ == "__main__":
    main()
