# scripts/30_build_ml_dataset_gbm.py
from __future__ import annotations
from pathlib import Path
import sys, math, time, random
import numpy as np
import pandas as pd

# --- project root on path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- our libs ---
from src.pricing.vanilla import bs_price
from src.pricing.parisian_lattice import LatticeParams, price_parisian_binomial
from src.models.monte_carlo import MonteCarloEngine

# --------------------- knobs (edit here) ---------------------
SEED_GLOBAL = 42

# Pricing grid size (cartesian)  →  ~3,000 cases
T_LIST       = [1/12, 1/4, 1/2, 1.0]                  # years
SIGMA_LIST   = [0.10, 0.15, 0.20, 0.25, 0.30]
B_REL_LIST   = [0.60, 0.70, 0.80, 0.90, 0.95, 0.97]   # include 0.97 (near-spot)
K_REL_LIST   = [0.80, 0.90, 1.00, 1.10, 1.20]
D_FRAC_LIST  = [0.01, 0.02, 0.05, 0.10, 0.20]         # include very short windows

# Lattice / MC settings (GBM only)
LAT_STEPS         = 800          # good all-around; (optional) see adaptive note below
MC_STEPS          = 600
MC_PATHS_PRICE    = 10_000       # sanity-only column; lower to keep runtime sane
MC_PATHS_HEDGE    = 4_000        # used only for hedging episode generation
MC_ANTITH         = True
MC_CV             = "vanilla"    # CV for Parisian (discounted vanilla payoff)
r_FLAT            = 0.03         # flat rate; q(T) from forwards if available

# Hedging episodes
HEDGE_MAX_EPISODES_PER_CASE = 2000    # cap (per parameter case)
HEDGE_SAMPLE_TIMES          = 12      # re-hedges along each path (≈ monthly over 1y)
HEDGE_DELTA_BUMP            = 1e-3    # central FD bump for delta

# (Optional) only compute MC sanity for a subset of the grid:
MC_SAMPLE_FRAC_FOR_PRICE    = 0.20    # 20% of cases get MC columns; others set NaN
# -------------------------------------------------------------



def nearest_q_from_forwards(vol_df: pd.DataFrame, S0: float, r: float, T: float) -> float:
    """If you have SX5E forwards in volsurface_tidy, infer q(T); else return 0."""
    if vol_df is None or vol_df.empty or "F" not in vol_df.columns:
        return 0.0
    F_by_T = vol_df.groupby("T")["F"].first()
    Ts = F_by_T.index.to_numpy()
    j = int(np.argmin(np.abs(Ts - T)))
    F_T = float(F_by_T.iloc[j])
    # guard if T~0
    return r - (math.log(max(F_T, 1e-12) / max(S0, 1e-12)) / max(T, 1e-12))


def lattice_out_price(S0, K, T, r, q, sigma, B, D, steps=LAT_STEPS) -> float:
    p = LatticeParams(
        S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
        B=B, D=D, steps=steps,
        option_type="call", inout="out",
        direction="down", style="cumulative",
        in_via_parity=True
    )
    return float(price_parisian_binomial(p))


def lattice_delta_from_state(S, K, T_rem, r, q, sigma, B, D_rem, steps=LAT_STEPS, bump=HEDGE_DELTA_BUMP) -> float:
    """Central FD delta for OUT price at an intermediate state."""
    Su = S * (1 + bump)
    Sd = S * (1 - bump)
    Vu = lattice_out_price(Su, K, T_rem, r, q, sigma, B, D_rem, steps=steps)
    Vd = lattice_out_price(Sd, K, T_rem, r, q, sigma, B, D_rem, steps=steps)
    return (Vu - Vd) / (Su - Sd)


def build_pricing_dataset(S0: float, r: float, vol_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    total = len(T_LIST) * len(SIGMA_LIST) * len(B_REL_LIST) * len(K_REL_LIST) * len(D_FRAC_LIST)
    k = 0
    for T in T_LIST:
        for sigma in SIGMA_LIST:
            for b_rel in B_REL_LIST:
                for k_rel in K_REL_LIST:
                    for dfrac in D_FRAC_LIST:
                        k += 1
                        B = b_rel * S0
                        K = k_rel * S0
                        D = dfrac * T
                        q = nearest_q_from_forwards(vol_df, S0, r, T)

                        # labels
                        vanilla = bs_price(S0, K, T, r, q, sigma, "call")
                        lat_out = lattice_out_price(S0, K, T, r, q, sigma, B, D, LAT_STEPS)
                        lat_in  = vanilla - lat_out

                        # light GBM MC (sanity)
                        eng = MonteCarloEngine(
                            S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                            model="GBM", n_paths=MC_PATHS_PRICE, n_steps=MC_STEPS,
                            seed=SEED_GLOBAL, antithetic=MC_ANTITH,
                            control_variate=False, stratified=False, verbose=False
                        )
                        mc_out, se_out = eng.price_parisian(
                            K=K, barrier=B, window=D,
                            option_type="call", inout="out",
                            direction="down", style="cumulative", cv=MC_CV
                        )
                        mc_in = vanilla - mc_out

                        rows.append(dict(
                            T=T, D=D, K=K, B=B, sigma=sigma, S0=S0, r=r, q=q,
                            moneyness=K/S0, barrier_rel=B/S0, D_over_T=(D/max(T,1e-12)),
                            Vanilla=vanilla, LAT_OUT=lat_out, LAT_IN=lat_in,
                            MC_OUT=mc_out, SE_OUT=se_out, MC_IN=mc_in,
                            LAT_STEPS=LAT_STEPS, MC_STEPS=MC_STEPS, MC_PATHS=MC_PATHS_PRICE
                        ))

                        if k % 25 == 0:
                            print(f"[pricing] {k}/{total} cases done...")

    df = pd.DataFrame(rows)
    return df


def build_hedging_dataset(S0: float, r: float, vol_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    For each pricing case, simulate GBM paths and sample hedging 'states':
      state features → label delta (lattice FD) for OUT option.
    """
    rng = np.random.default_rng(SEED_GLOBAL)
    episodes = []

    total = len(T_LIST) * len(SIGMA_LIST) * len(B_REL_LIST) * len(K_REL_LIST) * len(D_FRAC_LIST)
    case_id = 0

    for T in T_LIST:
        for sigma in SIGMA_LIST:
            for b_rel in B_REL_LIST:
                for k_rel in K_REL_LIST:
                    for dfrac in D_FRAC_LIST:
                        case_id += 1
                        B = b_rel * S0
                        K = k_rel * S0
                        D = dfrac * T
                        q = nearest_q_from_forwards(vol_df, S0, r, T)
                        # Sim paths (GBM)
                        eng = MonteCarloEngine(
                            S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                            model="GBM", n_paths=MC_PATHS_HEDGE, n_steps=MC_STEPS,
                            seed=SEED_GLOBAL + case_id, antithetic=MC_ANTITH,
                            control_variate=False, stratified=False, verbose=False
                        )
                        paths = eng.simulate_paths()   # shape (n_steps+1, n_paths)
                        dt = T / MC_STEPS
                        times = np.linspace(0.0, T, MC_STEPS + 1)

                        # choose sampling times (skip t=0 and very end)
                        # e.g. ~monthly equally spaced
                        idx_times = np.linspace(1, MC_STEPS - 1, num=min(HEDGE_SAMPLE_TIMES, MC_STEPS-1), dtype=int)
                        # sample subset of paths to cap episodes
                        paths_idx = np.arange(paths.shape[1])
                        rng.shuffle(paths_idx)
                        # rough cap: target episodes per case
                        per_time_quota = max(50, HEDGE_MAX_EPISODES_PER_CASE // max(1, len(idx_times)))
                        selected_paths = paths_idx[:min(per_time_quota, len(paths_idx))]

                        # precompute under-barrier booleans per time
                        under = (paths[1:, :] < B)   # (MC_STEPS, n_paths)
                        under_cum = np.cumsum(under, axis=0) * dt  # cumulative occupation time up to each step

                        for it in idx_times:
                            t_now = times[it]
                            T_rem = max(0.0, T - t_now)
                            if T_rem <= 0.0:
                                continue

                            # vectorized clocks at this time for selected paths
                            D_spent = under_cum[it-1, selected_paths] if it > 0 else np.zeros_like(selected_paths, dtype=float)
                            D_rem = np.maximum(0.0, D - D_spent)
                            S_now = paths[it, selected_paths]

                            # filter: if already knocked-out (for OUT), price is 0 → delta ~ 0; keep a slice of those too
                            # but don’t waste all episodes on KO = trivial states
                            keep_mask = rng.random(S_now.size) < 0.8  # keep 80% randomly
                            S_now = S_now[keep_mask]
                            D_rem = D_rem[keep_mask]
                            D_spent = D_spent[keep_mask]

                            # compute lattice delta labels (loop; this is the heavy part)
                            # choose steps proportional to remaining time for stable dt
                            deltas = []
                            for s_val, drem in zip(S_now, D_rem):
                                steps_local = max(200, int(round(LAT_STEPS * (T_rem / T))))
                                dlt = lattice_delta_from_state(
                                    S=s_val, K=K, T_rem=T_rem, r=r, q=q, sigma=sigma,
                                    B=B, D_rem=drem, steps=steps_local, bump=HEDGE_DELTA_BUMP
                                )
                                deltas.append(dlt)

                            # build rows
                            for s_val, drem, dspent, dlt in zip(S_now, D_rem, D_spent, deltas):
                                episodes.append(dict(
                                    case_id=case_id,
                                    t=t_now, T=T, T_rem=T_rem,
                                    S=s_val, K=K, B=B, D=D,
                                    D_spent=dspent, D_rem=drem,
                                    sigma=sigma, r=r, q=q,
                                    moneyness_now=K/max(s_val,1e-12),
                                    barrier_rel_now=B/max(s_val,1e-12),
                                    D_over_T=(D/max(T,1e-12)),
                                    direction="down", style="cumulative", inout="out",
                                    delta_label=dlt
                                ))

                        print(f"[hedge] case {case_id}/{total} → episodes so far: {len(episodes)}")

    df = pd.DataFrame(episodes)
    return df


def main():
    random.seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)

    # Market snapshot (to get S0; forwards optional)
    try:
        spot = pd.read_csv(ROOT/"data"/"sx5e_daily.csv", parse_dates=["Date"])
        S0 = float(spot["SX5E_Close"].iloc[-1])
    except Exception:
        S0 = 100.0  # fallback
    r = r_FLAT

    try:
        vol = pd.read_csv(ROOT/"data"/"sx5e_volsurface_tidy.csv", parse_dates=["exp_date"])
    except Exception:
        vol = None

    # Build pricing dataset
    t0 = time.perf_counter()
    df_price = build_pricing_dataset(S0, r, vol)
    t1 = time.perf_counter()

    # Build hedging episodes
    #df_hedge = build_hedging_dataset(S0, r, vol)
    #t2 = time.perf_counter()

    # Save
    out_dir = ROOT/"results"
    out_dir.mkdir(exist_ok=True)
    df_price.to_csv(out_dir/"dataset_parisian_gbm_pricing.csv", index=False)
    #df_hedge.to_csv(out_dir/"dataset_parisian_gbm_hedging.csv", index=False)

    # Quick summary
    print("\n=== DATASETS BUILT ===")
    print(f"Pricing rows: {len(df_price)}  -> {out_dir/'dataset_parisian_gbm_pricing.csv'} (time: {t1-t0:.2f}s)")
    #print(f"Hedging rows: {len(df_hedge)}  -> {out_dir/'dataset_parisian_gbm_hedging.csv'} (time: {t2-t1:.2f}s)")
    print("\nPricing columns:", list(df_price.columns))
    #print("Hedging columns:", list(df_hedge.columns))


if __name__ == "__main__":
    main()
