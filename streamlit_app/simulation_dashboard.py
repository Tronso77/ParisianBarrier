import os, sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.simulator import simulate_paths

@st.cache_data
def run_sim(model: str, S0: float, r: float, nsteps: int, nsim: int, dt: float, seed: int):
    """
    Simulate paths with optional S0 and r overrides.
    """
    return simulate_paths(model, nsteps, nsim, dt, seed=seed, S0=S0, r=r)


def show_simulation_dashboard():
    st.header("1️⃣ Path Simulation")

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox(
            "Model",
             ["BM","ABM","GBM","VG","NIG","MJD","KJD","POI",
              "GAMMA","CIR","HESTON","CEV","SABR", "VGCIR", "CGMY"],
            key="sim_model"
        )
        S0 = st.number_input("Spot price S₀", 1.0, 1e5, 100.0, step=1.0, key="sim_S0")
        r  = st.number_input("Drift / rate r", -1.0, 1.0, 0.0, step=0.001, key="sim_r")
        seed = st.number_input("RNG seed", value=42, step=1, key="sim_seed")
    with col2:
        nsim   = st.slider("Number of paths", 1000, 200000, 20000, step=1000, key="sim_nsim")
        nsteps = st.slider("Number of steps", 10, 1000, 252, step=10, key="sim_nsteps")
    with col3:
        T = st.slider("Time horizon (yrs)", 0.1, 5.0, 1.0, step=0.1, key="sim_T")

    if st.button("Simulate Paths", key="sim_button"):
        dt = T / nsteps
        paths = run_sim(model, S0, r, nsteps, nsim, dt, seed=int(seed))

        # ── sample paths ─────────────────────────────────────────────────────────
        st.subheader(f"Sample of first 20 {model} paths")
        sample = paths.iloc[:, :min(20, nsim)]
        st.line_chart(sample)

        # ── ensemble mean ±1σ ───────────────────────────────────────────────────
        st.subheader("Ensemble mean ±1 σ over time")
        mu = paths.mean(axis=1)
        sd = paths.std(axis=1)
        fig, ax = plt.subplots()
        ax.plot(paths.index * dt, mu, label="mean")
        ax.fill_between(paths.index * dt, mu - sd, mu + sd, alpha=0.3, label="±1 σ")
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        # ── distribution at intermediate time ──────────────────────────────────
        st.subheader("Distribution at selected time")
        t_frac = st.slider("Time (fraction of T)", 0.0, 1.0, 1.0, step=0.25, key="dist_time")
        idx = int(t_frac * nsteps)
        data_t = paths.iloc[idx].values
        fig2, ax2 = plt.subplots()
        ax2.hist(data_t, bins=50, density=True)
        ax2.set_xlabel(f"Value at t={t_frac:.2f}·T")
        ax2.set_ylabel("Density")
        st.pyplot(fig2)

        # ── maturity distribution & risk stats ─────────────────────────────────
        st.subheader("Maturity (t=T) distribution & risk metrics")
        final = paths.iloc[-1].values
        stats = {
            "mean":  np.mean(final),
            "std":   np.std(final, ddof=1),
            "skew":  skew(final),
            "kurtosis": kurtosis(final),
        }
        st.table(pd.DataFrame(stats, index=[""]).T)

        # Value‐at‐Risk and Expected Shortfall
        pct_levels = [5, 25, 50, 75, 95]
        pct_values = np.percentile(final, pct_levels)
        var95 = pct_values[0]
        es95  = final[final <= var95].mean()
        df_pct = pd.DataFrame({
            "percentile": pct_levels + ["VaR(95%)", "ES(95%)"],
            "value":       list(pct_values) + [var95, es95]
        })
        st.table(df_pct)

        # ── histogram at maturity ─────────────────────────────────────────────
        st.subheader("Histogram at maturity")
        fig3, ax3 = plt.subplots()
        ax3.hist(final, bins=50)
        ax3.axvline(stats["mean"], color="red", linestyle="--", label="mean")
        ax3.axvline(var95,           color="black", linestyle=":",  label="VaR(95%)")
        ax3.legend()
        st.pyplot(fig3)
