# streamlit_app/simulation_dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# models import (src/ is added in app.py)
from models.params import param_assign
from models.simulator import simulate_paths

@st.cache_data
def run_sim(model: str, nsteps: int, nsim: int, dt: float, seed: int):
    return simulate_paths(model, nsteps, nsim, dt, seed=seed)


def show_simulation_dashboard():
    st.header("1️⃣ Path Simulation")

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox(
            "Model",
            [m for m in ["BM","ABM","GBM","VG","NIG","MJD","KJD","POI","GAMMA","CIR","HESTON","CEV","SABR"] if m!="CGMY"],
            key="sim_model"
        )
        seed = st.number_input("RNG seed", value=42, step=1, key="sim_seed")
    with col2:
        nsim = st.slider("Number of paths", 1000, 200000, 20000, step=1000, key="sim_nsim")
        nsteps = st.slider("Number of steps", 10, 1000, 252, step=10, key="sim_nsteps")
    with col3:
        T = st.slider("Time horizon (yrs)", 0.1, 5.0, 1.0, step=0.1, key="sim_T")

    if st.button("Simulate Paths", key="sim_button"):
        dt = T / nsteps
        params = param_assign(model)
        paths = run_sim(model, nsteps, nsim, dt, seed=int(seed))

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
            "mean": np.mean(final),
            "std": np.std(final, ddof=1),
            "skew": skew(final),
            "kurtosis": kurtosis(final),
        }
        st.table(pd.DataFrame(stats, index=[""]).T)

        # Value‐at‐Risk and Expected Shortfall
        pct_levels = [5, 25, 50, 75, 95]
        pct_values = np.percentile(final, pct_levels)
        var95 = pct_values[0]
        es95 = final[final <= var95].mean()
        df_pct = pd.DataFrame({
            "percentile": pct_levels + ["VaR(95%)", "ES(95%)"],
            "value": list(pct_values) + [var95, es95]
        })
        st.table(df_pct)

        # ── histogram at maturity ─────────────────────────────────────────────
        st.subheader("Histogram at maturity")
        fig3, ax3 = plt.subplots()
        ax3.hist(final, bins=50)
        ax3.axvline(stats["mean"], color="red", linestyle="--", label="mean")
        ax3.axvline(var95, color="black", linestyle=":", label="VaR(95%)")
        ax3.legend()
        st.pyplot(fig3)
