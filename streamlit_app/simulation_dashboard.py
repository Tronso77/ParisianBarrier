# streamlit_app/simulation_dashboard.py

import os, sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make sure src/ is on the path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from models.simulator import simulate_paths

def show_simulation_dashboard():
    st.title("Path Simulation")
    st.write("Generate and explore sample paths for various stochastic models.")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("Simulation Settings")

    model = st.sidebar.selectbox(
        "Model",
        [
            "BM", "ABM", "GBM", "VG", "MJD", "NIG",
            "POI", "GAMMA", "CEV", "HESTON", "CIR"
        ]
    )

    # always‐needed
    S0     = st.sidebar.number_input("Spot price S₀", 0.01, 1e6, 100.0, step=0.1)
    T      = st.sidebar.slider("Horizon T (yrs)", 0.1, 5.0, 1.0, step=0.1)
    nsteps = st.sidebar.slider("Time steps",        10, 1000, 252,   step=1)
    nsim   = st.sidebar.slider("Number of paths",   10, 20000,5000,  step=10)
    seed   = st.sidebar.number_input("Random seed",   0, 99999,  42,    step=1)

    # model‐specific overrides
    #  • drift/r  for ABM, GBM, CEV, Heston
    if model in {"ABM", "GBM", "CEV", "HESTON"}:
        r = st.sidebar.number_input("Drift / rate r", -1.0, 1.0, 0.05, step=0.001)
    else:
        r = None

    #  • volatility σ  for diffusions
    if model in {"BM", "ABM", "GBM", "CEV", "HESTON"}:
        sigma = st.sidebar.number_input("Volatility σ", 0.0, 10.0, 0.2, step=0.01)
    else:
        sigma = None

    #  • CEV exponent β
    if model == "CEV":
        beta = st.sidebar.number_input("CEV β", -2.0, 2.0, -2.0, step=0.1)
    else:
        beta = None

    # how many to plot
    n_display = st.sidebar.slider(
        "Sample paths to plot", 1, min(20, nsim), min(5, nsim), step=1
    )

    if st.sidebar.button("Run Simulation"):
        dt = T / nsteps

        # simulate (we only support S0/r overrides today)
        paths = simulate_paths(
            model,
            nsteps,
            nsim,
            dt,
            seed=seed,
            S0=S0,
            r=r
        )

        t = np.linspace(0, T, nsteps+1)

        # ── Sample paths ──────────────────────────────────────────────────────
        st.subheader(f"Sample of {n_display} paths ({model})")
        sample_idx = np.linspace(0, nsim-1, n_display, dtype=int)
        fig, ax = plt.subplots()
        for i in sample_idx:
            ax.plot(t, paths.iloc[:, i], alpha=0.8)
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Value")
        ax.set_title("Path Samples")
        st.pyplot(fig, use_container_width=True)

        # ── Ensemble mean ±1σ ────────────────────────────────────────────────
        st.subheader("Ensemble mean ± 1 σ over time")
        mu = paths.mean(axis=1)
        sd = paths.std(axis=1)
        fig2, ax2 = plt.subplots()
        ax2.plot(t, mu, lw=2, color="black", label="mean")
        ax2.fill_between(t, mu-sd, mu+sd, color="gray", alpha=0.3, label="± 1 σ")
        ax2.set_xlabel("Time (yrs)")
        ax2.set_ylabel("Value")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

        # ── Histogram at maturity ────────────────────────────────────────────
        st.subheader(f"Histogram at T = {T:.2f} yrs")
        final = paths.iloc[-1].values
        fig3, ax3 = plt.subplots()
        ax3.hist(final, bins=30, density=True, alpha=0.6)
        ax3.axvline(final.mean(), color="red", linestyle="--", label="mean")
        ax3.set_xlabel("Value")
        ax3.set_ylabel("Density")
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)

# entry‐point
if __name__ == "__main__":
    show_simulation_dashboard()
