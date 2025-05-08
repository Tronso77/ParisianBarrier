import os, sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.params import param_assign
from models.simulator import simulate_paths
from pricing.vanilla import price_european_option

@st.cache_data
def run_paths(model: str, S0: float, r: float, nsteps: int, nsim: int, T: float, seed: int):
    dt = T / nsteps
    paths = simulate_paths(model, nsteps, nsim, dt, seed=seed, S0=S0, r=r)
    return paths


def show_pricing_dashboard():
    st.header("Option Pricing — BS Benchmark & MC Comparison")

    # Sidebar inputs
    with st.sidebar:
        st.subheader("Market & Payoff Inputs")
        S0 = st.number_input("Spot price S₀", 1.0, 1e5, 100.0, step=1.0)
        K = st.number_input("Strike K", 0.01, 1e6, 100.0, step=1.0)
        T = st.number_input("Maturity (yrs)", 0.01, 5.0, 1.0, step=0.01)
        r = st.number_input("Risk-free rate r", 0.0, 0.2, 0.05, step=0.001)
        sigma_bs = st.number_input("Volatility σ (BS)", 0.01, 1.0, 0.20, step=0.01)

        st.markdown("---")
        st.subheader("MC Simulation Settings")
        models = st.multiselect("MC Models", options=["GBM", "Heston"], default=["GBM"])
        nsim = st.slider("Paths (MC)", 1000, 500000, 100000, step=1000)
        nsteps = st.slider("Time steps", 10, 1000, 252, step=10)
        seed = st.number_input("RNG seed", 0, 10000, 42, step=1)

        st.markdown("---")
        payoff_type = st.selectbox("Payoff", ["Call", "Put"])

    # BS analytic benchmark via QuantLib
    bs_price = price_european_option(
        S0=S0, K=K, T=T, r=r, sigma=sigma_bs,
        option_type=payoff_type.lower(), engine="analytic"
    )
    st.subheader("Black-Scholes Benchmark")
    st.metric(f"BS {payoff_type}", f"{bs_price:.4f}")

    # Monte Carlo comparisons
    records = []
    for model in models:
        paths = run_paths(model, S0, r, nsteps, nsim, T, seed)
        # final payoffs
        S_T = paths.iloc[-1].values
        if payoff_type == "Call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        discount = np.exp(-r * T)
        price_mc = discount * np.mean(payoffs)
        stderr = discount * np.std(payoffs, ddof=1) / np.sqrt(nsim)
        diff = 100 * (price_mc - bs_price) / bs_price
        records.append({
            "Model": model,
            "MC Price": price_mc,
            "Std Error": stderr,
            "Δ vs BS (%)": diff
        })

    df = pd.DataFrame(records).set_index("Model")
    st.subheader("MC vs BS Comparison")
    st.table(df.style.format({
        "MC Price": "{:.4f}",
        "Std Error": "{:.4f}",
        "Δ vs BS (%)": "+.2f%"
    }))

    # Convergence plot
    st.subheader("Convergence of MC Estimate")
    fig, ax = plt.subplots()
    for model in models:
        paths = run_paths(model, S0, r, nsteps, nsim, T, seed)
        S_T = paths.iloc[-1].values
        if payoff_type == "Call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        cum_mean = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)
        ax.plot(cum_mean, lw=1, label=model)

    ax.axhline(bs_price, color="k", lw=2, linestyle="--", label="BS analytic")
    ax.set_xlabel("Path index")
    ax.set_ylabel("Cumulative estimate")
    ax.legend(loc="lower right")
    st.pyplot(fig, use_container_width=True)

    # Payoff distribution
    st.subheader("Payoff Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(payoffs, bins=50, density=True, alpha=0.6)
    ax2.set_xlabel("Payoff")
    ax2.set_ylabel("Density")
    st.pyplot(fig2, use_container_width=True)
