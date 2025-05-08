# streamlit_app/pricing_dashboard.py

import os, sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpmath as mp

# 1) ensure your src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.params       import param_assign
from models.monte_carlo  import MonteCarloEngine
from pricing.payoff      import payoff_european_call, payoff_european_put

# — Black–Scholes closed-form
def bs_call_price(S0, K, T, r, sigma):
    d1 = (mp.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*mp.sqrt(T))
    d2 = d1 - sigma*mp.sqrt(T)
    return S0*mp.ncdf(d1) - K*mp.e**(-r*T)*mp.ncdf(d2)

def bs_put_price(S0, K, T, r, sigma):
    d1 = (mp.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*mp.sqrt(T))
    d2 = d1 - sigma*mp.sqrt(T)
    return K*mp.e**(-r*T)*mp.ncdf(-d2) - S0*mp.ncdf(-d1)


# cache
@st.experimental_memo(suppress_st_warning=True)
def run_paths(model: str, S0: float, r: float, nsteps: int, nsim: int, T: float, seed: int):
    dt     = T / nsteps
    params = param_assign(model, S0=S0, r=r)
    engine = MonteCarloEngine(
        model=model,
        params=params,
        nsteps=nsteps,
        nsim=nsim,
        dt=dt,
        seed=seed
    )
    return engine.simulate()


def show_pricing_dashboard():
    st.header("Option Pricing — BS Benchmark & MC Comparison")

    # ── sidebar inputs ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ Market & Payoff Inputs")
        S0       = st.number_input("Spot price S₀",     1.0, 1e5,   100.0, step=1.0)
        K        = st.number_input("Strike K",          0.01,1e6,   100.0, step=1.0)
        T        = st.number_input("Maturity (yrs)",   0.01,5.0,     1.0, step=0.01)
        r        = st.number_input("Risk-free rate r",  0.0, 0.2,   0.05, step=0.001)
        sigma_bs = st.number_input("Volatility σ (BS)", 0.01,1.0,   0.20, step=0.01)

        st.markdown("---")
        st.subheader(" MC Simulation Settings")
        models = st.multiselect("MC Models",
                        options=["GBM","Heston"],
                        default=["GBM"])
        nsim   = st.slider("Paths (MC)",    1_000, 500_000, 100_000, step=1_000)
        nsteps = st.slider("Time steps",     10,   1_000,    252,   step=10)
        seed   = st.number_input("RNG seed", 0,    10_000,   42,    step=1)

        st.markdown("---")
        payoff_type = st.selectbox("Payoff", ["Call","Put"])

    # ── compute BS benchmark ───────────────────────────────────────────────────
    if payoff_type == "Call":
        bs_price = float(bs_call_price(S0, K, T, r, sigma_bs))
    else:
        bs_price = float(bs_put_price (S0, K, T, r, sigma_bs))

    st.subheader("Black-Scholes Benchmark")
    st.metric(f"BS {payoff_type}", f"{bs_price:.4f}")

    # ── Monte Carlo comparisons ────────────────────────────────────────────────
    records = []
    for model in models:
        paths  = run_paths(model, S0, r, nsteps, nsim, T, seed)
        payoff = (payoff_european_call if payoff_type=="Call" else payoff_european_put)(paths, K)

        engine = MonteCarloEngine(model, param_assign(model,S0=S0,r=r), nsteps, nsim, T/nsteps, seed)
        mc     = engine.price_option(
                     payoff,
                     discount=np.exp(-r*T),
                     method="antithetic"
                 )

        diff   = 100*(mc["price"] - bs_price)/bs_price
        records.append({
            "Model":       model,
            "MC Price":    mc["price"],
            "Std Error":   mc["stderr"],
            "Δ vs BS (%)": diff
        })

    df = pd.DataFrame(records).set_index("Model")
    st.subheader("MC vs BS Comparison")
    st.table(df.style.format({
        "MC Price":"{:.4f}",
        "Std Error":"{:.4f}",
        "Δ vs BS (%)":"{:+.2f}%"
    }))

    # ── Convergence chart ──────────────────────────────────────────────────────
    st.subheader("Convergence of MC Estimate")
    fig, ax = plt.subplots()
    for model in models:
        paths   = run_paths(model, S0, r, nsteps, nsim, T, seed)
        payoff  = (payoff_european_call if payoff_type=="Call" else payoff_european_put)(paths, K)
        cum_mean= np.cumsum(payoff) / np.arange(1, len(payoff)+1)
        ax.plot(cum_mean, lw=1, label=model)

    ax.axhline(bs_price, color="k", lw=2, linestyle="--", label="BS analytic")
    ax.set_xlabel("Path index")
    ax.set_ylabel("Cumulative estimate")
    ax.legend(loc="lower right")
    st.pyplot(fig, use_container_width=True)

    # ── Payoff distribution ────────────────────────────────────────────────────
    st.subheader("Payoff Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(payoff, bins=50, density=True, alpha=0.6)
    ax2.set_xlabel("Payoff")
    ax2.set_ylabel("Density")
    st.pyplot(fig2, use_container_width=True)