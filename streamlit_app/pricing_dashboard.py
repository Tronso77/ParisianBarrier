# streamlit_app/dashboards/pricing.py

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ensure src/ on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from pricing.vanilla import price_european_option


def show_pricing():
    st.title("💰 European Option Pricing")
    st.write("Compare analytic Black–Scholes prices to Monte Carlo estimates.")

    # Sidebar inputs
    st.sidebar.header("Market & Payoff Inputs")
    S0       = st.sidebar.number_input("Spot price S₀",       min_value=0.01, max_value=1e6,   value=100.0)
    K        = st.sidebar.number_input("Strike K",            min_value=0.01, max_value=1e6,   value=100.0)
    T        = st.sidebar.number_input("Maturity (yrs)",      min_value=0.01, max_value=5.0,   value=1.0, step=0.01)
    r        = st.sidebar.number_input("Risk-free rate r",    min_value=0.0,  max_value=1.0,   value=0.05, step=0.001)
    sigma_bs = st.sidebar.number_input("Volatility σ (BS)",    min_value=0.01, max_value=5.0,   value=0.2,  step=0.01)
    payoff   = st.sidebar.selectbox("Payoff", ["Call", "Put"])

    st.sidebar.markdown("---")
    st.sidebar.header("Monte Carlo Settings")
    engine      = st.sidebar.selectbox("Engine", ["analytic", "mc"], index=0)
    mc_samples  = st.sidebar.slider("MC Samples", 1000, 200_000, 20_000, step=1000)
    mc_steps    = st.sidebar.slider("MC Steps", 10, 1000, 100, step=10)
    antithetic  = st.sidebar.checkbox("Antithetic variates", True)
    mc_seed     = st.sidebar.number_input("MC seed", 0, 99999, 42)

    # Compute prices
    # Analytic BS
    bs_price = price_european_option(
        S0=S0, K=K, T=T, r=r, sigma=sigma_bs,
        option_type=payoff.lower(), engine="analytic"
    )
    st.subheader("Black–Scholes Analytic Price")
    st.metric(f"BS {payoff}", f"{bs_price:.4f}")

    # Monte Carlo if selected
    if engine == "mc":
        mc_price = price_european_option(
            S0=S0, K=K, T=T, r=r, sigma=sigma_bs,
            option_type=payoff.lower(), engine="mc",
            mc_samples=mc_samples, mc_steps=mc_steps,
            mc_seed=mc_seed, antithetic=antithetic
        )
        st.subheader("Monte Carlo Estimate")
        st.metric(f"MC {payoff}", f"{mc_price:.4f}")

        # Optional: show convergence
        st.subheader("MC Convergence (batch means)")
        # generate incremental batches
        batch_size = mc_samples // 10
        prices = []
        for i in range(1, 11):
            price_i = price_european_option(
                S0=S0, K=K, T=T, r=r, sigma=sigma_bs,
                option_type=payoff.lower(), engine="mc",
                mc_samples=batch_size*i, mc_steps=mc_steps,
                mc_seed=mc_seed, antithetic=antithetic
            )
            prices.append(price_i)
        fig, ax = plt.subplots()
        ax.plot([batch_size*i for i in range(1,11)], prices, marker='o')
        ax.axhline(bs_price, color='k', linestyle='--', label='BS analytic')
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("MC price")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
