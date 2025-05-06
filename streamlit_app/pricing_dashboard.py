# streamlit_app/pricing_dashboard.py

import os, sys

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.params       import param_assign
from models.simulator    import simulate_paths
from models.monte_carlo  import MonteCarloEngine
from pricing.payoff      import (
    payoff_european_call,
    payoff_european_put,
    autocallable_payoff,
)

def show_pricing_dashboard():
    st.header("3️⃣ Option Pricing")

    # ── Sidebar inputs ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Pricing Settings")

        # model & simulation
        model    = st.selectbox("Underlying Model", ["GBM","VG","NIG","MJD","KJD","HESTON"])
        maturity = st.number_input("Maturity (years)", 0.01, 5.0, 1.0, step=0.01)
        nsteps   = st.number_input("Time steps", 10, 2000, 252, step=1)
        nsim     = st.number_input("Paths", 1_000, 1_000_000, 100_000, step=1_000)
        seed     = st.number_input("RNG seed", 0, 10_000, 42, step=1)
        S0 = st.number_input("Spot price S₀", 1.0, 10_000.0, 100.0, step=1.0)
        r  = st.number_input("Risk‑free rate r", 0.0, 0.20, 0.05, step=0.005)


        # payoff
        payoff_type = st.selectbox("Payoff type", ["European Call","European Put","Autocallable"])
        if payoff_type in ["European Call","European Put"]:
            K = st.number_input("Strike (K)", 0.01, 1e6, 100.0, step=1.0)
        else:
            notional        = st.number_input("Notional", 1.0, 1e6, 100.0)
            barrier         = st.slider("Autocall barrier (×S₀)", 0.1, 3.0, 1.0, step=0.01)
            coupon          = st.number_input("Coupon per call", 0.0, 1.0, 0.05)
            mat_coupon      = st.number_input("Coupon at maturity", 0.0, 1.0, 0.10)
            knock_in_level  = st.slider("Knock‑in level (×S₀)", 0.0, 1.0, 0.7, step=0.01)
            call_fracs      = st.multiselect(
                                   "Call dates (fractions of T)",
                                   [0.25,0.5,0.75,1.0],
                                   default=[0.25,0.5,0.75]
                               )

        # variance reduction
        st.subheader("Variance Reduction")
        use_antithetic = st.checkbox("Antithetic Variates", value=True)
        use_stratified = st.checkbox("Stratified Sampling", value=False)
        strata         = st.slider("Number of strata", 2, 50, 10) if use_stratified else None

    # ── Run pricing ──────────────────────────────────────────────────────────────
    if st.button("Price"):
        dt     = maturity / nsteps
        params = param_assign(model, S0=S0, r=r)

        # build engine with new signature: MonteCarloEngine(model: str,…)
        engine = MonteCarloEngine(
            model=model,
            params=params,
            nsteps=int(nsteps),
            nsim=int(nsim),
            dt=dt,
            seed=int(seed)
        )

        # simulate
        paths = engine.simulate()

        # show a few sample paths
        st.subheader("Sample Paths (first 10)")
        st.line_chart(paths.iloc[:, :min(10, paths.shape[1])])

        # construct payoff
        if payoff_type == "European Call":
            payoff = payoff_european_call(paths, K)
        elif payoff_type == "European Put":
            payoff = payoff_european_put(paths, K)
        else:
            call_dates = [int(f * nsteps) for f in sorted(call_fracs)]
            payoff, called = autocallable_payoff(
                paths,
                call_dates=call_dates,
                notional=notional,
                barrier=barrier,
                coupon=coupon,
                maturity_coupon=mat_coupon,
                knock_in_level=knock_in_level
            )

        # price
        method = "antithetic" if use_antithetic else ("stratified" if use_stratified else "standard")
        discount = np.exp(-r * maturity)  # flat 5% rate
        result   = engine.price_option(
            payoff,
            discount=discount,
            method=method,
            strata=strata
        )

        # ── Results ───────────────────────────────────────────────────────────────
        st.subheader("📊 Pricing Results")
        col1, col2 = st.columns(2)
        col1.metric("Price", f"{result['price']:.4f}")
        col2.metric("Std. Error", f"{result['stderr']:.4f}")

        # convergence of estimate
        st.subheader("Convergence of Estimate")
        # show cumulative mean vs sample count
        cum_mean = np.cumsum(payoff) / np.arange(1, len(payoff)+1)
        fig, ax = plt.subplots()
        ax.plot(cum_mean, lw=1)
        ax.set_xlabel("Path index")
        ax.set_ylabel("Cumulative mean payoff")
        st.pyplot(fig)

        # payoff distribution
        st.subheader("Payoff Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(payoff, bins=50, density=True, alpha=0.6)
        ax2.set_xlabel("Payoff")
        ax2.set_ylabel("Density")
        st.pyplot(fig2)
