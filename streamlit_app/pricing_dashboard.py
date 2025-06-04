import os, sys

import streamlit as st
import numpy as np
import pandas as pd
import mpmath as mp

# ensure your src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pricing.vanilla        import price_european_option
from pricing.shifted_black  import price_shifted_black
from models.params          import param_assign
from models.monte_carlo     import MonteCarloEngine
from pricing.payoff         import payoff_european_call, payoff_european_put

# Black–Scholes closed-form (mpmath already used in vanilla, but kept here for context)
def bs_call_price(S0, K, T, r, sigma):
    return price_european_option(S0, K, T, r, sigma, option_type="call", engine="analytic")

def bs_put_price(S0, K, T, r, sigma):
    return price_european_option(S0, K, T, r, sigma, option_type="put", engine="analytic")


@st.cache_data
def run_mc_paths(model: str, S0: float, r: float, nsteps: int, nsim: int, T: float, seed: int):
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
    return engine.simulate(), engine  # return both simulation and engine for pricing


def show_pricing_dashboard():
    st.header("🔢 Option Pricing — BS vs Shifted-BS vs MC")

    # ── sidebar inputs ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Market & Payoff Inputs")
        S0       = st.number_input("Spot price S₀",     1.0, 1e5,   100.0, step=1.0)
        K        = st.number_input("Strike K",          0.01,1e6,   100.0, step=1.0)
        T        = st.number_input("Maturity (yrs)",   0.01,5.0,     1.0, step=0.01)
        r        = st.number_input("Risk-free rate r",  0.0, 0.2,   0.05, step=0.001)
        sigma_bs = st.number_input("Volatility σ (BS)", 0.01,1.0,   0.20, step=0.01)

        st.markdown("---")
        st.subheader("Choose Pricing Engine")
        engine = st.selectbox("Engine", ["Analytic-BS", "Shifted-BS", "MC"])

        # shift only for Shifted-BS
        shift = 0.0
        if engine == "Shifted-BS":
            shift = st.number_input("Shift amount", 0.0, S0, 10.0, step=1.0)

        # MC settings
        if engine == "MC":
            nsim   = st.slider("Paths (MC)",    1_000, 500_000, 100_000, step=1_000)
            nsteps = st.slider("Time steps",     10,   1_000,    252,   step=10)
            seed   = st.number_input("RNG seed", 0,    10_000,   42,    step=1)

        st.markdown("---")
        st.subheader("Payoff Type")
        payoff_type = st.selectbox("Payoff", ["European Call", "European Put", "Barrier Knock-Out"])

        # barrier settings
        barrier_type = None
        barrier      = None
        rebate       = None
        if payoff_type.startswith("Barrier"):
            barrier_type = st.selectbox("Barrier type", ["up-and-out", "down-and-out"])
            barrier      = st.number_input("Barrier level", 0.0, 10*S0, 120.0, step=1.0)
            rebate       = st.number_input("Rebate on knock-out", 0.0, S0, 0.0, step=0.1)

    # ── compute analytic price ─────────────────────────────────────────────────
    if payoff_type in {"European Call", "European Put"}:
        if engine == "Analytic-BS":
            price = (bs_call_price  if payoff_type=="European Call" else bs_put_price)(
                S0, K, T, r, sigma_bs)
        elif engine == "Shifted-BS":
            price = price_shifted_black(
                S0, K, T, r, sigma_bs, shift,
                option_type="call" if payoff_type=="European Call" else "put")
        else:  # MC
            paths, mc_engine = run_mc_paths("GBM", S0, r, nsteps, nsim, T, seed)
            payoff = (payoff_european_call if payoff_type=="European Call"
                      else payoff_european_put)(paths, K)
            mc   = mc_engine.price_option(payoff, discount=np.exp(-r*T), method="antithetic")
            price = mc["price"]

        st.subheader(f"{payoff_type} price")
        st.metric("Price", f"{price:.4f}")

    # ── barrier pricing via QL analytic engine ─────────────────────────────────
    else:
        import QuantLib as ql
        today = ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
        maturity = today + int(T*365+0.5)

        # market data
        spot = ql.QuoteHandle(ql.SimpleQuote(S0))
        div  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
        rf   = ql.YieldTermStructureHandle(ql.FlatForward(today, r,   ql.Actual365Fixed()))
        vol  = ql.BlackVolTermStructureHandle(
                   ql.BlackConstantVol(today, ql.NullCalendar(), sigma_bs, ql.Actual365Fixed()))

        # payoff & exercise
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity)

        # barrier option
        barrier_type_map = {
            "up-and-out":   ql.Barrier.UpOut,
            "down-and-out": ql.Barrier.DownOut
        }
        barrier_option = ql.BarrierOption(
            barrier_type_map[barrier_type],
            barrier,
            rebate,
            payoff,
            exercise
        )

        # engine
        process = ql.BlackScholesMertonProcess(spot, div, rf, vol)
        barrier_option.setPricingEngine(ql.AnalyticBarrierEngine(process))

        st.subheader(f"{barrier_type.capitalize()} KO Call @ Barrier={barrier}")
        st.metric("Analytic KO price", f"{barrier_option.NPV():.4f}")

    # ── end of show_pricing_dashboard
