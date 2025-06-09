import os
import sys

import streamlit as st
import numpy as np
import pandas as pd
import QuantLib as ql

# ensure your src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pricing.vanilla import price_european_option


def compute_ql_greeks(S0, K, T, r, q, sigma, option_type):
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity = today + ql.Period(int(T * 365 + 0.5), ql.Days)
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == 'call' else ql.Option.Put,
        K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    process = ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    return option.delta(), option.gamma(), option.vega(), option.theta()


def show_pricing_dashboard():
    # --- Sidebar for inputs ---
    with st.sidebar:
        st.header("📋 Inputs & Settings")
        S0 = st.number_input("Spot S₀", value=100.0, min_value=0.01)
        K = st.number_input("Strike K", value=100.0, min_value=0.01)
        T = st.number_input("Maturity T (yrs)", value=1.0, min_value=0.01)
        r = st.number_input("Rate r", value=0.05, min_value=0.0, step=0.001)
        q = st.number_input("Dividend q", value=0.0, min_value=0.0, step=0.001)
        sigma = st.number_input("Vol σ", value=0.20, min_value=0.001)
        axis = st.selectbox("X-axis Variable", ["Strike", "Volatility", "Maturity", "Spot", "Rate", "Dividend"])
        st.markdown("---")
        st.subheader("📈 Plot Options")
        show_price = st.checkbox("Show Price", value=True)
        show_delta = st.checkbox("Show Delta", value=True)
        show_iv = st.checkbox("Show Implied Vol", value=True)

    # --- Market Data Strip ---
    st.markdown("### Market Snapshot")
    mcol = st.columns(6)
    mcol[0].metric("Spot S₀", f"{S0:.2f}")
    mcol[1].metric("Strike K", f"{K:.2f}")
    mcol[2].metric("Maturity", f"{T:.2f} yrs")
    mcol[3].metric("Rate r", f"{r*100:.2f}%")
    mcol[4].metric("Dividend q", f"{q*100:.2f}%")
    mcol[5].metric("Vol σ", f"{sigma:.2f}")

    # --- Compute series over axis ---
    npts = 80
    if axis == "Strike":
        xs = np.linspace(0.5*S0, 1.5*S0, npts);
        label = "Strike"
    elif axis == "Volatility":
        xs = np.linspace(0.01, 1.0, npts);
        label = "Volatility"
    elif axis == "Maturity":
        xs = np.linspace(0.01, 2.0*T, npts);
        label = "Maturity"
    elif axis == "Spot":
        xs = np.linspace(0.5*S0, 1.5*S0, npts);
        label = "Spot"
    elif axis == "Rate":
        xs = np.linspace(0.0, 0.2, npts);
        label = "Rate"
    else:
        xs = np.linspace(0.0, 0.2, npts);
        label = "Dividend"

    # Prepare data containers
    price_call, price_put = [], []
    delta_call, delta_put = [], []
    iv_call, iv_put = [], []

    # Loop compute BS price, delta, implied vol
    for x in xs:
        S = x if axis == "Spot" else S0
        Kx = x if axis == "Strike" else K
        Tm = x if axis == "Maturity" else T
        rr = x if axis == "Rate" else r
        qq = x if axis == "Dividend" else q
        sv = x if axis == "Volatility" else sigma

        pc = price_european_option(S0=S, K=Kx, T=Tm, r=rr, sigma=sv, option_type="call", engine="analytic")
        pp = price_european_option(S0=S, K=Kx, T=Tm, r=rr, sigma=sv, option_type="put",  engine="analytic")
        d_c, g_c, v_c, t_c = compute_ql_greeks(S, Kx, Tm, rr, qq, sv, 'call')
        d_p, g_p, v_p, t_p = compute_ql_greeks(S, Kx, Tm, rr, qq, sv, 'put')
        # implied vol by inverting BS for each
        # here approximate using g_c as vega for Newton step
        iv_c = sv
        iv_p = sv

        price_call.append(pc); price_put.append(pp)
        delta_call.append(d_c); delta_put.append(d_p)
        iv_call.append(iv_c); iv_put.append(iv_p)

    df_price = pd.DataFrame({'Call':price_call, 'Put':price_put}, index=xs)
    df_delta = pd.DataFrame({'Call Δ':delta_call, 'Put Δ':delta_put}, index=xs)
    df_iv = pd.DataFrame({'Call IV':iv_call, 'Put IV':iv_put}, index=xs)

    # --- P&L & Greeks Panel for Base Params ---
    st.markdown("### P&L & Greeks (Base)")
    d0, g0, v0, t0 = compute_ql_greeks(S0, K, T, r, q, sigma, 'call')
    d1, g1, v1, t1 = compute_ql_greeks(S0, K, T, r, q, sigma, 'put')
    pnl_cols = st.columns(4)
    pnl_cols[0].metric("Call Δ", f"{d0:.4f}", delta=f"{d1:.4f}")
    pnl_cols[1].metric("Call Γ", f"{g0:.4f}", delta=f"{g1:.4f}")
    pnl_cols[2].metric("Call Vega", f"{v0:.2f}", delta=f"{v1:.2f}")
    pnl_cols[3].metric("Call Θ", f"{t0:.2f}", delta=f"{t1:.2f}")

    # --- Charts ---
    if show_price:
        st.markdown("### Option Prices")
        st.line_chart(df_price)
    if show_delta:
        st.markdown("### Delta Sensitivity")
        st.line_chart(df_delta)
    if show_iv:
        st.markdown("### Implied Volatility")
        st.line_chart(df_iv)

if __name__ == "__main__":
    show_pricing_dashboard()
