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
    """Return delta, gamma, vega, theta for a vanilla option with continuous dividend yield q."""
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity = today + ql.Period(int(T*365+0.5), ql.Days)
    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )
    payoff   = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type=='call' else ql.Option.Put,
        K
    )
    exercise = ql.EuropeanExercise(maturity)
    option   = ql.VanillaOption(payoff, exercise)
    process  = ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)
    engine   = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    return option.delta(), option.gamma(), option.vega(), option.theta()


def compute_implied_vol(target_price, S0, K, T, r, q, option_type, tol=1e-6, max_iter=100):
    """Solve implied volatility by bisection including dividend q."""
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity = today + ql.Period(int(T*365+0.5), ql.Days)
    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    def bs_price(sigma):
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
        )
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type=='call' else ql.Option.Put,
            K
        )
        exercise = ql.EuropeanExercise(maturity)
        opt = ql.VanillaOption(payoff, exercise)
        process = ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return opt.NPV()
    low, high = 1e-6, 5.0
    f_low, f_high = bs_price(low)-target_price, bs_price(high)-target_price
    if f_low*f_high>0: return np.nan
    for _ in range(max_iter):
        mid = 0.5*(low+high)
        f_mid = bs_price(mid)-target_price
        if abs(f_mid)<tol: return mid
        if f_low*f_mid<0: high, f_high = mid, f_mid
        else: low, f_low = mid, f_mid
    return 0.5*(low+high)


def show_pricing_dashboard():
    st.title("Option Price & Implied Vol Explorer")

    # Sidebar inputs
    with st.sidebar:
        st.header("Parameters & Axis")
        S0       = st.number_input("Spot S₀", value=100.0, min_value=0.01)
        r        = st.number_input("Rate r", value=0.05,   min_value=0.0, step=0.001)
        q        = st.number_input("Dividend yield q", value=0.0, min_value=0.0, step=0.001)
        sigma    = st.number_input("Volatility σ", value=0.20, min_value=0.001)
        T        = st.number_input("Maturity (yrs)", value=1.0, min_value=0.01)
        axis     = st.selectbox("X-axis Variable", ["Strike", "Volatility", "Maturity", "Spot", "Rate", "Dividend"])

    # Build x-axis values
    npts = 100
    if axis=="Strike":
        xs = np.linspace(0.5*S0,1.5*S0,npts);        x_label="Strike K"
    elif axis=="Volatility":
        xs = np.linspace(0.01,1.0,npts);               x_label="Volatility σ"
    elif axis=="Maturity":
        xs = np.linspace(0.01,2.0*T,npts);             x_label="Maturity T"
    elif axis=="Spot":
        xs = np.linspace(0.5*S0,1.5*S0,npts);        x_label="Spot S₀"
    elif axis=="Rate":
        xs = np.linspace(0.0,0.2,npts);                x_label="Rate r"
    else:
        xs = np.linspace(0.0,0.2,npts);                x_label="Dividend q"

    # Containers
    call_prices, put_prices = [],[]
    call_ivs, put_ivs       = [],[]
    call_deltas, put_deltas = [],[]

    for x in xs:
        # set varying parameter
        S = S0; K=x if axis=="Strike" else S0
        v = sigma if axis!="Volatility" else x
        t = T     if axis!="Maturity"   else x
        rr= r     if axis!="Rate"       else x
        qq= q     if axis!="Dividend"   else x
        # prices
        call = price_european_option(S0=S, K=K, T=t, r=rr, sigma=v, option_type="call", engine="analytic")
        put  = price_european_option(S0=S, K=K, T=t, r=rr, sigma=v, option_type="put",  engine="analytic")
        call_prices.append(call); put_prices.append(put)
        # implied vols
        call_ivs.append(compute_implied_vol(call, S, K, t, rr, qq, 'call'))
        put_ivs.append(compute_implied_vol(put,  S, K, t, rr, qq, 'put'))
        # delta
        d_call,_,_,_ = compute_ql_greeks(S, K, t, rr, qq, v, 'call')
        d_put,_,_,_  = compute_ql_greeks(S, K, t, rr, qq, v, 'put')
        call_deltas.append(d_call); put_deltas.append(d_put)

    # DataFrames
    df_price = pd.DataFrame({"Call":call_prices, "Put":put_prices}, index=xs); df_price.index.name=x_label
    df_iv    = pd.DataFrame({"Call IV":call_ivs,  "Put IV":put_ivs},      index=xs); df_iv.index.name=x_label
    df_delta = pd.DataFrame({"Call Δ":call_deltas,"Put Δ":put_deltas},    index=xs); df_delta.index.name=x_label

    # Charts and tables
    st.subheader("Option Prices")
    st.line_chart(df_price)

    st.subheader("Delta Sensitivity")
    st.line_chart(df_delta)

    st.subheader("Vanilla Greeks (Base Params)")
    # compute base greeks table
    base_call_greeks = compute_ql_greeks(S0, S0, T, r, q, sigma, 'call')
    base_put_greeks  = compute_ql_greeks(S0, S0, T, r, q, sigma, 'put')
    greeks_df = pd.DataFrame(
        [base_call_greeks, base_put_greeks],
        index=["Call","Put"],
        columns=["Delta","Gamma","Vega","Theta"]
    )
    st.table(greeks_df)

    st.subheader("Implied Volatility")
    st.line_chart(df_iv)


if __name__=="__main__":
    show_pricing_dashboard()
