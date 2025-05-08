import QuantLib as ql

def price_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    engine: str = "analytic",    # "analytic" or "mc"
    mc_samples: int = 100_000,
    mc_steps: int = 100
) -> float:
    """
    Price a European call or put using QuantLib.

    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    engine : str
        "analytic" for Black-Scholes formula, "mc" for Monte Carlo
    mc_samples : int
        Number of MC samples (if engine="mc")
    mc_steps : int
        Number of time steps in MC (if engine="mc")

    Returns
    -------
    float
        Option price
    """
    # 1) Setup dates
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity = today + int(T * 365 + 0.5)

    # 2) Market data
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0,    ql.Actual365Fixed()))
    rf_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r,      ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )

    # 3) Payoff & exercise
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type.lower()=="call" else ql.Option.Put,
        K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    # 4) Process & engine
    process = ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)
    if engine == "analytic":
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    else:
        mc_engine = ql.MCEuropeanEngine(
            process,
            "PseudoRandom",
            timeSteps=mc_steps,
            requiredSamples=mc_samples
        )
        option.setPricingEngine(mc_engine)

    # 5) Return the NPV
    return option.NPV()