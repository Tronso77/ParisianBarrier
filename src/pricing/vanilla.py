import QuantLib as ql


def price_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    engine: str = "analytic",      # "analytic" or "mc"
    mc_samples: int = 100_000,
    mc_steps: int = 100,
    mc_seed: int = 42,
    antithetic: bool = False
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
    mc_seed : int
        RNG seed for MC engine
    antithetic : bool
        Whether to use antithetic variates in MC

    Returns
    -------
    float
        Option price
    """
    # 1) Setup evaluation date & maturity
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    # convert T in years to a QuantLib Date
    maturity = today + ql.Period(int(T * 365 + 0.5), ql.Days)

    # 2) Market data
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed())
    )
    rf_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )

    # 3) Payoff & Exercise
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type.lower() == "call" else ql.Option.Put,
        K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    # 4) Process setup
    process = ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)

    # 5) Choose engine
    if engine == "analytic":
        engine_ql = ql.AnalyticEuropeanEngine(process)
    else:
        # Monte Carlo engine with pseudo-random Sobol / PRNG
        engine_ql = ql.MCEuropeanEngine(
            process,
            "pseudorandom",
            timeSteps=mc_steps,
            antitheticVariate=antithetic,
            requiredSamples=mc_samples,
            seed=mc_seed
        )
    option.setPricingEngine(engine_ql)

    # 6) Return NPV
    return option.NPV()


