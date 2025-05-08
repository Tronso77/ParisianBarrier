import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.special import gamma
from models.params import param_assign

def simulate_paths(model, nsteps, nsim, dt, seed=None, *, S0=None, r=None):
    """
    Unified path simulation interface using QuantLib where available.
    Returns a DataFrame of simulated paths.
    """
    if seed is not None:
        np.random.seed(seed)

    model = model.upper()
    params = list(param_assign(model))
    if S0 is not None:
        params[0] = S0
    if r is not None:
        params[-1] = r
    params = tuple(params)

    if model == "BM":
        return _simulate_bm(params, nsteps, nsim, dt)
    elif model == "ABM":
        return _simulate_abm(params, nsteps, nsim, dt)
    elif model == "POI":
        return _simulate_poisson(params, nsteps, nsim, dt)
    elif model == "GAMMA":
        return _simulate_gamma(params, nsteps, nsim, dt)
    elif model == "GBM":
        return _simulate_gbm(params, nsteps, nsim, dt)
    elif model == "VG":
        return _simulate_vg(params, nsteps, nsim, dt)
    elif model == "MJD":
        return _simulate_mjd(params, nsteps, nsim, dt)
    elif model == "NIG":
        return _simulate_nig(params, nsteps, nsim, dt)
    elif model == "KJD":
        return _simulate_kjd(params, nsteps, nsim, dt)
    elif model == "CGMY":
        return _simulate_cgmy(params, nsteps, nsim, dt)
    elif model == "HESTON":
        return _simulate_heston(params, nsteps, nsim, dt)
    elif model == "CIR":
        return _simulate_cir(params, nsteps, nsim, dt)
    elif model == "CEV":
        return _simulate_cev(params, nsteps, nsim, dt)
    elif model == "SABR":
        return _simulate_sabr(params, nsteps, nsim, dt)
    elif model == "VGCIR":
        return _simulate_vgcir(params, nsteps, nsim, dt)
    else:
        raise ValueError(f"Model '{model}' not supported in simulator.")


# ---------------------------------------------
# Poisson
def _simulate_poisson(params, nsteps, nsim, dt):
    λ = params[0]
    dN = np.random.poisson(λ * dt, (nsteps, nsim))
    N = np.vstack([np.zeros((1, nsim)), np.cumsum(dN, axis=0)])
    return pd.DataFrame(N)


# ---------------------------------------------
# Gamma
def _simulate_gamma(params, nsteps, nsim, dt):
    α, λG = params
    θ = 1 / λG
    G = np.random.gamma(dt * α, θ, (nsteps, nsim))
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(G, axis=0)])
    return pd.DataFrame(X)


# ---------------------------------------------
# Brownian Motion (no drift) via QuantLib
def _simulate_bm(params, nsteps, nsim, dt):
    _, σ = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    # Gaussian with zero drift
    ugrng = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng  = ql.GaussianRandomSequenceGenerator(ugrng)
    process = ql.BrownianMotion(σ, 0, dt)  # μ=0
    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# Arithmetic Brownian Motion
def _simulate_abm(params, nsteps, nsim, dt):
    μ, σ = params
    dW = σ * np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dX = μ * dt + dW
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(X)


# ---------------------------------------------
# Geometric Brownian Motion (QuantLib)
def _simulate_gbm(params, nsteps, nsim, dt):
    S0, μ, σ = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, μ,       ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), σ, ql.Actual365Fixed())
    )

    process = ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# Variance Gamma (QuantLib)
def _simulate_vg(params, nsteps, nsim, dt):
    S0, θ, σ, ν = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))

    process = ql.VarianceGammaProcess(spot, div_ts, rf_ts, σ, ν, θ)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# Merton Jump Diffusion (QuantLib)
def _simulate_mjd(params, nsteps, nsim, dt):
    S0, σ, λ, μJ, σJ = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), σ, ql.Actual365Fixed())
    )

    process = ql.MertonJumpProcess(spot, div_ts, rf_ts, vol_ts, λ, μJ, σJ)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# Normal Inverse Gaussian
def _simulate_nig(params, nsteps, nsim, dt):
    μ, θ, σ, κ = params
    from scipy.stats import invgauss
    lam  = dt / np.sqrt(κ); nu = 1 / np.sqrt(κ)
    GY1T = invgauss.rvs(mu=lam, scale=nu, size=(nsteps, nsim))
    dW   = np.random.randn(nsteps, nsim)
    dX   = θ * GY1T + σ * np.sqrt(GY1T) * dW + μ * dt
    log_paths = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(np.exp(log_paths))


# ---------------------------------------------
# Kou Jump Diffusion (QuantLib)
def _simulate_kjd(params, nsteps, nsim, dt):
    S0, σ, λ, p, η1, η2 = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))

    process = ql.KouJumpDiffusionProcess(spot, div_ts, rf_ts, σ, λ, p, η1, η2)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# CGMY - Need fix
def _simulate_cgmy(params, nsteps, nsim, dt, trunc_eps=1e-4):
    μ, C, Gp, M, Y = params
    T = nsteps * dt
    time_grid = np.linspace(0, T, nsteps + 1)
    lam = C * gamma(1 - Y) * (Gp**(Y - 1) + M**(Y - 1)); lam = float(abs(lam))
    num_jumps = np.random.poisson(lam * T, size=nsim)
    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        nj = num_jumps[i]
        if nj == 0: continue
        jump_times = np.random.uniform(0, T, nj)
        jump_sizes = np.random.exponential(1/Gp, nj) - np.random.exponential(1/M, nj)
        for jt, js in zip(jump_times, jump_sizes):
            idx = np.searchsorted(time_grid, jt)
            paths[idx:, i] += js
    paths += μ * time_grid[:, None]
    return pd.DataFrame(np.exp(paths))


# ---------------------------------------------
# Heston (QuantLib)
def _simulate_heston(params, nsteps, nsim, dt):
    S0, v0, κ, θ, σ_v, ρ, r = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0,     ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, r,       ql.Actual365Fixed()))

    process = ql.HestonProcess(rf_ts, div_ts, spot, v0, κ, θ, σ_v, ρ)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# CIR (QuantLib)
def _simulate_cir(params, nsteps, nsim, dt):
    θ, κ, σ_c, v0 = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

    process = ql.CoxIngersollRossProcess(rf_ts, κ, θ, σ_c, v0)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# CEV (QuantLib)
def _simulate_cev(params, nsteps, nsim, dt):
    S0, μ, β, σ_c = params
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot   = ql.QuoteHandle(ql.SimpleQuote(S0))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    rf_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, μ,   ql.Actual365Fixed()))

    process = ql.CEVProcess(spot, div_ts, rf_ts, σ_c, β)
    ugrng   = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    grng    = ql.GaussianRandomSequenceGenerator(ugrng)

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        pg   = ql.PathGenerator(process, dt * nsteps, nsteps, grng, False)
        ql_p = pg.next().value()
        for j in range(nsteps + 1):
            paths[j, i] = ql_p[j]
    return pd.DataFrame(paths)


# ---------------------------------------------
# SABR - Need fix
def _simulate_sabr(params, nsteps, nsim, dt):
    S0, α0, β, ρ, γ = params
    F = np.zeros((nsteps + 1, nsim))
    v = np.zeros((nsteps + 1, nsim))
    F[0, :], v[0, :] = S0, α0
    dZ = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dW = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    for j in range(1, nsteps + 1):
        v_prev = np.clip(v[j - 1, :], 1e-8, None)
        F_prev = np.clip(F[j - 1, :], 1e-8, None)
        v[j, :] = np.maximum(v_prev + γ * v_prev * dZ[j - 1, :], 0)
        dB      = ρ * dZ[j - 1, :] + np.sqrt(1 - ρ**2) * dW[j - 1, :]
        F[j, :] = np.maximum(F_prev + v_prev * (F_prev**β) * dB, 0)
    return pd.DataFrame(F)


# ---------------------------------------------
# VG-CIR - Need fix
def _simulate_vgcir(params, nsteps, nsim, dt):
    mu, theta, sigma, kappa_vg = params[:4]
    theta_cir, kappa_cir, eta_cir, v0 = params[4:]
    v = np.zeros((nsteps + 1, nsim)); v[0, :] = v0
    dW = np.random.randn(nsteps, nsim)
    for j in range(1, nsteps + 1):
        v_prev = np.maximum(v[j - 1, :], 0)
        v[j, :] = np.maximum(
            v_prev + kappa_cir * (theta_cir - v_prev) * dt + eta_cir * np.sqrt(v_prev * dt) * dW[j - 1, :],
            0
        )
    tau = np.cumsum(0.5 * (v[:-1, :] + v[1:, :]) * dt, axis=0)
    dG = np.random.gamma(tau / kappa_vg, kappa_vg)
    dW = np.random.randn(nsteps, nsim)
    dX = mu * dt + theta * dG + sigma * np.sqrt(dG) * dW
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(X)