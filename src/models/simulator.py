# src/models/simulator.py

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import invgauss
from scipy.special import gamma as gamma_func
from models.params import param_assign


def simulate_paths(model, nsteps, nsim, dt, seed=None, *, S0=None, r=None, sigma=None):
    """
    Unified interface: QL for GBM; NumPy for others.
    Returns a DataFrame of shape (nsteps+1, nsim).
    Overrides:
      - S0: initial spot (first entry in params)
      - r: drift/rate for GBM and CEV
      - sigma: volatility for BM/GBM/CEV
    """
    if seed is not None:
        np.random.seed(seed)

    model = model.upper()
    params = list(param_assign(model))

    # apply overrides
    if S0 is not None:
        params[0] = S0
    if r is not None and model in {"GBM", "CEV"}:
        params[1] = r
    if sigma is not None and model in {"BM", "GBM", "CEV"}:
        params[-1] = sigma
    params = tuple(params)

    # QL-driven GBM
    if model == "GBM":
        return _simulate_gbm_ql(params, nsteps, nsim, dt)

    # NumPy fallbacks
    if model == "BM":        return _simulate_bm(params, nsteps, nsim, dt)
    if model == "ABM":       return _simulate_abm(params, nsteps, nsim, dt)
    if model == "VG":        return _simulate_vg_numpy(params, nsteps, nsim, dt)
    if model == "MJD":       return _simulate_mjd_numpy(params, nsteps, nsim, dt)
    if model == "NIG":       return _simulate_nig(params, nsteps, nsim, dt)
    if model == "CEV":       return _simulate_cev(params, nsteps, nsim, dt)
    if model == "POI":       return _simulate_poisson(params, nsteps, nsim, dt)
    if model == "GAMMA":     return _simulate_gamma(params, nsteps, nsim, dt)
    if model == "HESTON":    return _simulate_heston_numpy(params, nsteps, nsim, dt)
    if model == "CIR":       return _simulate_cir_numpy(params, nsteps, nsim, dt)

    raise ValueError(f"Model '{model}' not supported.")


# ─── QuantLib GBM helper ──────────────────────────────────────────────────────

def _ql_path(process, nsteps, nsim, dt):
    grid  = ql.TimeGrid(dt * nsteps, nsteps)
    urng  = ql.UniformRandomSequenceGenerator(nsteps, ql.UniformRandomGenerator())
    gauss = ql.GaussianRandomSequenceGenerator(urng)
    pg    = ql.GaussianPathGenerator(process, grid, gauss, False)

    arr = np.zeros((nsteps+1, nsim))
    for i in range(nsim):
        path = pg.next().value()
        for t in range(nsteps+1):
            arr[t, i] = path[t]
    return arr


def _simulate_gbm_ql(params, nsteps, nsim, dt):
    S0, mu, sigma = params
    proc = ql.GeometricBrownianMotionProcess(S0, mu, sigma)
    arr  = _ql_path(proc, nsteps, nsim, dt)
    return pd.DataFrame(arr)


# ─── NumPy fallbacks ───────────────────────────────────────────────────────────

def _simulate_bm(params, nsteps, nsim, dt):
    S0, sigma = params
    dW  = sigma * np.sqrt(dt) * np.random.randn(nsteps, nsim)
    X   = np.zeros((nsteps+1, nsim))
    X[0]    = S0
    X[1:]   = S0 + np.cumsum(dW, axis=0)
    return pd.DataFrame(X)


def _simulate_abm(params, nsteps, nsim, dt):
    S0, mu, sigma = params
    inc = mu*dt + sigma*np.sqrt(dt)*np.random.randn(nsteps, nsim)
    cum = np.cumsum(inc, axis=0)
    X   = np.zeros((nsteps+1, nsim))
    X[0]   = S0
    X[1:]  = S0 + cum
    return pd.DataFrame(X)


def _simulate_poisson(params, nsteps, nsim, dt):
    lam = params[0]
    dN  = np.random.poisson(lam*dt, (nsteps, nsim))
    N   = np.vstack([np.zeros((1, nsim)), np.cumsum(dN, axis=0)])
    return pd.DataFrame(N)


def _simulate_gamma(params, nsteps, nsim, dt):
    alpha, lam = params
    theta      = 1.0/lam
    G          = np.random.gamma(dt*alpha, theta, (nsteps, nsim))
    X          = np.vstack([np.zeros((1, nsim)), np.cumsum(G, axis=0)])
    return pd.DataFrame(X)


def _simulate_vg_numpy(params, nsteps, nsim, dt):
    S0, theta, sigma, kappa = params
    G  = np.random.gamma(dt/kappa, kappa, (nsteps, nsim))
    dW = np.random.randn(nsteps, nsim)
    dX = theta*G + sigma*np.sqrt(G)*dW
    lp = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(S0 * np.exp(lp))


def _simulate_mjd_numpy(params, nsteps, nsim, dt):
    S0, sigma, lam, muJ, sigmaJ = params
    dW = sigma*np.sqrt(dt)*np.random.randn(nsteps, nsim)
    dN = np.random.poisson(lam*dt, (nsteps, nsim))
    dJ = muJ*dN + sigmaJ*np.sqrt(dN)*np.random.randn(nsteps, nsim)
    dX = dW + dJ
    lp = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(S0 * np.exp(lp))


def _simulate_nig(params, nsteps, nsim, dt):
    if len(params) == 4:
        S0, theta, sigma, kappa = params
    else:
        S0, theta, sigma = params
        kappa = 1.0
    lam = dt / np.sqrt(kappa)
    nu  = 1.0 / np.sqrt(kappa)
    GY  = invgauss.rvs(mu=lam, scale=nu, size=(nsteps, nsim))
    dW  = np.random.randn(nsteps, nsim)
    dX  = theta * GY + sigma * np.sqrt(GY) * dW
    lp  = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(S0 * np.exp(lp))


def _simulate_cev(params, nsteps, nsim, dt):
    """
    CEV process: dS = μ S_t dt + σ S_t^beta dW_t
    Euler‐Maruyama discretization.
    """
    S0, mu, beta, sigma = params
    # allocate array: rows 0..nsteps, cols 0..nsim-1
    X = np.zeros((nsteps+1, nsim))
    X[0, :] = S0

    # pre‐draw all increments
    dW = np.sqrt(dt) * np.random.randn(nsteps, nsim)

    for j in range(1, nsteps+1):
        S_prev = np.clip(X[j-1, :], 1e-8, None)    # avoid zero
        # Euler step
        X[j, :] = (
            S_prev
            + mu    * S_prev * dt
            + sigma * (S_prev ** beta) * dW[j-1, :]
        )
        # enforce non‐negative
        X[j, :] = np.clip(X[j, :], 0.0, None)

    return pd.DataFrame(X)



def _simulate_heston_numpy(params, nsteps, nsim, dt):
    S0, v0, kappa, theta, sigma_v, rho, r = params
    v = np.zeros((nsteps+1, nsim)); logS = np.zeros_like(v)
    v[0] = v0; logS[0] = np.log(S0)
    dWv = np.sqrt(dt)*np.random.randn(nsteps, nsim)
    dWz = np.sqrt(dt)*np.random.randn(nsteps, nsim)
    dZ  = rho*dWv + np.sqrt(1-rho**2)*dWz
    for j in range(1, nsteps+1):
        vp      = np.maximum(v[j-1], 0)
        v[j]    = np.maximum(vp + kappa*(theta-vp)*dt + sigma_v*np.sqrt(vp)*dWv[j-1], 0)
        logS[j] = logS[j-1] - 0.5*vp*dt + np.sqrt(vp)*dZ[j-1]
    return pd.DataFrame(np.exp(logS))


def _simulate_cir_numpy(params, nsteps, nsim, dt):
    theta, kappa, sigma_c, v0 = params
    v = np.zeros((nsteps+1, nsim))
    v[0] = v0
    dW = np.random.randn(nsteps, nsim)
    for j in range(1, nsteps+1):
        vp     = np.maximum(v[j-1], 0)
        v[j]   = np.maximum(vp + kappa*(theta-vp)*dt + sigma_c*np.sqrt(vp*dt)*dW[j-1], 0)
    return pd.DataFrame(v)
