import numpy as np
import pandas as pd
from scipy.special import gamma
from models.params import param_assign

def simulate_paths(model, nsteps, nsim, dt, seed=None,*, S0=None, r=None):
    """
    Unified path simulation interface for cumulant-based models.
    Returns a DataFrame of simulated paths.
    """
    if seed is not None:
        np.random.seed(seed)
    
    model = model.upper()
    # pull default params (possibly including S0 as first entry)
    params = list(param_assign(model)) # override the initial spot if supplied
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
        raise ValueError(f"Model '{model}' not supported in cumulant-based simulator.")

# ---------------------------------------------
# Poisson
def _simulate_poisson(params, nsteps, nsim, dt):
    lambdaP = params[0]
    dN = np.random.poisson(lambdaP * dt, (nsteps, nsim))
    N = np.vstack([np.zeros((1, nsim)), np.cumsum(dN, axis=0)])
    return pd.DataFrame(N)
# ---------------------------------------------
# Gamma
def _simulate_gamma(params, nsteps, nsim, dt):
    alpha, lambdaG = params
    theta = 1 / lambdaG
    G = np.random.gamma(dt * alpha, theta, (nsteps, nsim))
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(G, axis=0)])
    return pd.DataFrame(X)

# ---------------------------------------------
# Brownian Motion (no drift)
def _simulate_bm(params, nsteps, nsim, dt):
    _, sigma = params  # Ignore mu if provided
    dW = sigma * np.sqrt(dt) * np.random.randn(nsteps, nsim)
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(dW, axis=0)])
    return pd.DataFrame(X)

# ---------------------------------------------
#  Arithmetic Brownian Motion
def _simulate_abm(params, nsteps, nsim, dt):
    mu, sigma = params
    dW = sigma * np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dX = mu * dt + dW
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(X)
# ---------------------------------------------
#  Geometric Brownian Motion
def _simulate_gbm(params, nsteps, nsim, dt):
    S0, mu, sigma = params
    dW = np.random.randn(nsteps, nsim)
    dX = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW
    log_paths = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    # start at S0 and exponentiate the log‐increments
    return pd.DataFrame(S0 * np.exp(log_paths))

# ---------------------------------------------
# Variance Gamma
def _simulate_vg(params, nsteps, nsim, dt):
    mu, theta, sigma, kappa = params
    G = np.random.gamma(dt / kappa, kappa, (nsteps, nsim))
    dW = np.random.randn(nsteps, nsim)
    dX = mu * dt + theta * G + sigma * np.sqrt(G) * dW
    log_paths = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(np.exp(log_paths))

# ---------------------------------------------
# Merton Jump Diffusion
def _simulate_mjd(params, nsteps, nsim, dt):
    mu, sigma, lamb, muZ, sigmaZ = params
    dW = sigma * np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dN = np.random.poisson(lamb * dt, (nsteps, nsim))
    dJ = muZ * dN + sigmaZ * np.sqrt(dN) * np.random.randn(nsteps, nsim)
    dX = mu * dt + dW + dJ
    # simulate log‐price path and then exponentiate so S0=1
    log_paths = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(np.exp(log_paths))

# ---------------------------------------------
# Normal Inverse Gaussian
def _simulate_nig(params, nsteps, nsim, dt):
    mu, theta, sigma, kappa = params
    from scipy.stats import invgauss
    lam = dt / np.sqrt(kappa)
    nu = 1 / np.sqrt(kappa)
    GY1T = invgauss.rvs(mu=lam, scale=nu, size=(nsteps, nsim))
    dW = np.random.randn(nsteps, nsim)
    dX = theta * GY1T + sigma * np.sqrt(GY1T) * dW + mu * dt
    log_paths = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(np.exp(log_paths))

# ---------------------------------------------
# Kou Jump Diffusion
def _simulate_kjd(params, nsteps, nsim, dt):
    mu, sigma, lamb, p, eta1, eta2 = params
    dW = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    N = np.random.poisson(lamb * dt, (nsteps, nsim))
    U = np.random.rand(nsteps, nsim)
    J = (U < p) * np.random.exponential(1 / eta1, (nsteps, nsim)) - (U >= p) * np.random.exponential(1 / eta2, (nsteps, nsim))
    dX = mu * dt + sigma * dW + J * N
    log_paths = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(np.exp(log_paths))

# ---------------------------------------------
# CGMY
def _simulate_cgmy(params, nsteps, nsim, dt, trunc_eps=1e-4):
    mu, C, G, M, Y = params

    # Time grid
    T = nsteps * dt
    time_grid = np.linspace(0, T, nsteps + 1)

    # Approximate Lévy measure truncation
    # Simulate compound Poisson process for large jumps
    lam = C * gamma(1 - Y) * (G**(Y - 1) + M**(Y - 1))
    lam = float(abs(lam))            # ensure scalar ≥0
    num_jumps = np.random.poisson(lam * T, size=nsim)
   

    paths = np.zeros((nsteps + 1, nsim))
    for i in range(nsim):
        nj = num_jumps[i]
        if nj == 0:
            continue
        jump_times = np.random.uniform(0, T, nj)
        jump_sizes = np.random.exponential(scale=1.0/G, size=nj) - np.random.exponential(scale=1.0/M, size=nj)
        for jt, js in zip(jump_times, jump_sizes):
            idx = np.searchsorted(time_grid, jt)
            paths[idx:, i] += js

    paths += mu * time_grid[:, None]
    return pd.DataFrame(np.exp(paths))
# ---------------------------------------------
# Heston Model
def _simulate_heston(params, nsteps, nsim, dt):
    S0, v0, kappa, theta, eta, rho, r = params
    v = np.zeros((nsteps + 1, nsim))
    logS = np.zeros((nsteps + 1, nsim))
    v[0, :] = v0
    logS[0, :] = np.log(S0)
    dWv = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dWz = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dZ = rho * dWv + np.sqrt(1 - rho ** 2) * dWz
    for j in range(1, nsteps + 1):
        v_prev = np.maximum(v[j-1, :], 0)
        v[j, :] = np.maximum(v_prev + kappa * (theta - v_prev) * dt + eta * np.sqrt(v_prev) * dWv[j-1, :], 0)
        logS[j, :] = logS[j-1, :] - 0.5 * v_prev * dt + np.sqrt(v_prev) * dZ[j-1, :]
    S = np.exp(logS)
    return pd.DataFrame(S)

# ---------------------------------------------
# CIR Process
def _simulate_cir(params, nsteps, nsim, dt):
    theta, kappa, eta, v0 = params
    v = np.zeros((nsteps + 1, nsim))
    v[0, :] = v0
    dW = np.random.randn(nsteps, nsim)
    for j in range(1, nsteps + 1):
        v_prev = np.maximum(v[j-1, :], 0)
        v[j, :] = np.maximum(v_prev + kappa * (theta - v_prev) * dt + eta * np.sqrt(v_prev * dt) * dW[j-1, :], 0)
    return pd.DataFrame(v)

# ---------------------------------------------
# CEV Process
def _simulate_cev(params, nsteps, nsim, dt):
    S0, mu, beta, sigma = params
    X = np.zeros((nsteps+1, nsim))
    X[0,:] = S0
    dW = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    for j in range(1, nsteps+1):
        X_prev = np.clip(X[j-1,:], 1e-8, None)          # avoid zero
        X[j,:] = X_prev \
            + mu * dt \
            + sigma * (X_prev ** beta) * dW[j-1,:]
        X[j,:] = np.clip(X[j,:], 0, None)               # enforce non‑negative
    return pd.DataFrame(X)


# ---------------------------------------------
# SABR Process
def _simulate_sabr(params, nsteps, nsim, dt):
    S0, alpha0, beta, rho, gamma = params
    F = np.zeros((nsteps+1, nsim))
    v = np.zeros((nsteps+1, nsim))
    F[0,:], v[0,:] = S0, alpha0
    dZ = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    dW = np.sqrt(dt) * np.random.randn(nsteps, nsim)
    for j in range(1, nsteps+1):
        v_prev = np.clip(v[j-1,:], 1e-8, None)
        F_prev = np.clip(F[j-1,:], 1e-8, None)
        v[j,:] = np.maximum(v_prev + gamma * v_prev * dZ[j-1,:], 0)
        dB     = rho * dZ[j-1,:] + np.sqrt(1 - rho**2) * dW[j-1,:]
        F[j,:] = np.maximum(
            F_prev + v_prev * (F_prev ** beta) * dB,
            0
        )
    return pd.DataFrame(F)


# ---------------------------------------------
# Variance Gamma subordinated by CIR
def _simulate_vgcir(params, nsteps, nsim, dt):
    # Split params: VG first, then CIR
    mu, theta, sigma, kappa_vg = params[:4]
    theta_cir, kappa_cir, eta_cir, v0 = params[4:]

    # 1. Simulate CIR variance process
    v = np.zeros((nsteps + 1, nsim))
    v[0, :] = v0
    dW = np.random.randn(nsteps, nsim)
    for j in range(1, nsteps + 1):
        v_prev = np.maximum(v[j - 1, :], 0)
        v[j, :] = np.maximum(
            v_prev + kappa_cir * (theta_cir - v_prev) * dt + eta_cir * np.sqrt(v_prev * dt) * dW[j - 1, :],
            0
        )

    # 2. Time-change using integral of CIR
    tau = np.cumsum(0.5 * (v[:-1, :] + v[1:, :]) * dt, axis=0)  # Trapezoidal approx

    # 3. Generate VG process with time-changed time grid
    dG = np.random.gamma(tau / kappa_vg, kappa_vg)
    dW = np.random.randn(nsteps, nsim)
    dX = mu * dt + theta * dG + sigma * np.sqrt(dG) * dW
    X = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
    return pd.DataFrame(X)
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------