# src/models/simulator.py
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.stats import invgauss

from .params import param_assign
from .variance_reduction import apply_stratified  # Z ~ N(0,1), shape (nsteps, nsim)


def simulate_paths(
    model: str,
    nsteps: int,
    nsim: int,
    dt: float,
    seed: int | None = None,
    *,
    # optional overrides (use any subset)
    S0: float | None = None,
    mu: float | None = None,     # direct drift override (e.g., r - q for GBM)
    r: float | None = None,      # if provided with q for GBM, we set mu := r - q
    q: float | None = None,
    sigma: float | None = None,
    stratified: bool = False,
    antithetic: bool = False,
    gbm_exact_terminal: bool = True,  # <-- NEW: exact one-step draw for GBM terminal
) -> pd.DataFrame:
    """
    Unified simulator returning a DataFrame of shape (nsim, nsteps+1) with columns 0..nsteps.

    Conventions (per PARAMETER layout):
      - GBM/ABM/CEV: (S0, mu, sigma) where 'mu' is the arithmetic drift.
        For GBM under Q: pass (r, q) and we will set mu := r - q.
      - Antithetic: total paths remains = nsim (pairs Z and -Z).
    """
    rng = np.random.default_rng(seed)
    model = model.upper()

    # start from defaults; override below
    p = list(param_assign(model))

    # S0 override if relevant
    if S0 is not None and len(p) > 0 and model in {"BM", "ABM", "GBM", "HESTON", "CEV", "VG", "MJD", "VGCIR"}:
        p[0] = float(S0)

    # direct sigma override where the last entry is sigma
    if sigma is not None and model in {"BM", "GBM", "CEV"}:
        p[-1] = float(sigma)

    if model == "GBM":
        # p = (S0_p, mu_p, sig)
        S0_p, mu_p, sig = p

        # effective drift:
        # 1) if 'mu' is given, use it
        # 2) else if 'r' is given, force risk-neutral mu := r - q (q may be 0)
        # 3) else use default mu_p
        if mu is not None:
            mu_eff = float(mu)
        elif r is not None:
            mu_eff = float(r) - float(q or 0.0)
        else:
            mu_eff = float(mu_p)

        # helper to build normals (time along rows)
        def draw_normals(m_paths: int) -> np.ndarray:
            if stratified:
                return apply_stratified(m_paths, nsteps)  # (nsteps, m_paths)
            return rng.standard_normal((nsteps, m_paths))

        # antithetic support (keeps total = nsim)
        def apply_antithetic(Z: np.ndarray) -> np.ndarray:
            if not antithetic:
                return Z
            half = (Z.shape[1] + 1) // 2
            Zh = Z[:, :half]
            Z2 = np.hstack([Zh, -Zh])
            return Z2[:, :Z.shape[1]]

        # We offer two GBM schemes:
        # (A) exact-terminal (single N(0,1) per path, then expand to a full path)
        # (B) multi-step exact log scheme (sum of independent normals)
        if gbm_exact_terminal:
            # ---- (A) exact terminal draw ----
            # Draw one Z_T per path (with antithetic pairing if requested)
            m = nsim
            if antithetic:
                half = (nsim + 1) // 2
                zT_half = rng.standard_normal(half)
                zT = np.concatenate([zT_half, -zT_half])[:nsim]
            else:
                zT = rng.standard_normal(nsim)

            # Terminal price
            T = nsteps * dt
            log_ST = math.log(S0 if S0 is not None else S0_p) + (mu_eff - 0.5 * sig * sig) * T + sig * math.sqrt(T) * zT
            ST = np.exp(log_ST)

            # Build a simple geometric bridge path for output shape (nsteps+1)
            # (We don't bias pricing since payoff only depends on ST; paths are for barrier bookkeeping.)
            times = np.linspace(0.0, T, nsteps + 1)
            paths = np.empty((nsim, nsteps + 1), dtype=float)
            S0_used = (S0 if S0 is not None else S0_p)
            for i in range(nsim):
                # linear in log-space between ln S0 and ln ST_i
                paths[i, :] = np.exp(np.linspace(math.log(S0_used), log_ST[i], nsteps + 1))
            return pd.DataFrame(paths, columns=range(nsteps + 1))

        else:
            # ---- (B) multi-step exact log scheme ----
            Z = draw_normals(nsim)                 # (nsteps, nsim)
            Z = apply_antithetic(Z)                # (nsteps, nsim)
            v = sig * math.sqrt(dt)
            drift = (mu_eff - 0.5 * sig * sig) * dt
            X = np.empty((nsteps + 1, nsim))
            X[0] = math.log(S0 if S0 is not None else S0_p)
            X[1:] = X[0:1, :] + np.cumsum(drift + v * Z, axis=0)
            S = np.exp(X)                          # (nsteps+1, nsim)
            return pd.DataFrame(S.T, columns=range(nsteps + 1))

    if model == "BM":
        S0_p, sig = (p[0], p[1])
        dW = sig * math.sqrt(dt) * rng.standard_normal((nsteps, nsim))
        X = np.empty((nsteps + 1, nsim)); X[0] = S0_p; X[1:] = S0_p + np.cumsum(dW, axis=0)
        return pd.DataFrame(X.T, columns=range(nsteps + 1))

    if model == "ABM":
        S0_p, mu_p, sig = p
        mu_eff = float(mu) if mu is not None else float(mu_p)
        dX = mu_eff * dt + sig * math.sqrt(dt) * rng.standard_normal((nsteps, nsim))
        X = np.empty((nsteps + 1, nsim)); X[0] = S0_p; X[1:] = S0_p + np.cumsum(dX, axis=0)
        return pd.DataFrame(X.T, columns=range(nsteps + 1))

    if model == "POI":
        (lam,) = p
        dN = rng.poisson(lam * dt, (nsteps, nsim))
        N = np.vstack([np.zeros((1, nsim)), np.cumsum(dN, axis=0)])
        return pd.DataFrame(N.T, columns=range(nsteps + 1))

    if model == "GAMMA":
        alpha, lam = p
        theta = 1.0 / lam
        G = rng.gamma(dt * alpha, theta, (nsteps, nsim))
        X = np.vstack([np.zeros((1, nsim)), np.cumsum(G, axis=0)])
        return pd.DataFrame(X.T, columns=range(nsteps + 1))

    if model == "VG":
        S0_p, theta, sig, kappa = p
        G = rng.gamma(dt / kappa, kappa, (nsteps, nsim))
        dW = rng.standard_normal((nsteps, nsim))
        dX = theta * G + sig * np.sqrt(G) * dW
        lp = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
        return pd.DataFrame((S0_p * np.exp(lp)).T, columns=range(nsteps + 1))

    if model == "MJD":
        S0_p, sig, lam, muJ, sigJ = p
        dW = sig * math.sqrt(dt) * rng.standard_normal((nsteps, nsim))
        dN = rng.poisson(lam * dt, (nsteps, nsim))
        dJ = muJ * dN + sigJ * np.sqrt(np.maximum(dN, 0)) * rng.standard_normal((nsteps, nsim))
        dX = dW + dJ
        lp = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
        return pd.DataFrame((S0_p * np.exp(lp)).T, columns=range(nsteps + 1))

    if model == "NIG":
        if len(p) == 4:
            S0_p, theta, sig, kappa = p
        else:
            S0_p, theta, sig = p; kappa = 1.0
        lam = dt / math.sqrt(kappa); nu = 1.0 / math.sqrt(kappa)
        # SciPy's random_state can take a Generator in recent versions
        GY = invgauss.rvs(mu=lam, scale=nu, size=(nsteps, nsim), random_state=rng)
        dW = rng.standard_normal((nsteps, nsim))
        dX = theta * GY + sig * np.sqrt(GY) * dW
        lp = np.vstack([np.zeros((1, nsim)), np.cumsum(dX, axis=0)])
        return pd.DataFrame((S0_p * np.exp(lp)).T, columns=range(nsteps + 1))

    if model == "CEV":
        S0_p, mu_p, beta, sig = p
        mu_eff = float(mu) if mu is not None else float(mu_p)
        X = np.zeros((nsteps + 1, nsim)); X[0] = S0_p
        dW = math.sqrt(dt) * rng.standard_normal((nsteps, nsim))
        for j in range(1, nsteps + 1):
            S_prev = np.clip(X[j - 1], 1e-10, None)
            X[j] = S_prev + mu_eff * S_prev * dt + sig * (S_prev ** beta) * dW[j - 1]
            X[j] = np.clip(X[j], 0.0, None)
        return pd.DataFrame(X.T, columns=range(nsteps + 1))

    if model == "HESTON":
        S0_p, v0, kappa, theta, sigma_v, rho, r_p = p
        v = np.zeros((nsteps + 1, nsim)); v[0] = v0
        logS = np.zeros_like(v); logS[0] = math.log(S0_p)
        dWv = math.sqrt(dt) * rng.standard_normal((nsteps, nsim))
        dWz = math.sqrt(dt) * rng.standard_normal((nsteps, nsim))
        dZ = rho * dWv + math.sqrt(max(1.0 - rho * rho, 0.0)) * dWz
        for j in range(1, nsteps + 1):
            vp = np.maximum(v[j - 1], 0.0)
            v[j] = np.maximum(vp + kappa * (theta - vp) * dt + sigma_v * np.sqrt(vp) * dWv[j - 1], 0.0)
            logS[j] = logS[j - 1] - 0.5 * vp * dt + np.sqrt(vp) * dZ[j - 1]
        return pd.DataFrame(np.exp(logS).T, columns=range(nsteps + 1))

    if model == "CIR":
        theta_p, kappa, sigma_c, v0 = p
        v = np.zeros((nsteps + 1, nsim)); v[0] = v0
        dW = rng.standard_normal((nsteps, nsim))
        for j in range(1, nsteps + 1):
            vp = np.maximum(v[j - 1], 0.0)
            v[j] = np.maximum(vp + kappa * (theta_p - vp) * dt + sigma_c * np.sqrt(vp * dt) * dW[j - 1], 0.0)
        return pd.DataFrame(v.T, columns=range(nsteps + 1))

    raise ValueError(f"Model '{model}' not supported.")
