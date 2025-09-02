# src/models/variance_reduction.py
from __future__ import annotations

import math
import numpy as np
from typing import Optional
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1) Control variate on S_T (q-aware)
# ---------------------------------------------------------------------------
def apply_control_variate(
    ST: np.ndarray,
    discounted_payoff: np.ndarray,
    S0: float,
    r: float,
    T: float,
    q: float = 0.0,
) -> np.ndarray:
    """
    Classic control variate using S_T with known expectation E[S_T] = S0 * exp((r - q) * T).

    Parameters
    ----------
    ST : (n_paths,) terminal spot levels
    discounted_payoff : (n_paths,) array of e^{-rT} * payoff
    S0, r, T, q : model inputs under Q

    Returns
    -------
    adjusted : (n_paths,) adjusted discounted payoff with reduced variance
    """
    ST = np.asarray(ST, dtype=float)
    Y = np.asarray(discounted_payoff, dtype=float)

    exp_ST = S0 * math.exp((r - q) * T)
    var_ST = float(np.var(ST, ddof=1))
    if var_ST <= 0.0:
        return Y.copy()

    cov = float(np.cov(Y, ST, ddof=1)[0, 1])
    beta = cov / var_ST
    return Y - beta * (ST - exp_ST)


# ---------------------------------------------------------------------------
# 2) Brownian Bridge utilities for GBM in log-space
# ---------------------------------------------------------------------------
def brownian_bridge_crossing_prob(
    x0: np.ndarray, x1: np.ndarray, b: float, sigma2_dt: float
) -> np.ndarray:
    """
    Crossing probability of a *Brownian bridge* between x0 and x1 against a *constant level* b
    over one step of variance = sigma^2 * Δt (in *log* space).

    Formula (standard):  P(hit) = exp( -2 * (x0 - b) * (x1 - b) / (sigma^2 Δt) )
    when x0 and x1 are on the same side; if on opposite sides, P(hit)=1.

    Inputs are broadcastable arrays. Returns array clipped to [0,1].
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    same_side = (x0 - b) * (x1 - b) > 0.0
    p = np.ones_like(x0, dtype=float)
    expo = -2.0 * (x0 - b) * (x1 - b) / max(sigma2_dt, 1e-16)
    p[same_side] = np.exp(np.minimum(expo[same_side], 0.0))
    return np.clip(p, 0.0, 1.0)


def refine_gbm_bridge(
    paths: np.ndarray,
    T: float,
    sigma: float,
    refine_factor: int = 4,
) -> np.ndarray:
    """
    Densify GBM paths using a *log-space Brownian bridge* between every two consecutive
    time steps. Does NOT change endpoints; preserves terminal distribution exactly and
    samples correct conditional interiors.

    Parameters
    ----------
    paths : array (n_steps+1, n_paths) of spot levels
    T     : maturity
    sigma : diffusion vol of GBM
    refine_factor : k ≥ 2 ⇒ each coarse step is split into k substeps.
                    Output has n_steps*k + 1 rows.

    Returns
    -------
    refined : array (n_steps*refine_factor + 1, n_paths)
    """
    if refine_factor <= 1:
        return np.array(paths, copy=True)

    paths = np.asarray(paths, dtype=float)
    n_steps_coarse = paths.shape[0] - 1
    n_paths = paths.shape[1]
    dt = T / n_steps_coarse
    k = int(refine_factor)

    # log-paths
    X = np.log(np.clip(paths, 1e-300, None))
    out = np.empty((n_steps_coarse * k + 1, n_paths), dtype=float)
    out[0] = X[0]

    rng = np.random.default_rng()

    for j in range(n_steps_coarse):
        x0 = X[j]
        x1 = X[j + 1]
        out[j * k] = x0
        # Brownian bridge within the step (conditional on endpoints)
        for m in range(1, k):
            theta = m / k  # relative time inside the coarse interval
            mean = (1.0 - theta) * x0 + theta * x1
            var = (sigma * sigma) * dt * theta * (1.0 - theta)
            std = math.sqrt(max(var, 0.0))
            out[j * k + m] = mean + std * rng.standard_normal(n_paths)

    out[-1] = X[-1]
    return np.exp(out)


def apply_brownian_bridge(
    paths: np.ndarray,
    T: float,
    sigma: float,
    refine_factor: int = 4,
    mode: str = "refine",
) -> np.ndarray:
    """
    Brownian Bridge helper for GBM paths.

    - mode='refine' (recommended): return a *densified* path using conditional Brownian bridge
      between coarse points. Use this for barrier/Parisian to reduce discrete-monitoring bias.
    - mode='resample' (legacy): re-sample interior points in-place with a single draw per time layer.
      Not recommended for barrier pricing; kept for backwards-compatibility.

    Returns array with the same or increased number of time steps depending on mode.
    """
    if mode == "refine":
        return refine_gbm_bridge(paths, T, sigma, refine_factor=refine_factor)

    # Legacy heuristic "resample" (single interior layer redraw). Prefer 'refine'.
    n_steps, n_paths = paths.shape
    log_paths = np.log(np.clip(paths, 1e-300, None))
    rng = np.random.default_rng()
    for t in range(1, n_steps - 1):
        theta = t / (n_steps - 1)
        mean = (1.0 - theta) * log_paths[0] + theta * log_paths[-1]
        var = (sigma * sigma) * T * theta * (1.0 - theta)
        std = math.sqrt(max(var, 0.0))
        log_paths[t] = mean + std * rng.standard_normal(n_paths)
    return np.exp(log_paths)


# ---------------------------------------------------------------------------
# 3) Stratified normals (time-step-wise stratification)
# ---------------------------------------------------------------------------
def apply_stratified(nsim: int, nsteps: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Return standard normals Z with per-step stratification across paths.

    Shape: (nsteps, nsim)

    Implementation details:
    - Each step draws a random offset and uses stratified uniforms on [0,1].
    - Values are clipped away from {0,1} to avoid infinities in norm.ppf.
    - A fresh random permutation per step avoids latent path ordering artifacts.
    """
    rng = rng or np.random.default_rng()
    eps = 1e-12
    Z = np.empty((nsteps, nsim), dtype=float)
    idx = np.arange(nsim)
    for t in range(nsteps):
        # stratified uniforms with random offset
        u = (idx + rng.random(nsim)) / nsim
        u = np.clip(u, eps, 1.0 - eps)
        rng.shuffle(u)  # decorrelate across paths
        Z[t, :] = norm.ppf(u)
    return Z
