from typing import Tuple

# ─── Model Defaults ────────────────────────────────────────────
# Every tuple now starts with S0.  For the Lévy models it's
# common to choose S0=100 (or 1.0), but pick whatever baseline you like.
PARAMETERS = {
    "BM":      (100.0, 1.0),                               # S0, sigma
    "ABM":     (100.0, 0.0, 1.0),                          # S0, mu, sigma
    "GBM":     (100.0, 0.05, 0.2),                         # S0, r, sigma
    "POI":     (100.0, 5.0),                               # S0, lambda
    "GAMMA":   (100.0, 5.0, 10.0),                         # S0, alpha, lambda
    "VG":      (100.0, -0.09, 0.2, 0.017),                 # S0, theta, sigma, nu
    "MJD":     (100.0, 0.2, 0.75, -0.05, 0.1),              # S0, sigma, lambda, muJ, sigmaJ
    "NIG":     (100.0, -0.032705, 0.184253536, 0.170829388),# S0, theta, sigma, nu
    "KJD":     (100.0, 0.2, 0.75, 0.4, 0.3, 0.2),           # S0, sigma, lambda, p, eta1, eta2
    "CGMY":    (100.0, 0.6509, 5.853, 18.27, 1.8),         # S0, C, G, M, Y
    "CIR":     (0.04, 0.3, 0.2, 0.04),                     # θ, kappa, sigma, v0
    "HESTON":  (100.0, 0.04, 0.3, 0.2, 0.2, -0.2, 0.01),    # S0, v0, kappa, theta, sigma_v, rho, r
    "CEV":     (100.0, 0.1, -2.0, 0.2),                    # S0, mu, beta, sigma
    "SABR":    (100.0, 0.04, 0.2, -0.2, 0.1),              # S0, alpha0, beta, rho, gamma
    "VGCIR":   (100.0, -0.09, 0.2, 0.017, 0.04, 0.3, 0.2, 0.04),  # S0, θ, σ, ν, CIR θ, κ, σ, v0
}

# ─── Override Logic ───────────────────────────────────────────
def param_assign(model: str, *, S0: float = None, r: float = None) -> Tuple[float, ...]:
    """
    Fetch the default tuple for `model` and optionally override:
      - S0 (always the first entry)
      - r  (the last entry), if the model uses a rate parameter.
    """
    key = model.upper()
    if key not in PARAMETERS:
        raise ValueError(f"Model '{model}' not supported.")
    params = list(PARAMETERS[key])

    if S0 is not None:
        params[0] = S0

    # If they passed an r, replace the last element (most models now
    # treat the final parameter as their 'rate' or drift input).
    if r is not None and len(params) > 1:
        params[-1] = r

    return tuple(params)