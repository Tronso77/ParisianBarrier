"""
params.py

Default parameter sets for each supported model.
Each entry is a tuple in the order expected by the simulator & cumulant routines.
"""
from typing import Tuple

PARAMETERS = {
    "BM":      (0.0, 1.0),
    "ABM":     (0.0, 1.0),
    "GBM":     (100.0, 0.05, 0.2),   # S0, mu, sigma
    "POI":     (5.0,),
    "GAMMA":   (5.0, 10.0),
    "VG":      (0.1, -0.09, 0.2, 0.017),
    "MJD":     (0.0, 0.2, 0.75, -0.05, 0.1),
    "NIG":     (0.0, -0.032705, 0.184253536, 0.170829388),
    "KJD":     (0.0, 0.2, 0.75, 0.4, 0.3, 0.2),
    "CGMY":    (0.0, 0.6509, 5.853, 18.27, 1.8),
    "CIR":     (0.2, 0.3, 0.2, 0.04),
    "HESTON":  (100.0, 0.04, 0.3, 0.2, 0.2, -0.2, 0.01),# S0 v0  kappa theta eta rho  r
    "CEV":     (100.0, 0.1, -2.0, 0.2), # S0,  mu, beta, sigma
    "SABR":    (100.0, 0.04, 0.2, -0.2, 0.1),# S0     alpha0 beta rho  gamma
}

def param_assign(model: str, S0: float = 100.0, r: float = 0.05) -> Tuple[float, ...]:
    """
    Returns the tuple of parameters for `model`, always starting with S0 and
    ending with r, so downstream code can rely on a uniform signature.
    """
    key = model.upper()
    try:
        base = PARAMETERS[key]
    except KeyError:
        raise ValueError(f"Model '{model}' not supported in param_assign.")
    # e.g. (S0, μ, σ, ..., r)
    return (S0, *base, r)