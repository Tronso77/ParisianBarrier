
# src/models/params.py
from typing import Tuple

# ─── Model Defaults ──────────────────────────────────────────────────────
# Tuples match the parameter order required by each simulator:
PARAMETERS = {
    "BM":      (100.0, 1.0),                              # S0, sigma
    "ABM":     (100.0, 0.0, 1.0),                         # S0, mu, sigma
    "GBM":     (100.0, 0.05, 0.2),                        # S0, r (mu), sigma

    # Pure Lévy/jump models: no S0 here; simulation uses only these params
    "POI":     (5.0,),                                    # lambda
    "GAMMA":   (5.0, 10.0),                               # alpha, lambda
    "NIG":     (-0.032705, 0.184253536, 0.170829388),     # theta, sigma, kappa
    "KJD":     (0.2, 0.75, 0.4, 0.3, 0.2),                # sigma, lambda, p, eta1, eta2
    "CGMY":    (0.6509, 5.853, 18.27, 1.8),               # C, G, M, Y

    # Mean-reverting / diffusions
    "CIR":     (0.3, 0.04, 0.2, 0.04),                    # kappa, theta, sigma, v0
    "HESTON":  (100.0, 0.04, 0.3, 0.2, 0.2, -0.2, 0.01),   # S0, v0, kappa, theta, sigma_v, rho, r
    "CEV":     (100.0, 0.1, -2.0, 0.2),                   # S0, mu, beta, sigma
    "SABR":    (100.0, 0.04, 0.2, -0.2, 0.1),             # S0, alpha0, beta, rho, nu

    # VG-CIR & others keep original fallback order
    "VG":      (100.0, -0.09, 0.2, 0.017),                # S0, theta, sigma, nu
    "MJD":     (100.0, 0.2, 0.75, -0.05, 0.1),             # S0, sigma, lambda, meanJ, volJ
    "VGCIR":   (100.0, -0.09, 0.2, 0.017, 0.3, 0.04, 0.2, 0.04),  # S0, theta, sigma, nu, CIR kappa, theta, sigma, v0
}


def param_assign(model: str, *, S0: float = None, r: float = None) -> Tuple[float, ...]:
    """
    Fetch default parameters for `model` and optionally override:
      - S0: initial spot (for diffusion/jump with S0)
      - r : drift or risk-free rate (for GBM, ABM, Heston)
    """
    key = model.upper()
    if key not in PARAMETERS:
        raise ValueError(f"Model '{model}' not supported.")
    p = list(PARAMETERS[key])

    # override S0 when present as first entry
    if S0 is not None and len(p) > 0 and key in {"BM","ABM","GBM","HESTON","CEV","VG","MJD","VGCIR"}:
        p[0] = S0

    # override risk-free/drift
    if r is not None:
        if key in {"GBM","ABM","CEV"}:
            p[1] = r
        elif key == "HESTON":
            p[-1] = r
        # others ignore r

    return tuple(p)

