# src/models/params.py
from __future__ import annotations
from typing import Tuple

# Default parameter tuples for each model.
# IMPORTANT: for GBM/ABM/CEV the second entry is the arithmetic drift 'mu'.
# Under the risk-neutral measure for GBM, you should use mu = r - q.
PARAMETERS = {
    "BM":      (100.0, 1.0),                              # S0, sigma
    "ABM":     (100.0, 0.0, 1.0),                         # S0, mu, sigma
    "GBM":     (100.0, 0.05, 0.2),                        # S0, mu, sigma  (mu ≈ r - q under Q)

    # Pure Lévy/jump models (no S0 used by the increment process)
    "POI":     (5.0,),                                    # lambda
    "GAMMA":   (5.0, 10.0),                               # alpha, lambda
    "NIG":     (-0.032705, 0.184253536, 0.170829388),     # theta, sigma, kappa
    "KJD":     (0.2, 0.75, 0.4, 0.3, 0.2),                # sigma, lambda, p, eta1, eta2
    "CGMY":    (0.6509, 5.853, 18.27, 1.8),               # C, G, M, Y

    # Mean-reverting / diffusions
    "CIR":     (0.3, 0.04, 0.2, 0.04),                    # kappa, theta, sigma, v0
    "HESTON":  (100.0, 0.04, 0.3, 0.2, 0.2, -0.2, 0.01),  # S0, v0, kappa, theta, sigma_v, rho, r
    "CEV":     (100.0, 0.1, -2.0, 0.2),                   # S0, mu, beta, sigma
    "SABR":    (100.0, 0.04, 0.2, -0.2, 0.1),             # S0, alpha0, beta, rho, nu

    "VG":      (100.0, -0.09, 0.2, 0.017),                # S0, theta, sigma, nu
    "MJD":     (100.0, 0.2, 0.75, -0.05, 0.1),            # S0, sigma, lambda, meanJ, volJ
    "VGCIR":   (100.0, -0.09, 0.2, 0.017, 0.3, 0.04, 0.2, 0.04),
    # S0, theta, sigma, nu, CIR kappa, theta, sigma, v0
}

def param_assign(model: str, *, S0: float | None = None, r: float | None = None) -> Tuple[float, ...]:
    """
    Fetch default parameters for `model` and optionally override:
      - S0: initial spot (for models where S0 is first element)
      - r : risk-free/drift for models where the second (or last) slot is 'mu'/'r'
            (GBM/ABM/CEV -> overrides mu; HESTON -> overrides r)
    NOTE: For GBM under Q, pass (r,q) to simulate_paths(...) and it will use mu = r - q.
    """
    key = model.upper()
    if key not in PARAMETERS:
        raise ValueError(f"Model '{model}' not supported.")
    p = list(PARAMETERS[key])

    if S0 is not None and len(p) > 0 and key in {"BM","ABM","GBM","HESTON","CEV","VG","MJD","VGCIR"}:
        p[0] = float(S0)

    if r is not None:
        if key in {"GBM","ABM","CEV"}:
            p[1] = float(r)        # treated as 'mu' by the simulator
        elif key == "HESTON":
            p[-1] = float(r)       # last entry is r in our Heston layout

    return tuple(p)
