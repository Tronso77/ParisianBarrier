"""
payoff.py

Derivatives,  payoff, and coupon logic.
"""

import numpy as np
import pandas as pd

def autocallable_payoff(
    paths,
    call_dates,
    notional=100,
    barrier=1.0,
    coupon=0.05,
    maturity_coupon=0.10,
    knock_in_level=0.7,
):
    n_paths = paths.shape[1]
    call_flags = np.zeros(n_paths, dtype=bool)
    payoffs = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=bool)

    for i, date in enumerate(call_dates):
        spot = paths.iloc[date].values if hasattr(paths, "iloc") else paths[date]
        new_calls = (spot >= barrier) & (~called)
        call_flags[new_calls] = True
        payoffs[new_calls] = notional * (1 + (i + 1) * coupon)
        called |= new_calls

    not_called = ~called
    final_spot = paths.iloc[-1].values if hasattr(paths, "iloc") else paths[-1]
    breached = np.any(paths < knock_in_level, axis=0) & not_called

    payoffs[breached] = final_spot[breached] * notional
    payoffs[not_called & ~breached] = notional * (1 + maturity_coupon)

    return payoffs, call_flags

def payoff_european_call(paths: pd.DataFrame, K: float) -> np.ndarray:
    ST = paths.iloc[-1].values
    return np.maximum(ST - K, 0.0)

def payoff_european_put(paths: pd.DataFrame, K: float) -> np.ndarray:
    ST = paths.iloc[-1].values
    return np.maximum(K - ST, 0.0)
