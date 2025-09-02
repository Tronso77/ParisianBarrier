# src/pricing/vanilla.py
from __future__ import annotations

from math import exp, log, sqrt, erf
import numpy as np
from scipy.stats import norm

__all__ = ["bs_price", "bs_delta_call", "vanilla_payoff", "discounted_vanilla_payoff"]

def _N(z: float) -> float:
    """Standard normal CDF (via error function)."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str = "call") -> float:
    """
    Black–Scholes price with continuous dividend yield q.
    option_type: 'call' or 'put'
    """
    opt = option_type.lower()
    if T <= 0:
        return max(0.0, (S - K) if opt == "call" else (K - S))
    if sigma <= 0:
        fwd = S * exp((r - q) * T)
        disc = exp(-r * T)
        intrinsic = max(0.0, fwd - K) if opt == "call" else max(0.0, K - fwd)
        return disc * intrinsic

    v = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    if opt == "call":
        return S * exp(-q * T) * _N(d1) - K * exp(-r * T) * _N(d2)
    else:
        return K * exp(-r * T) * _N(-d2) - S * exp(-q * T) * _N(-d1)

def bs_delta_call(S: np.ndarray | float, K: float, T: float, r: float, q: float, sigma: float) -> np.ndarray | float:
    """
    Vectorized Black–Scholes call delta with dividend yield q.
    Returns a NumPy array if S is array-like, otherwise a float.
    """
    arr = np.asarray(S, dtype=float)
    T = float(max(T, 1e-12))
    v = sigma * np.sqrt(T)
    if v <= 0:
        delta = np.where(arr > K, np.exp(-q * T), 0.0)
    else:
        d1 = (np.log(arr / K) + (r - q + 0.5 * sigma * sigma) * T) / v
        delta = np.exp(-q * T) * norm.cdf(d1)
    return float(delta) if np.isscalar(S) else delta

def vanilla_payoff(ST: np.ndarray, K: float, option_type: str = "call") -> np.ndarray:
    """
    Vectorized terminal payoff for a vanilla European option.
    ST: array-like of terminal prices.
    """
    ST = np.asarray(ST, dtype=float)
    opt = option_type.lower()
    if opt == "call":
        return np.maximum(ST - K, 0.0)
    elif opt == "put":
        return np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def discounted_vanilla_payoff(ST: np.ndarray, K: float, r: float, T: float, option_type: str = "call") -> np.ndarray:
    """Discounted vanilla payoff."""
    return np.exp(-r * T) * vanilla_payoff(ST, K, option_type)
