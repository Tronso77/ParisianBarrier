from __future__ import annotations

import numpy as np

from .vanilla import vanilla_payoff

__all__ = [
    "parisian_indicator",
    "parisian_option_payoff",
    "parisian_occupation_time",
    "parisian_first_hit_time",
]


def _step_fractions(
    paths: np.ndarray,
    barrier: float,
    direction: str = "down",
    within: str = "loglinear",
) -> np.ndarray:
    """
    Fraction of each step spent on the barrier side.

    Returns an array of shape (n_steps, n_paths) with values in [0, 1].
    """
    S = np.asarray(paths, dtype=float)
    side_down = direction.lower() == "down"
    steps = S.shape[0] - 1

    if within.lower() == "grid":
        # crude grid monitoring: full dt when endpoint is on the barrier side
        side = (S[1:] < barrier) if side_down else (S[1:] > barrier)
        return side.astype(float)

    # log-linear interpolation within each step (best for GBM)
    x_prev = np.log(S[:-1] + 1e-300)
    x_curr = np.log(S[1:] + 1e-300)
    b = np.log(barrier + 1e-300)

    below_prev = x_prev < b if side_down else x_prev > b
    below_curr = x_curr < b if side_down else x_curr > b

    both_true = below_prev & below_curr
    both_false = (~below_prev) & (~below_curr)
    straddle = ~(both_true | both_false)

    frac = np.zeros((steps, S.shape[1]), dtype=float)
    frac[both_true] = 1.0
    frac[both_false] = 0.0

    denom = x_curr - x_prev
    denom = np.where(denom == 0.0, np.nan, denom)
    tstar = np.clip((b - x_prev) / denom, 0.0, 1.0)

    if side_down:
        frac[straddle] = np.where(below_prev[straddle], tstar[straddle], 1.0 - tstar[straddle])
    else:
        frac[straddle] = np.where(below_prev[straddle], 1.0 - tstar[straddle], tstar[straddle])

    # replace NaNs (which can happen when denom=0) with 0.0
    np.nan_to_num(frac, copy=False)
    return frac


def parisian_occupation_time(
    paths: np.ndarray,
    barrier: float,
    dt: float,
    direction: str = "down",
    within: str = "loglinear",
) -> np.ndarray:
    """
    Fractional time (in years) each path spends on the barrier side.
    """
    fractions = _step_fractions(paths, barrier, direction=direction, within=within)
    return (fractions * dt).sum(axis=0)


def parisian_indicator(
    paths: np.ndarray,
    barrier: float,
    dt: float,
    window: float,
    direction: str = "down",
    style: str = "cumulative",
    within: str = "loglinear",
) -> np.ndarray:
    """
    Return boolean vector (n_paths,) indicating if the Parisian condition is met.
    """
    if style.lower() == "cumulative":
        occ = parisian_occupation_time(paths, barrier, dt, direction=direction, within=within)
        return occ >= window

    if style.lower() == "consecutive":
        need = int(np.ceil(window / dt))
        side = (paths[1:] < barrier) if direction.lower() == "down" else (paths[1:] > barrier)
        run = np.zeros(side.shape[1], dtype=int)
        hit = np.zeros(side.shape[1], dtype=bool)
        for t in range(side.shape[0]):
            run = (run + 1) * side[t]
            hit |= (run >= need)
        return hit

    raise ValueError("style must be 'cumulative' or 'consecutive'")


def parisian_first_hit_time(
    paths: np.ndarray,
    barrier: float,
    dt: float,
    window: float,
    direction: str = "down",
    style: str = "cumulative",
    within: str = "loglinear",
) -> np.ndarray:
    """
    First time (in years) the Parisian condition is met. NaN if never triggered.
    """
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    hit_time = np.full(n_paths, np.nan, dtype=float)

    if style.lower() == "cumulative":
        fractions = _step_fractions(paths, barrier, direction=direction, within=within)
        increments = fractions * dt
        cumulative = np.cumsum(increments, axis=0)
        mask = cumulative >= window
        any_hit = mask.any(axis=0)
        if not np.any(any_hit):
            return hit_time

        first_idx = np.argmax(mask, axis=0)
        for i in np.where(any_hit)[0]:
            idx = first_idx[i]
            prev = cumulative[idx - 1, i] if idx > 0 else 0.0
            missing = window - prev
            step_time = increments[idx, i]
            if step_time <= 0.0:
                hit_time[i] = (idx + 1) * dt
            else:
                frac_within = np.clip(missing / step_time, 0.0, 1.0)
                hit_time[i] = idx * dt + frac_within * dt
        return hit_time

    if style.lower() == "consecutive":
        side = (paths[1:] < barrier) if direction.lower() == "down" else (paths[1:] > barrier)
        run = np.zeros(n_paths, dtype=int)
        need = window / dt
        for t in range(n_steps):
            run = np.where(side[t], run + 1, 0)
            newly_hit = np.isnan(hit_time) & (run * dt >= window)
            if newly_hit.any():
                overshoot = run[newly_hit] * dt - window
                hit_time[newly_hit] = (t + 1) * dt - overshoot
        return hit_time

    raise ValueError("style must be 'cumulative' or 'consecutive'")


def parisian_option_payoff(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    barrier: float,
    dt: float,
    window: float,
    option_type: str = "call",
    inout: str = "in",
    direction: str = "down",
    style: str = "cumulative",
    within: str = "loglinear",
) -> np.ndarray:
    """
    Discounted payoff for Parisian IN/OUT options on the supplied paths.
    """
    ST = paths[-1]
    vanilla = vanilla_payoff(ST, K, option_type)
    trig = parisian_indicator(paths, barrier, dt, window, direction, style, within=within)

    eff = trig if inout.lower() == "in" else ~trig
    payoff = vanilla * eff.astype(vanilla.dtype)
    return np.exp(-r * T) * payoff
