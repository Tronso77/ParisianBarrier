from __future__ import annotations
import numpy as np
from .vanilla import vanilla_payoff

__all__ = ["parisian_indicator", "parisian_option_payoff"]

def parisian_indicator(paths: np.ndarray,
                       barrier: float,
                       dt: float,
                       window: float,
                       direction: str = "down",
                       style: str = "cumulative",
                       within: str = "grid",      # NEW: 'grid' or 'loglinear'
                       ) -> np.ndarray:
    """
    Return boolean vector (n_paths,) indicating if the Parisian condition is met.
    paths: (n_steps+1, n_paths)
    direction: 'down' (clock runs when S < B) or 'up' (S > B)
    style: 'cumulative' or 'consecutive'
    within: for 'cumulative', how to count within-step occupation:
            - 'grid'     : old behavior, whole dt if endpoint is on the barrier side
            - 'loglinear': fractional dt using a log-linear bridge between endpoints
    """
    S = paths
    side_down = (direction.lower() == "down")

    if style.lower() == "cumulative":
        if within.lower() == "grid":
            side = (S[1:] < barrier) if side_down else (S[1:] > barrier)
            occ  = side.sum(axis=0) * dt
            return occ >= window

        # ---- loglinear fractional occupation (cumulative) ----
        # Work in log space for GBM
        x_prev = np.log(S[:-1] + 1e-300)
        x_curr = np.log(S[ 1:] + 1e-300)
        b = np.log(barrier + 1e-300)

        # For 'down': count time with x < b. For 'up': time with x > b.
        if side_down:
            below_prev = (x_prev < b)
            below_curr = (x_curr < b)
        else:
            below_prev = (x_prev > b)
            below_curr = (x_curr > b)

        both_true  = below_prev & below_curr
        both_false = (~below_prev) & (~below_curr)
        # straddle where the indicator flips within the step
        straddle = ~(both_true | both_false)

        frac = np.zeros_like(x_prev, dtype=float)
        frac[both_true]  = 1.0
        frac[both_false] = 0.0

        # crossing fraction using a linear bridge in log-space
        # x(t) = x_prev + t * (x_curr - x_prev), t in [0,1]
        # t* = (b - x_prev) / (x_curr - x_prev); clamp to [0,1]
        denom = (x_curr - x_prev)
        tstar = np.clip((b - x_prev) / np.where(denom == 0.0, np.nan, denom), 0.0, 1.0)
        # down: count portion with x < b
        if side_down:
            # if we start above and end below -> fraction below = 1 - t*
            # if we start below and end above -> fraction below = t*
            frac[straddle] = np.where(below_prev[straddle], tstar[straddle], 1.0 - tstar[straddle])
        else:
            # 'up': count portion with x > b
            frac[straddle] = np.where(below_prev[straddle], 1.0 - tstar[straddle], tstar[straddle])

        # accumulate occupation
        occ = (frac * dt).sum(axis=0)
        return occ >= window

    elif style.lower() == "consecutive":
        # longest consecutive run exceeding the window length
        need = int(np.ceil(window / dt))
        side = (S[1:] < barrier) if side_down else (S[1:] > barrier)
        run = np.zeros(side.shape[1], dtype=int)
        hit = np.zeros(side.shape[1], dtype=bool)
        for t in range(side.shape[0]):
            run = (run + 1) * side[t]
            hit |= (run >= need)
        return hit

    else:
        raise ValueError("style must be 'cumulative' or 'consecutive'")

def parisian_option_payoff(paths: np.ndarray,
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
                           within: str = "grid",     # pass-through
                           ) -> np.ndarray:
    ST = paths[-1]
    vanilla = vanilla_payoff(ST, K, option_type)
    trig = parisian_indicator(paths, barrier, dt, window, direction, style, within=within)

    eff = trig if inout.lower() == "in" else ~trig
    payoff = vanilla * eff.astype(vanilla.dtype)
    return np.exp(-r * T) * payoff
