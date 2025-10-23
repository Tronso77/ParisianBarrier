"""Utility functions to read Monte Carlo paths like a trading desk analyst."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..pricing.payoff_parisian import (
    parisian_first_hit_time,
    parisian_indicator,
    parisian_occupation_time,
)


@dataclass(frozen=True)
class ParisianSummary:
    """Compact snapshot of how the barrier clock behaved across the simulation."""

    hit_ratio: float
    avg_occupation: float
    avg_occupation_hit: float
    window: float
    avg_terminal: float
    avg_hit_time: float
    avg_hit_time_hit: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "hit_ratio": self.hit_ratio,
            "avg_occupation": self.avg_occupation,
            "avg_occupation_hit": self.avg_occupation_hit,
            "window": self.window,
            "avg_terminal": self.avg_terminal,
            "avg_hit_time": self.avg_hit_time,
            "avg_hit_time_hit": self.avg_hit_time_hit,
        }


def summarise_parisian_paths(
    paths: np.ndarray,
    barrier: float,
    dt: float,
    window: float,
    *,
    direction: str = "down",
    style: str = "cumulative",
    within: str = "loglinear",
) -> ParisianSummary:
    """
    Turn a stack of simulated paths into trading-friendly stats.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot levels (n_steps+1, n_paths).
    barrier : float
        Knock level.
    dt : float
        Time step of the simulation.
    window : float
        Parisian clock window (in years).
    direction, style, within :
        Same conventions as `parisian_indicator`.
    """
    hits = parisian_indicator(
        paths,
        barrier=barrier,
        dt=dt,
        window=window,
        direction=direction,
        style=style,
        within=within,
    )
    occupation = parisian_occupation_time(
        paths, barrier=barrier, dt=dt, direction=direction, within=within
    )
    first_hit = parisian_first_hit_time(
        paths,
        barrier=barrier,
        dt=dt,
        window=window,
        direction=direction,
        style=style,
        within=within,
    )

    hit_ratio = float(hits.mean())
    avg_occ = float(occupation.mean())
    avg_occ_hit = float(occupation[hits].mean()) if np.any(hits) else 0.0
    avg_terminal = float(paths[-1].mean())
    avg_hit_time = (
        float(np.nanmean(first_hit)) if np.isfinite(first_hit).any() else float("nan")
    )
    avg_hit_time_hit = (
        float(np.nanmean(first_hit[hits])) if np.any(hits) else float("nan")
    )

    return ParisianSummary(
        hit_ratio=hit_ratio,
        avg_occupation=avg_occ,
        avg_occupation_hit=avg_occ_hit,
        window=float(window),
        avg_terminal=avg_terminal,
        avg_hit_time=avg_hit_time,
        avg_hit_time_hit=avg_hit_time_hit,
    )
