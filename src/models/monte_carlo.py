from models.simulator import simulate_paths
import numpy as np
import pandas as pd
from typing import Optional, Dict


class MonteCarloEngine:
    def __init__(
        self,
        model: str,
        params: tuple,
        nsteps: int,
        nsim:   int,
        dt:     float,
        seed:   Optional[int] = None,
    ):
        self.model      = model
        self.params          = params
        self.nsteps          = nsteps
        self.nsim            = nsim
        self.dt              = dt
        self.seed            = seed

    def simulate(self, **kwargs) -> pd.DataFrame:
        # now simulate_paths is defined
        return simulate_paths(
            self.model,
            self.nsteps,
            self.nsim,
            self.dt,
            seed=self.seed,
            **kwargs
        )

    def payoff_european(self, paths: pd.DataFrame, strike: float, option: str = "call") -> np.ndarray:
        ST = paths.iloc[-1].values
        if option == "call":
            return np.maximum(ST - strike, 0)
        else:
            return np.maximum(strike - ST, 0)

    def price_option(
        self,
        payoff: np.ndarray,
        discount: float = 1.0,
        method: str = "standard",
        control_variate: Optional[Dict] = None,
        strata: Optional[int] = None,
        importance_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        P = payoff.copy()
        if method == "antithetic":
            P = 0.5 * (P.reshape(-1, 2).mean(axis=1))
        if method == "stratified" and strata:
            u = np.random.rand(len(P))
            bins = np.linspace(0, 1, strata+1)
            out = []
            for i in range(strata):
                idx = (u >= bins[i]) & (u < bins[i+1])
                if idx.any():
                    out.append(P[idx].mean())
            P = np.array(out)
        if method == "control_variate" and control_variate is not None:
            Y = control_variate['paths']
            mu_Y = control_variate['analytic_mean']
            b = np.cov(P, Y)[0,1] / np.var(Y)
            P = P - b*(Y - mu_Y)
        if method == "importance" and importance_weights is not None:
            P = P * importance_weights
        price = discount * P.mean()
        stderr = discount * P.std(ddof=1) / np.sqrt(len(P))
        return {"price": price, "stderr": stderr}
