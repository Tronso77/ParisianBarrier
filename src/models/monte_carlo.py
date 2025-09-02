# src/models/monte_carlo.py
from __future__ import annotations

import math
import numpy as np
from typing import Literal

# alias to avoid name clash with the method below
from .simulator import simulate_paths as sim_paths
from .variance_reduction import apply_brownian_bridge, apply_stratified
from ..pricing.vanilla import bs_price
from ..pricing.payoff_parisian import parisian_option_payoff


class MonteCarloEngine:
    """
    Monte Carlo pricer with optional variance reductions.

    Conventions:
      - Paths are (n_steps+1, n_paths): time-first, then paths.
      - GBM is simulated internally (supports antithetic + stratified).
      - Other models call src.models.simulator (which returns (nsim, nsteps+1)) and are transposed.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        model: str = "GBM",
        n_paths: int = 10000,
        n_steps: int = 100,
        seed: int | None = None,
        brownian_bridge: bool = False,  # keep False for Parisian unless you densify consistently
        antithetic: bool = False,
        control_variate: bool = False,  # used by price_option (vanilla)
        stratified: bool = False,
        verbose: bool = False,
    ):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)
        self.model = str(model).upper()
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.seed = seed
        self.brownian_bridge = bool(brownian_bridge)
        self.antithetic = bool(antithetic)
        self.control_variate = bool(control_variate)
        self.stratified = bool(stratified)
        self.verbose = bool(verbose)

    # ---------------- public path generator (time-first) ----------------
    def simulate_paths(self) -> np.ndarray:
        """
        Returns array of shape (n_steps+1, n_paths) with spot levels.
        """
        dt = self.T / self.n_steps

        if self.model == "GBM":
            # risk-neutral drift
            drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt
            vol = self.sigma * math.sqrt(dt)

            rng = np.random.default_rng(self.seed)

            if self.antithetic:
                half = (self.n_paths + 1) // 2  # robust to odd n_paths
                Z_half = (apply_stratified(half, self.n_steps, rng=rng)
                          if self.stratified else rng.standard_normal((self.n_steps, half)))
                Z = np.concatenate([Z_half, -Z_half], axis=1)[:, :self.n_paths]
            else:
                Z = (apply_stratified(self.n_paths, self.n_steps, rng=rng)
                     if self.stratified else rng.standard_normal((self.n_steps, self.n_paths)))

            logS = np.empty((self.n_steps + 1, Z.shape[1]), dtype=float)
            logS[0, :] = np.log(self.S0)
            logS[1:, :] = logS[[0], :] + np.cumsum(drift + vol * Z, axis=0)
            paths = np.exp(logS)  # (n_steps+1, n_paths)

        else:
            # Delegate to shared simulator (nsim rows, nsteps+1 cols), then transpose
            df = sim_paths(
                model=self.model,
                nsteps=self.n_steps,
                nsim=self.n_paths,
                dt=dt,
                seed=self.seed,
                S0=self.S0,
                mu=self.r - self.q,      # drift under Q for drifted models
                sigma=self.sigma,
                stratified=self.stratified,
                antithetic=self.antithetic,
            )
            paths = df.values.T  # (n_steps+1, n_paths)

        if self.brownian_bridge:
            paths = apply_brownian_bridge(paths, self.T, self.sigma)

        return paths

    # ---------------- vanilla pricing (with optional CV) ----------------
    def price_option(
        self,
        option_type: str = "call",
        payoff_fn=None,
        greek: str | None = None,
        bump: float = 1e-4,
    ):
        if greek is not None:
            base_price, _ = self.price_option(option_type=option_type, payoff_fn=payoff_fn, greek=None)
            g = greek.lower()
            if g == "delta":
                S0_up = self.S0 * (1 + bump)
                up = MonteCarloEngine(**{**self.__dict__, "S0": S0_up, "verbose": False})
                p_up, _ = up.price_option(option_type=option_type, payoff_fn=payoff_fn)
                return (p_up - base_price) / (S0_up - self.S0)
            if g == "gamma":
                S0_up = self.S0 * (1 + bump); S0_dn = self.S0 * (1 - bump)
                up = MonteCarloEngine(**{**self.__dict__, "S0": S0_up, "verbose": False})
                dn = MonteCarloEngine(**{**self.__dict__, "S0": S0_dn, "verbose": False})
                p_up, _ = up.price_option(option_type=option_type, payoff_fn=payoff_fn)
                p_dn, _ = dn.price_option(option_type=option_type, payoff_fn=payoff_fn)
                return (p_up - 2 * base_price + p_dn) / ((S0_up - self.S0) ** 2)
            if g == "vega":
                sig_up = self.sigma + bump
                up = MonteCarloEngine(**{**self.__dict__, "sigma": sig_up, "verbose": False})
                p_up, _ = up.price_option(option_type=option_type, payoff_fn=payoff_fn)
                return (p_up - base_price) / (sig_up - self.sigma)
            if g == "theta":
                T_dn = max(1e-8, self.T - bump)
                dn = MonteCarloEngine(**{**self.__dict__, "T": T_dn, "verbose": False})
                p_dn, _ = dn.price_option(option_type=option_type, payoff_fn=payoff_fn)
                return (p_dn - base_price) / (-bump)
            if g == "rho":
                r_up = self.r + bump
                up = MonteCarloEngine(**{**self.__dict__, "r": r_up, "verbose": False})
                p_up, _ = up.price_option(option_type=option_type, payoff_fn=payoff_fn)
                return (p_up - base_price) / (r_up - self.r)
            raise ValueError(f"Greek '{greek}' not implemented.")

        paths = self.simulate_paths()
        ST = paths[-1]
        if payoff_fn is not None:
            payoff = payoff_fn(ST)
        else:
            payoff = np.maximum(ST - self.K, 0.0) if option_type.lower() == "call" else np.maximum(self.K - ST, 0.0)

        disc = math.exp(-self.r * self.T) * payoff

        if self.control_variate:
            # CV on S_T with E[S_T] = S0 * exp((r - q) T)
            exp_ST = self.S0 * math.exp((self.r - self.q) * self.T)
            var_ST = float(np.var(ST, ddof=1))
            if var_ST > 0.0:
                beta = float(np.cov(disc, ST, ddof=1)[0, 1]) / var_ST
                disc = disc - beta * (ST - exp_ST)

        price = float(disc.mean())
        stderr = float(disc.std(ddof=1) / math.sqrt(disc.size))
        if self.verbose:
            print(f"[Price] {option_type} = {price:.6f} ± {stderr:.6f}")
        return price, stderr

    # ---------------- Parisian pricing (cumulative or consecutive) ----------------

    def price_parisian(
        self,
        K: float,
        barrier: float,
        window: float,
        option_type: str = "call",
        inout: str = "in",
        direction: str = "down",
        style: str = "cumulative",
        cv: str = "vanilla",
        within: str = "loglinear",   # 'loglinear
    ):
        """
        Parisian IN/OUT price via pathwise payoff and discounting.
        Optional control variate 'vanilla' reduces variance.
        'within' controls how much of each dt is counted toward the occupation clock.
        """
        paths = self.simulate_paths()                # (n_steps_eff+1, n_paths)
        # IMPORTANT: dt must reflect the actual path sampling
        dt = self.T / (paths.shape[0] - 1)

        disc_payoff = parisian_option_payoff(
            paths, K, self.r, self.T, barrier, dt, window,
            option_type=option_type, inout=inout, direction=direction, style=style,
            within=within,   # pass through
        )

        if cv == "vanilla":
            from ..pricing.vanilla import discounted_vanilla_payoff, bs_price
            ST = paths[-1]
            Y  = discounted_vanilla_payoff(ST, K, self.r, self.T, option_type)
            EY = bs_price(self.S0, K, self.T, self.r, self.q, self.sigma, option_type)
            cov = np.cov(disc_payoff, Y, ddof=1)[0, 1]
            var = np.var(Y, ddof=1)
            beta = cov / var if var > 0 else 0.0
            adj  = disc_payoff - beta * (Y - EY)
            return float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(adj.size))

        return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(disc_payoff.size))
