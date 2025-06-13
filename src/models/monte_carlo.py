import numpy as np
from models.simulator import simulate_paths

class MonteCarloEngine:
    def __init__(self, S0, K, T, r, sigma, model="GBM", n_paths=10000, n_steps=100, seed=None):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def simulate_paths(self):
        dt = self.T / self.n_steps
        return simulate_paths(
            model=self.model, 
            nsteps=self.n_steps,
            nsim=self.n_paths,
            dt=dt,
            seed=self.seed,
            S0=self.S0,
            r=self.r,
            sigma=self.sigma
        ).values
    
    def price_option(self, option_type="call"):
        paths = self.simulate_paths()
        ST = paths[-1]

        if option_type == "call":
            payoff = np.maximum(ST - self.K, 0)
        elif option_type == "put":
            payoff = np.maximum(self.K - ST, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        discounted_payoff = np.exp(-self.r * self.T) * payoff
        price = np.mean(discounted_payoff)
        stderr = np.std(discounted_payoff, ddof=1) / np.sqrt(self.n_paths)
        return price, stderr