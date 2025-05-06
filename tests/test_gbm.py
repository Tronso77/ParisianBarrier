import pytest
from src.models.simulator import simulate_paths
from src.validation.cumulants import analytical_mean_var


def test_gbm_mean_variance():
    nsteps, nsim, dt = 252, 100_000, 1/252
    paths = simulate_paths("GBM", nsteps=nsteps, nsim=nsim, dt=dt, seed=42)
    terminals = paths.iloc[:, -1].values

    samp_mean, samp_var = terminals.mean(), terminals.var()
    analytic_mean, analytic_var = analytical_mean_var(
        model="GBM", S0=100.0, mu=0.05, sigma=0.2, T=1.0
    )

    tol = 0.01
    assert pytest.approx(analytic_mean, rel=tol) == samp_mean
    assert pytest.approx(analytic_var,  rel=tol) == samp_var
