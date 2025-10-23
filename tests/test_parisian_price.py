import numpy as np

from src.models import MonteCarloEngine
from src.pricing import (
    parisian_first_hit_time,
    parisian_indicator,
    parisian_occupation_time,
)
from src.utils import summarise_parisian_paths


def test_fractional_clock_and_indicator():
    paths = np.array(
        [
            [100, 100],
            [95, 105],
            [94, 106],
            [93, 107],
            [92, 108],
            [91, 109],
            [90, 110],
        ],
        dtype=float,
    )
    dt = 0.1
    barrier = 96
    window = 0.2

    occupation = parisian_occupation_time(paths, barrier, dt, direction="down", within="loglinear")
    assert np.allclose(occupation, [0.5204146, 0.0], atol=1e-6)

    first_hit = parisian_first_hit_time(
        paths,
        barrier=barrier,
        dt=dt,
        window=window,
        direction="down",
        style="cumulative",
        within="loglinear",
    )
    assert np.isfinite(first_hit[0]) and np.isnan(first_hit[1])
    assert np.isclose(first_hit[0], 0.279585, atol=1e-6)

    trig = parisian_indicator(
        paths,
        barrier=barrier,
        dt=dt,
        window=window,
        direction="down",
        style="cumulative",
        within="loglinear",
    )
    assert trig.tolist() == [True, False]

    summary = summarise_parisian_paths(
        paths,
        barrier=barrier,
        dt=dt,
        window=window,
        direction="down",
        style="cumulative",
        within="loglinear",
    )
    assert summary.hit_ratio == 0.5
    assert np.isclose(summary.avg_occupation, 0.2602073, atol=1e-6)
    assert np.isclose(summary.avg_occupation_hit, 0.5204146, atol=1e-6)
    assert summary.avg_terminal == 100.0
    assert np.isclose(summary.avg_hit_time_hit, first_hit[0])


def test_in_out_parity_matches_vanilla():
    base = dict(S0=100.0, K=100.0, T=0.25, r=0.02, sigma=0.20, q=0.0)
    mc_kwargs = dict(n_paths=4096, n_steps=96, seed=11, antithetic=True, stratified=True)

    engine_in = MonteCarloEngine(**base, **mc_kwargs)
    price_in, se_in = engine_in.price_parisian(
        K=base["K"],
        barrier=95.0,
        window=0.25,
        option_type="call",
        inout="in",
        direction="down",
        style="cumulative",
        within="loglinear",
    )

    engine_out = MonteCarloEngine(**base, **mc_kwargs)
    price_out, se_out = engine_out.price_parisian(
        K=base["K"],
        barrier=95.0,
        window=0.25,
        option_type="call",
        inout="out",
        direction="down",
        style="cumulative",
        within="loglinear",
    )

    vanilla_engine = MonteCarloEngine(**base, **mc_kwargs, control_variate=True)
    vanilla_price, vanilla_se = vanilla_engine.price_option(option_type="call")

    tol = 8.0 * max(se_in, se_out, vanilla_se)
    assert abs((price_in + price_out) - vanilla_price) <= tol
