# src/models/monte_carlo.py

import QuantLib as ql

class MonteCarloEngine:
    """
    A unified Monte Carlo pricer that uses QuantLib's MCEuropeanEngine under the hood.

    Usage:
        engine = MonteCarloEngine(
            process,           # a QuantLib GeneralizedBlackScholesProcess (or HestonProcess)
            payoff,            # a QuantLib Payoff (e.g. PlainVanillaPayoff)
            exercise,          # a QuantLib Exercise (e.g. EuropeanExercise)
            time_steps=100,    # number of time‐steps in the MC
            samples=50000,     # number of MC paths
            seed=42,           # RNG seed
            antithetic=False   # whether to use antithetic variates
        )
        price = engine.price()
        stderr = engine.error_estimate()
    """

    def __init__(
        self,
        process: ql.GeneralizedBlackScholesProcess,
        payoff: ql.Payoff,
        exercise: ql.Exercise,
        time_steps: int = 100,
        samples: int = 50_000,
        seed: int = 42,
        antithetic: bool = False
    ):
        # build the QL option
        self.option = ql.VanillaOption(payoff, exercise)

        # choose the pseudo-random trait
        rng_trait = "pseudorandom"

        # attach the Monte Carlo engine
        mc_engine = ql.MCEuropeanEngine(
            process,
            rng_trait,
            timeSteps=time_steps,
            antitheticVariate=antithetic,
            requiredSamples=samples,
            seed=seed
        )
        self.option.setPricingEngine(mc_engine)

    def price(self) -> float:
        """Run the MC and return the NPV (price)."""
        return self.option.NPV()

    def error_estimate(self) -> float:
        """Return the MC standard error."""
        return self.option.errorEstimate()
