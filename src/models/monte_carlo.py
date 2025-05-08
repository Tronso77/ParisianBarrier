import QuantLib as ql

class MonteCarloEngine:
    """
    A unified Monte Carlo pricer that uses QuantLib's engines under the hood.
    Usage:
        engine = MonteCarloEngine(
            model_process,    # a QuantLib StochasticProcess
            payoff,           # a QuantLib Payoff (e.g. PlainVanillaPayoff)
            exercise_date,    # a QuantLib Exercise (e.g. EuropeanExercise)
            time_steps=100,
            samples=25000
        )
        price = engine.price()
    """
    def __init__(
        self,
        process: ql.GeneralizedBlackScholesProcess,
        payoff: ql.Payoff,
        exercise: ql.Exercise,
        time_steps: int = 100,
        samples: int = 50_000,
        seed: int = 42
    ):
        # Build the option
        self.option = ql.VanillaOption(payoff, exercise)

        # Attach a QuantLib MC engine
        mc_engine = ql.MCEuropeanEngine(
            process,
            "PseudoRandom",
            timeSteps=time_steps,
            requiredSamples=samples,
            seed=seed
        )
        self.option.setPricingEngine(mc_engine)

    def price(self) -> float:
        """Run the MC and return the price."""
        return self.option.NPV()

    def error_estimate(self) -> float:
        """Get the MC standard error."""
        return self.option.errorEstimate()