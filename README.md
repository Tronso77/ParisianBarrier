# Parisian Barrier Desk Prototype

Small internal tool used on the structuring desk to price Parisian barriers, keep tabs on the barrier clock, and sanity-check hedging numbers before we speak with trading. The code stays narrow on purpose: a Monte Carlo engine, Parisian payoff logic, and analytics that tell us how often the deal knocks in and how the clock behaves.

The emphasis is commercial, not academic. Everything here answers the standard desk questions – *what’s the hit ratio, how much time do we actually sit inside the window, and what happens if we tighten the barrier or tweak vol?*

## What’s in the box
- Monte Carlo engine tuned for Parisian monitoring (antithetic, stratified sampling, optional vanilla control variate).
- Parisian payoff logic with fractional barrier occupation so discrete monitoring bias stays under control.
- Utilities to summarise hits, clock time, and quick scenario tables the desk cares about.
- One notebook that walks through the trade narrative we use with the desk head.

## Run this demo
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
jupyter lab
```
Then open `notebooks/01_Parisian_Barrier_Trading_Desk_Demo.ipynb` for the walkthrough.

Sample trades used in the walkthrough sit in `data/sample/trade_params.csv`. Adjust or load them straight into your own analyses.

## License
MIT License – this remains an internal prototype, so please keep it in-house.
