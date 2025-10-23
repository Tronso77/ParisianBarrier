# Parisian Barrier Prototype

Small tool to price Parisian barriers, keep tabs on the barrier clock, and sanity-check hedging numbers. The code: a Monte Carlo engine, Parisian payoff logic, and analytics that tell us how knocks in and clock behave.

The emphasisquestions – *what’s the hit ratio, how much time do we actually sit inside the window, and what happens if we tighten the barrier or tweak vol?*

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
Then open `notebooks/01_Parisian_Barrier_Demo.ipynb` for the walkthrough.

Sample trades used in the walkthrough sit in `data/sample/trade_params.csv`. Adjust or load them straight into your own analyses.

