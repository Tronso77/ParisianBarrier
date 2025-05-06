# Monte Carlo Simulator Dashboard

A Python package and Streamlit application for simulating, validating, and pricing financial derivatives under a variety of stochastic models.

## Live Demo

A deployed version is available here:

**https://simulator-mc.streamlit.app/**


## Features

* **Path Simulation**: Generate Monte Carlo paths for models including:

  * Brownian Motion (BM, ABM)
  * Geometric Brownian Motion (GBM)
  * Variance Gamma (VG)
  * Normal Inverse Gaussian (NIG)
  * Merton & Kou Jump Diffusions (MJD, KJD)
  * CIR and Heston stochastic volatility
  * CEV, Poisson, Gamma, SABR (partial)
* **Variance Reduction**: Antithetic variates, stratified sampling, control variates, importance sampling.
* **Cumulant Validation**: Compare Monte Carlo moments against analytic cumulants over time.
* **Option Pricing**: Price European calls/puts and autocallable structures under GBM & Heston (others coming soon).
* **Interactive Dashboard**: Streamlit app to explore simulations, validate models, and price payoffs.

## 📁 Repository Structure

```
Simulator/
├── src/                    # Core library code
│   ├── models/             # Simulation engines & Monte Carlo framework
│   ├── validation/         # Analytic cumulant calculations
│   ├── pricing/            # Payoff definitions & hedging logic
│   └── utils/              # Helper functions
├── streamlit_app/          # Streamlit dashboards (Simulation, Validation, Pricing)
├── notebooks/              # Jupyter notebooks for experimentation
├── data/                   # (ignored) sample data
├── plots/                  # (ignored) report figures
└── README.md               # Project overview (this file)
```

## 📦 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/Tronso77/Simulator.git
   cd Simulator
   ```
2. Create & activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

##  Running Locally

To launch the Streamlit dashboard:

```bash
streamlit run streamlit_app/app.py --server.enableCORS false
```

Then open your browser to the URL shown (e.g. [http://localhost:8501](http://localhost:8501)).

##  Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.

## 📜 License

This project is released under the MIT License.
