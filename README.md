# Monte Carlo Simulator Dashboard

A Python package and Streamlit application for simulating, validating, and pricing financial derivatives under stochastic models.

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
  * CEV, Poisson, Gamma, SABR (not yet)
* **Variance Reduction**: Antithetic variates, stratified sampling, control variates, importance sampling.(CV and IS not yet)
* **Option Pricing**: Price European calls/puts and autocallable structures under GBM & Heston (others coming soon).


## Installation

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
