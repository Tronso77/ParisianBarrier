# streamlit_app/validation_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.simulator    import simulate_paths
from validation.cumulants import cumulants
from models.params       import param_assign


def show_validation_dashboard():
    """Render the cumulant‑validation dashboard with enriched visuals."""
    # Sidebar controls
    st.sidebar.title("📊 Validation Settings")
    model   = st.sidebar.selectbox("Model", [
        m for m in ["BM","ABM","GBM","VG","NIG","MJD","KJD","POI",
                   "GAMMA","CIR","HESTON","CEV","SABR"] if m != "CGMY"
    ])
    nsim    = st.sidebar.number_input("Paths", 1000, 200000, 50000, step=1000)
    nsteps  = st.sidebar.number_input("Steps", 10, 1000, 252, step=10)
    T       = st.sidebar.slider("Horizon T (yrs)", 0.1, 5.0, 1.0)
    seed    = st.sidebar.number_input("RNG seed", value=42)
    show_cum = st.sidebar.checkbox("Show cumulant comparison", True)

    dt = T / nsteps

    st.title("🔍 Cumulant Validation")
    params = param_assign(model)
    paths  = simulate_paths(model, nsteps, nsim, dt, seed=int(seed))

    # Sample paths
    st.subheader(f"Sample paths: {model}")
    st.line_chart(paths.iloc[:, :min(20, nsim)])

    if show_cum:
        st.subheader("Cumulant vs MC Statistics")
        # choose slices
        fracs = [0.25, 0.5, 0.75, 1.0]
        idxs  = [int(f * nsteps) for f in fracs]
        X     = paths.iloc[idxs].values
        if model in {"GBM","VG","NIG","MJD","KJD","HESTON","SABR"}:
            X = np.log(X / paths.iloc[0].values)
        else:
            X = X - paths.iloc[0].values

        # MC moments
        EX_mc = X.mean(axis=1)
        VX_mc = X.var(axis=1)

        # analytic cumulants
        c1, c2, *_ = cumulants(params, model, dt, tj=np.linspace(0, T, nsteps+1))
        times      = np.array(fracs) * T
        EX_an      = c1 * times if c1 is not None else np.full_like(EX_mc, np.nan)
        VX_an      = c2 * times if c2 is not None else np.full_like(VX_mc, np.nan)

        # build DataFrame
        df = pd.DataFrame({
            "Time": times,
            "E_MC": EX_mc,      "E_Analytic": EX_an,
            "Var_MC": VX_mc,    "Var_Analytic": VX_an
        })
        df["Bias_E"] = df.E_MC - df.E_Analytic
        df["Bias_V"] = df.Var_MC - df.Var_Analytic

        st.dataframe(df.style.format({"E_MC": "{:.4f}", "E_Analytic": "{:.4f}",
                                      "Var_MC": "{:.4f}", "Var_Analytic": "{:.4f}"}))

        # plot bias percentages
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(df.Time, 100*df.Bias_E/np.where(np.abs(df.E_Analytic)>0, df.E_Analytic, 1), marker='o')
        axes[0].set_title('Bias % in Mean')
        axes[1].plot(df.Time, 100*df.Bias_V/np.where(np.abs(df.Var_Analytic)>0, df.Var_Analytic, 1), marker='o')
        axes[1].set_title('Bias % in Variance')
        for ax in axes:
            ax.set_xlabel('Time (yrs)')
            ax.set_ylabel('Percent')
            ax.grid(True)
        st.pyplot(fig)
