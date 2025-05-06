"""
utils.py

Helper functions for time grid, discounting, visualization, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LOG_BASED_MODELS = {"GBM", "VG", "NIG", "CGMY", "KJD"}

def estimate_sample_cumulants(paths, model=None):
    """
    Estimate cumulants from the final values of simulated paths.
    Applies log-return only for log-based processes.
    """
    x = paths.iloc[-1].values
    x0 = paths.iloc[0].values

    # Models that require log returns
    log_models = ["GBM", "VG", "NIG", "CGMY", "KJD"]

    if model is not None and model.upper() in log_models:
        x = np.log(np.maximum(x, 1e-12) / np.maximum(x0, 1e-12))  # log-safe
    else:
        x = x - x0  # simple increments

    c1 = np.mean(x)
    c2 = np.var(x)
    c3 = np.mean((x - c1)**3)
    c4 = np.mean((x - c1)**4)
    return c1, c2, c3, c4

def rel_error(true, est):
    eps = 1e-12
    denom = np.maximum(np.abs(true), eps)
    return np.abs(est - true) / denom

def plot_cumulant_errors(df, title="Relative Error per Cumulant by Model"):
    """
    Plots bar charts of relative errors for cumulants C1 to C4.
    Expects a DataFrame with columns like: 'Model', 'C1_RelErr', ..., 'C4_RelErr'.
    """
    df = df.copy()
    df = df.set_index("Model")

    cumulants = ["C1_RelErr", "C2_RelErr", "C3_RelErr", "C4_RelErr"]
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    
    plt.figure(figsize=(12, 6))
    for i, cum in enumerate(cumulants):
        if cum in df.columns:
            plt.bar(df.index, df[cum], label=cum.replace("_RelErr", ""), color=colors[i], alpha=0.8)
    
    plt.xticks(rotation=45)
    plt.ylabel("Relative Error")
    plt.title(title)
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()