# src/pricing/vol_surface.py
import numpy as np, pandas as pd

def load_t(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, parse_dates=['exp_date'])
    return df  # expects columns: exp_date,T,cp,delta_abs,delta_signed,iv,F,K

def _interp_1d(x, xp, fp):
    xp = np.asarray(xp, float); fp = np.asarray(fp, float); x = float(x)
    if x <= xp.min(): return float(fp[xp.argmin()])
    if x >= xp.max(): return float(fp[xp.argmax()])
    i = np.searchsorted(xp, x) - 1
    x0, x1 = xp[i], xp[i+1]; y0, y1 = fp[i], fp[i+1]
    w = (x - x0) / (x1 - x0)
    return float((1 - w) * y0 + w * y1)

def sigma_from_KT(tidy_df: pd.DataFrame, K: float, T: float) -> float:
    exps = np.sort(tidy_df['T'].unique())
    # bracket T
    if T <= exps.min():
        T0 = T1 = exps.min()
    elif T >= exps.max():
        T0 = T1 = exps.max()
    else:
        idx = np.searchsorted(exps, T)
        T0, T1 = exps[idx - 1], exps[idx]

    def sig_at(Tfix: float) -> float:
        # robust filter with isclose; fallback to nearest expiry if empty
        sub = tidy_df[np.isclose(tidy_df['T'].to_numpy(), Tfix, rtol=0, atol=1e-9)]
        if sub.size == 0 or len(sub) == 0:
            nearest = exps[np.argmin(np.abs(exps - Tfix))]
            sub = tidy_df[np.isclose(tidy_df['T'].to_numpy(), nearest, rtol=0, atol=1e-9)]
        sub = sub.sort_values('K')

        F = float(sub['F'].iloc[0])
        k = np.log(sub['K'].to_numpy() / F)
        s = sub['iv'].to_numpy()

        kx = float(np.log(K / F))
        return _interp_1d(kx, k, s)

    s0 = sig_at(T0); s1 = sig_at(T1)
    return s0 if T0 == T1 else _interp_1d(T, [T0, T1], [s0, s1])
