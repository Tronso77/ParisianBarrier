# src/pricing/parisian_cjy.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .vanilla import bs_price

__all__ = ["CJYParams", "StepGrid", "price_step_option_pde", "price_parisian_cjy"]

# --------------------------- Data classes ---------------------------

@dataclass
class CJYParams:
    S0: float; K: float; T: float; r: float; q: float; sigma: float
    B: float; D: float
    option_type: str = "call"   # "call" | "put"
    direction: str = "down"     # implemented: "down" (flip mask for "up")
    inout: str = "out"          # "out" | "in"

@dataclass
class StepGrid:
    """
    Grid for the 1D step-option PDE (x = ln S) solved with Crank–Nicolson.
    """
    Nx: int = 700
    S_left: float | None = None
    S_right: float | None = None
    x_left: float | None = None
    x_right: float | None = None
    Nt: int | None = None       # if None -> auto from hx and sigma*sqrt(T)

# --------------------------- Numerics helpers ---------------------------

def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Tri-diagonal solver (Thomas algorithm)."""
    n = d.size
    ac = a.copy(); bc = b.copy(); cc = c.copy(); dc = d.copy()
    for i in range(1, n):
        w = ac[i-1] / bc[i-1]
        bc[i] -= w * cc[i-1]
        dc[i] -= w * dc[i-1]
    x = np.empty(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i]*x[i+1]) / bc[i]
    return x

# --------------------------- Step option PDE ---------------------------

def price_step_option_pde(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
    B: float, alpha: float,
    option_type: str = "call",
    direction: str = "down",
    grid: StepGrid | None = None,
) -> float:
    """
    Solve the step (soft-barrier) option:
      v_t + 0.5 σ^2 S^2 v_SS + (r−q) S v_S − r v − α·1_{S<B} v = 0,  v(S,T)=vanilla payoff.
    This yields G(α) = E[e^{-rT} payoff · e^{-α A_T}], where A_T is occupation time below B.
    CJY: Parisian OUT(D) = Laplace^{-1}_α [ G(α)/α ] at D.
    """
    assert alpha >= 0.0, "alpha must be ≥ 0"
    g = grid or StepGrid()

    # Domain in S → x=ln S
    if g.S_left is None or g.S_right is None:
        Smin = min(S0, B) / 12.0
        Smax = max(S0, B) * 6.0
    else:
        Smin, Smax = g.S_left, g.S_right

    if g.x_left is not None and g.x_right is not None:
        xL, xR = float(g.x_left), float(g.x_right)
    else:
        xL, xR = math.log(Smin), math.log(Smax)

    Nx = int(g.Nx)
    x = np.linspace(xL, xR, Nx)
    S = np.exp(x)
    hx = (xR - xL) / (Nx - 1)

    # Time grid heuristic (stable & not too big)
    if g.Nt is None:
        # ~ proportional to (variance span / space step)^2
        Nt = max(60, int(2.5 * (sigma * math.sqrt(T) / max(hx, 1e-12))**2))
    else:
        Nt = int(g.Nt)
    dt = T / max(Nt, 1)

    # PDE coefficients in x
    a = 0.5 * sigma * sigma
    b = (r - q) - 0.5 * sigma * sigma
    c0 = -r

    # Kill on barrier side
    if direction.lower() == "down":
        kill_mask = (S < B)
    else:
        kill_mask = (S > B)
    alpha_d = alpha * kill_mask.astype(float)

    # Terminal payoff
    if option_type.lower() == "call":
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # CN system L v := a v_xx + b v_x + (c0 - α·mask) v
    al = a/(hx*hx) - b/(2*hx)
    be = -2*a/(hx*hx) + (c0 - alpha_d)         # array (due to alpha_d)
    ga = a/(hx*hx) + b/(2*hx)

    low  = -0.5*dt*al * np.ones(Nx-1)
    diag = (1 - 0.5*dt*be)                      # array (Nx,)
    upp  = -0.5*dt*ga * np.ones(Nx-1)

    lowR =  0.5*dt*al * np.ones(Nx-1)
    diagR= (1 + 0.5*dt*be)                      # array (Nx,)
    uppR =  0.5*dt*ga * np.ones(Nx-1)

    # Backward in time
    t = T
    for _ in range(Nt):
        t_next = t - dt

        # Asymptotic (Dirichlet-like) boundaries
        if option_type.lower() == "call":
            V_left  = 0.0
            V_right = S[-1]*math.exp(-q*t_next) - K*math.exp(-r*t_next)
        else:
            V_left  = K*math.exp(-r*t_next)
            V_right = 0.0

        rhs = diagR * V
        rhs[1:]  += lowR * V[:-1]
        rhs[:-1] += uppR * V[1:]
        rhs[0]   += 0.5*dt*al * V_left
        rhs[-1]  += 0.5*dt*ga * V_right

        V = _thomas(low, diag.copy(), upp, rhs)
        t = t_next

    # Interpolate to S0
    x0 = math.log(S0)
    if x0 <= xL: return float(V[0])
    if x0 >= xR: return float(V[-1])
    i = int((x0 - xL) / hx)
    w = (x0 - (xL + i*hx)) / hx
    return float((1 - w) * V[i] + w * V[i+1])

# --------------------------- Laplace inversions ---------------------------

def _aw_invert_F_over_alpha_at_D(F_over_alpha, D: float, M: int = 64, euler: bool = True) -> float:
    """
    Abate–Whitt inversion of f(D) with F(s) = Laplace[f](s).
    Here F_over_alpha(s) = StepPrice(s)/s, so OUT(D) = L^{-1}[F_over_alpha](D).
    IMPORTANT: include (2/π) factor; Euler acceleration reduces truncation error.
    """
    if D <= 0.0:
        return 0.0
    L = math.log(2.0) / D
    partials = []
    S_alt = 0.0
    for k in range(M):
        s = (k + 0.5) * L
        S_alt += ((-1) ** k) * F_over_alpha(s)
        partials.append(S_alt)
    if euler and M >= 8:
        w = [math.comb(M - 1, k) for k in range(M)]
        S_alt = sum(w[k] * partials[k] for k in range(M)) / (2 ** (M - 1))
    return (2.0 / math.pi) * L * S_alt

def _stehfest_invert_F_over_alpha(F_over_alpha, D: float, N: int = 14) -> float:
    """
    Gaver–Stehfest real-axis inversion (cross-check). N must be even (10–14 typical).
    """
    if D <= 0.0:
        return 0.0
    if N % 2:
        N += 1
    ln2 = math.log(2.0)
    def V(k):
        s = 0.0
        for j in range(int((k+1)/2), min(k, N//2)+1):
            num = j**(N//2) * math.comb(N//2, j) * math.comb(2*j, j) * math.comb(j, k-j)
            den = math.factorial(N//2)
            s += num / den
        return s * ((-1) ** (k + N//2))
    acc = 0.0
    for k in range(1, N+1):
        s = (k * ln2) / D
        acc += V(k) * F_over_alpha(s)
    return (ln2 / D) * acc

# --------------------------- Public CJY price ---------------------------

def price_parisian_cjy(
    p: CJYParams,
    *, grid: StepGrid | None = None,
    M: int = 96,
    method: str = "auto",        # "auto" | "aw" | "stehfest"
    enforce_bounds: bool = True,
    calibrate: bool = True,      # normalize so L^{-1}{1/α}(D) = 1
) -> float:
    if p.direction.lower() != "down":
        raise NotImplementedError("Only down barriers implemented; flip mask for 'up'.")

    vanilla = bs_price(p.S0, p.K, p.T, p.r, p.q, p.sigma, p.option_type)

    def F_over_alpha(s: float) -> float:
        step = price_step_option_pde(
            p.S0, p.K, p.T, p.r, p.q, p.sigma, p.B, s,
            option_type=p.option_type, direction=p.direction, grid=grid
        )
        return step / max(s, 1e-16)

    # -------- choose inverter (AUTO) --------
    m = method.lower()
    if m == "auto":
        # If the transform is very flat (rare KO), Stehfest is safer; else AW is fine.
        z = abs(math.log(p.S0 / p.B)) / max(1e-12, p.sigma * math.sqrt(p.T))
        rel_D = p.D / max(p.T, 1e-12)
        L = math.log(2.0) / max(p.D, 1e-12)
        probe = [(k + 0.5) * L for k in range(6)]
        vals = np.array([F_over_alpha(s) for s in probe], float)
        flat = float(np.std(vals) / max(1e-12, abs(np.mean(vals))))
        m = "stehfest" if (flat < 1e-3 or rel_D <= 0.10 or z >= 0.8) else "aw"

    # -------- invert with calibration --------
    if m == "aw":
        inv = _aw_invert_F_over_alpha_at_D
        v_out = inv(F_over_alpha, p.D, M=M, euler=True)
        if calibrate:
            c = inv(lambda s: 1.0 / max(s, 1e-16), p.D, M=M, euler=True)
            if c > 0:
                v_out *= (1.0 / c)
    elif m == "stehfest":
        inv = _stehfest_invert_F_over_alpha
        v_out = inv(F_over_alpha, p.D, N=14)
        if calibrate:
            c = inv(lambda s: 1.0 / max(s, 1e-16), p.D, N=14)
            if c > 0:
                v_out *= (1.0 / c)
    else:
        raise ValueError("method must be 'auto', 'aw' or 'stehfest'")

    # -------- guard rails --------
    if enforce_bounds and (v_out < 0.0 or v_out > vanilla):
        # try other inverter, blend, and project
        alt = (_stehfest_invert_F_over_alpha if m == "aw" else _aw_invert_F_over_alpha_at_D)(
            F_over_alpha, p.D
        )
        if calibrate:
            c_alt = (_stehfest_invert_F_over_alpha if m == "aw" else _aw_invert_F_over_alpha_at_D)(
                lambda s: 1.0 / max(s, 1e-16), p.D
            )
            if c_alt > 0:
                alt *= (1.0 / c_alt)
        v_out = 0.5 * (v_out + alt)
        v_out = min(vanilla, max(0.0, v_out))

    return float(v_out if p.inout.lower() == "out" else vanilla - v_out)
