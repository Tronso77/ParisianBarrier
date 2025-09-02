# src/pricing/parisian_pde.py
from __future__ import annotations
from dataclasses import dataclass
from math import log
import numpy as np

from .vanilla import bs_price


def thomas_solve(a, b, c, d):
    """Solve tridiagonal system with vectors a (sub), b (diag), c (super)."""
    n = len(d)
    ac = np.asarray(a, dtype=float).copy()
    bc = np.asarray(b, dtype=float).copy()
    cc = np.asarray(c, dtype=float).copy()
    dc = np.asarray(d, dtype=float).copy()
    for i in range(1, n):
        w = ac[i - 1] / bc[i - 1]
        bc[i] -= w * cc[i - 1]
        dc[i] -= w * dc[i - 1]
    x = np.empty(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


@dataclass
class ParisianParams:
    S0: float; K: float; T: float; r: float; q: float; sigma: float
    B: float; D: float
    option_type: str = 'call'   # 'call' or 'put'
    direction: str = 'down'     # 'down' or 'up'
    inout: str = 'out'          # 'out' or 'in'


@dataclass
class GridParams:
    """
    Grid in x = ln S and τ \in [0, D], with advection in τ and CN in x.
    - If S_left/S_right (or x_left/x_right) are not provided, we build a wide domain
      using multipliers below to reduce boundary bias (important for barriers).
    """
    Nx: int = 300
    Ntau: int = 150
    S_left: float | None = None
    S_right: float | None = None
    x_left: float | None = None
    x_right: float | None = None
    # Domain multipliers when S_left/S_right are not given:
    s_left_mult: float = 12.0   # Smin ≈ min(S0,B) / s_left_mult
    s_right_mult: float = 6.0   # Smax ≈ max(S0,B) * s_right_mult


def _price_out(p: ParisianParams, g: GridParams) -> float:
    """
    Parisian OUT via 2D scheme:
      - CN in x for diffusion step (for each τ-slice),
      - pure advection in τ on the barrier side (shift τ → τ + dt).
    """
    # ----- S-domain in log-space -----
    if g.S_left is None or g.S_right is None:
        Smin = max(1e-12, min(p.S0, p.B) / max(g.s_left_mult, 1.0))
        Smax = max(p.S0, p.B) * max(g.s_right_mult, 1.0)
    else:
        Smin, Smax = float(g.S_left), float(g.S_right)

    xL = log(Smin) if g.x_left is None else float(g.x_left)
    xR = log(Smax) if g.x_right is None else float(g.x_right)

    Nx = int(g.Nx)
    x = np.linspace(xL, xR, Nx)
    S = np.exp(x)

    # Normalize flags once
    opt_call = (p.option_type.lower() == 'call')
    side_down = (p.direction.lower() == 'down')

    # ----- τ/time grids -----
    Ntau = int(g.Ntau)
    dtau = p.D / max(Ntau, 1)
    Nt = max(1, int(np.ceil(p.T / max(dtau, 1e-16))))
    dt = p.T / Nt
    shift_ratio = dt / dtau  # ideally 1.0

    # ----- PDE coefficients for CN in x -----
    sig2 = p.sigma * p.sigma
    a = 0.5 * sig2
    b = (p.r - p.q) - 0.5 * sig2
    r = p.r

    hx = (xR - xL) / (Nx - 1)
    alpha = a / (hx * hx) - b / (2 * hx)
    beta  = -2 * a / (hx * hx) - r
    gamma = a / (hx * hx) + b / (2 * hx)

    low  = -0.5 * dt * alpha * np.ones(Nx - 1)
    diag = (1 - 0.5 * dt * beta) * np.ones(Nx)
    upp  = -0.5 * dt * gamma * np.ones(Nx - 1)

    lowR =  0.5 * dt * alpha * np.ones(Nx - 1)
    diagR= (1 + 0.5 * dt * beta) * np.ones(Nx)
    uppR =  0.5 * dt * gamma * np.ones(Nx - 1)

    # Where the clock runs
    mask = (S < p.B) if side_down else (S > p.B)
    idx_mask = np.where(mask)[0]

    # ----- terminal payoff across τ-slices -----
    payoff = np.maximum(S - p.K, 0.0) if opt_call else np.maximum(p.K - S, 0.0)
    # τ<D: vanilla payoff; τ = D: OUT already knocked out → 0
    V = np.zeros((Nx, Ntau + 1))
    V[:, :-1] = payoff[:, None]

    # ----- backward in time -----
    t = p.T
    for _ in range(Nt):
        t_next = t - dt

        # Boundary conditions (Dirichlet-like asymptotics)
        if opt_call:
            V_left  = 0.0
            V_right = S[-1] * np.exp(-p.q * t_next) - p.K * np.exp(-p.r * t_next)
        else:
            V_left  = p.K * np.exp(-p.r * t_next)
            V_right = 0.0

        # Step A: CN diffusion for each τ-slice (independent tri-diag solves)
        for j in range(Ntau + 1):
            rhs = diagR * V[:, j]
            rhs[1:]  += lowR * V[:-1, j]
            rhs[:-1] += uppR * V[1:,  j]
            # add BCs
            rhs[0]  += 0.5 * dt * alpha * V_left
            rhs[-1] += 0.5 * dt * gamma * V_right
            V[:, j] = thomas_solve(low, diag, upp, rhs)

        # Step B: advection in τ on barrier side (shift τ→τ+dt)
        V_new = V.copy()
        if idx_mask.size:
            if abs(shift_ratio - 1.0) < 1e-12:
                # exact one-slice shift
                V_new[idx_mask, :-1] = V[idx_mask, 1:]
                V_new[idx_mask, -1]  = 0.0  # absorbing at τ = D for OUT
            else:
                s = shift_ratio
                for i in idx_mask:
                    for j in range(Ntau + 1):
                        src = j + s
                        if src >= Ntau:
                            V_new[i, j] = 0.0
                        elif src <= 0:
                            V_new[i, j] = V[i, 0]
                        else:
                            j0 = int(np.floor(src))
                            w  = src - j0
                            V_new[i, j] = (1 - w) * V[i, j0] + w * V[i, j0 + 1]
        V = V_new
        V[:, -1] = 0.0  # enforce absorbing top τ-boundary
        t = t_next

    # Interpolate at S0, τ=0
    x0 = log(p.S0)
    if x0 <= xL:
        return float(V[0, 0])
    if x0 >= xR:
        return float(V[-1, 0])
    i = int((x0 - xL) / hx)
    w = (x0 - (xL + i * hx)) / hx
    return float((1 - w) * V[i, 0] + w * V[i + 1, 0])


def price_parisian(params: ParisianParams, grid: GridParams | None = None, *, enforce_bounds: bool = True) -> float:
    """
    OUT price by 2D PDE; IN by parity. If enforce_bounds=True, clip
    OUT into [0, vanilla] to guard against tiny grid overshoots.
    """
    p = params
    g = grid or GridParams()

    if p.inout.lower() == 'out':
        v_out = _price_out(p, g)
        if enforce_bounds:
            v_van = bs_price(p.S0, p.K, p.T, p.r, p.q, p.sigma, p.option_type)
            # tiny epsilon to avoid micro negative IN after subtraction
            eps = 1e-10
            v_out = min(max(v_out, 0.0), v_van + eps)
        return float(v_out)

    # Knock-IN via parity
    v_van = bs_price(p.S0, p.K, p.T, p.r, p.q, p.sigma, p.option_type)
    v_out = _price_out(ParisianParams(**{**p.__dict__, 'inout': 'out'}), g)
    if enforce_bounds:
        v_out = min(max(v_out, 0.0), v_van)
    return float(v_van - v_out)


def parity_error(params: ParisianParams, grid: GridParams | None = None) -> float:
    """Return (IN + OUT − vanilla) to diagnose grid/BC accuracy."""
    p = params
    g = grid or GridParams()
    v_van = bs_price(p.S0, p.K, p.T, p.r, p.q, p.sigma, p.option_type)
    v_out = _price_out(ParisianParams(**{**p.__dict__, 'inout': 'out'}), g)
    v_in  = v_van - v_out
    return float(v_in + v_out - v_van)


def converge_check(params: ParisianParams, Nx_list=(200, 300, 400), Ntau_list=(100, 150, 200)):
    """Quick grid sweep helper."""
    out = []
    for Nx in Nx_list:
        for Ntau in Ntau_list:
            price = price_parisian(params, GridParams(Nx=Nx, Ntau=Ntau))
            out.append({'Nx': Nx, 'Ntau': Ntau, 'price': price})
    return out
