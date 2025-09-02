# src/pricing/parisian_lattice.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .vanilla import bs_price

__all__ = ["LatticeParams", "price_parisian_binomial"]

@dataclass
class LatticeParams:
    S0: float; K: float; T: float; r: float; q: float; sigma: float
    B: float; D: float
    steps: int = 220
    option_type: str = "call"      # "call" or "put"
    inout: str = "out"             # "out" or "in"
    direction: str = "down"        # "down" or "up"
    style: str = "cumulative"      # "cumulative" or "consecutive"
    in_via_parity: bool = True     # prefer IN = vanilla - OUT (lower bias)

def price_parisian_binomial(p: LatticeParams) -> float:
    """
    CRR binomial with an extra Parisian 'clock' state k in {0..need}.
      - cumulative:   clock increments by 1 when the *arrival* node lies on the barrier side
      - consecutive:  clock increments by 1 on barrier side, else resets to 0
      - OUT:          price is set to 0 when clock hits 'need' (absorbing KO)
      - IN (default): computed via parity (vanilla - OUT) for lower bias; direct IN also supported

    Conventions:
      - 'down' barrier: barrier side = {S < B};  'up' barrier: barrier side = {S > B}
      - Arrival rule: we test the node *after* the move (this matches MC/PDE “occupation over the next dt”)

    Edge cases handled:
      - T <= 0:
          * If D <= 0: OUT = 0 (immediate KO), IN = vanilla
          * If D > 0:  OUT = vanilla (no time to spend), IN = 0
      - need := ceil(D/dt). If need > N (cannot reach threshold within maturity):
          * OUT = vanilla, IN = 0
      - Numerical safety: clamp CRR probability to [0,1]
    """
    N = int(max(0, p.steps))
    T = float(p.T)
    S0 = float(p.S0)
    K = float(p.K)
    r = float(p.r)
    q = float(p.q)
    sig = float(p.sigma)
    call = (p.option_type.lower() == "call")
    down = (p.direction.lower() == "down")
    cumulative = (p.style.lower() == "cumulative")
    out_flag = (p.inout.lower() == "out")

    # ---- T <= 0: payoff determined at inception ----
    if T <= 0.0 or N == 0:
        # vanilla intrinsic at S0
        van0 = max(S0 - K, 0.0) if call else max(K - S0, 0.0)
        if p.D <= 0.0:
            # zero window → OUT instantly KO; IN matches vanilla
            return 0.0 if out_flag else float(van0)
        else:
            # positive window but zero time → no KO possible
            return float(van0) if out_flag else 0.0

    # ---- CRR parameters ----
    dt = T / N
    u = math.exp(sig * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    # risk-neutral prob under carry r - q
    pu = (math.exp((r - q) * dt) - d) / (u - d)
    pu = 1.0 if pu > 1.0 else (0.0 if pu < 0.0 else pu)
    pd = 1.0 - pu

    # ---- Parisian threshold in steps ----
    need = max(1, int(math.floor(p.D / dt)) + 1)

    # If D == 0: OUT instantly KO (price 0); IN == vanilla
    if need == 0:
        if out_flag:
            return 0.0
        else:
            # vanilla under BS is fine (quicker and less biased than lattice vanilla)
            return bs_price(S0, K, T, r, q, sig, p.option_type)

    # If need > N: target consecutive/cumulative window cannot be reached
    if need > N:
        # OUT never KO => OUT == vanilla; IN == 0 by parity
        if out_flag:
            return bs_price(S0, K, T, r, q, sig, p.option_type)
        else:
            return 0.0

    # ---- Precompute prices at each layer for barrier-side tests ----
    # S_nodes[i] = array length (i+1) with prices at time i*dt
    S_nodes = [None] * (N + 1)
    for i in range(N + 1):
        j = np.arange(i + 1)
        S_nodes[i] = S0 * (u ** j) * (d ** (i - j))

    # ---- Helpers: OUT value via rolling layers ----
    def _value_out() -> float:
        # Terminal layer at i=N
        # V_next[j] is a vector (need+1,), clock state at node (N, j)
        V_next = []
        ST = S_nodes[N]
        for j in range(N + 1):
            van = (ST[j] - K) if call else (K - ST[j])
            van = van if van > 0.0 else 0.0
            arr = np.empty(need + 1, dtype=float)
            # alive (k < need) → vanilla payoff at maturity
            arr[:need] = van
            # already KO (k == need) → 0
            arr[need] = 0.0
            V_next.append(arr)

        # Backward induction
        for i in range(N - 1, -1, -1):
            S_up = S_nodes[i + 1][1:]    # arrival after UP  from (i,j)
            S_dn = S_nodes[i + 1][:-1]   # arrival after DOWN from (i,j)
            on_up = (S_up < p.B) if down else (S_up > p.B)
            on_dn = (S_dn < p.B) if down else (S_dn > p.B)

            V_cur = [np.zeros(need + 1, dtype=float) for _ in range(i + 1)]
            for j in range(i + 1):
                # Determine how the clock updates on arrival
                inc_u = 1 if on_up[j] else 0
                inc_d = 1 if on_dn[j] else 0
                for k in range(need + 1):
                    if cumulative:
                        ku = min(need, k + inc_u)
                        kd = min(need, k + inc_d)
                    else:  # consecutive
                        ku = min(need, (k + 1) if inc_u else 0)
                        kd = min(need, (k + 1) if inc_d else 0)

                    # OUT: if clock hits need at arrival → KO → continuation 0
                    up_val = 0.0 if ku == need else V_next[j + 1][ku]
                    dn_val = 0.0 if kd == need else V_next[j    ][kd]
                    V_cur[j][k] = disc * (pu * up_val + pd * dn_val)

            V_next = V_cur  # roll

        return float(V_next[0][0])

    # ---- Direct IN value (optional; parity is preferred) ----
    def _value_in_direct() -> float:
        # Terminal layer
        V_next = []
        ST = S_nodes[N]
        for j in range(N + 1):
            van = (ST[j] - K) if call else (K - ST[j])
            van = van if van > 0.0 else 0.0
            arr = np.zeros(need + 1, dtype=float)
            # only when clock already hit need by maturity do we pay vanilla
            arr[need] = van
            V_next.append(arr)

        # Backward induction; once IN (k==need), continue as vanilla (no barrier anymore)
        for i in range(N - 1, -1, -1):
            S_up = S_nodes[i + 1][1:]
            S_dn = S_nodes[i + 1][:-1]
            on_up = (S_up < p.B) if down else (S_up > p.B)
            on_dn = (S_dn < p.B) if down else (S_dn > p.B)

            V_cur = [np.zeros(need + 1, dtype=float) for _ in range(i + 1)]
            for j in range(i + 1):
                # vanilla continuation for “already-in” layer
                V_cur[j][need] = disc * (pu * V_next[j + 1][need] + pd * V_next[j][need])
                inc_u = 1 if on_up[j] else 0
                inc_d = 1 if on_dn[j] else 0
                for k in range(need):
                    if cumulative:
                        ku = min(need, k + inc_u)
                        kd = min(need, k + inc_d)
                    else:
                        ku = min(need, (k + 1) if inc_u else 0)
                        kd = min(need, (k + 1) if inc_d else 0)
                    up_val = V_next[j + 1][ku]
                    dn_val = V_next[j    ][kd]
                    V_cur[j][k] = disc * (pu * up_val + pd * dn_val)

            V_next = V_cur

        return float(V_next[0][0])

    # ---- Return according to IN/OUT flag ----
    if out_flag:
        return _value_out()

    # IN:
    if p.in_via_parity:
        v_out = _value_out()
        v_van = bs_price(S0, K, T, r, q, sig, p.option_type)
        return float(v_van - v_out)
    else:
        return _value_in_direct()
