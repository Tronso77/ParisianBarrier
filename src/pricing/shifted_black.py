import mpmath as mp
from typing import Literal

OptionType = Literal["call", "put"]

def price_shifted_black(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    shift: float,
    option_type: OptionType = "call",
) -> float:
    """
    Shifted-Black pricing: treat S0+shift and K+shift as underlying/strike,
    then apply standard BS formula.
    """
    # shift
    S = S0 + shift
    X = K  + shift

    # d1, d2
    d1 = (mp.log(S/X) + (r + 0.5 * sigma**2) * T) / (sigma * mp.sqrt(T))
    d2 = d1 - sigma * mp.sqrt(T)

    if option_type == "call":
        return float(S * mp.ncdf(d1) - X * mp.e**(-r * T) * mp.ncdf(d2))
    else:
        return float(X * mp.e**(-r * T) * mp.ncdf(-d2) - S * mp.ncdf(-d1))
