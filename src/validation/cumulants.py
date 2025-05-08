import numpy as np
import scipy.special as sc
from typing import Tuple, Optional

def norm_cum(params):
    """BM / ABM: returns (c1,c2,c3,c4)."""
    mu, sigma = params
    return mu, sigma**2, 0.0, 0.0

def gbm_cum(params):
    """GBM: (c1,c2,c3,c4) per unit time."""
    mu, sigma = params
    return mu - 0.5*sigma**2, sigma**2, 0.0, 0.0

def vg_cum(params):    
    mu, theta, sigma, kappa = params
    c1 = mu + theta
    c2 = sigma**2 + theta**2 * kappa
    c3 = kappa * (3*sigma**2*theta + 2*theta**3*kappa)
    c4 = kappa * (3*sigma**4 + 4*sigma**2*theta**2*kappa + 2*theta**4*kappa**2)
    return c1, c2, c3, c4

def nig_cum(params):   
    mu, theta, sigma, kappa = params
    c1 = mu + theta
    c2 = sigma**2 + theta**2*kappa
    c3 = 3*theta*kappa*(sigma**2 + theta**2*kappa)
    c4 = kappa*(sigma**4 + 5*theta**4*kappa**2 + 6*sigma**2*theta**2*kappa)
    return c1, c2, c3, c4

def cgmy_cum(params):  
    mu, C, G, M, Y = params
    c1 = mu + C*sc.gamma(1-Y)*(M**(Y-1) - G**(Y-1))
    c2 = C*sc.gamma(2-Y)*(M**(Y-2) + G**(Y-2))
    c3 = C*sc.gamma(3-Y)*(M**(Y-3) - G**(Y-3))
    c4 = C*sc.gamma(4-Y)*(M**(Y-4) + G**(Y-4))
    return c1, c2, c3, c4

def mjd_cum(params):   
    mu, sig, lam, muZ, sigmaZ = params
    c1 = mu + lam*muZ
    c2 = sig**2 + lam*(muZ**2 + sigmaZ**2)
    c3 = lam*(muZ**3 + 3*sigmaZ**2*muZ)
    c4 = lam*(3*sigmaZ**4 + 6*sigmaZ**2*muZ**2 + muZ**4)
    return c1, c2, c3, c4

def kjd_cum(params):  
    mu, sig, lam, p, e1, e2 = params
    c1 = mu + lam*(p/e1 - (1-p)/e2)
    c2 = sig**2 + lam*(2*p/e1**2 + 2*(1-p)/e2**2)
    c3 = lam*(6*p/e1**3 - 6*(1-p)/e2**3)
    c4 = lam*(24*p/e1**4 + 24*(1-p)/e2**4)
    return c1, c2, c3, c4

def gamma_cum(params):
    alpha, lam = params
    c1 = alpha/lam
    c2 = alpha/lam**2
    c3 = 2*alpha/lam**3
    c4 = 6*alpha/lam**4
    return c1, c2, c3, c4

def poisson_cum(params):
    lam = params[0]
    return lam, lam, lam, lam

def sv_cum(params, model, T):
    """Total cumulants over horizon T for CIR and Heston."""
    if model == "CIR":
        theta, kappa, eta, v0 = params
        c1_T = (theta - v0)*(1 - np.exp(-kappa*T))
        c2_T = (
            v0*eta**2*np.exp(-kappa*T)*(1 - np.exp(-kappa*T))/kappa
            + theta*eta**2*(1 - np.exp(-kappa*T))**2/(2*kappa)
        )
        return c1_T, c2_T

    if model == "HESTON":
        # params = (S0, v0, kappa, theta, eta, rho, r)
        # drop S0 here, so pass params[1:]
        v0, kappa, theta, eta, rho, r = params
        # first cumulant
        c1_T = (r - 0.5*theta)*T + (theta - v0)/(2*kappa)*(1 - np.exp(-kappa*T))
        # second cumulant (variance of log-price)
        term1 = eta**2/(4*kappa**3) * (2*kappa*T - 3 + 4*np.exp(-kappa*T) - np.exp(-2*kappa*T))
        term2 = (rho*eta)**2/(2*kappa**3) * (kappa*T - 1 + np.exp(-kappa*T))
        c2_T = term1 + term2
        return c1_T, c2_T

    raise ValueError(f"No SV-cum code for {model}")

def cumulants(params, model, dt, tj=None):
    """
    Returns (c1,c2,c3,c4) per unit time for every model.
    """
    M = model.upper()

    if M in ["BM","ABM"]:
        return norm_cum(params)
    if M == "GBM":
        return gbm_cum(params[1:])
    if M == "VG":
        return vg_cum(params)
    if M == "NIG":
        return nig_cum(params)
    if M == "CGMY":
        return cgmy_cum(params)
    if M == "MJD":
        return mjd_cum(params)
    if M == "KJD":
        return kjd_cum(params)
    if M == "GAMMA":
        return gamma_cum(params)
    if M == "POI":
        return poisson_cum(params)

    if M == "CIR":
        c1_T, c2_T = sv_cum(params, "CIR", dt)
        return c1_T/dt, c2_T/dt, 0.0, 0.0

    if M == "HESTON":
        # horizon for Heston is full tj[-1] or dt
        T = (tj[-1] if tj is not None else dt)
        # drop S0 from params
        slice_params = params[1:]
        c1_T, c2_T = sv_cum(slice_params, "HESTON", T)
        return c1_T/T, c2_T/T, 0.0, 0.0

    if M == "CEV":
        # params = (S0, mu, beta, sigma)
        S0, mu, beta, sigma = params
        c1 = mu
        c2 = sigma**2 * S0**(2*beta)
        return c1, c2, 0.0, 0.0

    if M == "SABR":
        return None, None, None, None

    raise ValueError(f"Unsupported model: {model}")