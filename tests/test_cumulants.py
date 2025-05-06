import pytest
import numpy as np
import scipy.special as sc
from src.validation.cumulants import cumulants


def test_abm_bm_cumulants():
    mu, sigma = 0.1, 0.3
    dt = 1.0
    # ABM
    c1, c2, c3, c4 = cumulants((mu, sigma), "ABM", dt)
    assert c1 == pytest.approx(mu * dt)
    assert c2 == pytest.approx(sigma**2 * dt)
    assert c3 == pytest.approx(0.0)
    assert c4 == pytest.approx(0.0)
    # BM is same as ABM
    c1b, c2b, c3b, c4b = cumulants((mu, sigma), "BM", dt)
    assert (c1b, c2b, c3b, c4b) == (c1, c2, c3, c4)


def test_gbm_cumulants():
    mu, sigma = 0.05, 0.2
    dt = 1.0
    c1, c2, c3, c4 = cumulants((mu, sigma), "GBM", dt)
    assert c1 == pytest.approx((mu - 0.5 * sigma**2) * dt)
    assert c2 == pytest.approx(sigma**2 * dt)
    assert c3 == pytest.approx(0.0)
    assert c4 == pytest.approx(0.0)


def test_vg_cumulants():
    mu, theta, sigma, kappa = 0.01, 0.02, 0.3, 0.5
    dt = 1.0
    c1, c2, c3, c4 = cumulants((mu, theta, sigma, kappa), "VG", dt)
    assert c1 == pytest.approx(mu + theta)
    assert c2 == pytest.approx(sigma**2 + theta**2 * kappa)
    assert c3 == pytest.approx(
        kappa * (3 * sigma**2 * theta + 2 * theta**3 * kappa)
    )
    assert c4 == pytest.approx(
        kappa * (3 * sigma**4 + 4 * sigma**2 * theta**2 * kappa + 2 * theta**4 * kappa**2)
    )


def test_nig_cumulants():
    mu, theta, sigma, kappa = 0.02, 0.01, 0.25, 0.4
    dt = 1.0
    c1, c2, c3, c4 = cumulants((mu, theta, sigma, kappa), "NIG", dt)
    assert c1 == pytest.approx(mu + theta)
    assert c2 == pytest.approx(sigma**2 + theta**2 * kappa)
    assert c3 == pytest.approx(3 * theta * kappa * (sigma**2 + theta**2 * kappa))
    assert c4 == pytest.approx(
        kappa * (sigma**4 + 5 * theta**4 * kappa**2 + 6 * sigma**2 * theta**2 * kappa)
    )


def test_cgmy_cumulants():
    mu, C, G, M, Y = 0.03, 1.5, 2.0, 3.0, 0.5
    dt = 1.0
    c1, c2, c3, c4 = cumulants((mu, C, G, M, Y), "CGMY", dt)
    assert c1 == pytest.approx(mu + C * sc.gamma(1 - Y) * (M**(Y - 1) - G**(Y - 1)))
    assert c2 == pytest.approx(C * sc.gamma(2 - Y) * (M**(Y - 2) + G**(Y - 2)))
    assert c3 == pytest.approx(C * sc.gamma(3 - Y) * (M**(Y - 3) - G**(Y - 3)))
    assert c4 == pytest.approx(C * sc.gamma(4 - Y) * (M**(Y - 4) + G**(Y - 4)))


def test_mjd_cumulants():
    mu, sig, lamb, muZ, sigmaZ = 0.04, 0.2, 1.0, 0.01, 0.02
    dt = 1.0
    c1, c2, c3, c4 = cumulants((mu, sig, lamb, muZ, sigmaZ), "MJD", dt)
    assert c1 == pytest.approx(mu + lamb * muZ)
    assert c2 == pytest.approx(sig**2 + lamb * (muZ**2 + sigmaZ**2))
    assert c3 == pytest.approx(lamb * (muZ**3 + 3 * sigmaZ**2 * muZ))
    assert c4 == pytest.approx(
        lamb * (3 * sigmaZ**4 + 6 * sigmaZ**2 * muZ**2 + muZ**4)
    )


def test_kjd_cumulants():
    mu, sig, lamb, p, eta1, eta2 = 0.01, 0.15, 0.8, 0.3, 1.2, 2.5
    dt = 1.0
    c1, c2, c3, c4 = cumulants((mu, sig, lamb, p, eta1, eta2), "KJD", dt)
    assert c1 == pytest.approx(mu + lamb * (p / eta1 - (1 - p) / eta2))
    assert c2 == pytest.approx(sig**2 + lamb * (2 * p / eta1**2 + 2 * (1 - p) / eta2**2))
    assert c3 == pytest.approx(lamb * (6 * p / eta1**3 - 6 * (1 - p) / eta2**3))
    assert c4 == pytest.approx(
        lamb * (24 * p / eta1**4 + 24 * (1 - p) / eta2**4)
    )


def test_gamma_cumulants():
    alpha, lambdaG = 2.0, 3.0
    dt = 1.0
    c1, c2, c3, c4 = cumulants((alpha, lambdaG), "GAMMA", dt)
    assert c1 == pytest.approx(alpha / lambdaG)
    assert c2 == pytest.approx(alpha / lambdaG**2)
    assert c3 == pytest.approx(2 * alpha / lambdaG**3)
    assert c4 == pytest.approx(6 * alpha / lambdaG**4)


def test_poisson_cumulants():
    lambdaP = 2.5
    dt = 1.0
    c1, c2, c3, c4 = cumulants((lambdaP,), "POI", dt)
    assert c1 == pytest.approx(lambdaP)
    assert c2 == pytest.approx(lambdaP)
    assert c3 == pytest.approx(lambdaP)
    assert c4 == pytest.approx(lambdaP)
