import numpy as np
from src.black_scholes import bs_price, bs_greeks


def fd_greeks(S, K, T, r, sigma, option_type="call", q=0.0, h_S=None, h_sigma=None, h_T=None, h_r=None):
    """
    Greeks computed by finite differences on the Black-Scholes price.

    Uses central differences where possible; one-sided for T and r near zero.

    Default bump sizes are chosen to balance truncation vs roundoff error.
    Returns dict with keys: delta, gamma, vega, theta, rho.
    """
    if h_S is None:
        h_S = max(1e-4 * S, 1e-4)
    if h_sigma is None:
        h_sigma = 1e-4
    if h_T is None:
        h_T = max(1e-4 * T, 1e-5)
    if h_r is None:
        h_r = 1e-5

    # Delta: dV/dS, central
    delta = (
        bs_price(S + h_S, K, T, r, sigma, option_type, q)
        - bs_price(S - h_S, K, T, r, sigma, option_type, q)
    ) / (2 * h_S)

    # Gamma: d2V/dS2, central
    gamma = (
        bs_price(S + h_S, K, T, r, sigma, option_type, q)
        - 2 * bs_price(S, K, T, r, sigma, option_type, q)
        + bs_price(S - h_S, K, T, r, sigma, option_type, q)
    ) / (h_S ** 2)

    # Vega: dV/dsigma, central
    vega = (
        bs_price(S, K, T, r, sigma + h_sigma, option_type, q)
        - bs_price(S, K, T, r, sigma - h_sigma, option_type, q)
    ) / (2 * h_sigma)

    # Theta: -dV/dT (note sign convention: theta is per year, negative for most options)
    if T > h_T:
        theta = -(
            bs_price(S, K, T + h_T, r, sigma, option_type, q)
            - bs_price(S, K, T - h_T, r, sigma, option_type, q)
        ) / (2 * h_T)
    else:
        # Near expiry, use forward difference
        theta = -(
            bs_price(S, K, T, r, sigma, option_type, q)
            - bs_price(S, K, max(T - h_T, 1e-6), r, sigma, option_type, q)
        ) / h_T

    # Rho: dV/dr, central
    rho = (
        bs_price(S, K, T, r + h_r, sigma, option_type, q)
        - bs_price(S, K, T, r - h_r, sigma, option_type, q)
    ) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def compare_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Return a dict of {greek: (analytic, finite_diff, abs_error)} for comparison.
    """
    analytic = bs_greeks(S, K, T, r, sigma, option_type, q)
    numeric = fd_greeks(S, K, T, r, sigma, option_type, q)
    out = {}
    for k in analytic:
        a = analytic[k]
        n = numeric[k]
        out[k] = (a, n, abs(a - n))
    return out