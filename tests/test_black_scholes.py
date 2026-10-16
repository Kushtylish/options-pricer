import numpy as np
import pytest
from src.black_scholes import bs_price, bs_greeks, put_call_parity_gap


def test_atm_call_textbook_value():
    # Classic textbook case: S=K=100, T=1, r=5%, sigma=20%
    price = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
    assert abs(price - 10.450584) < 1e-5


def test_atm_put_textbook_value():
    price = bs_price(100, 100, 1.0, 0.05, 0.2, "put")
    assert abs(price - 5.573526) < 1e-5


def test_put_call_parity():
    residual = put_call_parity_gap(100, 100, 1.0, 0.05, 0.2)
    assert abs(residual) < 1e-10


def test_call_price_bounds():
    # Call price must be in [max(0, S-Ke^-rT), S]
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    price = bs_price(S, K, T, r, sigma, "call")
    lower = max(S - K * np.exp(-r * T), 0.0)
    assert lower <= price <= S


def test_expiry_returns_intrinsic():
    # T=0 -> intrinsic value
    assert bs_price(120, 100, 0, 0.05, 0.2, "call") == 20.0
    assert bs_price(80, 100, 0, 0.05, 0.2, "call") == 0.0
    assert bs_price(80, 100, 0, 0.05, 0.2, "put") == 20.0


def test_dividend_yield_reduces_call_price():
    no_div = bs_price(100, 100, 1.0, 0.05, 0.2, "call", q=0.0)
    with_div = bs_price(100, 100, 1.0, 0.05, 0.2, "call", q=0.03)
    assert with_div < no_div


def test_analytic_delta_call_in_range():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    assert 0.0 < g["delta"] < 1.0


def test_analytic_put_delta_negative():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "put")
    assert -1.0 < g["delta"] < 0.0


def test_gamma_positive():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    assert g["gamma"] > 0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, 0.2, "forward")