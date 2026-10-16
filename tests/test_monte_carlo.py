import numpy as np
import pytest
from src.black_scholes import bs_price
from src.monte_carlo import mc_price


def test_mc_close_to_bs_call():
    bs = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
    mc, se = mc_price(100, 100, 1.0, 0.05, 0.2, "call", n_paths=100_000, seed=42)
    assert abs(mc - bs) < 4 * se


def test_mc_close_to_bs_put():
    bs = bs_price(100, 100, 1.0, 0.05, 0.2, "put")
    mc, se = mc_price(100, 100, 1.0, 0.05, 0.2, "put", n_paths=100_000, seed=42)
    assert abs(mc - bs) < 4 * se


def test_std_error_decreases_with_more_paths():
    _, se_small = mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=10_000, seed=1)
    _, se_large = mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=100_000, seed=1)
    assert se_large < se_small


def test_control_variate_reduces_std_error():
    _, se_plain = mc_price(
        100, 100, 1.0, 0.05, 0.2, n_paths=10_000,
        antithetic=False, control_variate=False, seed=1,
    )
    _, se_cv = mc_price(
        100, 100, 1.0, 0.05, 0.2, n_paths=10_000,
        antithetic=True, control_variate=True, seed=1,
    )
    assert se_cv < se_plain


def test_antithetic_reduces_std_error():
    _, se_plain = mc_price(
        100, 100, 1.0, 0.05, 0.2, n_paths=10_000,
        antithetic=False, control_variate=False, seed=1,
    )
    _, se_anti = mc_price(
        100, 100, 1.0, 0.05, 0.2, n_paths=10_000,
        antithetic=True, control_variate=False, seed=1,
    )
    assert se_anti < se_plain


def test_expiry_returns_intrinsic():
    price, se = mc_price(120, 100, 0, 0.05, 0.2, "call")
    assert price == 20.0