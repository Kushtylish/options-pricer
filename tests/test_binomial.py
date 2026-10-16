import numpy as np
import pytest
from src.black_scholes import bs_price
from src.binomial import binomial_price


def test_converges_to_bs_call():
    bs = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
    crr = binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=1000, option_type="call")
    assert abs(crr - bs) < 5e-3


def test_converges_to_bs_put():
    bs = bs_price(100, 100, 1.0, 0.05, 0.2, "put")
    crr = binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=1000, option_type="put")
    assert abs(crr - bs) < 5e-3


def test_american_put_geq_european_put():
    euro = binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=500, option_type="put", american=False)
    amer = binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=500, option_type="put", american=True)
    assert amer >= euro - 1e-6


def test_deep_itm_american_put_is_intrinsic():
    # S=50, K=100, american put: immediate exercise optimal
    price = binomial_price(50, 100, 1.0, 0.05, 0.2, n_steps=500, option_type="put", american=True)
    assert abs(price - 50.0) < 1e-6


def test_american_call_no_dividend_equals_european():
    # With no dividends, American call = European call
    euro = binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=500, option_type="call", american=False)
    amer = binomial_price(100, 100, 1.0, 0.05, 0.2, n_steps=500, option_type="call", american=True)
    assert abs(euro - amer) < 1e-6


def test_expiry_intrinsic():
    assert binomial_price(120, 100, 0, 0.05, 0.2, n_steps=100, option_type="call") == 20.0