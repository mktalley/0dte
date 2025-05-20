import numpy as np
import pytest
from datetime import datetime
from scipy.stats import norm

from src.main import calculate_iv, calculate_delta

# Helper functions for Black-Scholes pricing

def bs_put_price(S, K, r, T, sigma):
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_price(S, K, r, T, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def test_calculate_iv_roundtrip():
    S = 100.0
    K = 100.0
    r = 0.01
    T = 0.5
    sigma_true = 0.25
    price = bs_put_price(S, K, r, T, sigma_true)
    sigma_est = calculate_iv(price, S, K, T, r, 'put')
    assert pytest.approx(sigma_true, rel=1e-2) == sigma_est


def test_calculate_delta_call_put():
    S = 100.0
    K = 100.0
    r = 0.01
    expiry = datetime.utcnow()
    # Approximate time to expiry
    price_put = bs_put_price(S, K, r, 1/365, 0.2)
    price_call = bs_call_price(S, K, r, 1/365, 0.2)
    delta_put = calculate_delta(price_put, K, expiry, S, r, 'put')
    delta_call = calculate_delta(price_call, K, expiry, S, r, 'call')
    assert delta_put is not None and delta_put < 0
    assert delta_call is not None and delta_call > 0
    # ATM call ~0.5
    assert 0.3 < delta_call < 0.7
    # ATM put ~ -0.5
    assert -0.7 < delta_put < -0.3