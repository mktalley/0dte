#!/usr/bin/env python3
"""
optimize_strangle_spy.py

Grid search for 0DTE short strangle parameters on SPY over the past year,
using theoretical Black-Scholes entry and intrinsic exit at EOD.

Usage:
    python optimize_strangle_spy.py
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import math

# Grid of delta targets (absolute values)
put_short_deltas = [0.30, 0.35, 0.40, 0.45]
put_long_deltas  = [0.10, 0.15, 0.20, 0.25]
call_short_deltas = [0.35, 0.40, 0.45]
call_long_deltas  = [0.15, 0.20, 0.25]

# Risk-free rate
RISK_FREE_RATE = 0.01

# Black-Scholes helper

def bs_d1(S, K, r, T, sigma):
    return (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

# Black-Scholes call price
def bs_call_price(S, K, r, T, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# Black-Scholes put price
def bs_put_price(S, K, r, T, sigma):
    if T <= 0:
        return max(K - S, 0)
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Delta functions

def call_delta(S, K, r, T, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, r, T, sigma)
    return norm.cdf(d1)

def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, T, sigma)
    return norm.cdf(d1) - 1.0

# Solve for strike given a target delta

def strike_from_delta(delta_func, S, r, T, sigma, target_delta):
    # target_delta should be signed for put/call
    f = lambda K: delta_func(S, K, r, T, sigma) - target_delta
    a = 1e-6
    b = S * 2
    try:
        return brentq(f, a, b)
    except Exception:
        return None

# Backtest for a given parameter set
def backtest_strangle(sp_sd, sp_ld, sc_sd, sc_ld, df_spy, df_vix):
    # sp_sd: short put delta abs (positive number), use -sp_sd
    # sp_ld: long put delta abs
    # sc_sd: short call delta abs
    # sc_ld: long call delta abs
    r = RISK_FREE_RATE
    T = 1/252
    pnls = []
    trade_dates = []
    # Align dates
    dates = sorted(set(df_spy.index).intersection(df_vix.index))
    # Only Fridays
    dates = [d for d in dates if d.weekday() == 4]
    for dt in dates:
        S_open = df_spy.at[dt, 'Open']
        S_close = df_spy.at[dt, 'Close']
        sigma = df_vix.at[dt, 'Close'] / 100.0
        # Compute strikes
        K_ps = strike_from_delta(put_delta, S_open, r, T, sigma, -sp_sd)
        K_pl = strike_from_delta(put_delta, S_open, r, T, sigma, -sp_ld)
        K_cs = strike_from_delta(call_delta, S_open, r, T, sigma, sc_sd)
        K_cl = strike_from_delta(call_delta, S_open, r, T, sigma, sc_ld)
        if None in (K_ps, K_pl, K_cs, K_cl):
            continue
        # Entry prices
        pu_short = bs_put_price(S_open, K_ps, r, T, sigma)
        pu_long  = bs_put_price(S_open, K_pl, r, T, sigma)
        ca_short = bs_call_price(S_open, K_cs, r, T, sigma)
        ca_long  = bs_call_price(S_open, K_cl, r, T, sigma)
        credit = (pu_short - pu_long) + (ca_short - ca_long)
        # Exit cost: intrinsic at close
        exit_put  = max(K_ps - S_close, 0) - max(K_pl - S_close, 0)
        exit_call = max(S_close - K_cs, 0) - max(S_close - K_cl, 0)
        exit_cost = exit_put + exit_call
        pnl = (credit - exit_cost) * 100  # per 1 contract per leg
        pnls.append(pnl)
        trade_dates.append(dt)
    total = sum(pnls)
    wins = sum(1 for x in pnls if x > 0)
    count = len(pnls)
    win_rate = wins / count * 100 if count else 0.0
    return total, count, win_rate

if __name__ == '__main__':
    # Load data
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    df_spy = yf.download('SPY', start=start_date, end=end_date + timedelta(days=1), progress=False)
    # flatten columns if MultiIndex
    if isinstance(df_spy.columns, pd.MultiIndex):
        df_spy.columns = df_spy.columns.droplevel(level=1)

    df_vix = yf.download('^VIX', start=start_date, end=end_date + timedelta(days=1), progress=False)

    # flatten VIX columns if MultiIndex
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(level=1)

    best = (None, float('-inf'))  # ((params), total_pnl)
    print('Running grid search...')
    for sp_sd in put_short_deltas:
        for sp_ld in put_long_deltas:
            if sp_ld >= sp_sd:
                continue
            for sc_sd in call_short_deltas:
                for sc_ld in call_long_deltas:
                    if sc_ld >= sc_sd:
                        continue
                    total, count, wr = backtest_strangle(sp_sd, sp_ld, sc_sd, sc_ld, df_spy, df_vix)
                    if total > best[1]:
                        best = ((sp_sd, sp_ld, sc_sd, sc_ld, count, wr), total)
    (spd, pld, scd, cld, cnt, winr), pnl = best
    print(f"Best parameters:")
    print(f"  Short Put Delta:  -{spd}")
    print(f"  Long  Put Delta:  -{pld}")
    print(f"  Short Call Delta: +{scd}")
    print(f"  Long  Call Delta: +{cld}")
    print(f"Trades: {cnt}, Total PnL: ${pnl:.2f}, Win rate: {winr:.1f}%")

