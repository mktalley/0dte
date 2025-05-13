#!/usr/bin/env python3
"""
optimize_put_credit_spread_spy.py

Grid-search for 0DTE SPY put credit spread delta targets over the past year.
Usage:
    python optimize_put_credit_spread_spy.py
"""
import math
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

# Candidate delta targets (absolute values)
short_put_deltas = [0.25, 0.30, 0.35, 0.40, 0.45]
long_put_deltas  = [0.05, 0.10, 0.15, 0.20]

# Risk-free rate
RISK_FREE_RATE = 0.01

# Black-Scholes helpers
def bs_d1(S, K, r, T, sigma):
    return (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_put_price(S, K, r, T, sigma):
    if T <= 0:
        return max(K - S, 0)
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return 0 if S >= K else -1
    d1 = bs_d1(S, K, r, T, sigma)
    return norm.cdf(d1) - 1

def strike_from_delta(delta_func, S, r, T, sigma, target_delta):
    # target_delta is negative (e.g. -0.35)
    f = lambda K: delta_func(S, K, r, T, sigma) - target_delta
    try:
        return brentq(f, 1e-6, S)
    except:
        return None

# Backtest a given delta combination
def backtest_put_spread(sp_sd, sp_ld, df_spy, df_vix):
    r = RISK_FREE_RATE
    T = 1/252
    pnls = []
    # align dates; only Fridays
    dates = sorted(set(df_spy.index).intersection(df_vix.index))
    dates = [d for d in dates if d.weekday() == 4]
    for dt in dates:
        S_o = df_spy.at[dt, 'Open']
        S_c = df_spy.at[dt, 'Close']
        sigma = df_vix.at[dt, 'Close'] / 100.0
        K_s = strike_from_delta(put_delta, S_o, r, T, sigma, -sp_sd)
        K_l = strike_from_delta(put_delta, S_o, r, T, sigma, -sp_ld)
        if K_s is None or K_l is None or K_l >= K_s:
            continue
        # Entry credit
        p_s = bs_put_price(S_o, K_s, r, T, sigma)
        p_l = bs_put_price(S_o, K_l, r, T, sigma)
        credit = p_s - p_l
        # Exit cost intrinsic at close
        exit_cost = (max(K_s - S_c, 0) - max(K_l - S_c, 0))
        pnl = (credit - exit_cost) * 100
        pnls.append(pnl)
    total = sum(pnls)
    count = len(pnls)
    win_rate = sum(1 for x in pnls if x > 0) / count * 100 if count else 0.0
    return total, count, win_rate

if __name__ == '__main__':
    # load data
    end_d = date.today()
    start_d = end_d - timedelta(days=365)
    df_spy = yf.download('SPY', start=start_d, end=end_d + timedelta(days=1), progress=False)
    if isinstance(df_spy.columns, pd.MultiIndex):
        df_spy.columns = df_spy.columns.droplevel(level=1)
    df_vix = yf.download('^VIX', start=start_d, end=end_d + timedelta(days=1), progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(level=1)

    best = ((None, None), float('-inf'))
    print('Optimizing put credit spread...')
    for sp_sd in short_put_deltas:
        for sp_ld in long_put_deltas:
            if sp_ld >= sp_sd:
                continue
            total, cnt, wr = backtest_put_spread(sp_sd, sp_ld, df_spy, df_vix)
            if total > best[1]:
                best = ((sp_sd, sp_ld, cnt, wr), total)
    (best_sd, best_ld, cnt, wr), pnl = best
    print(f'Best Put Spread: Short delta = -{best_sd}, Long delta = -{best_ld}')
    print(f'Trades: {cnt}, Total PnL: ${pnl:.2f}, Win rate: {wr:.1f}%')
