#!/usr/bin/env python3
"""
optimize_real_put_credit_spread.py

Grid-search real-data 0DTE SPY put credit spread using downloaded minute bars.

Usage:
    python optimize_real_put_credit_spread.py [--data-dir PATH]
"""
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Black-Scholes helpers
def bs_d1(S, K, r, T, sigma):
    return (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    return norm.cdf(bs_d1(S,K,r,T,sigma)) - 1.0

def strike_from_delta(S, r, T, sigma, target_delta):
    f = lambda K: put_delta(S, K, r, T, sigma) - target_delta
    try:
        return brentq(f, 1e-6, S)
    except:
        return None

def parse_strike(sym: str) -> float:
    return int(sym[-8:]) / 1000.0

# Compute mid price of a bar
def mid_price(o, h, l, c):
    return (o + h + l + c) / 4.0

# Backtest single parameter set
def backtest_params(sp_sd, sp_ld, pt, sl, df_daily, data_dir):
    r = 0.01
    T = 1/252
    pnls = []
    wins = 0
    # select Fridays
    dates = [d.date() for d in df_daily.index if d.weekday() == 4]
    for dt in dates:
        day_dir = data_dir / dt.isoformat()
        if not day_dir.exists():
            continue
        # daily data
        row = df_daily.loc[pd.to_datetime(dt)]
        S_o = row['Open']
        S_c = row['Close']
        sigma = row['VIX'] / 100.0
        # compute theoretical strikes
        K_s = strike_from_delta(S_o, r, T, sigma, -sp_sd)
        K_l = strike_from_delta(S_o, r, T, sigma, -sp_ld)
        if K_s is None or K_l is None or K_l >= K_s:
            continue
        # list all put bar files
        files = list(day_dir.glob('*.csv'))
        if not files:
            continue
        # map sym->strike
        strikes = {f.stem: parse_strike(f.stem) for f in files}
        # pick nearest
        sym_s = min(strikes, key=lambda s: abs(strikes[s] - K_s))
        sym_l = min(strikes, key=lambda s: abs(strikes[s] - K_l))
        # load minute bars
        df_s = pd.read_csv(day_dir/sym_s + '.csv', parse_dates=['t'])
        df_l = pd.read_csv(day_dir/sym_l + '.csv', parse_dates=['t'])
        if df_s.empty or df_l.empty:
            continue
        # merge
        df = pd.merge(df_s, df_l, on='t', suffixes=('_s','_l'))
        if df.empty:
            continue
        # compute spread series
        df['spread'] = df.apply(lambda row: mid_price(row.open_s,row.high_s,row.low_s,row.close_s)
                                 - mid_price(row.open_l,row.high_l,row.low_l,row.close_l), axis=1)
        # entry spread
        entry = df.iloc[0]['spread']
        exit_price = df.iloc[-1]['spread']
        pnl = None
        # intraday exits
        for price in df['spread'].iloc[1:]:
            if price <= entry*(1 - pt):
                pnl = (entry - price) * 100
                wins += 1
                break
            if price >= entry*(1 + sl):
                pnl = -(price - entry) * 100
                break
        if pnl is None:
            pnl = (entry - exit_price) * 100
            if pnl > 0:
                wins += 1
        pnls.append(pnl)
    total = sum(pnls)
    trades = len(pnls)
    win_rate = wins / trades * 100 if trades else 0.0
    return total, trades, win_rate

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='backtest_data/SPY', help='Path to SPY backtest_data directory')
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    # load daily
    df_daily = pd.read_csv(data_dir/'underlying.csv', skiprows=[1,2], index_col=0, parse_dates=[0])
    best = (None, float('-inf'))
    # grid
    short_puts = [0.25, 0.30, 0.35, 0.40, 0.45]
    long_puts  = [0.05, 0.10, 0.15, 0.20]
    profits    = [0.25, 0.50]
    stops      = [0.50, 1.00]
    print('Running real-data grid search...')
    for sp_sd in short_puts:
        for sp_ld in long_puts:
            if sp_ld >= sp_sd: continue
            for pt in profits:
                for sl in stops:
                    tot, cnt, wr = backtest_params(sp_sd, sp_ld, pt, sl, df_daily, data_dir)
                    if tot > best[1]:
                        best = ((sp_sd,sp_ld,pt,sl,cnt,wr), tot)
    (spd,pld,pt,sl,cnt,wr), pnl = best
    print(f"Best real-data put spread:")
    print(f"  Short Put Δ = -{spd}, Long Put Δ = -{pld}")
    print(f"  Profit-take = {pt*100:.0f}%, Stop-loss = {sl*100:.0f}%")
    print(f"Trades: {cnt}, Total PnL = ${pnl:.2f}, Win rate = {wr:.1f}%")
