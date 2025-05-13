#!/usr/bin/env python3
"""
optimize_real_put_spread.py

Grid-search real-data 0DTE SPY put credit spreads using minute bars.
Performs a grid sweep over short/long put deltas and intraday profit/stop exits.

Usage:
    python optimize_real_put_spread.py
"""
import os
from datetime import date, datetime, timedelta
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import ContractType

# Grid parameters
SHORT_PUT_DELTAS = [0.25, 0.30, 0.35, 0.40, 0.45]
LONG_PUT_DELTAS  = [0.05, 0.10, 0.15, 0.20]
PROFIT_TAKES     = [0.25, 0.50]
STOP_LOSSES      = [0.50, 1.00]

# Constants
RISK_FREE_RATE = 0.01
T_EXPIRY = 1/252

# Black-Scholes put delta
 def bs_d1(S, K, r, T, sigma):
    return (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, T, sigma)
    return norm.cdf(d1) - 1.0

# Solve K from target delta
 def strike_from_delta(S, r, T, sigma, target_delta):
    f = lambda K: put_delta(S, K, r, T, sigma) - target_delta
    try:
        return brentq(f, 1e-6, S)
    except:
        return None

# Helper to compute mid price of bar
 def mid_price(bar):
    return (bar.open + bar.high + bar.low + bar.close) / 4.0

# Backtest for one parameter set
def backtest_params(sp_sd, sp_ld, profit_take, stop_loss, dates, price_df, vix_df, client):
    total_pnl = 0.0
    wins = 0
    trades = 0
    for dt in dates:
        S_o = price_df.at[dt, 'Open']
        S_c = price_df.at[dt, 'Close']
        sigma = vix_df.at[dt, 'Close'] / 100.0
        # compute strikes
        K_s = strike_from_delta(S_o, RISK_FREE_RATE, T_EXPIRY, sigma, -sp_sd)
        K_l = strike_from_delta(S_o, RISK_FREE_RATE, T_EXPIRY, sigma, -sp_ld)
        if K_s is None or K_l is None or K_l >= K_s:
            continue
        # fetch option chain
        chain = client.get_option_chain(OptionChainRequest(
            underlying_symbol='SPY', expiration_date=dt, type=ContractType.PUT
        ))
        if not chain:
            continue
        # map underlying symbol strings to strike prices
        strikes = {sym: int(sym[-8:]) / 1000.0 for sym in chain.keys()}
        # find nearest symbols
        sym_s = min(strikes, key=lambda s: abs(strikes[s] - K_s))
        sym_l = min(strikes, key=lambda s: abs(strikes[s] - K_l))
        # fetch minute bars
        start_dt = datetime.combine(dt, datetime.min.time().replace(hour=9, minute=30))
        end_dt   = datetime.combine(dt, datetime.min.time().replace(hour=16, minute=0))
        bars = client.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=[sym_s, sym_l], start=start_dt, end=end_dt, timeframe=TimeFrame.Minute
        ))
        data = bars.data
        if sym_s not in data or sym_l not in data:
            continue
        # build aligned bar lists
        df_s = pd.DataFrame([{ 't': b.timestamp, 'open':b.open, 'high':b.high, 'low':b.low, 'close':b.close } for b in data[sym_s]])
        df_l = pd.DataFrame([{ 't': b.timestamp, 'open':b.open, 'high':b.high, 'low':b.low, 'close':b.close } for b in data[sym_l]])
        if df_s.empty or df_l.empty:
            continue
        # merge on timestamps
        df = pd.merge(df_s, df_l, on='t', suffixes=('_s','_l'))
        if df.empty:
            continue
        # compute spread price series
        df['spread'] = df.apply(lambda r: mid_price(r[['open_s','high_s','low_s','close_s']]._asdict()) - mid_price(r[['open_l','high_l','low_l','close_l']]._asdict()), axis=1)
        entry = df.iloc[0]['spread']
        # intraday exit
        exit_price = df.iloc[-1]['spread']
        pnl = None
        for price in df['spread'].iloc[1:]:
            if price <= entry * (1 - profit_take):
                pnl = (entry - price) * 100
                wins += 1
                break
            if price >= entry * (1 + stop_loss):
                pnl = -(price - entry) * 100
                break
        if pnl is None:
            pnl = (entry - exit_price) * 100
            if pnl > 0: wins += 1
        total_pnl += pnl
        trades += 1
    win_rate = wins / trades * 100 if trades else 0.0
    return total_pnl, trades, win_rate

if __name__ == '__main__':
    # load daily data
    end_d = date.today()
    start_d = end_d - timedelta(days=365)
    df_spy = yf.download('SPY', start=start_d, end=end_d + timedelta(days=1), progress=False)
    if isinstance(df_spy.columns, pd.MultiIndex): df_spy.columns = df_spy.columns.droplevel(1)
    df_vix = yf.download('^VIX', start=start_d, end=end_d + timedelta(days=1), progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.droplevel(1)
    # get all Fridays
    dates = sorted(set(df_spy.index).intersection(df_vix.index))
    dates = [d for d in dates if d.weekday() == 4]
    # Alpaca client
    client = OptionHistoricalDataClient()
    best = (None, float('-inf'))
    print('Optimizing real-data backtest...')
    for sp_sd in SHORT_PUT_DELTAS:
        for sp_ld in LONG_PUT_DELTAS:
            if sp_ld >= sp_sd: continue
            for pt in PROFIT_TAKES:
                for sl in STOP_LOSSES:
                    tot, cnt, wr = backtest_params(sp_sd, sp_ld, pt, sl, dates, df_spy[['Open','Close']], df_vix[['Close']], client)
                    if tot > best[1]: best = ((sp_sd,sp_ld,pt,sl,cnt,wr), tot)
    (spd, pld, pt, sl, cnt, wr), pnl = best
    print(f"Best real backtest:")
    print(f"  Short δ: -{spd}, Long δ: -{pld}, Take: {pt*100:.0f}%, Stop: {sl*100:.0f}%")
    print(f"  Trades: {cnt}, PnL: ${pnl:.2f}, Win: {wr:.1f}%")
