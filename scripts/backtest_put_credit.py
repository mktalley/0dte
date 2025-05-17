#!/usr/bin/env python3
"""
backtest_put_credit.py

Backtest the 0DTE SPY put-credit-spread strategy over the past year.
Generates a CSV of daily trades with credit, width, and strikes.
"""
import os
import csv
from datetime import date, datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

# === Strategy Parameter Logic ===
timezone = ZoneInfo("America/New_York")
# Risk parameters
max_risk_per_trade = 1000  # USD max risk per trade

# Functions copied from main.py

def calculate_iv(option_price, S, K, T, r, option_type):
    intrinsic = max(0, (S - K) if option_type == 'call' else (K - S))
    if option_price <= intrinsic + 1e-6:
        return 0.0
    def f(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - option_price
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - option_price
    try:
        return brentq(f, 1e-6, 5.0)
    except Exception:
        return None


def calculate_delta(option_price, strike, expiry, spot, r, option_type):
    now = datetime.now(tz=timezone)
    T = max((expiry - now).total_seconds() / (365 * 24 * 3600), 1e-6)
    iv = calculate_iv(option_price, spot, strike, T, r, option_type)
    if iv is None:
        return None
    d1 = (np.log(spot / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)


def get_params_for_date(d):
    dow = d.weekday()  # 0=Mon,4=Fri
    if dow <= 1:
        # Mon/Tue aggressive
        return {
            'MIN_CREDIT_PCT': 0.15,
            'OI_THRESHOLD': 200,
            'SHORT_PUT_DELTA_RANGE': (-0.5, -0.3),
            'LONG_PUT_DELTA_RANGE': (-0.3, -0.1),
            'STRIKE_RANGE': 0.2,
            'r': 0.01,
        }
    elif dow <= 3:
        # Wed/Thu relaxed
        return {
            'MIN_CREDIT_PCT': 0.15,
            'OI_THRESHOLD': 100,
            'SHORT_PUT_DELTA_RANGE': (-0.5, -0.3),
            'LONG_PUT_DELTA_RANGE': (-0.3, -0.1),
            'STRIKE_RANGE': 0.25,
            'r': 0.01,
        }
    else:
        # Fri tight
        return {
            'MIN_CREDIT_PCT': 0.25,
            'OI_THRESHOLD': 500,
            'SHORT_PUT_DELTA_RANGE': (-0.45, -0.35),
            'LONG_PUT_DELTA_RANGE': (-0.25, -0.15),
            'STRIKE_RANGE': 0.1,
            'r': 0.01,
        }


def backtest_symbol(symbol, start_date, end_date, out_csv):
    # initialize yfinance ticker and expiration dates
    ticker = yf.Ticker(symbol)
    exp_dates = set(ticker.options)
    # download all daily price history
    hist_df = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat(), interval='1d', auto_adjust=False)
    # map each date to the open price
    hist_dict = {dt.date(): op for dt, op in hist_df['Open'].items()}
    # Prepare output CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'symbol', 'short_strike', 'long_strike',
            'credit', 'width'
        ])

    # Iterate business days
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            params = get_params_for_date(d)
            # get spot price from pre-downloaded history
            spot = hist_dict.get(d)
            if spot is None:
                d += timedelta(days=1)
                continue
            # fetch option chain for today expiration
            date_str = d.strftime('%Y-%m-%d')
            if date_str not in exp_dates:
                d += timedelta(days=1)
                continue
            chain = ticker.option_chain(date_str).puts
            # filter strikes
            lo = spot * (1 - params['STRIKE_RANGE'])
            hi = spot * (1 + params['STRIKE_RANGE'])
            filt = chain[(chain['strike'] >= lo) & (chain['strike'] <= hi) &
                         (chain['openInterest'] >= params['OI_THRESHOLD'])]
            if len(filt) < 5:
                d += timedelta(days=1)
                continue
            # compute mid price
            filt = filt.copy()
            filt['mid'] = (filt['bid'] + filt['ask']) / 2
            # find short and long
            short = long = None
            expiry = datetime.combine(d, dt_time(16, 0)).replace(tzinfo=timezone)
            for _, row in filt.iterrows():
                delta = calculate_delta(row['mid'], row['strike'], expiry, spot, params['r'], 'put')
                if delta is None:
                    continue
                if params['SHORT_PUT_DELTA_RANGE'][0] <= delta <= params['SHORT_PUT_DELTA_RANGE'][1]:
                    short = (row['strike'], row['mid'])
                elif params['LONG_PUT_DELTA_RANGE'][0] <= delta <= params['LONG_PUT_DELTA_RANGE'][1]:
                    long = (row['strike'], row['mid'])
                if short and long:
                    break
            if short and long:
                credit = short[1] - long[1]
                width = abs(short[0] - long[0])
                if credit >= params['MIN_CREDIT_PCT'] * width and (width * 100) <= max_risk_per_trade:
                    with open(out_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([d.isoformat(), symbol,
                                         short[0], long[0], round(credit, 4), round(width, 4)])
        d += timedelta(days=1)


def main():
    today = date.today()
    start = today - timedelta(days=365)
    out = 'logs/backtest_spy.csv'
    os.makedirs('logs', exist_ok=True)
    print(f"Backtesting {start} to {today} on SPY...")
    backtest_symbol('SPY', start, today, out)
    print(f"Results written to {out}")

if __name__ == '__main__':
    main()
