#!/usr/bin/env python3
"""
download_backtest_data.py

Download SPY/VIX daily and 0DTE SPY put option chains & minute bars between a date range.
Usage:
    pip install pandas yfinance python-dotenv alpaca-py
    python download_backtest_data.py --start 2024-05-01 --end 2025-05-01 --output-dir backtest_data
"""
import os
import argparse
from pathlib import Path
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionBarsRequest

from alpaca.data.timeframe import TimeFrame

import backtest_spy as bt
import time as time_module  # for retry backoff


def parse_args():
    parser = argparse.ArgumentParser(description="Download backtest data for SPY 0DTE strategy")
    parser.add_argument('--start', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('--end', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('--output-dir', default='backtest_data', help="Directory to store downloaded data")
    return parser.parse_args()


def iso_str(dt: date) -> str:
    return dt.isoformat()


def parse_strike(sym: str) -> float:
    # Last 8 characters encode strike price multiplied by 1000
    return int(sym[-8:]) / 1000.0


def main():
    args = parse_args()
    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    client = OptionHistoricalDataClient(api_key, api_secret)

    # Download daily SPY & VIX
    print("Downloading SPY & VIX daily data...")
    spy = yf.download('SPY', start=start_date, end=end_date + timedelta(days=1), interval='1d')
    vix = yf.download('^VIX', start=start_date, end=end_date + timedelta(days=1), interval='1d')
    # Extract VIX close as a DataFrame column
    vix_close = vix[['Close']].rename(columns={'Close': 'VIX'})
    df_daily = pd.concat([spy[['Open', 'Close']], vix_close], axis=1).dropna()
    underlying_path = output_dir / 'underlying.csv'
    df_daily.to_csv(underlying_path)
    print(f"Saved underlying daily data to {underlying_path}")

    # Strategy parameters
    # Prepare yfinance ticker for options
    ticker = yf.Ticker('SPY')

    r = bt.RISK_FREE_RATE
    T = 1 / 252
    short_delta = sum(bt.SHORT_PUT_DELTA_RANGE) / 2
    long_delta = sum(bt.LONG_PUT_DELTA_RANGE) / 2

    # Fetch full option chain once
    print("Fetching full SPY option chain...")
    full_chain = client.get_option_chain(OptionChainRequest(
        underlying_symbol='SPY'
    ))
    print(f"Fetched {len(full_chain)} option symbols")

    # Iterate trading days
    for dt, row in df_daily.iterrows():
        dt_date = dt.date()
        date_str = iso_str(dt_date)
        day_dir = output_dir / date_str
        day_dir.mkdir(exist_ok=True)

        # Fetch option chain expiring same day with retries
        print(f"[{date_str}] Fetching option chain...")
        for attempt in range(3):
            try:
                full_chain = client.get_option_chain(OptionChainRequest(
                    underlying_symbol='SPY',
                    expiration_date=dt_date
                ))
                break
            except Exception as e:
                print(f"[{date_str}] Error fetching option chain (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time_module.sleep(2 ** attempt)
                else:
                    print(f"[{date_str}] Failed to fetch option chain after 3 attempts, skipping")
                    full_chain = {}
        # Filter for options expiring today (symbol format: root + yymmdd)
        exp_str = dt_date.strftime('%y%m%d')
        chain = {sym: snap for sym, snap in full_chain.items() if sym[3:9] == exp_str}
        # Save raw chain
        import pickle
        with open(day_dir / 'chain.pkl', 'wb') as f:
            pickle.dump(chain, f)
        if not chain:
            print(f"[{date_str}] No options expiring that day.")
            continue

        # Compute target strikes
        S_open = float(row['Open'])
        sigma = float(row['VIX']) / 100.0
        try:
            K_short = bt.strike_from_delta(S_open, r, T, sigma, short_delta)
            K_long = bt.strike_from_delta(S_open, r, T, sigma, long_delta)
            if K_short <= K_long:
                K_short, K_long = max(K_short, K_long), min(K_short, K_long)
        except Exception as e:
            print(f"[{date_str}] Could not compute strikes: {e}")
            continue

        # Select the closest available puts
        puts = [sym for sym in chain.keys() if sym[9] == 'P']
        strikes = {sym: parse_strike(sym) for sym in puts}
        symbol_short = min(strikes.keys(), key=lambda s: abs(strikes[s] - K_short))
        symbol_long = min(strikes.keys(), key=lambda s: abs(strikes[s] - K_long))
        print(f"[{date_str}] Selected strikes: short={symbol_short} ({strikes[symbol_short]}), long={symbol_long} ({strikes[symbol_long]})")

        # Fetch intraday bars
        start_dt = datetime.combine(dt_date, time(9, 30), tzinfo=ZoneInfo('America/New_York'))
        end_dt = datetime.combine(dt_date, time(16, 0), tzinfo=ZoneInfo('America/New_York'))
        print(f"[{date_str}] Fetching minute bars for options...")
        bars = client.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=[symbol_short, symbol_long],
            start=start_dt,
            end=end_dt,
            timeframe=TimeFrame.Minute
        ))
        for sym in (symbol_short, symbol_long):
            bar_list = bars.get(sym, [])
            df_bars = pd.DataFrame([
                { 't': b.t, 'o': b.o, 'h': b.h, 'l': b.l, 'c': b.c, 'v': b.v }
                for b in bar_list
            ])
            csv_path = day_dir / f"{sym}.csv"
            df_bars.to_csv(csv_path, index=False)
        print(f"[{date_str}] Saved bars for {symbol_short} & {symbol_long}")

    print("Done downloading backtest data.")


if __name__ == '__main__':
    main()
