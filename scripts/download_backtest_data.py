#!/usr/bin/env python3
"""
download_backtest_data.py

Download daily OHLC and implied vol plus 0DTE put option chains & minute bars for one or more underlyings over a date range.
Usage:
    pip install pandas yfinance python-dotenv alpaca-py
    python download_backtest_data.py --start 2024-05-01 --end 2025-05-01 --symbols SPY QQQ --output-dir backtest_data
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
from alpaca.trading.enums import ContractType

import backtest_spy as bt
import time as time_module  # for retry backoff


def parse_args():
    parser = argparse.ArgumentParser(description="Download 0DTE backtest data for one or more underlyings")
    parser.add_argument('--symbols', nargs='+', default=['SPY'], help="Underlying ticker symbols to download (e.g. SPY QQQ)")
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

    # Per-symbol data download for underlyings
    vol_map = {'SPY': '^VIX', 'QQQ': '^VXN', 'IWM': '^VIX'}  # map underlying to its volatility index
    for symbol in args.symbols:
        sym_dir = output_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        vol_symbol = vol_map.get(symbol, '^VIX')
        print(f"Downloading daily data for {symbol} & {vol_symbol}...")
        df_stock = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), interval='1d', progress=False)
        df_vol   = yf.download(vol_symbol, start=start_date, end=end_date + timedelta(days=1), interval='1d', progress=False)
        df_vol_close = df_vol[['Close']].rename(columns={'Close': 'VIX'})
        df_daily_sym = pd.concat([df_stock[['Open', 'Close']], df_vol_close], axis=1).dropna()
        df_daily_sym.to_csv(sym_dir / 'underlying.csv')
        print(f"Saved underlying.csv for {symbol}")

        # Strategy parameters
        r = bt.RISK_FREE_RATE
        T = 1 / 252
        short_delta = sum(bt.SHORT_PUT_DELTA_RANGE) / 2
        long_delta  = sum(bt.LONG_PUT_DELTA_RANGE) / 2

        def format_strike(K):
            return f"{int(round(K * 1000)):08d}"

        # Fetch 0DTE weekly expirations (Fridays)
        for dt, row in df_daily_sym.iterrows():
            dt_date = dt.date()
            if dt_date.weekday() != 4:
                continue
            date_str = iso_str(dt_date)
            day_dir = sym_dir / date_str
            day_dir.mkdir(exist_ok=True)

            # Compute target strikes
            S_open = float(row['Open'])
            sigma  = float(row['VIX']) / 100.0
            try:
                K_short = bt.strike_from_delta(S_open, r, T, sigma, short_delta)
                K_long  = bt.strike_from_delta(S_open, r, T, sigma, long_delta)
                if K_short <= K_long:
                    K_short, K_long = max(K_short, K_long), min(K_short, K_long)
            except Exception as e:
                print(f"[{symbol} {date_str}] Strike computation failed: {e}")
                continue

            # Fetch and save full option chain for this expiration
            chain = client.get_option_chain(OptionChainRequest(
                underlying_symbol=symbol,
                type=ContractType.PUT,
                expiration_date=dt_date
            ))
            if not chain:
                print(f"[{symbol} {date_str}] Empty chain")
                continue
            # Convert chain dict to DataFrame and save
            records_chain = []
            for symb, snap in chain.items():
                strike = parse_strike(symb)
                bid = snap.latest_quote.bid_price if snap.latest_quote else None
                ask = snap.latest_quote.ask_price if snap.latest_quote else None
                iv = snap.implied_volatility
                records_chain.append({'symbol': symb, 'strike': strike, 'bid': bid, 'ask': ask, 'iv': iv})
            df_chain = pd.DataFrame(records_chain)
            df_chain.to_csv(day_dir / 'chain.csv', index=False)
            # Fetch minute bars for all contracts in chain
            start_dt = datetime.combine(dt_date, time(9, 30), tzinfo=ZoneInfo('America/New_York'))
            end_dt   = datetime.combine(dt_date, time(16, 0), tzinfo=ZoneInfo('America/New_York'))
            try:
                bars = client.get_option_bars(OptionBarsRequest(
                    symbol_or_symbols=list(chain.keys()),
                    start=start_dt,
                    end=end_dt,
                    timeframe=TimeFrame.Minute
                ))
            except Exception as e:
                print(f"[{symbol} {date_str}] Error fetching bars: {e}")
                continue
            # Write out legs
            for leg, bar_list in bars.data.items():
                df_bars = pd.DataFrame([
                    {'t': b.timestamp, 'o': b.open, 'h': b.high, 'l': b.low, 'c': b.close, 'v': b.volume}
                    for b in bar_list
                ])
                df_bars.to_csv(day_dir / f"{leg}.csv", index=False)
            print(f"[{symbol} {date_str}] Saved chain and bars")

    print("Done downloading backtest data.")

    print("Done downloading backtest data.")


if __name__ == '__main__':
    main()
