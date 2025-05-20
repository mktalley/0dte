#!/usr/bin/env python3
"""
backtest_spy_alpaca.py

Intraday minute-level backtest of 0DTE SPY put credit spreads using Alpaca historical data.
Usage:
    pip install pandas numpy alpaca-trade-api scipy
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"
    python backtest_spy_alpaca.py --start YYYY-MM-DD --end YYYY-MM-DD --output results.csv
"""
import os
import sys
# allow importing src as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from scipy.stats import norm
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import ContractType

# Strategy parameters
from src.main import (
    RISK_PER_TRADE_PERCENTAGE,
    STOP_LOSS_PERCENTAGE,
    PROFIT_TAKE_PERCENTAGE,
    SHORT_PUT_DELTA_RANGE,
    LONG_PUT_DELTA_RANGE,
    MIN_CREDIT_PERCENTAGE,
)

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
if not API_KEY or not API_SECRET:
    print("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment.")
    sys.exit(1)

NY = ZoneInfo("America/New_York")
CONTRACT_SIZE = 100

# Fill cost model
COMMISSION_PER_CONTRACT = 0.65  # per contract per leg
SLIPPAGE_PER_CONTRACT = 0.02    # per contract per leg


def compute_pnl(entry_short, entry_long, mid_short, mid_long):
    # credit collected (per contract)
    credit = (entry_short - entry_long) * CONTRACT_SIZE
    # cost to close (per contract)
    cost = (mid_short - mid_long) * CONTRACT_SIZE
    gross_pnl = credit - cost
    # subtract commission and slippage (two legs per contract)
    total_commission = COMMISSION_PER_CONTRACT * 2
    total_slippage = SLIPPAGE_PER_CONTRACT * 2
    net_pnl = gross_pnl - total_commission - total_slippage
    return net_pnl, credit


def backtest_spy(start_date: date, end_date: date, output_csv: str):
    # Alpaca clients
    stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    option_client = OptionHistoricalDataClient(API_KEY, API_SECRET)

    records = []
    current = start_date
    while current <= end_date:
        # skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        session_start = datetime.combine(current, time(9, 30), tzinfo=NY)
        session_end = datetime.combine(current, time(16, 0), tzinfo=NY)

        try:
            full_chain = option_client.get_option_chain(OptionChainRequest(
                underlying_symbol="SPY",
                type=ContractType.PUT,
            ))
        except Exception as e:
            print(f"{current} - Chain fetch error: {e}")
            current += timedelta(days=1)
            continue
                # determine weekly expiration (Friday) and filter chain
        days_until_expiry = (4 - current.weekday()) % 7
        expiry_date = current + timedelta(days=days_until_expiry)
        expiry_code = expiry_date.strftime("%y%m%d")
        snapshots = {sym: snap for sym, snap in full_chain.items() if sym[3:9] == expiry_code}
        if not snapshots:
            print(f"{current} - No options expiring on {expiry_date} (expiry code {expiry_code})")
            current += timedelta(days=1)
            continue

        # 2. filter by delta and open interest
        shorts = []
        longs = []
        for snap in snapshots.values():
            if not hasattr(snap, 'greeks') or snap.greeks is None:
                continue
            delta = snap.greeks.delta
            oi = getattr(snap, 'open_interest', 0)
            if SHORT_PUT_DELTA_RANGE[0] <= delta <= SHORT_PUT_DELTA_RANGE[1]:
                shorts.append(snap)
            if LONG_PUT_DELTA_RANGE[0] <= delta <= LONG_PUT_DELTA_RANGE[1]:
                longs.append(snap)
        if not shorts or not longs:
            print(f"{current} - No valid leg candidates (shorts: {len(shorts)}, longs: {len(longs)})")
            current += timedelta(days=1)
            continue
        # choose highest open interest
        short_snap = max(shorts, key=lambda s: s.open_interest)
        long_snap = max(longs, key=lambda s: s.open_interest)
        width = abs(short_snap.strike_price - long_snap.strike_price)
        # 3. fetch minute bars for each leg
        req_bars_short = OptionBarsRequest(
            symbol=snap.symbol if (snap := short_snap) else None,
            timeframe=TimeFrame.Minute,
            start=session_start.isoformat(),
            end=session_end.isoformat(),
        )
        req_bars_long = OptionBarsRequest(
            symbol=snap.symbol if (snap := long_snap) else None,
            timeframe=TimeFrame.Minute,
            start=session_start.isoformat(),
            end=session_end.isoformat(),
        )
        try:
            bars_short = option_client.get_option_bars(req_bars_short).df
            bars_long = option_client.get_option_bars(req_bars_long).df
        except Exception as e:
            print(f"{current} - Bar fetch error: {e}")
            current += timedelta(days=1)
            continue

        if bars_short.empty or bars_long.empty:
            print(f"{current} - Missing bars for legs")
            current += timedelta(days=1)
            continue

        # entry at first open
        entry_short = bars_short.iloc[0].o
        entry_long  = bars_long.iloc[0].o
        # simulate intraday
        exit_reason = 'end_of_day'
        for i in range(len(bars_short)):
            mid_short = bars_short.iloc[i].c
            mid_long  = bars_long.iloc[i].c
            pnl, credit = compute_pnl(entry_short, entry_long, mid_short, mid_long)
            # thresholds in $ per contract
            max_loss = width * CONTRACT_SIZE * STOP_LOSS_PERCENTAGE
            target = width * CONTRACT_SIZE * PROFIT_TAKE_PERCENTAGE
            if pnl <= -max_loss:
                exit_reason = 'stop_loss'
            elif pnl >= target:
                exit_reason = 'profit_take'
            if exit_reason != 'end_of_day':
                exit_time = bars_short.index[i]
                break
        else:
            # exit at last minute close
            mid_short = bars_short.iloc[-1].c
            mid_long  = bars_long.iloc[-1].c
            pnl, credit = compute_pnl(entry_short, entry_long, mid_short, mid_long)
            exit_time = bars_short.index[-1]

        records.append({
            'date': current,
            'short_strike': short_snap.strike_price,
            'long_strike': long_snap.strike_price,
            'entry_time': bars_short.index[0],
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'credit': credit,
            'pnl': pnl,
        })

        current += timedelta(days=1)

    # results
    # enforce consistent columns even if no trades executed
    cols = ['date','short_strike','long_strike','entry_time','exit_time','exit_reason','credit','pnl']
    df = pd.DataFrame(records, columns=cols)
    df.to_csv(output_csv, index=False)
    total = df['pnl'].sum()
    wins = (df['pnl'] > 0).sum()
    total_trades = len(df)
    win_rate = wins / total_trades * 100 if total_trades else 0.0
    print(f"Trades: {total_trades}, Wins: {wins}, Win rate: {win_rate:.1f}%, Total P&L: ${total:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    p.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    p.add_argument('--output', default='backtest_spy_alpaca.csv', help='Output CSV file')
    args = p.parse_args()
    start = datetime.fromisoformat(args.start).date()
    end   = datetime.fromisoformat(args.end).date()
    backtest_spy(start, end, args.output)
