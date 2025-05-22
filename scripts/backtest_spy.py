#!/usr/bin/env python3
"""
Backtest 0DTE put credit spreads for given ticker symbols over a date range using VIX as implied vol proxy.
Requirements: pandas, numpy, scipy, yfinance.
Usage:
    pip install pandas numpy scipy yfinance
    python backtest_spy.py SPY
    python backtest_spy.py SPY QQQ --start YYYY-MM-DD --end YYYY-MM-DD
"""

import sys
try:
    import numpy as np
    import pandas as pd
    import os
    from dotenv import load_dotenv
    from zoneinfo import ZoneInfo

    # Alpaca historical option data client
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest, OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from scipy.stats import norm
    from scipy.optimize import brentq
    import yfinance as yf
except ImportError as e:
    print(f"Missing dependency: {e.name}. Please install required packages: pandas, numpy, scipy, yfinance")
    sys.exit(1)

from datetime import date, timedelta
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Utility to parse strike price from symbol

def parse_strike(sym: str) -> float:
    # Last 8 characters encode strike price multiplied by 1000
    return int(sym[-8:]) / 1000.0


def real_backtest_symbol(symbol, start_date, end_date, output_dir, data_dir):
    """
    Backtest using pre-downloaded SPY 0DTE option bars from data_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load daily underlying data
    df_daily = pd.read_csv(Path(data_dir) / 'underlying.csv', skiprows=[1,2], index_col=0, parse_dates=True)  # skip ticker and date header rows
    r = RISK_FREE_RATE
    initial_capital = CAPITAL_POOL
    contract_size = 100
    records = []
    capital = initial_capital
    for idx, row in df_daily.iterrows():
        dt_date = idx.date()
        date_str = dt_date.isoformat()
        day_dir = Path(data_dir) / date_str
        if not day_dir.exists():
            continue
        # Load minute bar CSVs
        bars = {}
        for f in day_dir.glob('*.csv'):
            sym = f.stem
            dfb = pd.read_csv(f, parse_dates=['t'])
            bars[sym] = dfb
        if not bars:
            continue
        # Select short and long by strike
        strikes = {sym: parse_strike(sym) for sym in bars}
        symbol_short = max(strikes, key=strikes.get)
        symbol_long = min(strikes, key=strikes.get)
        df_short = bars[symbol_short]
        df_long = bars[symbol_long]
        # Entry and exit prices
        entry_short = df_short.iloc[0]['o']
        entry_long = df_long.iloc[0]['o']
        exit_short = df_short.iloc[-1]['c']
        exit_long = df_long.iloc[-1]['c']
        # Compute credit and PnL
        credit_share = entry_short - entry_long
        credit = credit_share * contract_size
        exit_cost_share = exit_short - exit_long
        exit_cost = exit_cost_share * contract_size
        pnl = credit - exit_cost
        capital += pnl
        records.append({
            'date': dt_date,
            'K_short': strikes[symbol_short],
            'K_long': strikes[symbol_long],
            'credit': credit,
            'payoff': exit_cost,
            'pnl': pnl,
            'capital': capital,
            'win': pnl > 0,
        })
    results = pd.DataFrame(records)
    # Summary statistics
    total = len(results)
    wins = int(results['win'].sum()) if total > 0 else 0
    losses = total - wins
    total_pnl = results['pnl'].sum() if total > 0 else 0.0
    win_rate = wins / total * 100 if total > 0 else 0.0
    print(f"===== Real Data Backtest {symbol} 0DTE Put Credit Spread =====")
    print(f"Period: {start_date} to {end_date}")
    print(f"Trades: {total}")
    print(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    if total > 0:
        print(results[['date', 'K_short', 'K_long', 'credit', 'pnl']].to_string(index=False))
        csv_path = Path(output_dir) / f"{symbol}_real_results.csv"
        results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")



# Strategy configurations (align with main.py; can be overridden via env variables)
import os
SHORT_PUT_DELTA_RANGE = (-0.45, -0.35)
LONG_PUT_DELTA_RANGE = (-0.25, -0.15)
MIN_CREDIT_PERCENTAGE = 0.25  # fraction of spread width per share
MAX_RISK_PER_TRADE = 2000    # maximum per-contract risk in dollars
STOP_LOSS_PERCENTAGE = 0.5
PROFIT_TAKE_PERCENTAGE = 0.75
RISK_FREE_RATE = 0.01
CAPITAL_POOL = 100000



def bs_put_price(S, K, r, T, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_delta(S, K, r, T, sigma):
    """Black-Scholes put delta."""
    if T <= 0:
        return 0 if S >= K else -1
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def strike_from_delta(S, r, T, sigma, target_delta):
    """Solve for strike such that put delta equals target_delta."""
    f = lambda K: put_delta(S, K, r, T, sigma) - target_delta
    a = 1e-6
    b = S
    try:
        K = brentq(f, a, b)
    except ValueError:
        raise ValueError(f"Could not find strike for target delta {target_delta}")
    return K


def backtest_symbol(symbol, start_date, end_date, output_dir):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    """Backtest 0DTE SPY put credit spread from start_date to end_date."""
    # Initialize parameters from main config
    r = RISK_FREE_RATE
    initial_capital = CAPITAL_POOL
    # Target deltas as midpoints of ranges
    short_delta = sum(SHORT_PUT_DELTA_RANGE) / 2
    long_delta = sum(LONG_PUT_DELTA_RANGE) / 2
    # Download SPY and VIX data
    spy = yf.download(symbol, start=start_date, end=end_date)
    # Flatten SPY MultiIndex columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(level=1)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    # Flatten VIX MultiIndex columns if present
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(level=1)

    # Combine SPY and VIX into single DataFrame
    vix_close = vix[['Close']].rename(columns={'Close': 'VIX'})
    data = spy[['Open', 'High', 'Low', 'Close']].join(vix_close, how='inner').dropna()
    # Compute median VIX to set dynamic max-risk thresholds
    median_vix = data['VIX'].median()

    # Time to expiration = 1 trading day ~ 1/252 year
    T = 1 / 252
    # Contract size (shares per option contract)
    contract_size = 100
    # Initialize capital
    capital = initial_capital
    records = []
    for dt, row in data.iterrows():
        S_open = row['Open']
        sigma = row['VIX'] / 100.0
        # Find strikes for target deltas
        try:
            K_short = strike_from_delta(S_open, r, T, sigma, short_delta)
            K_long = strike_from_delta(S_open, r, T, sigma, long_delta)
        except Exception:
            continue
        # Ensure short strike > long strike
        if K_short <= K_long:
            K_short, K_long = max(K_short, K_long), min(K_short, K_long)
        # Entry prices
        price_short = bs_put_price(S_open, K_short, r, T, sigma)
        price_long = bs_put_price(S_open, K_long, r, T, sigma)
        # Credit and payoff per share
        credit_share = price_short - price_long
        width_share = abs(K_short - K_long)
        # Skip if max per-contract risk is too high
        if width_share * contract_size > MAX_RISK_PER_TRADE:
            continue
        # Minimum credit as fraction of spread width
        if credit_share < width_share * MIN_CREDIT_PERCENTAGE:
            continue
        high = row['High']
        low = row['Low']
        S_close = row['Close']
        # Profit-taking and stop-loss thresholds (share terms)
        exit_type = 'eod'
        profit_pct = PROFIT_TAKE_PERCENTAGE
        sl_pct = STOP_LOSS_PERCENTAGE
        # Compute exit payoff thresholds
        exit_payoff_share_profit = credit_share * (1 - profit_pct)
        threshold_profit_price = K_short - exit_payoff_share_profit
        exit_payoff_share_stop = credit_share * (1 + sl_pct)
        threshold_stop_price = K_short - exit_payoff_share_stop
        # Determine exit condition
        if low <= threshold_stop_price:
            # Stop-loss triggered
            exit_payoff_share = exit_payoff_share_stop
            exit_price = threshold_stop_price
            exit_type = 'sl'
        elif high >= threshold_profit_price:
            # Profit-take triggered
            exit_payoff_share = exit_payoff_share_profit
            exit_price = threshold_profit_price
            exit_type = 'tp'
        else:
            # Exit at close
            exit_payoff_share = max(K_short - S_close, 0) - max(K_long - S_close, 0)
            exit_price = S_close
        # Per-contract values
        credit_per_contract = credit_share * contract_size
        payoff_per_contract = exit_payoff_share * contract_size
        n_contracts = 1
        credit = credit_per_contract * n_contracts
        payoff = payoff_per_contract * n_contracts
        pnl = credit - payoff
        capital += pnl
        records.append({
            'date': dt.date(),
            'K_short': K_short,
            'K_long': K_long,
            'n_contracts': n_contracts,
            'credit': credit,
            'payoff': payoff,
            'pnl': pnl,
            'capital': capital,
            'win': pnl > 0,
            'exit_type': exit_type,
        })
    results = pd.DataFrame(records)
    total = len(results)
    wins = int(results['win'].sum()) if total > 0 else 0
    losses = total - wins
    total_pnl = results['pnl'].sum() if total > 0 else 0.0
    win_rate = wins / total * 100 if total > 0 else 0.0

    print(f"===== Backtest {symbol} 0DTE Put Credit Spread =====")
    print(f"Period: {start_date} to {end_date}")
    print(f"Trades: {total}")
    print(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    # Detailed results and outputs
    if total > 0:
        print(results[['date', 'K_short', 'K_long', 'credit', 'pnl']].to_string(index=False))
        # Save to CSV
        csv_path = output_dir / f"{symbol}_results.csv"
        results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Generate charts
        # Ensure date column is datetime
        results['date'] = pd.to_datetime(results['date'])

        chart_dir = output_dir / "charts"
        chart_dir.mkdir(exist_ok=True)

        # Equity Curve
        plt.figure(figsize=(10, 6))
        plt.plot(results['date'], results['capital'], lw=2)
        plt.title(f"{symbol} Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Capital ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_dir / "equity_curve.png")
        plt.close()

        # Drawdown Curve
        cum_max = results['capital'].cummax()
        drawdown = (results['capital'] - cum_max) / cum_max * 100
        plt.figure(figsize=(10, 6))
        plt.plot(results['date'], drawdown, lw=2, color='red')
        plt.title(f"{symbol} Drawdown (%)")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_dir / "drawdown_curve.png")
        plt.close()

        # PnL Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['pnl'], bins=50, color='skyblue', edgecolor='gray')
        plt.title(f"{symbol} PnL per Trade Distribution")
        plt.xlabel("PnL ($)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_dir / "pnl_distribution.png")
        plt.close()

        print(f"Charts saved to {chart_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtests for given symbols")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to backtest")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (default 1 year ago)", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD (default today)", default=None)
    args = parser.parse_args()

    # Parse dates
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today()
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=365)

    # Loop over symbols and run backtests
    for sym in args.symbols:
        outdir = Path("backtests") / f"{date.today().isoformat()}_{sym}"
        backtest_symbol(sym, start_date, end_date, outdir)

    # Combined results for all symbols
    if len(args.symbols) > 1:
        combined_dir = Path("backtests") / f"{date.today().isoformat()}_combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        # Load and concatenate individual results
        dfs = []
        for sym in args.symbols:
            csv_path = Path("backtests") / f"{date.today().isoformat()}_{sym}" / f"{sym}_results.csv"
            df = pd.read_csv(csv_path)
            df['symbol'] = sym
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        # Print combined table
        print("===== Combined Backtest Results =====")
        print(combined[['symbol','date','K_short','K_long','credit','pnl']].to_string(index=False))
        # Save combined CSV
        combined_csv = combined_dir / "combined_results.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"Combined results saved to {combined_csv}")
        # Combined charts
        chart_dir = combined_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        # Equity curves per symbol
        plt.figure(figsize=(10, 6))
        for sym in args.symbols:
            df = combined[combined['symbol'] == sym]
            df['date'] = pd.to_datetime(df['date'])
            plt.plot(df['date'], df['capital'], lw=2, label=sym)
        plt.title("Equity Curves")
        plt.xlabel("Date")
        plt.ylabel("Capital ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_dir / "equity_curves.png")
        plt.close()
        # PnL distribution across all symbols
        plt.figure(figsize=(10, 6))
        plt.hist(combined['pnl'], bins=50, color='skyblue', edgecolor='gray')
        plt.title("Combined PnL per Trade Distribution")
        plt.xlabel("P&L ($)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_dir / "pnl_distribution.png")
        plt.close()
        print(f"Combined charts saved to {chart_dir}")
