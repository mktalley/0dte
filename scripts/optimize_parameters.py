#!/usr/bin/env python3
"""
optimize_parameters.py

Grid search for 0DTE put credit spread parameters (OI threshold, delta targets, min credit) over a date range.
Usage:
    python scripts/optimize_parameters.py SPY IWM QQQ --start YYYY-MM-DD --end YYYY-MM-DD --output results.csv
"""
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# === Parameter Grids ===
OI_THRESHOLDS   = [50, 100, 200, 500]
SHORT_DELTAS    = [0.25, 0.30, 0.35, 0.40, 0.45]
LONG_DELTAS     = [0.05, 0.10, 0.15, 0.20]
MIN_CREDIT_PCTS = [0.10, 0.15, 0.20, 0.25, 0.30]

# Constants
RISK_FREE_RATE = 0.01
T_EXPIRY       = 1 / 252
CONTRACT_SIZE  = 100

# Black-Scholes helpers

def bs_d1(S, K, r, T, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return 0 if S >= K else -1
    d1 = bs_d1(S, K, r, T, sigma)
    return norm.cdf(d1) - 1


def run_backtest(symbol, start_date, end_date):
    # Download daily data
    df_spy = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
    if isinstance(df_spy.columns, pd.MultiIndex):
        df_spy.columns = df_spy.columns.droplevel(level=1)
    df_vix = yf.download('^VIX', start=start_date, end=end_date + timedelta(days=1), progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(level=1)

    # Merge open, close, and VIX
    data = (
        df_spy[['Open', 'Close']]
        .join(df_vix[['Close']].rename(columns={'Close': 'VIX'}), how='inner')
        .dropna()
    )

        ticker = yf.Ticker(symbol)
    exp_dates = set(ticker.options)
    # Pre-fetch option chains (puts) per date to avoid repeated API calls
    chain_data = {}
    for dt in data.index:
        date_str = dt.date().isoformat()
        if date_str not in exp_dates:
            continue
        try:
            puts = ticker.option_chain(date_str).puts
            puts = puts.copy()
            puts['mid'] = (puts['bid'] + puts['ask']) / 2
            chain_data[date_str] = puts
        except Exception:
            continue
    records = []

    for oi in OI_THRESHOLDS:
        for sd in SHORT_DELTAS:
            for ld in LONG_DELTAS:
                if ld >= sd:
                    continue
                for mc in MIN_CREDIT_PCTS:
                    pnls = []
                    wins = 0
                    trades = 0

                    for dt, row in data.iterrows():
                        trade_date = dt.date()
                        date_str = trade_date.isoformat()
                        if date_str not in exp_dates:
                            continue

                        S_open = row['Open']
                        S_close = row['Close']
                        sigma = row['VIX'] / 100.0

                        try:
                            chain = ticker.option_chain(date_str).puts
                        except Exception:
                            continue

                        # Filter by OI
                        chain = chain[chain['openInterest'] >= oi]
                        if chain.empty:
                            continue

                        chain = chain.copy()
                        chain['mid'] = (chain['bid'] + chain['ask']) / 2
                        chain['delta'] = chain.apply(
                            lambda r: put_delta(r['mid'], r['strike'], RISK_FREE_RATE, T_EXPIRY, sigma), axis=1
                        )

                        chain['sd_diff'] = (chain['delta'] - (-sd)).abs()
                        chain['ld_diff'] = (chain['delta'] - (-ld)).abs()
                        try:
                            short_row = chain.loc[chain['sd_diff'].idxmin()]
                            long_row = chain.loc[chain['ld_diff'].idxmin()]
                        except Exception:
                            continue

                        K_s = short_row['strike']
                        K_l = long_row['strike']
                        if K_l >= K_s:
                            continue

                        credit = short_row['mid'] - long_row['mid']
                        width = abs(K_s - K_l)
                        if credit < mc * width:
                            continue

                        exit_cost = (max(K_s - S_close, 0) - max(K_l - S_close, 0))
                        pnl = (credit - exit_cost) * CONTRACT_SIZE

                        pnls.append(pnl)
                        trades += 1
                        if pnl > 0:
                            wins += 1

                    total_pnl = sum(pnls)
                    win_rate = wins / trades * 100 if trades > 0 else 0.0

                    records.append({
                        'symbol': symbol,
                        'OI_THRESHOLD': oi,
                        'SHORT_DELTA': sd,
                        'LONG_DELTA': ld,
                        'MIN_CREDIT_PCT': mc,
                        'TRADES': trades,
                        'TOTAL_PNL': total_pnl,
                        'WIN_RATE': win_rate,
                    })
    return records


def main():
    parser = argparse.ArgumentParser(description="Optimize 0DTE put-credit spread parameters")
    parser.add_argument('symbols', nargs='+', help='Ticker symbols to optimize')
    parser.add_argument('--start', type=lambda s: date.fromisoformat(s),
                        default=date.today() - timedelta(days=365), help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=lambda s: date.fromisoformat(s),
                        default=date.today(), help='End date YYYY-MM-DD')
    parser.add_argument('--output', default='optimize_parameters_results.csv',
                        help='Output CSV file path')
    args = parser.parse_args()

    all_records = []
    for sym in args.symbols:
        print(f"Running backtest for {sym} from {args.start} to {args.end}...")
        recs = run_backtest(sym, args.start, args.end)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    for sym in df['symbol'].unique():
        df_sym = df[df['symbol'] == sym]
        if df_sym.empty:
            continue
        best = df_sym.sort_values('TOTAL_PNL', ascending=False).iloc[0]
        print(f"Best parameters for {sym}: {best.to_dict()}")

if __name__ == '__main__':
    main()
