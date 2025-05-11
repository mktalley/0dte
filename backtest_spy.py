#!/usr/bin/env python3
"""
Backtest 0DTE SPY put credit spread for the past year using VIX as implied vol proxy.
Requirements: pandas, numpy, scipy, yfinance.
Usage:
    pip install pandas numpy scipy yfinance
    python backtest_spy.py
"""

import sys
try:
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    from scipy.optimize import brentq
    import yfinance as yf
except ImportError as e:
    print(f"Missing dependency: {e.name}. Please install required packages: pandas, numpy, scipy, yfinance")
    sys.exit(1)

from datetime import date, timedelta


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


def backtest_spy(start_date, end_date, r=0.01,
                 short_delta=-0.40, long_delta=-0.20,
                 min_credit=0.25,
                 initial_capital=100000, daily_target_pct=0.01):
    # Download SPY and VIX data
    spy = yf.download("SPY", start=start_date, end=end_date)
    # Flatten SPY MultiIndex columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(level=1)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    # Flatten VIX MultiIndex columns if present
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(level=1)

    # Combine SPY and VIX into single DataFrame
    vix_close = vix[['Close']].rename(columns={'Close': 'VIX'})
    data = spy[['Open', 'Close']].join(vix_close, how='inner').dropna()

    # Time to expiration = 1 trading day ~ 1/252 year
    T = 1 / 252
    # Contract size (shares per option contract)
    contract_size = 100
    # Initialize capital and daily target
    capital = initial_capital
    daily_target = initial_capital * daily_target_pct
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
        if credit_share < min_credit:
            continue
        S_close = row['Close']
        payoff_share = max(K_short - S_close, 0) - max(K_long - S_close, 0)
        pnl_share = credit_share - payoff_share
        # Determine per-contract credit and payoff
        credit_per_contract = credit_share * contract_size
        payoff_per_contract = payoff_share * contract_size
        # Compute daily target based on current capital
        daily_target = capital * daily_target_pct
        # Determine number of contracts
        n_contracts = max(1, int(daily_target // credit_per_contract))
        # Compute total values
        credit = credit_per_contract * n_contracts
        payoff = payoff_per_contract * n_contracts
        pnl = credit - payoff
        # Update capital
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
        })
    results = pd.DataFrame(records)
    total = len(results)
    wins = int(results['win'].sum()) if total > 0 else 0
    losses = total - wins
    total_pnl = results['pnl'].sum() if total > 0 else 0.0
    win_rate = wins / total * 100 if total > 0 else 0.0

    print("===== Backtest SPY 0DTE Put Credit Spread =====")
    print(f"Period: {start_date} to {end_date}")
    print(f"Trades: {total}")
    print(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    # Detailed results
    if total > 0:
        print(results[['date', 'K_short', 'K_long', 'credit', 'pnl']].to_string(index=False))
        results.to_csv("backtest_spy_results.csv", index=False)
        print("Results saved to backtest_spy_results.csv")


if __name__ == "__main__":
    end = date.today()
    start = end - timedelta(days=365)
    backtest_spy(start, end)
