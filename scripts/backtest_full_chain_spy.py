#!/usr/bin/env python3
"""
backtest_full_chain_spy.py

Full-chain scan backtest for SPY 0DTE put credit spreads over a date range.
Scans the full option chain at open, filters by OI, delta bands, credit %, and max risk;
computes PnL at expiration (intrinsic payoff).

Usage:
    pip install pandas numpy scipy yfinance
    python backtest_full_chain_spy.py \
        --symbol SPY \
        --start YYYY-MM-DD \
        --end YYYY-MM-DD \
        --oi-threshold 100 \
        --min-credit-pct 0.15 \
        --strike-range 0.30 \
        --short-delta -0.6 -0.4 \
        --long-delta -0.4 -0.2 \
        --risk-caps 100 200 300 1000
"""
import argparse
from datetime import date, timedelta, datetime, time
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import os

# Black-Scholes put delta (midpoint pricing)
def calculate_iv(option_price, S, K, T, r, option_type):
    intrinsic = max(0, (S - K) if option_type == 'call' else (K - S))
    if option_price <= intrinsic + 1e-8:
        return 0.0
    def f(sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - option_price
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1) - option_price
    try:
        return brentq(f, 1e-6, 5.0)
    except Exception:
        return None

def put_delta(S, K, r, T, sigma):
    # standard BS put delta
    if T <= 0:
        return 0 if S >= K else -1
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1


def main():
    parser = argparse.ArgumentParser(description="Full-chain SPY 0DTE put credit spread backtest")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--oi-threshold", type=int, default=100, help="Minimum open interest")
    parser.add_argument("--min-credit-pct", type=float, default=0.15, help="Min credit fraction of width")
    parser.add_argument("--strike-range", type=float, default=0.30, help="Strike range fraction around spot")
    parser.add_argument("--short-delta", nargs=2, type=float, metavar=("LOW","HIGH"), default=[-0.6,-0.4],
                        help="Short put delta range (e.g. -0.6 -0.4)")
    parser.add_argument("--long-delta", nargs=2, type=float, metavar=("LOW","HIGH"), default=[-0.4,-0.2],
                        help="Long put delta range (e.g. -0.4 -0.2)")
    parser.add_argument("--risk-caps", nargs='+', type=float, default=[300.0], help="List of max per-contract risk in dollars")
    args = parser.parse_args()

    sym = args.symbol.upper()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    oi_thr = args.oi_threshold
    min_credit_pct = args.min_credit_pct
    strike_range = args.strike_range
    short_min, short_max = args.short_delta
    long_min, long_max = args.long_delta
    risk_caps = args.risk_caps

    # Prepare SPY daily history
    ticker = yf.Ticker(sym)
    hist = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat(), interval='1d', auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No price history for {sym}")
    hist['date'] = hist.index.date
    hist = hist.set_index('date')

    # Available expirations
    expirations = set(ticker.options)

    results_all = []
    for cap in risk_caps:
        records = []
        for single in pd.date_range(start_date, end_date, freq='B'):
            d = single.date()
            ds = d.isoformat()
            if ds not in expirations:
                continue
            # spot prices
            row = hist.loc.get(d)
            if row is None:
                continue
            S_open = float(row['Open'])
            S_close = float(row['Close'])
            # implied sigma proxy
            # approximate with VIX close; use VIX daily if available
            # here use fixed sigma from last VIX close (could be added)
            # for delta calculation, use T = 1/252
            T = 1/252
            r = 0.01

            # fetch option chain
            chain = ticker.option_chain(ds).puts
            # filter by strike range and OI
            lo = S_open * (1 - strike_range)
            hi = S_open * (1 + strike_range)
            df = chain[(chain['strike'] >= lo) & (chain['strike'] <= hi) &
                       (chain['openInterest'] >= oi_thr)].copy()
            if df.empty:
                continue
            df['mid'] = (df['bid'] + df['ask']) / 2
            # compute delta per row
            # approximate sigma by mid VIX percent ratio if available else fallback to intrinsic
            # using df row price, approximate sigma = 0.2
            sigma = 0.2
            cand_short = df.apply(lambda r0: (r0['strike'], r0['mid'],
                calculate_iv(r0['mid'], S_open, r0['strike'], T, r, 'put') and
                calculate_iv(r0['mid'], S_open, r0['strike'], T, r, 'put') and 
                None), axis=1)  # placeholder
            # to simplify, compute delta via BS put delta using approx sigma from VIX close if loaded
            # currently skip delta filter; pick strikes nearest target deltas theoretically
            # compute theoretical strike targets
            def bs_put_delta(op_price, K):
                iv = calculate_iv(op_price, S_open, K, T, r, 'put')
                return put_delta(S_open, K, r, T, iv) if iv is not None else None
            df['delta'] = df.apply(lambda rr: bs_put_delta(rr['mid'], rr['strike']), axis=1)
            shorts = df[(df['delta'] >= short_min) & (df['delta'] <= short_max)]
            longs = df[(df['delta'] >= long_min) & (df['delta'] <= long_max)]
            if shorts.empty or longs.empty:
                continue
            # build pairs
            best = None
            for _, rs in shorts.iterrows():
                for _, rl in longs.iterrows():
                    K_s, mid_s, del_s = rs['strike'], rs['mid'], rs['delta']
                    K_l, mid_l, del_l = rl['strike'], rl['mid'], rl['delta']
                    if K_s <= K_l:
                        continue
                    width = K_s - K_l
                    credit = mid_s - mid_l
                    if credit < min_credit_pct * width:
                        continue
                    if width * 100 > cap:
                        continue
                    if best is None or credit > best['credit']:
                        best = dict(date=d, cap=cap,
                                    K_short=K_s, K_long=K_l,
                                    mid_short=mid_s, mid_long=mid_l,
                                    delta_short=del_s, delta_long=del_l,
                                    width=width, credit=credit)
            if best is None:
                continue
            payoff = max(best['K_short'] - S_close, 0) - max(best['K_long'] - S_close, 0)
            pnl_share = best['credit'] - payoff
            pnl = pnl_share * 100
            win = pnl > 0
            best.update(dict(S_open=S_open, S_close=S_close,
                             payoff=payoff, pnl=pnl, win=win))
            records.append(best)
        # summarize
        df_rec = pd.DataFrame(records)
        total = len(df_rec)
        wins = int(df_rec['win'].sum()) if total>0 else 0
        losses = total - wins
        total_pnl = df_rec['pnl'].sum() if total>0 else 0.0
        win_rate = wins/total*100 if total>0 else 0.0
        print(f"\n=== Risk cap ${cap}: Trades={total}, Wins={wins}, Losses={losses}, WinRate={win_rate:.1f}%, PnL=${total_pnl:.2f} ===")
        # save CSV
        out_dir = f"backtests/{date.today().isoformat()}_{sym}_fullchain"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(records).to_csv(f"{out_dir}/{sym}_fullchain_cap{int(cap)}.csv", index=False)
    print("Done.")

if __name__ == '__main__':
    main()
