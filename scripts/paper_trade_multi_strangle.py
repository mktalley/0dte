#!/usr/bin/env python3
"""
paper_trade_multi_strangle.py

Daily-dated and weekly-dated 0DTE short strangle paper-trade.
- Trades a short strangle (put credit spread + call credit spread) for each symbol provided
  if that symbol has options expiring today.
- Defaults to SPY (weekly expirations), SPX, and XSP (daily expirations).

Requirements:
  - .env with APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
  - yfinance, scipy, alpaca-py, python-dotenv
Usage:
  ./paper_trade_multi_strangle.py [--symbols SPY,SPX,XSP] [--dry-run]
"""
import os
import sys
import csv
from datetime import date

env_vars = ('APCA_API_KEY_ID', 'APCA_API_SECRET_KEY', 'APCA_API_BASE_URL')
for var in env_vars:
    if not os.getenv(var):
        # defer dotenv load in main
        from dotenv import load_dotenv
        load_dotenv()
        break

import argparse
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scripts.fetch_spy_options import _parse_strike

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
from alpaca.trading.enums import (
    ContractType, TimeInForce, OrderClass,
    OrderSide, PositionIntent, OrderType
)

# Strategy parameters
SHORT_PUT_DELTA_RANGE = (-0.45, -0.35)
LONG_PUT_DELTA_RANGE  = (-0.25, -0.15)
SHORT_CALL_DELTA_RANGE = ( 0.35,  0.45)
LONG_CALL_DELTA_RANGE  = ( 0.15,  0.25)
RISK_FREE_RATE         = 0.01  # 1% annual

# Map underlying to vol index symbol
VOL_MAP = {
    'SPY': '^VIX',  # weekly expirations
    'SPX': '^VIX',  # daily expirations
}


def call_delta(S, K, r, T, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)


def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1


def strike_from_delta(delta_func, S, r, T, sigma, target_delta):
    f = lambda K: delta_func(S, K, r, T, sigma) - target_delta
    # bracket search between near-zero and twice spot
    return brentq(f, 1e-6, S*2)


def trade_strangle(symbol, today, data_client, trade_client, dry_run=False):
    # get underlying open price
    df = yf.download(symbol, start=today, end=today, progress=False)
    if df.empty:
        print(f"[{symbol}] no market data on {today}")
        return
    S_open = float(df.iloc[0]['Open'])

    # get vol index
    vmap = VOL_MAP.get(symbol, '^VIX')
    vdf = yf.download(vmap, start=today, end=today, progress=False)
    if vdf.empty:
        print(f"[{symbol}] no vol data ({vmap}) on {today}")
        return
    sigma = float(vdf.iloc[0]['Close']) / 100.0

    # time to expiration in years (0DTE)
    T = 1/252

    # compute target deltas
    sp_sd = sum(SHORT_PUT_DELTA_RANGE)/2
    sp_ld = sum(LONG_PUT_DELTA_RANGE)/2
    sc_sd = sum(SHORT_CALL_DELTA_RANGE)/2
    sc_ld = sum(LONG_CALL_DELTA_RANGE)/2

    # solve strikes
    K_ps = strike_from_delta(put_delta, S_open, RISK_FREE_RATE, T, sigma, sp_sd)
    K_pl = strike_from_delta(put_delta, S_open, RISK_FREE_RATE, T, sigma, sp_ld)
    K_cs = strike_from_delta(call_delta, S_open, RISK_FREE_RATE, T, sigma, sc_sd)
    K_cl = strike_from_delta(call_delta, S_open, RISK_FREE_RATE, T, sigma, sc_ld)

    # fetch option chains for expiration today
    exp = today
    put_chain = data_client.get_option_chain(
        OptionChainRequest(underlying_symbol=symbol, expiration_date=exp, type=ContractType.PUT)
    )
    call_chain = data_client.get_option_chain(
        OptionChainRequest(underlying_symbol=symbol, expiration_date=exp, type=ContractType.CALL)
    )
    if not put_chain or not call_chain:
        print(f"[{symbol}] no 0DTE chain on {today}")
        return

    # pick nearest strikes
    pick = lambda chain, K: min(chain, key=lambda o: abs(_parse_strike(o.symbol) - K))
    ps_opt = pick(put_chain, K_ps)
    pl_opt = pick(put_chain, K_pl)
    cs_opt = pick(call_chain, K_cs)
    cl_opt = pick(call_chain, K_cl)

    # build legs
    legs = [
        OptionLegRequest(symbol=ps_opt.symbol, ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.OPEN),
        OptionLegRequest(symbol=pl_opt.symbol, ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.OPEN),
        OptionLegRequest(symbol=cs_opt.symbol, ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.OPEN),
        OptionLegRequest(symbol=cl_opt.symbol, ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.OPEN),
    ]

    order = MarketOrderRequest(
        qty=1,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        type=OrderType.MARKET,
        legs=legs
    )

    if dry_run:
        print(f"[DRY RUN] {symbol}: would submit order with legs:")
        print(f"    Sell Put: {ps_opt.symbol}")
        print(f"    Buy  Put: {pl_opt.symbol}")
        print(f"    Sell Call: {cs_opt.symbol}")
        print(f"    Buy  Call: {cl_opt.symbol}")
        return
    try:
        resp = trade_client.submit_order(order)
        print(f"[{symbol}] submitted short strangle, order ID: {resp.id}")
        # log to CSV
        logf = 'trade_log.csv'
        hdr = ['date','symbol','order_id','ps','pl','cs','cl']
        row = [today.isoformat(), symbol, resp.id,
               ps_opt.symbol, pl_opt.symbol, cs_opt.symbol, cl_opt.symbol]
        write_hdr = not os.path.exists(logf)
        with open(logf, 'a', newline='') as f:
            w = csv.writer(f)
            if write_hdr:
                w.writerow(hdr)
            w.writerow(row)
    except Exception as e:
        print(f"[{symbol}] error submitting order: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-symbol 0DTE short strangle")
    parser.add_argument('--symbols', default='SPY,SPX,XSP',  # SPY=weekly, SPX/XSP=daily
                        help='Comma-separated symbols with daily (SPX) or weekly (SPY) 0DTE options')
    parser.add_argument('--dry-run', action='store_true', help='Dry run: show legs without submitting orders')
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]

    # Alpaca clients
    API_KEY = os.getenv('APCA_API_KEY_ID')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY')
    BASE_URL = os.getenv('APCA_API_BASE_URL')
    if not all((API_KEY, API_SECRET, BASE_URL)):
        print("Missing Alpaca credentials. Set APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL in .env or env.")
        sys.exit(1)

    data_client = OptionHistoricalDataClient()
    trade_client = TradingClient(API_KEY, API_SECRET, paper=True, base_url=BASE_URL)

    today = date.today()
    for sym in symbols:
        trade_strangle(sym, today, data_client, trade_client)

if __name__ == '__main__':
    main()
