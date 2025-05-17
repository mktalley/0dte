#!/usr/bin/env python3
"""
paper_trade_short_strangle.py

Daily paper-trade of a 0DTE SPY short strangle (put credit spread + call credit spread) on expiration Fridays.
Requires:
  - .env with APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
  - yfinance, scipy, alpaca-py, python-dotenv
"""
import os
import sys
from datetime import date
from zoneinfo import ZoneInfo
import csv

import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv

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
RISK_FREE_RATE         = 0.01  # assume 1% annual

VOL_MAP = { 'SPY': '^VIX' }


def bs_d1(S, K, r, T, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*(T**0.5))


def call_delta(S, K, r, T, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*(T**0.5))
    return norm.cdf(d1)


def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*(T**0.5))
    return norm.cdf(d1) - 1


def strike_from_delta(func_delta, S, r, T, sigma, target_delta, bracket=None):
    # Solve for strike K such that func_delta(S,K,...) == target_delta
    f = lambda K: func_delta(S, K, r, T, sigma) - target_delta
    a, b = bracket or (1e-6, S*2)
    return brentq(f, a, b)


def main():
    load_dotenv()
    # today's date
    today = date.today()

    # allow any underlying via CLI
    import argparse
    parser = argparse.ArgumentParser(description="0DTE short strangle paper-trade")
    parser.add_argument('--symbol', default='SPY', help='Underlying symbol')
    args = parser.parse_args()
    symbol = args.symbol

    # fetch 0DTE chain only if expiration = today exists (supports daily and weekly expiries)
    # will skip if no chain available for this symbol


    # Alpaca clients
    API_KEY = os.getenv('APCA_API_KEY_ID')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY')
    BASE_URL = os.getenv('APCA_API_BASE_URL')
    if not all((API_KEY, API_SECRET, BASE_URL)):
        print("Missing Alpaca credentials in .env. Exiting.")
        sys.exit(1)

    data_client = OptionHistoricalDataClient()
    trade_client = TradingClient(API_KEY, API_SECRET, paper=True, base_url=BASE_URL)

    symbol = 'SPY'
    # Get today's open
    df = yf.download(symbol, start=today, end=today, progress=False)
    if df.empty:
        print(f"No market data for {symbol} on {today}")
        sys.exit(1)
    S_open = float(df.iloc[0]['Open'])

    # Get VIX
    vix_sym = VOL_MAP.get(symbol, '^VIX')
    vix_df = yf.download(vix_sym, start=today, end=today, progress=False)
    sigma = float(vix_df.iloc[0]['Close']) / 100.0
    T = 1/252

    # Compute target strikes
    spd_short = sum(SHORT_PUT_DELTA_RANGE)/2
    spd_long  = sum(LONG_PUT_DELTA_RANGE)/2
    scd_short = sum(SHORT_CALL_DELTA_RANGE)/2
    scd_long  = sum(LONG_CALL_DELTA_RANGE)/2

    K_sp_short = strike_from_delta(put_delta, S_open, RISK_FREE_RATE, T, sigma, spd_short)
    K_sp_long  = strike_from_delta(put_delta, S_open, RISK_FREE_RATE, T, sigma, spd_long)
    K_sc_short = strike_from_delta(call_delta, S_open, RISK_FREE_RATE, T, sigma, scd_short)
    K_sc_long  = strike_from_delta(call_delta, S_open, RISK_FREE_RATE, T, sigma, scd_long)

    # Fetch option chains
    exp_date = today
    put_chain  = data_client.get_option_chain(OptionChainRequest(
        underlying_symbol=symbol, expiration_date=exp_date, type=ContractType.PUT
    ))
    call_chain = data_client.get_option_chain(OptionChainRequest(
        underlying_symbol=symbol, expiration_date=exp_date, type=ContractType.CALL
    ))
    if not put_chain or not call_chain:
        print(f"No 0DTE expiration chain for {symbol} on {today}. Exiting.")
        sys.exit(0)

    # pick nearest strikes
    def best_opt(chain, K):
        return min(chain, key=lambda o: abs(o.strike_price - K))

    put_short_opt  = best_opt(put_chain, K_sp_short)
    put_long_opt   = best_opt(put_chain, K_sp_long)
    call_short_opt = best_opt(call_chain, K_sc_short)
    call_long_opt  = best_opt(call_chain, K_sc_long)

    legs = [
        OptionLegRequest(symbol=put_short_opt.symbol,  ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.OPEN),
        OptionLegRequest(symbol=put_long_opt.symbol,   ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.OPEN),
        OptionLegRequest(symbol=call_short_opt.symbol, ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.OPEN),
        OptionLegRequest(symbol=call_long_opt.symbol,  ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.OPEN),
    ]

    order = MarketOrderRequest(
        qty=1,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        type=OrderType.MARKET,
        legs=legs
    )

    try:
        resp = trade_client.submit_order(order)
        print("Submitted short strangle order, ID:", resp.id)
        # Log trade
        log_file = 'trade_log.csv'
        header = ['date','symbol','order_id',
                  'put_short','put_long','call_short','call_long']
        row = [today.isoformat(), symbol, resp.id,
               put_short_opt.symbol, put_long_opt.symbol,
               call_short_opt.symbol, call_long_opt.symbol]
        write_header = not os.path.exists(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print("Error submitting order:", e)
        sys.exit(1)


if __name__ == '__main__':
    # ensure numpy import for bs functions
    import numpy as np
    main()
