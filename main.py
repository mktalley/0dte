# 0DTE Bot (Fully Optimized)

import os
import csv
import time as time_module
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, date
from scipy.stats import norm
from scipy.optimize import brentq
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import smtplib
import re
from types import SimpleNamespace
# duplicate import removed (datetime already imported)

from email.mime.text import MIMEText
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderClass, OrderType, TimeInForce, AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest, OptionLegRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import OptionLatestQuoteRequest, OptionChainRequest
from alpaca.data.enums import OptionsFeed

# === CONFIGURATION ===
load_dotenv()
# PAPER mode for trading and data
PAPER = True
# Load API credentials based on mode
if PAPER:
    API_KEY = os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
else:
    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_SECRET_KEY")
# Market data credentials (live)
DATA_API_KEY = os.getenv("ALPACA_API_KEY")
DATA_API_SECRET = os.getenv("ALPACA_SECRET_KEY")
# Determine options market data feed: use INDICATIVE for paper to avoid OPRA agreement requirement
OPTIONS_FEED = OptionsFeed.INDICATIVE if PAPER else OptionsFeed.OPRA

capital_pool = 100000
max_risk_per_trade = 1000
STOP_LOSS_PERCENTAGE = 0.5
PROFIT_TAKE_PERCENTAGE = 0.5
MIN_CREDIT_PERCENTAGE = 0.25
OI_THRESHOLD = 500
SHORT_PUT_DELTA_RANGE = (-0.45, -0.35)
LONG_PUT_DELTA_RANGE = (-0.25, -0.15)
STRIKE_RANGE = 0.1
SCAN_INTERVAL = 120
risk_free_rate = 0.01
timezone = ZoneInfo("America/New_York")

# === CLIENTS ===
trade_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
option_data_client = OptionHistoricalDataClient(DATA_API_KEY, DATA_API_SECRET, use_basic_auth=False, sandbox=False)
stock_data_client = StockHistoricalDataClient(DATA_API_KEY, DATA_API_SECRET, use_basic_auth=False, sandbox=False)

# === LOG FILES ===
os.makedirs("logs", exist_ok=True)
TRADE_LOG = "logs/trade_log.csv"
OPEN_TRADES_FILE = "logs/open_trades.csv"


# Initialize log files with headers if they don't exist
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","symbol","short_strike","long_strike","credit","spread_width","take_profit_price","stop_loss_price","status"])
if not os.path.exists(OPEN_TRADES_FILE):
    with open(OPEN_TRADES_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol","short_leg","long_leg","credit","width","take_profit_price","stop_loss_price","timestamp"])

# === UTILITIES ===
def log(msg):
    print(f"[{datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def safe_divide(a, b):
    return a / b if b else 0

def calculate_iv(option_price, S, K, T, r, option_type):
    intrinsic = max(0, (S - K) if option_type == 'call' else (K - S))
    if option_price <= intrinsic + 1e-6:
        return 0.0
    def f(sigma):
        d1 = safe_divide((np.log(S / K) + (r + 0.5 * sigma**2) * T), sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - option_price
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - option_price
    try:
        return brentq(f, 1e-6, 5.0)
    except:
        return None

def calculate_delta(option_price, strike, expiry, spot, r, option_type):
    now = datetime.now(tz=timezone)
    T = max((expiry - now).total_seconds() / (365 * 24 * 3600), 1e-6)
    iv = calculate_iv(option_price, spot, strike, T, r, option_type)
    if not iv:
        return None
    d1 = safe_divide((np.log(spot / strike) + (r + 0.5 * iv**2) * T), iv * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)

def is_market_open():
    now = datetime.now(timezone)
    return now.weekday() < 5 and dt_time(9, 30) <= now.time() <= dt_time(16, 0)

def get_fallback_tickers():
    return ["SPY", "QQQ", "TSLA", "AAPL", "MSFT", "NVDA", "META", "AMZN", "AMD", "GOOG",
            "BA", "XLF", "XLK", "DIA", "IWM", "XLE", "XBI", "TSM", "GDX", "ARKK"]

def load_tickers():
    today_str = date.today().isoformat()
    filename = f"tickers_selected/tickers_selected_{today_str}.txt"
    if os.path.exists(filename):
        with open(filename) as f:
            return [line.strip() for line in f if line.strip()]
    log("⚠️ No ticker file for today. Using fallback list.")
    return get_fallback_tickers()

def get_all_underlying_prices(tickers):
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=tickers)
        res = stock_data_client.get_stock_latest_trade(req)
        return {symbol: res[symbol].price for symbol in tickers if symbol in res}
    except Exception as e:
        log(f"❌ Failed to get prices: {e}")
        return {}

def get_0dte_options(symbol):

    spot = get_all_underlying_prices([symbol]).get(symbol)
    if not spot: return []
    min_strike = str(spot * (1 - STRIKE_RANGE))
    max_strike = str(spot * (1 + STRIKE_RANGE))
    today = datetime.now(timezone).date()
    if PAPER:
        try:
            chain = option_data_client.get_option_chain(
                OptionChainRequest(
                    underlying_symbol=symbol,
                    feed=OPTIONS_FEED,
                    type=ContractType.PUT,
                    strike_price_gte=spot*(1-STRIKE_RANGE),
                    strike_price_lte=spot*(1+STRIKE_RANGE),
                    expiration_date=today,
                )
            )
            contracts = []
            for sym, snap in chain.items():
                m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', sym)
                if not m:
                    continue
                _, date_str, opt_type, strike_str = m.groups()
                exp = datetime.strptime(date_str, '%y%m%d').date()
                strike = int(strike_str) / 1000
                contracts.append(SimpleNamespace(
                    symbol=sym,
                    expiration_date=exp,
                    strike_price=str(strike),
                    open_interest=OI_THRESHOLD*2,
                ))
            return contracts
        except Exception as e:
            log(f"❌ Failed to get 0DTE contracts via market data for {symbol}: {e}")
            log("ℹ️ Falling back to trading API for option contracts")
            try:
                req2 = GetOptionContractsRequest(
                    underlying_symbols=[symbol],
                    strike_price_gte=min_strike,
                    strike_price_lte=max_strike,
                    expiration_date=today,
                    status=AssetStatus.ACTIVE,
                    root_symbol=symbol,
                    type=ContractType.PUT,
                )
                contracts2 = trade_client.get_option_contracts(req2).option_contracts
                return contracts2
            except Exception as e2:
                log(f"❌ Fallback trading API failed for {symbol}: {e2}")
                return []
    req = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        strike_price_gte=min_strike,
        strike_price_lte=max_strike,
        expiration_date=today,
        status=AssetStatus.ACTIVE,
        root_symbol=symbol,
        type=ContractType.PUT,
    )
    try:
        contracts = trade_client.get_option_contracts(req).option_contracts
        if len(contracts) < 5:
            log(f"⚠️ Low contract count for {symbol}, retrying...")
            time_module.sleep(2)
            contracts = trade_client.get_option_contracts(req).option_contracts
        return contracts
    except Exception as e:
        log(f"❌ Failed to get 0DTE contracts for {symbol}: {e}")
        return []

def log_trade(symbol, short_strike, long_strike, credit, spread_width, take_profit_price, stop_loss_price, status):
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), symbol, short_strike, long_strike, credit, spread_width, take_profit_price, stop_loss_price, status])

def trade(symbol, spot):
    options = get_0dte_options(symbol)
    short_put = long_put = None
    for opt in options:
        if not opt.open_interest or int(opt.open_interest) < OI_THRESHOLD:
            continue
        quote = option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=opt.symbol, feed=OPTIONS_FEED)).get(opt.symbol)
        if not quote or not quote.bid_price or not quote.ask_price:
            continue
        price = (quote.bid_price + quote.ask_price) / 2
        expiry = datetime.combine(opt.expiration_date, dt_time(16, 0)).replace(tzinfo=timezone)
        delta = calculate_delta(price, float(opt.strike_price), expiry, spot, risk_free_rate, 'put')
        if delta is None:
            continue
        if SHORT_PUT_DELTA_RANGE[0] <= delta <= SHORT_PUT_DELTA_RANGE[1]:
            short_put = (opt, price)
        elif LONG_PUT_DELTA_RANGE[0] <= delta <= LONG_PUT_DELTA_RANGE[1]:
            long_put = (opt, price)
        if short_put and long_put:
            break
    if not short_put or not long_put:
        log(f"[{symbol}] No valid spread found.")
        return
    credit = short_put[1] - long_put[1]
    width = abs(float(short_put[0].strike_price) - float(long_put[0].strike_price))
    take_profit_price = round(credit - credit * PROFIT_TAKE_PERCENTAGE, 2)
    stop_loss_price = round(credit + (width - credit) * STOP_LOSS_PERCENTAGE, 2)
    min_credit = MIN_CREDIT_PERCENTAGE * width
    if credit < min_credit:
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, take_profit_price, stop_loss_price, "rejected")
        return
    if width * 100 > max_risk_per_trade:
        log(f"[{symbol}] Skipped: spread too large (${width * 100:.2f})")
        return
    try:
        order = LimitOrderRequest(
                symbol=symbol,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
            qty=1,
            limit_price=round(credit, 2),
            order_class=OrderClass.BRACKET,
            time_in_force=TimeInForce.DAY,
            legs=[
                OptionLegRequest(symbol=short_put[0].symbol, side=OrderSide.SELL, ratio_qty=1),
                OptionLegRequest(symbol=long_put[0].symbol, side=OrderSide.BUY, ratio_qty=1),
            ],
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price, limit_price=stop_loss_price),
        )
        trade_client.submit_order(order)
        # Log to trade_log.csv
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, take_profit_price, stop_loss_price, "submitted")
        # Log to open_trades.csv
        with open(OPEN_TRADES_FILE, "a", newline="") as f_open:
            writer_open = csv.writer(f_open)
            writer_open.writerow([symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, take_profit_price, stop_loss_price, datetime.now().isoformat()])
        log(f"✅ {symbol} Spread placed: Credit ${credit:.2f}, Width ${width:.2f}")
    except Exception as e:
        log(f"❌ Order failed: {e}")
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, take_profit_price, stop_loss_price, "submission_failed")

# === MAIN LOOP ===
log("🟢 Bot started")
TICKERS = load_tickers()
while True:
    if is_market_open():
        prices = get_all_underlying_prices(TICKERS)
        for symbol in TICKERS:
            if symbol in prices:
                trade(symbol, prices[symbol])
        log(f"⏱ Waiting {SCAN_INTERVAL // 60} minutes for next scan...")
        time_module.sleep(SCAN_INTERVAL)
    else:
        log("🔴 Market closed. Sleeping 15 minutes...")
        time_module.sleep(900)
