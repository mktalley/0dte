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
from email.mime.text import MIMEText
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce, AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest, OptionLegRequest, LimitOrderRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import OptionLatestQuoteRequest

# === CONFIGURATION ===
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = True
capital_pool = 100000
max_risk_per_trade = 1000
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", "0.5"))
PROFIT_TAKE_PERCENTAGE = float(os.getenv("PROFIT_TAKE_PERCENTAGE", "0.5"))
# Dynamic parameter selection based on day of week
now_dt = datetime.now(timezone)
dow = now_dt.weekday()  # 0=Mon, 4=Fri
if dow <= 1:  # Mon/Tue: aggressive early-week settings
    log("⚙️ Using aggressive early-week parameters")
    MIN_CREDIT_PERCENTAGE = float(os.getenv("MIN_CREDIT_PERCENTAGE", "0.15"))
    OI_THRESHOLD = int(os.getenv("OI_THRESHOLD", "200"))
    SHORT_PUT_DELTA_RANGE = (-0.5, -0.3)
    LONG_PUT_DELTA_RANGE = (-0.3, -0.1)
    STRIKE_RANGE = float(os.getenv("STRIKE_RANGE", "0.2"))
elif dow <= 3:  # Wed/Thu: moderate settings
    log("⚙️ Using mid-week parameters")
    MIN_CREDIT_PERCENTAGE = float(os.getenv("MIN_CREDIT_PERCENTAGE", "0.2"))
    OI_THRESHOLD = int(os.getenv("OI_THRESHOLD", "300"))
    SHORT_PUT_DELTA_RANGE = (-0.45, -0.35)
    LONG_PUT_DELTA_RANGE = (-0.25, -0.15)
    STRIKE_RANGE = float(os.getenv("STRIKE_RANGE", "0.15"))
else:  # Fri: tight settings
    log("⚙️ Using Friday expiration parameters")
    MIN_CREDIT_PERCENTAGE = float(os.getenv("MIN_CREDIT_PERCENTAGE", "0.25"))
    OI_THRESHOLD = int(os.getenv("OI_THRESHOLD", "500"))
    SHORT_PUT_DELTA_RANGE = (-0.45, -0.35)
    LONG_PUT_DELTA_RANGE = (-0.25, -0.15)
    STRIKE_RANGE = float(os.getenv("STRIKE_RANGE", "0.1"))
# Scan every 5 minutes instead of 10 to catch more delta swings
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))
risk_free_rate = 0.01
timezone = ZoneInfo("America/New_York")

# === CLIENTS ===
trade_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
option_data_client = OptionHistoricalDataClient(API_KEY, API_SECRET)
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# === LOG FILES ===
os.makedirs("logs", exist_ok=True)
TRADE_LOG = "logs/trade_log.csv"

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
    req = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        strike_price_gte=min_strike,
        strike_price_lte=max_strike,
        expiration_date=today,
        status=AssetStatus.ACTIVE,
        root_symbol=symbol,
        type=ContractType.PUT,
    )
    contracts = trade_client.get_option_contracts(req).option_contracts
    if len(contracts) < 5:
        log(f"⚠️ Low contract count for {symbol}, retrying...")
        time_module.sleep(2)
        contracts = trade_client.get_option_contracts(req).option_contracts
    return contracts

def log_trade(symbol, short_strike, long_strike, credit, spread_width, status):
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), symbol, short_strike, long_strike, credit, spread_width, status])

def trade(symbol, spot):
    options = get_0dte_options(symbol)
    short_put = long_put = None
    for opt in options:
        if not opt.open_interest or int(opt.open_interest) < OI_THRESHOLD:
            continue
        quote = option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=opt.symbol)).get(opt.symbol)
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
    min_credit = MIN_CREDIT_PERCENTAGE * width
    if credit < min_credit:
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "rejected")
        return
    if width * 100 > max_risk_per_trade:
        log(f"[{symbol}] Skipped: spread too large (${width * 100:.2f})")
        return
    try:
        order = LimitOrderRequest(
            qty=1,
            limit_price=round(credit, 2),
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            legs=[
                OptionLegRequest(symbol=short_put[0].symbol, side=OrderSide.SELL, ratio_qty=1),
                OptionLegRequest(symbol=long_put[0].symbol, side=OrderSide.BUY, ratio_qty=1),
            ],
        )
        trade_client.submit_order(order)
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "submitted")
        log(f"✅ {symbol} Spread placed: Credit ${credit:.2f}, Width ${width:.2f}")
    except Exception as e:
        log(f"❌ Order failed: {e}")
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "submission_failed")

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
