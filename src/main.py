# 0DTE Bot (Fully Optimized)

import os
import csv
import time as time_module
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, date, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from zoneinfo import ZoneInfo
import smtplib
from email.mime.text import MIMEText
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce, AssetStatus, ContractType, AssetClass, PositionSide
from alpaca.trading.requests import GetOptionContractsRequest, OptionLegRequest, LimitOrderRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import OptionLatestQuoteRequest
import json
import logging
from logging.handlers import TimedRotatingFileHandler

import functools
from tenacity import retry, wait_exponential, stop_after_attempt
# === CONFIGURATION VALIDATION ===
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError
from typing import Optional
import sys
import schedule
import subprocess

class Settings(BaseSettings):
    email_host: str = Field("localhost", env="EMAIL_HOST")
    email_port: int = Field(25, env="EMAIL_PORT")
    email_user: Optional[str] = Field(None, env="EMAIL_USER")
    email_pass: Optional[str] = Field(None, env="EMAIL_PASS")
    email_from: str = Field("alerts@example.com", env="EMAIL_FROM")
    email_to: Optional[str] = Field(None, env="EMAIL_TO")

    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")

    stop_loss_percentage: float = Field(0.5, env="STOP_LOSS_PERCENTAGE")
    profit_take_percentage: float = Field(0.5, env="PROFIT_TAKE_PERCENTAGE")
    min_credit_percentage: float = Field(0.15, env="MIN_CREDIT_PERCENTAGE")
    oi_threshold: int = Field(100, env="OI_THRESHOLD")
    strike_range: float = Field(0.1, env="STRIKE_RANGE")
    scan_interval: int = Field(300, env="SCAN_INTERVAL")
    circuit_breaker_threshold: int = Field(5, env="CIRCUIT_BREAKER_THRESHOLD")
    risk_per_trade_percentage: float = Field(0.01, env="RISK_PER_TRADE_PERCENTAGE")
    max_concurrent_trades: int = Field(5, env="MAX_CONCURRENT_TRADES")
    max_total_delta_exposure: float = Field(200, env="MAX_TOTAL_DELTA_EXPOSURE")
    spy_min_credit_percentage: float = Field(0.10, env="SPY_MIN_CREDIT_PERCENTAGE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except ValidationError as e:
    print("Configuration error:")
    print(e)
    sys.exit(1)

# Map settings to global variables
# Risk and filter settings from config
RISK_PER_TRADE_PERCENTAGE = settings.risk_per_trade_percentage
MAX_CONCURRENT_TRADES = settings.max_concurrent_trades
MAX_TOTAL_DELTA_EXPOSURE = settings.max_total_delta_exposure
SPY_MIN_CREDIT_PERCENTAGE = settings.spy_min_credit_percentage

EMAIL_HOST = settings.email_host
EMAIL_PORT = settings.email_port
EMAIL_USER = settings.email_user
EMAIL_PASS = settings.email_pass
EMAIL_FROM = settings.email_from
EMAIL_TO = settings.email_to

API_KEY = settings.alpaca_api_key
API_SECRET = settings.alpaca_secret_key

STOP_LOSS_PERCENTAGE = settings.stop_loss_percentage
PROFIT_TAKE_PERCENTAGE = settings.profit_take_percentage

MIN_CREDIT_PERCENTAGE = settings.min_credit_percentage
OI_THRESHOLD = settings.oi_threshold
STRIKE_RANGE = settings.strike_range
SCAN_INTERVAL = settings.scan_interval
CIRCUIT_BREAKER_THRESHOLD = settings.circuit_breaker_threshold

# Symbol-specific override dictionaries
SYMBOL_FILTER_OVERRIDES: dict[str, dict] = {}
SYMBOL_RISK_CAP: dict[str, float] = {}
# Default max risk per trade (dollars): no hard cap
max_risk_per_trade = float('inf')




def send_email(subject, body):
    if not EMAIL_TO:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        try:
            server.starttls()
        except Exception:
            pass
        if EMAIL_USER and EMAIL_PASS:
            server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO.split(","), msg.as_string())
        server.quit()
        log(f"Email alert sent: {subject}")
    except Exception as e:
        log(f"Failed to send email: {e}")

# === API ERROR HANDLING & CIRCUIT BREAKER ===
_api_failure_count = 0
_circuit_open = False

def api_guard(func):
    """
    Decorator for retry/backoff and circuit breaker on API calls.
    """
    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3), reraise=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _api_failure_count, _circuit_open
        if _circuit_open:
            msg = f"Circuit breaker open: blocking call to {func.__name__}"
            log(f"❌ {msg}")
            send_email("Circuit Breaker Open", msg)
            raise Exception(msg)
        try:
            result = func(*args, **kwargs)
            _api_failure_count = 0
            return result
        except Exception as e:
            _api_failure_count += 1
            msg = f"API call error in {func.__name__}: {e}"
            log(f"❌ {msg}")
            send_email(f"API Error: {func.__name__}", msg)
            if _api_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                _circuit_open = True
                send_email("Circuit Breaker Tripped",
                           f"{_api_failure_count} consecutive failures in {func.__name__}")
            raise
    return wrapper

# Wrapped API calls
@api_guard
def guarded_get_stock_latest_trade(req):
    return stock_data_client.get_stock_latest_trade(req)

@api_guard
def guarded_get_option_contracts(req):
    return trade_client.get_option_contracts(req)

@api_guard
def guarded_get_all_positions():
    return trade_client.get_all_positions()

@api_guard
def guarded_submit_order(order):
    return trade_client.submit_order(order)

@api_guard
def guarded_get_option_latest_quote(req):
    return option_data_client.get_option_latest_quote(req)



# Ensure timezone and logger are available for dynamic config
timezone = ZoneInfo("America/Los_Angeles")

# === STRUCTURED LOGGING ===
class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        record_dict = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # include custom attributes for structured logging
        for attr in ["symbol", "event", "candidates"]:
            if hasattr(record, attr):
                record_dict[attr] = getattr(record, attr)
        return json.dumps(record_dict)

# Ensure logs directory exists before initializing handlers
os.makedirs("logs", exist_ok=True)
# Configure root logger
logger = logging.getLogger("0dte")
logger.setLevel(logging.INFO)
# Convenience wrapper for simple logging calls
def log(message: str, **kwargs):
    logger.info(message, **kwargs)

# Console handler (plain text with timestamp)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
# === DAILY ROTATING LOG HANDLERS ===
import shutil

class DailyRotatingFileHandler(TimedRotatingFileHandler):
    """
    Keeps active logs at logs/orig_filename; archives yesterday's log into logs/YYYY-MM-DD/orig_filename on rollover.
    """
    def __init__(self, orig_filename, when, interval, backupCount, timezone, log_dir="logs"):
        self.orig_filename = orig_filename
        self.timezone = timezone
        self.log_dir = log_dir
        # Ensure top-level log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        # Active file stays at top level
        file_path = os.path.join(self.log_dir, self.orig_filename)
        super().__init__(file_path, when=when, interval=interval, backupCount=backupCount)

    def shouldRollover(self, record):
        return super().shouldRollover(record)

    def doRollover(self):
        # Close current log stream
        if self.stream:
            self.stream.close()
            self.stream = None
        # Determine date to archive (yesterday in timezone)
        now = datetime.now(tz=self.timezone)
        prev_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        subdir = os.path.join(self.log_dir, prev_date)
        os.makedirs(subdir, exist_ok=True)
        # Move the finished log into the dated subdirectory
        src = self.baseFilename
        dst = os.path.join(subdir, self.orig_filename)
        shutil.move(src, dst)
        # Compute next rollover time
        current_time = int(time_module.time())
        self.rolloverAt = self.computeRollover(current_time)
        # Reopen the active log file
        self.stream = self._open()

# JSON daily log handler
file_handler = DailyRotatingFileHandler(
    orig_filename="0dte.log",
    when="midnight",
    interval=1,
    backupCount=30,
    timezone=timezone,
)
file_handler.setFormatter(JsonLogFormatter())
logger.addHandler(file_handler)

# Human-readable daily log handler (Pacific Time)
class PacificFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=timezone)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.isoformat()
human_handler = DailyRotatingFileHandler(
    orig_filename="0dte_human.log",
    when="midnight",
    interval=1,
    backupCount=30,
    timezone=timezone,
)
human_formatter = PacificFormatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)
human_handler.setFormatter(human_formatter)
logger.addHandler(human_handler)

logger.info("🚀 Log handlers initialized: JSON and human-readable (PST)")

# === OVERRIDE FILTERS FOR WIDER SPREADS (Wider test) ===
MIN_CREDIT_PERCENTAGE = settings.min_credit_percentage  # allow firmer credit on wider spreads
OI_THRESHOLD = settings.oi_threshold                   # lower OI threshold to include deeper strikes
SHORT_PUT_DELTA_RANGE = (-0.6, -0.4)  # deeper OTM for short leg
LONG_PUT_DELTA_RANGE = (-0.4, -0.2)   # deeper OTM for long leg
STRIKE_RANGE = settings.strike_range  # ±30% from spot for strike scan

# Scan every 5 minutes instead of 10 to catch more delta swings
SCAN_INTERVAL = settings.scan_interval
risk_free_rate = 0.01


# Default to paper trading
PAPER = True
# === CLIENTS ===
trade_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
option_data_client = OptionHistoricalDataClient(API_KEY, API_SECRET)
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# === LOG FILES ===
os.makedirs("logs", exist_ok=True)
# === POSITION MANAGEMENT LOG FILES ===
OPEN_POS_LOG = "logs/open_positions.csv"
EXIT_LOG = "logs/exit_log.csv"
# PnL log directory; file rotates daily
PNL_LOG_DIR = "logs"
# Ensure daily PnL log file exists with header at startup
today_str = date.today().isoformat()
daily_pnl_file = os.path.join(PNL_LOG_DIR, f"pnl_log_{today_str}.csv")
if not os.path.exists(daily_pnl_file):
    with open(daily_pnl_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "side", "qty", "entry", "mid", "pnl_share", "pnl", "pnl_pct"])
# === POSITION MANAGEMENT ===

# === POSITION MANAGEMENT ===
def fetch_positions():
    """
    Fetch all open positions, record them to CSV, and return the list.
    """
    try:
        positions = trade_client.get_all_positions()
    except Exception as e:
        send_email("API Error: fetch_positions failed", f"Error fetching positions: {e}")
        return []
    # Record to CSV
    with open(OPEN_POS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        for p in positions:
            # Basic position fields
            writer.writerow([
                datetime.now().isoformat(),
                p.symbol,
                p.side.value if hasattr(p.side, 'value') else p.side,
                p.qty,
                p.avg_entry_price,
                p.cost_basis,
                (p.usd.unrealized_pl if p.usd and p.usd.unrealized_pl is not None else ""),
                (p.usd.unrealized_plpc if p.usd and p.usd.unrealized_plpc is not None else ""),
            ])
    return positions

def log_exit(symbol, side, qty, exit_price, pnl, ratio, status):
    """
    Log an exit or hedge trade to CSV.
    """
    with open(EXIT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            symbol,
            side,
            qty,
            exit_price,
            pnl,
            ratio,
            status,
        ])

TRADE_LOG = "logs/trade_log.csv"

# === UTILITIES ===

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
    # Handle naive and aware expiry datetimes
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone)
    else:
        expiry = expiry.astimezone(timezone)
    now = datetime.now(tz=timezone)
    T = max((expiry - now).total_seconds() / (365 * 24 * 3600), 1e-6)
    iv = calculate_iv(option_price, spot, strike, T, r, option_type)
    if not iv:
        return None
    d1 = safe_divide((np.log(spot / strike) + (r + 0.5 * iv**2) * T), iv * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)



def calculate_num_contracts(equity: float, width: float, max_risk_per_trade: float) -> int:
    """
    Calculate number of contracts based on equity, spread width, and max risk per trade.
    """
    risk_amt = equity * RISK_PER_TRADE_PERCENTAGE
    risk_cap_amt = min(max_risk_per_trade, risk_amt)
    return int(risk_cap_amt / (width * 100))


def calculate_pnl(entry_price: float, mid_price: float, qty: int, side, contract_size: int = 100) -> float:
    """
    Calculate PnL given entry price, current mid price, quantity, side, and contract size.
    """
    from alpaca.trading.enums import PositionSide
    # PnL per share
    pnl_share = (entry_price - mid_price) if side == PositionSide.SHORT else (mid_price - entry_price)
    return pnl_share * qty * contract_size


def should_exit(pnl_pct: float) -> str | None:
    """
    Determine if position should exit based on pnl percentage thresholds.
    Returns 'stop_loss', 'profit_take', or None.
    """
    if pnl_pct <= -STOP_LOSS_PERCENTAGE:
        return 'stop_loss'
    if pnl_pct >= PROFIT_TAKE_PERCENTAGE:
        return 'profit_take'
    return None

def is_market_open():
    # Check US equity market hours in Eastern Time
    now_et = datetime.now(tz=ZoneInfo("America/New_York"))
    return now_et.weekday() < 5 and dt_time(9, 30) <= now_et.time() <= dt_time(16, 0)

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
        log(f"API Error in get_all_underlying_prices: {e}")
        send_email("API Error: get_all_underlying_prices failed", str(e))
        return {}


def get_0dte_options(symbol):
    spot = get_all_underlying_prices([symbol]).get(symbol)
    if not spot: return []
    min_strike = str(spot * (1 - STRIKE_RANGE))
    max_strike = str(spot * (1 + STRIKE_RANGE))
    req = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        strike_price_gte=min_strike,
        strike_price_lte=max_strike,
        expiration_date=date.today(),
        status=AssetStatus.ACTIVE,
        root_symbol=symbol,
        type=ContractType.PUT,
    )
    contracts = trade_client.get_option_contracts(req).option_contracts
    if len(contracts) < 5:
        log(f"⚠️ Low contract count for {symbol}, retrying...")
    # Apply symbol-specific filter overrides
    overrides = SYMBOL_FILTER_OVERRIDES.get(symbol, {})
    min_credit_pct = overrides.get("MIN_CREDIT_PERCENTAGE", MIN_CREDIT_PERCENTAGE)
    short_delta_range = overrides.get("SHORT_PUT_DELTA_RANGE", SHORT_PUT_DELTA_RANGE)
    long_delta_range = overrides.get("LONG_PUT_DELTA_RANGE", LONG_PUT_DELTA_RANGE)

    return contracts

def log_trade(symbol, short_strike, long_strike, credit, spread_width, status):
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), symbol, short_strike, long_strike, credit, spread_width, status])

def trade(symbol, spot):
    # === Symbol-specific filter overrides ===
    overrides = SYMBOL_FILTER_OVERRIDES.get(symbol, {})
    threshold = overrides.get("OI_THRESHOLD", OI_THRESHOLD)
    short_range = overrides.get("SHORT_PUT_DELTA_RANGE", SHORT_PUT_DELTA_RANGE)
    long_range = overrides.get("LONG_PUT_DELTA_RANGE", LONG_PUT_DELTA_RANGE)
    min_credit_pct = overrides.get("MIN_CREDIT_PERCENTAGE", MIN_CREDIT_PERCENTAGE)
    # Initialize candidate logging for SPY
    candidates = [] if symbol == "SPY" else None

    options = get_0dte_options(symbol)
    short_put = long_put = None
    for opt in options:
        if not opt.open_interest or int(opt.open_interest) < threshold:
            continue
        quote = option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=opt.symbol)).get(opt.symbol)
        if not quote or not quote.bid_price or not quote.ask_price:
            continue
        price = (quote.bid_price + quote.ask_price) / 2
        expiry = datetime.combine(opt.expiration_date, dt_time(16, 0)).replace(tzinfo=timezone)
        delta = calculate_delta(price, float(opt.strike_price), expiry, spot, risk_free_rate, 'put')
        if delta is None:
            continue
        # Candidate logging for SPY
        if symbol == "SPY":
            candidate = {"option_symbol": opt.symbol, "open_interest": int(opt.open_interest), "delta": delta, "mid_price": price}
            
            candidates.append(candidate)
        # Delta filter
        if short_range[0] <= delta <= short_range[1]:
            short_put = (opt, price)
        elif long_range[0] <= delta <= long_range[1]:
            long_put = (opt, price)
        if short_put and long_put:
            break
    # Emit full candidate list for SPY before selection
    if symbol == "SPY":
        print(json.dumps(candidates))
    if not short_put or not long_put:
        log(f"[{symbol}] No valid spread found.")
        return
    credit = short_put[1] - long_put[1]
    width = abs(float(short_put[0].strike_price) - float(long_put[0].strike_price))
    min_credit = min_credit_pct * width
    if credit < min_credit:
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "rejected")
        return
    # Determine risk cap (symbol-specific override or default)
    cap = SYMBOL_RISK_CAP.get(symbol, max_risk_per_trade)
    if width * 100 > cap:
        log(f"[{symbol}] Skipped: spread too large (${width * 100:.2f} > ${cap:.2f})")
        return
    # === RISK SIZING ===
    try:
        account = trade_client.get_account()
        equity = float(account.equity)
    except Exception as e:
        log(f"[{symbol}] Error fetching account: {e}")
        return
    risk_amt = equity * RISK_PER_TRADE_PERCENTAGE
    risk_cap_amt = min(cap, risk_amt)
    num_contracts = int(risk_cap_amt / (width * 100))
    if num_contracts < 1:
        log(f"[{symbol}] Skipped: risk per trade too small for width ${width:.2f}")
        return
    # === CONCURRENCY LIMIT ===
    try:
        positions = trade_client.get_all_positions()
    except Exception as e:
        log(f"[{symbol}] Error fetching positions: {e}")
        return
    option_positions = [p for p in positions if p.asset_class == AssetClass.US_OPTION]
    num_spreads = len(option_positions) // 2
    if num_spreads >= MAX_CONCURRENT_TRADES:
        log(f"[{symbol}] Skipped: max concurrent trades reached ({MAX_CONCURRENT_TRADES})")
        return
    # === DELTA EXPOSURE ===
    short_delta = delta
    long_delta = calculate_delta(long_put[1], float(long_put[0].strike_price), expiry, spot, risk_free_rate, 'put')
    new_delta = (short_delta + long_delta) * 100 * num_contracts
    if abs(new_delta) > MAX_TOTAL_DELTA_EXPOSURE:
        log(f"[{symbol}] Skipped: trade delta exposure {new_delta:.2f} exceeds max {MAX_TOTAL_DELTA_EXPOSURE}")
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
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "submission_failed")
        log(f"[{symbol}] Error submitting order: {e}")

# === MAIN LOOP ===

def main_loop():
    # Initialize daily PnL log file with header
    today_str = datetime.now().date().isoformat()
    pnl_log_file = os.path.join(PNL_LOG_DIR, f"pnl_log_{today_str}.csv")
    if not os.path.exists(pnl_log_file):
        with open(pnl_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "side", "qty", "entry", "mid", "pnl_share", "pnl", "pnl_pct"])

    # Main loop entry point
    # Schedule end-of-day report at 16:05 ET
    def job_send_eod():
        log("📈 Triggering EOD report")
        try:
            subprocess.run(["python3", "scripts/send_eod_report.py"], check=True)
        except Exception as e:
            log(f"Error running EOD report: {e}")

    schedule.every().day.at("16:05").do(job_send_eod)

    log("🟢 Bot started")
    TICKERS = load_tickers()
    while True:
        schedule.run_pending()
        if is_market_open():
            prices = get_all_underlying_prices(TICKERS)
            # === POSITION MANAGEMENT ===
            positions = fetch_positions()
            for p in positions:
                # Only manage option positions
                if getattr(p, 'asset_class', None) != AssetClass.US_OPTION:
                    continue
                try:
                    quote = option_data_client.get_option_latest_quote(
                        OptionLatestQuoteRequest(symbol_or_symbols=p.symbol)
                    ).get(p.symbol)
                except Exception as e:
                    log(f"[{p.symbol}] Error fetching quote: {e}")
                    continue
                mid = (quote.bid_price + quote.ask_price) / 2
                qty = int(p.qty)
                entry = float(p.avg_entry_price)
                contract_size = 100
                # Compute PnL per share and total
                pnl_share = (entry - mid) if p.side == PositionSide.SHORT else (mid - entry)
                pnl = pnl_share * qty * contract_size
                # Percent change if available
                pnl_pct = None
                if p.usd and p.usd.unrealized_plpc is not None:
                    pnl_pct = p.usd.unrealized_plpc
                # Log PnL snapshot to daily rotated file
                date_str = datetime.now().date().isoformat()
                pnl_log_file = os.path.join(PNL_LOG_DIR, f"pnl_log_{date_str}.csv")
                write_header = not os.path.exists(pnl_log_file)
                with open(pnl_log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["timestamp", "symbol", "side", "qty", "entry", "mid", "pnl_share", "pnl", "pnl_pct"])
                    writer.writerow([
                        datetime.now().isoformat(),
                        p.symbol,
                        p.side.value if hasattr(p.side, 'value') else p.side,
                        qty,
                        entry,
                        mid,
                        pnl_share,
                        pnl,
                        pnl_pct,
                    ])
                # Check stop-loss or profit-take
                # Determine applicable profit-take threshold (SPY override)
                if p.symbol.startswith("SPY"):
                    profit_take_threshold = 0.75
                else:
                    profit_take_threshold = PROFIT_TAKE_PERCENTAGE
                if pnl_pct is not None and (pnl_pct <= -STOP_LOSS_PERCENTAGE or pnl_pct >= profit_take_threshold):
                    try:
                        resp = trade_client.close_position(p.symbol)
                        status = getattr(resp, 'status', 'closed')
                        log_exit(
                            p.symbol,
                            p.side.value if hasattr(p.side, 'value') else p.side,
                            qty,
                            mid,
                            pnl,
                            pnl_pct,
                            status,
                        )
                        action = 'stop loss' if pnl_pct <= -STOP_LOSS_PERCENTAGE else 'profit take'
                        msg = f"⚠️ Closed {p.symbol} due to {action}: PnL ${pnl:.2f}"
                        log(msg)
                        send_email(f"Position Closed: {p.symbol}", msg)
                    except Exception as e:
                        log(f"[{p.symbol}] Error closing position: {e}")
                        continue

            for symbol in TICKERS:
                if symbol in prices:
                    trade(symbol, prices[symbol])
            log(f"⏱ Waiting {SCAN_INTERVAL // 60} minutes for next scan...")
            time_module.sleep(SCAN_INTERVAL)
        else:
            log("🔴 Market closed. Sleeping 15 minutes...")
            time_module.sleep(900)

if __name__ == "__main__":
    import argparse
    import sys
    import os
    import runpy

    parser = argparse.ArgumentParser(description="0DTE Bot: backtest and trading for put credit and strangle strategies")
    parser.add_argument('--backtest', action='store_true', help='Run 0DTE backtest for the past year')
    parser.add_argument('--strangle', action='store_true', help='Use strangle strategy (requires --backtest for backtesting)')
    parser.add_argument('--start', help='Start date YYYY-MM-DD (optional for backtests)')
    parser.add_argument('--end', help='End date YYYY-MM-DD (optional for backtests)')
    args = parser.parse_args()

    if args.backtest:
        # Backtest mode
        if args.strangle:
            script = 'run_backtest_strangle_spy_last_year.py'
        else:
            script = 'run_backtest_spy_last_year.py'
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', script))
        # Reset sys.argv for the backtest script to avoid passing main.py args
        sys.argv = [script_path]
        runpy.run_path(script_path, run_name='__main__')
        sys.exit(0)
    elif args.strangle:
        # Live strangle trading bot
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'trading_bot.py'))
        runpy.run_path(script_path, run_name='__main__')
        sys.exit(0)
    else:
        # Live put credit spread bot
        log("✅ Starting live 0DTE put credit spread bot now; it will sleep every 15 minutes until market open at 09:30 ET if closed.")
        main_loop()
