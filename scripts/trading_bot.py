#!/usr/bin/env python3
"""
trading_bot.py

A professional-level 0DTE trading bot with entry, monitoring, and exit lifecycle.

Features:
  - Entry of 0DTE short strangles (put + call credit spread) on multiple underlyings at a scheduled time.
  - Continuous P/L monitoring and optional dynamic exit on profit target / stop loss.
  - Hard exit (close all positions) at end-of-day.
  - Persistent logging of trades and P/L.
  - Configurable via environment variables:
      SYMBOLS      = comma-separated list (default: SPY,SPX,XSP)
      ENTRY_TIME   = HH:MM (24h ET) entry time (default: 09:35)
      EXIT_TIME    = HH:MM (24h ET) hard exit time (default: 15:45)
      PROFIT_TARGET= float, total P/L in USD to trigger exit (default: 100)
      STOP_LOSS    = float, P/L in USD to trigger exit (default: -100)
  - Logs to `bot.log` with rotating file handler.

Usage:
  pip install -r requirements.txt  # ensure APScheduler, pytz
  ./trading_bot.py

"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, date, time as dtime, timedelta
import csv
import time

from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import io
import contextlib

import pytz

import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
from alpaca.trading.enums import (
    ContractType, TimeInForce, OrderClass,
    OrderSide, PositionIntent, OrderType
)

# Load environment
load_dotenv()

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger('trading_bot')
logger.setLevel(LOG_LEVEL)

import shutil
from logging.handlers import TimedRotatingFileHandler
import pytz
ET_ZONE = pytz.timezone('America/New_York')  # ensure timezone defined for log handler


# Create custom handler for daily rotating strangle logs with monthly archives
class MonthlyRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, orig_filename, when='midnight', interval=1, timezone=None, log_dir='logs/strangle'):
        self.orig_filename = orig_filename
        self.timezone = timezone
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        file_path = os.path.join(self.log_dir, orig_filename)
        super().__init__(file_path, when=when, interval=interval, backupCount=0, utc=False)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        # Archive yesterday's log
        prev = datetime.now(tz=self.timezone) - timedelta(days=1)
        date_str = prev.strftime('%Y-%m-%d')
        month_str = prev.strftime('%Y-%m')
        dest_dir = os.path.join(self.log_dir, month_str)
        os.makedirs(dest_dir, exist_ok=True)
        name, ext = os.path.splitext(self.orig_filename)
        dst = os.path.join(dest_dir, f"{name}_{date_str}{ext}")
        shutil.move(self.baseFilename, dst)
        # Schedule next rollover
        self.rolloverAt = self.computeRollover(int(time.time()))
        self.stream = self._open()

# Setup strangle bot log handler

class TzFormatter(logging.Formatter):
    """Logging formatter that applies a timezone to timestamps"""
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        # Use the specified timezone for timestamp
        dt = datetime.fromtimestamp(record.created, tz=self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return super().formatTime(record, datefmt)

# Log formatter using Pacific Time
PT_ZONE = pytz.timezone('America/Los_Angeles')
formatter = TzFormatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', tz=PT_ZONE)
handler = MonthlyRotatingFileHandler(orig_filename='strangle.log', timezone=PT_ZONE)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Prevent log messages from propagating to the root logger (avoid duplicates)
logger.propagate = False


# Config
# Account risk sizing
ACCOUNT_CAPITAL       = float(os.getenv('ACCOUNT_CAPITAL', '38000'))  # total cash available
DAILY_RISK_PCT        = float(os.getenv('DAILY_RISK_PCT', '0.05'))       # percent of capital to risk per day (default 5%)
RISK_PER_CONTRACT     = float(os.getenv('RISK_PER_CONTRACT', '100'))    # worst-case loss per contract
# Compute daily risk budget and contract sizing
daily_risk            = ACCOUNT_CAPITAL * DAILY_RISK_PCT
CONTRACTS_PER_DAY     = max(1, int(daily_risk / RISK_PER_CONTRACT))
# Override profit/stop based on daily budget
PROFIT_TARGET         = daily_risk
STOP_LOSS             = -daily_risk

# Base config from env (retained for backward compatibility)
ET_ZONE = pytz.timezone('America/New_York')
SYMBOLS = [s.strip().upper() for s in os.getenv('SYMBOLS', 'SPY,SPX,XSP').split(',')]
ENTRY_TIME = os.getenv('ENTRY_TIME', '09:35')  # ET
EXIT_TIME = os.getenv('EXIT_TIME', '15:45')    # ET



# Strategy parameters
SHORT_PUT_DELTA_RANGE = (-0.45, -0.35)
LONG_PUT_DELTA_RANGE  = (-0.25, -0.15)
SHORT_CALL_DELTA_RANGE = ( 0.35,  0.45)
LONG_CALL_DELTA_RANGE  = ( 0.15,  0.25)
RISK_FREE_RATE         = 0.01  # assume 1% annual

# Vol index mapping
VOL_MAP = { 'SPY': '^VIX', 'SPX': '^VIX', 'XSP': '^VIX' }

# Alpaca clients
# Alpaca API credentials (supports two naming conventions)
API_KEY    = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
# Optional Base URL (for paper or live)
BASE_URL   = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_API_BASE_URL')
if not all((API_KEY, API_SECRET)):
    logger.error('Missing Alpaca API key or secret in environment. Exiting.')
    sys.exit(1)

# Option historical data client (requires API key/secret)
data_client = OptionHistoricalDataClient(API_KEY, API_SECRET)
if BASE_URL:
    trade_client = TradingClient(API_KEY, API_SECRET, paper=True, base_url=BASE_URL)
else:
    trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Scheduler
scheduler = BlockingScheduler(timezone=ET_ZONE)


def bs_d1(S, K, r, T, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def call_delta(S, K, r, T, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(bs_d1(S,K,r,T,sigma))

def put_delta(S, K, r, T, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    return norm.cdf(bs_d1(S,K,r,T,sigma)) - 1

def strike_from_delta(delta_func, S, r, T, sigma, target_delta):
    f = lambda K: delta_func(S, K, r, T, sigma) - target_delta
    return brentq(f, 1e-6, S*2)


def trade_strangle(symbol, today):
    """Entry logic: compute strikes, submit short strangle."""
    try:
        # set end_date for yfinance downloads (inclusive today)
        end_date = today + timedelta(days=1)

        # fetch daily underlying price without yfinance warnings
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = yf.download(symbol, start=today, end=end_date, auto_adjust=False, progress=False)
        except Exception as e:
            logger.warning(f"{symbol}: market data download error: {e}")
            return None
        if df.empty:
            logger.warning(f"{symbol}: no market data on {today}")
            return None
        S = float(df.iloc[0]['Open'])

        # fetch daily volatility index without yfinance warnings
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                vdf = yf.download(VOL_MAP.get(symbol,'^VIX'), start=today, end=end_date, auto_adjust=False, progress=False)
        except Exception as e:
            logger.warning(f"{symbol}: vol data download error: {e}")
            return None
        if vdf.empty:
            logger.warning(f"{symbol}: no vol data on {today}")
            return None
        sigma = float(vdf.iloc[0]['Close']) / 100.0
        T = 1/252
        # target deltas
        spd = sum(SHORT_PUT_DELTA_RANGE)/2
        spl = sum(LONG_PUT_DELTA_RANGE)/2
        scd = sum(SHORT_CALL_DELTA_RANGE)/2
        scl = sum(LONG_CALL_DELTA_RANGE)/2
        # solve strikes
        K_ps = strike_from_delta(put_delta, S, RISK_FREE_RATE, T, sigma, spd)
        K_pl = strike_from_delta(put_delta, S, RISK_FREE_RATE, T, sigma, spl)
        K_cs = strike_from_delta(call_delta, S, RISK_FREE_RATE, T, sigma, scd)
        K_cl = strike_from_delta(call_delta, S, RISK_FREE_RATE, T, sigma, scl)
        # chains
        exp = today
        # fetch option chains
        raw_put_chain = data_client.get_option_chain(
            OptionChainRequest(underlying_symbol=symbol, expiration_date=exp, type=ContractType.PUT)
        )
        raw_call_chain = data_client.get_option_chain(
            OptionChainRequest(underlying_symbol=symbol, expiration_date=exp, type=ContractType.CALL)
        )
        # normalize chains: some APIs return dicts
        put_chain = list(raw_put_chain.values()) if isinstance(raw_put_chain, dict) else list(raw_put_chain)
        call_chain = list(raw_call_chain.values()) if isinstance(raw_call_chain, dict) else list(raw_call_chain)
        if not put_chain or not call_chain:
            logger.info(f"{symbol}: no 0DTE chain on {today}")
            return None
        # pick option snapshot with greeks.delta nearest to target_delta
        # pick option snapshot with greeks.delta nearest to target_delta
        pick = lambda chain, target_delta: min(
    chain,
    key=lambda snap: abs((snap.greeks.delta if getattr(snap, 'greeks', None) and snap.greeks.delta is not None else 0.0) - target_delta)
)
        ps = pick(put_chain, K_ps); pl = pick(put_chain, K_pl)
        cs = pick(call_chain, K_cs); cl = pick(call_chain, K_cl)
        legs = [
            OptionLegRequest(symbol=ps.symbol, ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.OPEN),
            OptionLegRequest(symbol=pl.symbol, ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.OPEN),
            OptionLegRequest(symbol=cs.symbol, ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.OPEN),
            OptionLegRequest(symbol=cl.symbol, ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.OPEN),
        ]
        order = MarketOrderRequest(qty=CONTRACTS_PER_DAY, time_in_force=TimeInForce.DAY,
                                  order_class=OrderClass.MLEG, type=OrderType.MARKET,
                                  legs=legs)
        resp = trade_client.submit_order(order)
        logger.info(f"{symbol}: entry order {resp.id}")
        return resp.id
    except Exception as e:
        logger.error(f"{symbol}: entry error: {e}")
        return None


def entry_job():
    today = date.today()
    logger.info(f"=== ENTRY JOB @ {datetime.now(ET_ZONE)} ===")
    for sym in SYMBOLS:
        trade_strangle(sym, today)


def monitor_job():
    logger.info(f"=== MONITOR JOB @ {datetime.now(ET_ZONE)} ===")
    # Skip monitoring when market is closed
    clock = trade_client.get_clock()
    if not clock.is_open:
        logger.info(f"Market is closed at {datetime.now(ET_ZONE)}, sleeping for 2 minutes.")
        time.sleep(120)
        return

    try:
        positions = trade_client.get_all_positions()
        total_pl = 0.0
        for pos in positions:
            if pos.asset_class == 'option' and pos.symbol.split('_')[0] in SYMBOLS:
                pl = float(pos.unrealized_pl)
                total_pl += pl
                logger.info(f"{pos.symbol}: qty={pos.qty}, P/L={pl:.2f}")
        logger.info(f"Total P/L across positions: {total_pl:.2f}")
        if total_pl >= PROFIT_TARGET or total_pl <= STOP_LOSS:
            logger.info(f"Threshold hit (P/L={total_pl:.2f}), early exit: closing positions and re-entering.")
            try:
                trade_client.close_all_positions()
                logger.info("All positions closed (early exit).")
            except Exception as e:
                logger.error(f"Early exit error: {e}")
            # Immediately re-enter fresh strangle
            logger.info("Re-deploying new strangle after early exit.")
            entry_job()
            return
    except Exception as e:
        logger.error(f"Monitor error: {e}")


def exit_job():
    logger.info(f"=== EXIT JOB @ {datetime.now(ET_ZONE)} ===")
    try:
        trade_client.close_all_positions()
        logger.info("All positions closed.")
    except Exception as e:
        logger.error(f"Exit error: {e}")
    # shutdown scheduler after exit
    scheduler.shutdown(wait=False)



def remind_job():
    """Daily reminder at 4 PM Pacific to revisit margin sizing logic."""
    logger.info("Reminder: revisit margin-based sizing logic.")


def heartbeat_job():
    """Periodic heartbeat until market open"""
    logger.info(f"Heartbeat: trading bot is alive @ {datetime.now(ET_ZONE)}")



def schedule_jobs():
    # parse entry and exit times
    eth, etm = map(int, ENTRY_TIME.split(':'))
    exh, exm = map(int, EXIT_TIME.split(':'))
    entry_trigger = CronTrigger(hour=eth, minute=etm, timezone=ET_ZONE)
    exit_trigger = CronTrigger(hour=exh, minute=exm, timezone=ET_ZONE)
    monitor_trigger = IntervalTrigger(
        seconds=60,
        start_date=datetime.combine(date.today(), dtime(hour=eth, minute=etm, tzinfo=ET_ZONE)) + timedelta(minutes=1),
        end_date=datetime.combine(date.today(), dtime(hour=exh, minute=exm, tzinfo=ET_ZONE)),
    )

    # schedule heartbeat until market open
    entry_dt = datetime.combine(date.today(), dtime(hour=eth, minute=etm, tzinfo=ET_ZONE))
    now_dt = datetime.now(ET_ZONE)
    if now_dt < entry_dt:
        heartbeat_trigger = IntervalTrigger(
            seconds=3600,
            start_date=now_dt,
            end_date=entry_dt,
            timezone=ET_ZONE
        )
        scheduler.add_job(heartbeat_job, trigger=heartbeat_trigger, id='heartbeat')
        logger.info(f"Scheduled heartbeat every hour until market open at {ENTRY_TIME} ET")

    # schedule trading lifecycle jobs
    scheduler.add_job(entry_job, trigger=entry_trigger, id='entry')
    scheduler.add_job(monitor_job, trigger=monitor_trigger, id='monitor')
    scheduler.add_job(exit_job, trigger=exit_trigger, id='exit')
    # schedule one-time reminder at 4 PM Pacific today
    pacific = pytz.timezone('US/Pacific')
    now_pac = datetime.now(pacific)
    run_date = now_pac.replace(hour=16, minute=0, second=0, microsecond=0)
    if run_date > now_pac:
        scheduler.add_job(remind_job,
                          trigger='date',
                          run_date=run_date,
                          timezone=pacific,
                          id='reminder')
    else:
        logger.info("4 PM Pacific time already passed; skipping today's reminder")


def main():
    logger.info("Starting trading bot")
    schedule_jobs()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down bot")

if __name__ == '__main__':
    main()
