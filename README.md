# 0DTE Bot

A fully optimized zero-day-to-expiry (0DTE) options trading bot and backtesting framework for SPY.

## Features

- Backtesting of SPY strategies for past year via `--backtest`, including strangle and credit spreads
- Live paper trading with strangle and put credit strategies
- Structured JSON logs and human-readable logs with daily rotation into date-based folders

## Usage

```bash
# Live trading (strangle strategy)
python src/main.py --strangle

# Backtest SPY last year (default credit spread strategy)
python src/main.py --backtest

# Backtest SPY strangle last year
python src/main.py --backtest --strangle
```

## Logging

Logs are now rotated daily into `logs/YYYY-MM-DD/` directories by default (using `DailyRotatingFileHandler`).

- **JSON logs**: `0dte.log` in each date folder, containing structured JSON entries for each log message.
- **Human-readable logs**: `0dte_human.log` in each date folder, with timestamps in America/Los_Angeles (PST/PDT) timezone.
- Retention: up to 30 days of logs (`backupCount=30`).

### Example

```bash
$ ls logs/2025-05-22
0dte.log  0dte_human.log
```

## Configuration

Environment variables are managed via Pydantic settings in `src/main.py`:
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` for trading and data APIs
- `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_FROM`, `EMAIL_TO` for alerts
- Strategy settings: `STOP_LOSS_PERCENTAGE`, `PROFIT_TAKE_PERCENTAGE`, `MIN_CREDIT_PERCENTAGE`, `OI_THRESHOLD`, `STRIKE_RANGE`, `SCAN_INTERVAL`

See `requirements.txt` for dependencies.
