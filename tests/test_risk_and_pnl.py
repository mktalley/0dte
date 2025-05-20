import pytest
from alpaca.trading.enums import PositionSide
from src.main import calculate_num_contracts, calculate_pnl, should_exit, RISK_PER_TRADE_PERCENTAGE, STOP_LOSS_PERCENTAGE, PROFIT_TAKE_PERCENTAGE


def test_calculate_num_contracts():
    equity = 100_000.0
    width = 1.0
    max_risk_per_trade = 500.0
    # risk_amt = equity * 1% = 1000, cap = min(500,1000)=500, num_contracts = 500/(1*100)=5
    assert calculate_num_contracts(equity, width, max_risk_per_trade) == 5

    equity = 10_000.0
    width = 2.0
    max_risk_per_trade = 1000.0
    # risk_amt = 100, cap = 100, num_contracts = 100/(2*100) = 0.5 -> int = 0
    assert calculate_num_contracts(equity, width, max_risk_per_trade) == 0


def test_calculate_pnl_short_and_long():
    entry = 10.0
    mid = 8.0
    qty = 2
    contract_size = 100
    # Short position: pnl_share = entry - mid = 2 -> total = 2 * qty * size = 400
    pnl_short = calculate_pnl(entry, mid, qty, PositionSide.SHORT, contract_size)
    assert pytest.approx((entry - mid) * qty * contract_size) == pnl_short

    # Long position: pnl_share = mid - entry = -2 -> total = -2 * qty * size = -400
    pnl_long = calculate_pnl(entry, mid, qty, PositionSide.LONG, contract_size)
    assert pytest.approx((mid - entry) * qty * contract_size) == pnl_long


def test_should_exit():
    # Using STOP_LOSS_PERCENTAGE and PROFIT_TAKE_PERCENTAGE from settings
    assert should_exit(-STOP_LOSS_PERCENTAGE - 0.1) == 'stop_loss'
    assert should_exit(PROFIT_TAKE_PERCENTAGE + 0.1) == 'profit_take'
    assert should_exit(0.0) is None
