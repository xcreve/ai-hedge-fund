"""Engine smoke tests on sample fixtures + look-ahead fill offset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.futures.backtest.engine import FuturesBacktestEngine
from src.futures.backtest.report import summary_text
from src.futures.data.csv_loader import load_directory
from src.futures.data.schema import InstrumentSpec
from src.futures.execution.portfolio_risk import PortfolioRiskLimits
from src.futures.signals.cct1 import CCT1Params, compute_cct1_signals

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "futures" / "sample"


@pytest.fixture
def rb_au_market():
    return load_directory(SAMPLE, symbols=["RB", "AU"])


@pytest.fixture
def instruments():
    return {
        "RB": InstrumentSpec(
            symbol="RB", multiplier=10, tick_size=1.0, margin_rate=0.10, sector="ferrous",
            commission_per_lot=3.0, slippage_ticks=1.0,
        ),
        "AU": InstrumentSpec(
            symbol="AU", multiplier=1000, tick_size=0.02, margin_rate=0.08, sector="precious",
            commission_per_lot=10.0, slippage_ticks=1.0,
        ),
    }


def test_engine_runs_on_sample(rb_au_market, instruments):
    params = CCT1Params(adx_threshold=None, atr_pct_threshold=20.0, time_stop_bars=8)
    engine = FuturesBacktestEngine(
        instruments=instruments,
        params=params,
        risk_limits=PortfolioRiskLimits(risk_per_trade=0.01, max_symbols=6),
        initial_equity=1_000_000,
    )
    result = engine.run(rb_au_market)
    assert len(result.equity_curve) > 50
    assert "sharpe_ratio" in result.metrics
    assert result.final_equity > 0
    text = summary_text(result)
    assert "CCT-1" in text


def test_entry_fill_is_next_open_not_signal_close(instruments):
    """Construct a single-symbol path where long_ok fires once; fill must be next open."""
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Build a slow uptrend so SMA filters pass, then one clean Donchian breakout
    close = np.concatenate([
        np.linspace(100, 110, n - 3),
        np.array([112.0, 118.0, 119.0]),  # breakout near end
    ])
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000},
        index=dates,
    )
    params = CCT1Params(
        n1=10,
        n2=5,
        sma_fast=5,
        sma_slow=20,
        atr_period=10,
        atr_pct_lookback=30,
        atr_pct_threshold=0.0,
        adx_threshold=None,
        max_pyramid_adds=0,
        time_stop_bars=None,
    )
    sig = compute_cct1_signals(df, params)
    ok_idx = np.where(sig["long_ok"].fillna(False).to_numpy())[0]
    assert len(ok_idx) >= 1
    t = int(ok_idx[0])
    expected_fill = float(df["open"].iloc[t + 1])

    engine = FuturesBacktestEngine(
        instruments={"RB": instruments["RB"]},
        params=params,
        risk_limits=PortfolioRiskLimits(risk_per_trade=0.02, max_symbol_margin_pct=0.5, max_portfolio_margin_pct=0.9),
        initial_equity=1_000_000,
        use_ma_exit=False,
        use_channel_exit=False,
    )
    result = engine.run({"RB": df})
    entries = [tr for tr in result.trades if tr.side == "buy" and tr.reason == "breakout_long"]
    assert entries, "expected at least one long entry"
    # First entry date should be the bar after signal
    assert entries[0].dt.date() == dates[t + 1].date()
    # Price near next open (± slippage ticks)
    assert abs(entries[0].price - expected_fill) <= instruments["RB"].tick_size * instruments["RB"].slippage_ticks + 1e-9


def test_csv_loader_schema(rb_au_market):
    assert set(rb_au_market) == {"RB", "AU"}
    assert list(rb_au_market["RB"].columns[:5])  # smoke
    for col in ("open", "high", "low", "close", "volume"):
        assert col in rb_au_market["RB"].columns
