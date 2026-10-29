"""Walk-forward window splitting + smoke evaluation on sample data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.futures.backtest.walkforward import (
    WalkForwardConfig,
    run_walk_forward,
    split_walk_forward_windows,
    walk_forward_config_from_mapping,
)
from src.futures.backtest.walkforward_report import summary_markdown, write_walkforward_reports
from src.futures.data.csv_loader import load_directory
from src.futures.data.schema import InstrumentSpec
from src.futures.execution.portfolio_risk import PortfolioRiskLimits
from src.futures.signals.cct1 import CCT1Params

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "futures" / "sample"


def test_split_walk_forward_windows_basic():
    cfg = WalkForwardConfig(train_bars=100, test_bars=20, step_bars=20, warmup_bars=30)
    folds = split_walk_forward_windows(200, cfg)
    assert len(folds) >= 4
    assert folds[0].train_start == 0
    assert folds[0].train_end == 100
    assert folds[0].test_start == 100
    assert folds[0].test_end == 120
    assert folds[0].warmup_start == 0  # clamped
    assert folds[1].train_start == 20
    assert folds[1].warmup_start == 0


def test_split_walk_forward_windows_warmup_and_count():
    cfg = WalkForwardConfig(train_bars=50, test_bars=10, step_bars=10, warmup_bars=15)
    folds = split_walk_forward_windows(100, cfg)
    # starts at 0,10,20,30,40 → last start 40 → 40+50+10=100 OK → 5 folds
    assert len(folds) == 5
    assert folds[-1].test_end == 100
    assert folds[2].warmup_start == folds[2].train_start - 15


def test_split_rejects_short_series():
    cfg = WalkForwardConfig(train_bars=100, test_bars=50, step_bars=10)
    with pytest.raises(ValueError, match="need at least"):
        split_walk_forward_windows(120, cfg)


def test_walk_forward_config_from_mapping():
    cfg = walk_forward_config_from_mapping(
        {
            "walkforward": {
                "train_bars": 180,
                "test_bars": 60,
                "step_bars": 60,
                "warmup_bars": 120,
                "selection": "tiny_grid",
                "param_grid": {"n1": [15, 20], "n2": [10]},
            }
        }
    )
    assert cfg.selection == "tiny_grid"
    assert cfg.param_grid["n1"] == [15, 20]


@pytest.fixture
def sample_market():
    return load_directory(SAMPLE, symbols=["RB", "AU"])


@pytest.fixture
def instruments():
    return {
        "RB": InstrumentSpec(
            symbol="RB",
            multiplier=10,
            tick_size=1.0,
            margin_rate=0.10,
            sector="ferrous",
            commission_per_lot=3.0,
            slippage_ticks=1.0,
        ),
        "AU": InstrumentSpec(
            symbol="AU",
            multiplier=1000,
            tick_size=0.02,
            margin_rate=0.08,
            sector="precious",
            commission_per_lot=10.0,
            slippage_ticks=1.0,
        ),
    }


def test_sample_data_long_enough_for_walkforward(sample_market):
    dates = sorted({d for df in sample_market.values() for d in df.index})
    # Default yaml: 180 train + 60 test; need multiple folds → prefer ≥ 420 bars
    assert len(dates) >= 420, f"sample data too short for walk-forward: {len(dates)}"


def test_walkforward_smoke_default_params(sample_market, instruments, tmp_path):
    # Slightly smaller windows so smoke stays fast on CI while still multi-fold
    cfg = WalkForwardConfig(
        train_bars=120,
        test_bars=40,
        step_bars=40,
        warmup_bars=80,
        selection="default_params",
    )
    params = CCT1Params(adx_threshold=None, atr_pct_threshold=20.0, time_stop_bars=8)
    result = run_walk_forward(
        sample_market,
        instruments=instruments,
        params=params,
        risk_limits=PortfolioRiskLimits(risk_per_trade=0.01, max_symbols=6),
        initial_equity=1_000_000,
        extras={"use_ma_exit": True, "use_channel_exit": True, "chandelier_atr_mult": 3.0},
        config=cfg,
    )
    assert len(result.folds) >= 2
    assert "total_return" in result.aggregate_metrics
    assert len(result.aggregate_equity) > 0
    assert "Portfolio Value" in result.aggregate_equity.columns

    md = summary_markdown(result)
    assert "Walk-Forward" in md
    assert "Fold" in md

    paths = write_walkforward_reports(result, tmp_path)
    assert paths["summary_md"].is_file()
    assert paths["fold_metrics"].is_file()
    fold_df = pd.read_csv(paths["fold_metrics"])
    assert len(fold_df) == len(result.folds)


def test_walkforward_tiny_grid_smoke(sample_market, instruments):
    cfg = WalkForwardConfig(
        train_bars=120,
        test_bars=40,
        step_bars=80,
        warmup_bars=60,
        selection="tiny_grid",
        selection_metric="total_return",
        param_grid={"n1": [15, 20], "n2": [10]},
    )
    params = CCT1Params(adx_threshold=None, atr_pct_threshold=15.0, time_stop_bars=None, max_pyramid_adds=0)
    # Single-symbol keeps the tiny grid smoke cheap
    market = {"RB": sample_market["RB"]}
    result = run_walk_forward(
        market,
        instruments={"RB": instruments["RB"]},
        params=params,
        risk_limits=PortfolioRiskLimits(
            risk_per_trade=0.02, max_symbol_margin_pct=0.5, max_portfolio_margin_pct=0.9
        ),
        initial_equity=1_000_000,
        extras={"use_ma_exit": True, "use_channel_exit": True, "chandelier_atr_mult": 3.0},
        config=cfg,
    )
    assert len(result.folds) >= 1
    # Chosen params must come from the grid
    assert result.folds[0].params.n1 in {15, 20}
    assert result.folds[0].params.n2 == 10
