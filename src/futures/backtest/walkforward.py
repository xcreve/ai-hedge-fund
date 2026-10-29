"""Rolling walk-forward evaluation for CCT-1 (spec §7 overfit protection).

Default path: research params on each OOS test fold (no train fitting).
Optional ``tiny_grid``: search a small param grid on the train window only,
then evaluate the chosen params on the held-out test fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.futures.backtest.engine import BacktestResult, FuturesBacktestEngine
from src.futures.data.schema import InstrumentSpec
from src.futures.execution.portfolio_risk import PortfolioRiskLimits
from src.futures.signals.cct1 import CCT1Params


@dataclass(frozen=True)
class FoldWindow:
    """Inclusive index ranges on an aligned date calendar."""

    fold_id: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive
    warmup_start: int

    @property
    def train_bars(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_bars(self) -> int:
        return self.test_end - self.test_start


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 180
    test_bars: int = 60
    step_bars: int = 60
    warmup_bars: int = 120
    selection: str = "default_params"  # default_params | tiny_grid
    selection_metric: str = "sharpe_ratio"
    param_grid: Dict[str, Sequence[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.train_bars < 1 or self.test_bars < 1 or self.step_bars < 1:
            raise ValueError("train_bars, test_bars, and step_bars must be >= 1")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if self.selection not in {"default_params", "tiny_grid"}:
            raise ValueError(f"unknown selection mode: {self.selection}")


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: CCT1Params
    metrics: dict
    final_equity: float
    initial_equity: float
    equity_curve: pd.DataFrame
    n_trades: int
    selection_score: Optional[float] = None


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    aggregate_equity: pd.DataFrame
    aggregate_metrics: dict
    config: WalkForwardConfig
    selection: str
    initial_equity: float


def walk_forward_config_from_mapping(raw: Dict[str, Any]) -> WalkForwardConfig:
    wf = dict(raw.get("walkforward", raw))
    grid = wf.get("param_grid") or {}
    if not isinstance(grid, dict):
        raise ValueError("walkforward.param_grid must be a mapping of lists")
    return WalkForwardConfig(
        train_bars=int(wf.get("train_bars", 180)),
        test_bars=int(wf.get("test_bars", 60)),
        step_bars=int(wf.get("step_bars", 60)),
        warmup_bars=int(wf.get("warmup_bars", 120)),
        selection=str(wf.get("selection", "default_params")),
        selection_metric=str(wf.get("selection_metric", "sharpe_ratio")),
        param_grid={str(k): list(v) for k, v in grid.items()},
    )


def split_walk_forward_windows(
    n_bars: int,
    config: WalkForwardConfig,
) -> List[FoldWindow]:
    """Split ``n_bars`` into rolling train/test folds.

    Each fold needs ``train_bars + test_bars`` consecutive bars. The next fold
    starts ``step_bars`` later. ``warmup_start`` is the earliest index that may
    be prepended for indicator warm-up (clamped to 0); it is not scored.
    """
    if n_bars < config.train_bars + config.test_bars:
        raise ValueError(
            f"need at least {config.train_bars + config.test_bars} bars for one fold, got {n_bars}"
        )

    folds: List[FoldWindow] = []
    fold_id = 0
    start = 0
    while start + config.train_bars + config.test_bars <= n_bars:
        train_start = start
        train_end = start + config.train_bars
        test_start = train_end
        test_end = test_start + config.test_bars
        warmup_start = max(0, train_start - config.warmup_bars)
        folds.append(
            FoldWindow(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                warmup_start=warmup_start,
            )
        )
        fold_id += 1
        start += config.step_bars
    if not folds:
        raise ValueError("no walk-forward folds produced")
    return folds


def _aligned_dates(market: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    dates = sorted({d for df in market.values() for d in df.index})
    if not dates:
        raise ValueError("market has no dates")
    return pd.DatetimeIndex(dates)


def _slice_market(
    market: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    date_set = set(dates)
    for sym, df in market.items():
        part = df.loc[df.index.isin(date_set)].copy()
        if not part.empty:
            out[sym] = part
    if not out:
        raise ValueError("sliced market is empty")
    return out


def _expand_param_grid(base: CCT1Params, grid: Dict[str, Sequence[Any]]) -> List[CCT1Params]:
    if not grid:
        return [base]
    keys = sorted(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos: List[CCT1Params] = []
    for combo in product(*values):
        kwargs = {k: v for k, v in zip(keys, combo)}
        # None YAML null already handled upstream; keep types for known ints/floats
        combos.append(replace(base, **kwargs))
    return combos


def _metric_score(metrics: dict, name: str) -> float:
    val = metrics.get(name)
    if val is None:
        # Prefer total return fallback when sharpe unavailable (short windows)
        ret = metrics.get("total_return")
        return float(ret) if ret is not None else float("-inf")
    return float(val)


def _run_engine(
    market: Dict[str, pd.DataFrame],
    *,
    instruments: Dict[str, InstrumentSpec],
    params: CCT1Params,
    risk_limits: PortfolioRiskLimits,
    initial_equity: float,
    extras: Dict[str, Any],
) -> BacktestResult:
    engine = FuturesBacktestEngine(
        instruments=instruments,
        params=params,
        risk_limits=risk_limits,
        initial_equity=initial_equity,
        use_ma_exit=bool(extras.get("use_ma_exit", True)),
        use_channel_exit=bool(extras.get("use_channel_exit", True)),
        chandelier_atr_mult=float(extras.get("chandelier_atr_mult", 3.0)),
    )
    return engine.run(market)


def _normalize_equity(eq: pd.DataFrame) -> pd.DataFrame:
    """Normalize engine equity columns to Date / Portfolio Value."""
    out = eq.copy()
    if "Date" not in out.columns and "date" in out.columns:
        out = out.rename(columns={"date": "Date"})
    if "Portfolio Value" not in out.columns and "equity" in out.columns:
        out = out.rename(columns={"equity": "Portfolio Value"})
    if "Date" not in out.columns or "Portfolio Value" not in out.columns:
        raise ValueError("equity_curve missing Date / Portfolio Value columns")
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def _metrics_on_segment(result: BacktestResult, segment_dates: pd.DatetimeIndex) -> tuple[dict, pd.DataFrame, float, float]:
    """Restrict equity curve to ``segment_dates`` and recompute metrics."""
    eq = _normalize_equity(result.equity_curve)
    date_set = set(pd.to_datetime(segment_dates))
    seg = eq[eq["Date"].isin(date_set)].reset_index(drop=True)
    if seg.empty:
        return (
            {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None, "total_return": 0.0},
            seg,
            result.initial_equity,
            result.initial_equity,
        )
    initial = float(seg["Portfolio Value"].iloc[0])
    final = float(seg["Portfolio Value"].iloc[-1])
    total_return = final / initial - 1.0 if initial else 0.0
    from src.backtesting.metrics import PerformanceMetricsCalculator

    points = seg[["Date", "Portfolio Value"]].to_dict(orient="records")
    metrics = dict(PerformanceMetricsCalculator().compute_metrics(points))
    metrics["total_return"] = total_return
    return metrics, seg, initial, final


def _select_params_on_train(
    market: Dict[str, pd.DataFrame],
    window: FoldWindow,
    dates: pd.DatetimeIndex,
    *,
    base_params: CCT1Params,
    config: WalkForwardConfig,
    instruments: Dict[str, InstrumentSpec],
    risk_limits: PortfolioRiskLimits,
    initial_equity: float,
    extras: Dict[str, Any],
) -> tuple[CCT1Params, Optional[float]]:
    if config.selection == "default_params":
        return base_params, None

    candidates = _expand_param_grid(base_params, config.param_grid)
    if len(candidates) > 12:
        raise ValueError(
            f"tiny_grid too large ({len(candidates)} combos); keep documented grid small (≤12)"
        )

    train_dates = dates[window.warmup_start : window.train_end]
    train_score_dates = dates[window.train_start : window.train_end]
    train_market = _slice_market(market, train_dates)

    best_params = candidates[0]
    best_score = float("-inf")
    for params in candidates:
        result = _run_engine(
            train_market,
            instruments=instruments,
            params=params,
            risk_limits=risk_limits,
            initial_equity=initial_equity,
            extras=extras,
        )
        metrics, _, _, _ = _metrics_on_segment(result, train_score_dates)
        score = _metric_score(metrics, config.selection_metric)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score if best_score != float("-inf") else None


def run_walk_forward(
    market: Dict[str, pd.DataFrame],
    *,
    instruments: Dict[str, InstrumentSpec],
    params: CCT1Params,
    risk_limits: PortfolioRiskLimits,
    initial_equity: float,
    extras: Optional[Dict[str, Any]] = None,
    config: Optional[WalkForwardConfig] = None,
) -> WalkForwardResult:
    """Run rolling walk-forward folds and stitch OOS equity."""
    extras = extras or {}
    config = config or WalkForwardConfig()
    dates = _aligned_dates(market)
    windows = split_walk_forward_windows(len(dates), config)

    folds: List[FoldResult] = []
    agg_parts: List[pd.DataFrame] = []
    equity_cursor = float(initial_equity)

    for window in windows:
        chosen, sel_score = _select_params_on_train(
            market,
            window,
            dates,
            base_params=params,
            config=config,
            instruments=instruments,
            risk_limits=risk_limits,
            initial_equity=initial_equity,
            extras=extras,
        )
        # Run from warmup through test end so indicators + positions can form,
        # then score only the OOS test segment.
        run_dates = dates[window.warmup_start : window.test_end]
        test_dates = dates[window.test_start : window.test_end]
        fold_market = _slice_market(market, run_dates)
        result = _run_engine(
            fold_market,
            instruments=instruments,
            params=chosen,
            risk_limits=risk_limits,
            initial_equity=equity_cursor,
            extras=extras,
        )
        metrics, seg_eq, seg_initial, _seg_final = _metrics_on_segment(result, test_dates)
        # Scale segment equity onto continuing aggregate path
        if not seg_eq.empty and seg_initial:
            scale = equity_cursor / seg_initial
            stitched = seg_eq.copy()
            stitched["Portfolio Value"] = stitched["Portfolio Value"] * scale
            if "Cash" in stitched.columns:
                stitched["Cash"] = stitched["Cash"] * scale
            stitched["fold_id"] = window.fold_id
            agg_parts.append(stitched)
            equity_cursor = float(stitched["Portfolio Value"].iloc[-1])

        trades_in_test = [
            t
            for t in result.trades
            if pd.Timestamp(t.dt) >= test_dates[0] and pd.Timestamp(t.dt) <= test_dates[-1]
        ]
        folds.append(
            FoldResult(
                fold_id=window.fold_id,
                train_start=pd.Timestamp(dates[window.train_start]),
                train_end=pd.Timestamp(dates[window.train_end - 1]),
                test_start=pd.Timestamp(test_dates[0]),
                test_end=pd.Timestamp(test_dates[-1]),
                params=chosen,
                metrics=metrics,
                final_equity=float(equity_cursor),
                initial_equity=float(initial_equity),
                equity_curve=seg_eq,
                n_trades=len(trades_in_test),
                selection_score=sel_score,
            )
        )

    if agg_parts:
        aggregate_equity = pd.concat(agg_parts, ignore_index=True)
        # Drop duplicate dates across overlapping folds (keep first fold's path)
        aggregate_equity = aggregate_equity.drop_duplicates(subset=["Date"], keep="first")
        from src.backtesting.metrics import PerformanceMetricsCalculator

        points = aggregate_equity[["Date", "Portfolio Value"]].to_dict(orient="records")
        aggregate_metrics = dict(PerformanceMetricsCalculator().compute_metrics(points))
        i0 = float(aggregate_equity["Portfolio Value"].iloc[0])
        i1 = float(aggregate_equity["Portfolio Value"].iloc[-1])
        aggregate_metrics["total_return"] = (i1 / i0 - 1.0) if i0 else 0.0
        aggregate_metrics["n_folds"] = len(folds)
    else:
        aggregate_equity = pd.DataFrame(columns=["Date", "Portfolio Value", "fold_id"])
        aggregate_metrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "total_return": 0.0,
            "n_folds": 0,
        }

    return WalkForwardResult(
        folds=folds,
        aggregate_equity=aggregate_equity,
        aggregate_metrics=aggregate_metrics,
        config=config,
        selection=config.selection,
        initial_equity=initial_equity,
    )
