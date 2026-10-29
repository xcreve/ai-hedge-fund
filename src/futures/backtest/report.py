"""Backtest reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.futures.backtest.engine import BacktestResult, TradeRecord


def trades_to_frame(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=["symbol", "dt", "side", "lots", "price", "reason", "pnl"]
        )
    return pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "dt": t.dt,
                "side": t.side,
                "lots": t.lots,
                "price": t.price,
                "reason": t.reason,
                "pnl": t.pnl,
            }
            for t in trades
        ]
    )


def summary_text(result: BacktestResult) -> str:
    m = result.metrics
    n_trades = len(result.trades)
    ret = (result.final_equity / result.initial_equity - 1.0) * 100.0
    lines = [
        "CCT-1 Futures Backtest Summary",
        f"  Initial equity : {result.initial_equity:,.2f}",
        f"  Final equity   : {result.final_equity:,.2f}",
        f"  Total return   : {ret:.2f}%",
        f"  Trades         : {n_trades}",
        f"  Sharpe         : {m.get('sharpe_ratio')}",
        f"  Sortino        : {m.get('sortino_ratio')}",
        f"  Max drawdown % : {m.get('max_drawdown')}",
        f"  Max DD date    : {m.get('max_drawdown_date')}",
    ]
    return "\n".join(lines)


def write_reports(result: BacktestResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = out_dir / "equity_curve.csv"
    trades_path = out_dir / "trades.csv"
    summary_path = out_dir / "summary.txt"

    result.equity_curve.to_csv(equity_path, index=False)
    trades_to_frame(result.trades).to_csv(trades_path, index=False)
    summary_path.write_text(summary_text(result) + "\n", encoding="utf-8")
    return {"equity": equity_path, "trades": trades_path, "summary": summary_path}
