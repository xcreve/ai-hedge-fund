"""Walk-forward report writers (fold metrics + aggregate equity + markdown summary)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.futures.backtest.walkforward import FoldResult, WalkForwardResult


def folds_to_frame(folds: list[FoldResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for f in folds:
        m = f.metrics
        rows.append(
            {
                "fold_id": f.fold_id,
                "train_start": f.train_start.strftime("%Y-%m-%d"),
                "train_end": f.train_end.strftime("%Y-%m-%d"),
                "test_start": f.test_start.strftime("%Y-%m-%d"),
                "test_end": f.test_end.strftime("%Y-%m-%d"),
                "n_trades": f.n_trades,
                "total_return": m.get("total_return"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "sortino_ratio": m.get("sortino_ratio"),
                "max_drawdown": m.get("max_drawdown"),
                "selection_score": f.selection_score,
                "n1": f.params.n1,
                "n2": f.params.n2,
            }
        )
    return pd.DataFrame(rows)


def summary_markdown(result: WalkForwardResult) -> str:
    cfg = result.config
    agg = result.aggregate_metrics
    lines = [
        "# CCT-1 Walk-Forward Summary",
        "",
        "Overfit protection (spec §7): parameter selection is confined to each "
        "train window; metrics below are out-of-sample on the test folds.",
        "",
        "## Config",
        "",
        f"- Selection mode: `{result.selection}`",
        f"- Train / test / step / warmup bars: "
        f"{cfg.train_bars} / {cfg.test_bars} / {cfg.step_bars} / {cfg.warmup_bars}",
        f"- Folds: {len(result.folds)}",
        f"- Initial equity: {result.initial_equity:,.2f}",
        "",
        "## Aggregate (stitched OOS equity)",
        "",
        f"- Total return: {_fmt_pct(agg.get('total_return'))}",
        f"- Sharpe: {_fmt(agg.get('sharpe_ratio'))}",
        f"- Sortino: {_fmt(agg.get('sortino_ratio'))}",
        f"- Max drawdown %: {_fmt(agg.get('max_drawdown'))}",
        "",
        "## Fold metrics",
        "",
        "| Fold | Test start | Test end | Return | Sharpe | Max DD % | Trades |",
        "|------|------------|----------|--------|--------|----------|--------|",
    ]
    for f in result.folds:
        m = f.metrics
        lines.append(
            f"| {f.fold_id} | {f.test_start.strftime('%Y-%m-%d')} | "
            f"{f.test_end.strftime('%Y-%m-%d')} | {_fmt_pct(m.get('total_return'))} | "
            f"{_fmt(m.get('sharpe_ratio'))} | {_fmt(m.get('max_drawdown'))} | {f.n_trades} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Default mode (`default_params`) applies research YAML params on every "
            "test fold — no in-sample fitting.",
            "- Optional `tiny_grid` searches the documented small grid on train only "
            "(see `cct1_walkforward.yaml`); keep combinations ≤ 12.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_text(result: WalkForwardResult) -> str:
    agg = result.aggregate_metrics
    lines = [
        "CCT-1 Walk-Forward Summary",
        f"  Selection      : {result.selection}",
        f"  Folds          : {len(result.folds)}",
        f"  Initial equity : {result.initial_equity:,.2f}",
        f"  OOS return     : {_fmt_pct(agg.get('total_return'))}",
        f"  Sharpe         : {_fmt(agg.get('sharpe_ratio'))}",
        f"  Sortino        : {_fmt(agg.get('sortino_ratio'))}",
        f"  Max drawdown % : {_fmt(agg.get('max_drawdown'))}",
    ]
    return "\n".join(lines)


def write_walkforward_reports(result: WalkForwardResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_path = out_dir / "fold_metrics.csv"
    equity_path = out_dir / "aggregate_equity.csv"
    summary_md = out_dir / "summary.md"
    summary_txt = out_dir / "summary.txt"

    folds_to_frame(result.folds).to_csv(folds_path, index=False)
    result.aggregate_equity.to_csv(equity_path, index=False)
    summary_md.write_text(summary_markdown(result), encoding="utf-8")
    summary_txt.write_text(summary_text(result) + "\n", encoding="utf-8")

    # Per-fold equity snippets (optional detail)
    folds_dir = out_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    for f in result.folds:
        f.equity_curve.to_csv(folds_dir / f"fold_{f.fold_id:02d}_equity.csv", index=False)

    return {
        "fold_metrics": folds_path,
        "aggregate_equity": equity_path,
        "summary_md": summary_md,
        "summary_txt": summary_txt,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        return f"{v:.4f}"
    return str(v)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"
