"""CLI entry for CCT-1 futures research backtests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.futures.backtest.engine import FuturesBacktestEngine
from src.futures.backtest.report import summary_text, write_reports
from src.futures.backtest.walkforward import run_walk_forward, walk_forward_config_from_mapping
from src.futures.backtest.walkforward_report import (
    summary_text as walkforward_summary_text,
    write_walkforward_reports,
)
from src.futures.data.csv_loader import load_directory
from src.futures.data.schema import InstrumentSpec, instrument_from_mapping
from src.futures.execution.portfolio_risk import PortfolioRiskLimits
from src.futures.signals.cct1 import CCT1Params

# Re-export for tests / discoverability
__all__ = ["main", "build_parser", "cmd_run", "cmd_walkforward"]

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PKG_DIR / "configs" / "cct1_research.yaml"
DEFAULT_WALKFORWARD_CONFIG = PKG_DIR / "configs" / "cct1_walkforward.yaml"
DEFAULT_INSTRUMENTS = PKG_DIR / "configs" / "instruments.yaml"
DEFAULT_DATA = Path("data/futures/sample")
DEFAULT_WALKFORWARD_OUT = Path("artifacts/futures/walkforward")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _params_from_config(cfg: Dict[str, Any]) -> tuple[CCT1Params, PortfolioRiskLimits, float, dict]:
    s = cfg.get("strategy", {})
    r = cfg.get("risk", {})
    adx = s.get("adx_threshold", 20)
    time_stop = s.get("time_stop_bars", 8)
    params = CCT1Params(
        n1=int(s.get("n1", 20)),
        n2=int(s.get("n2", 10)),
        sma_fast=int(s.get("sma_fast", 20)),
        sma_slow=int(s.get("sma_slow", 50)),
        atr_period=int(s.get("atr_period", 20)),
        atr_pct_lookback=int(s.get("atr_pct_lookback", 100)),
        atr_pct_threshold=float(s.get("atr_pct_threshold", 40)),
        adx_period=int(s.get("adx_period", 14)),
        adx_threshold=None if adx is None else float(adx),
        stop_atr_mult=float(s.get("stop_atr_mult", 2.0)),
        pyramid_atr_step=float(s.get("pyramid_atr_step", 0.5)),
        max_pyramid_adds=int(s.get("max_pyramid_adds", 3)),
        time_stop_bars=None if time_stop is None else int(time_stop),
    )
    limits = PortfolioRiskLimits(
        risk_per_trade=float(r.get("risk_per_trade", 0.0075)),
        max_symbol_margin_pct=float(r.get("max_symbol_margin_pct", 0.20)),
        max_portfolio_margin_pct=float(r.get("max_portfolio_margin_pct", 0.50)),
        max_symbols=int(r.get("max_symbols", 6)),
        daily_loss_halt_pct=float(r.get("daily_loss_halt_pct", 0.03)),
        max_drawdown_delever_pct=float(r.get("max_drawdown_delever_pct", 0.15)),
        sector_risk_mult=float(r.get("sector_risk_mult", 1.5)),
        cooldown_bars=int(r.get("cooldown_bars", 1)),
    )
    extras = {
        "use_ma_exit": bool(s.get("use_ma_exit", True)),
        "use_channel_exit": bool(s.get("use_channel_exit", True)),
        "chandelier_atr_mult": float(s.get("chandelier_atr_mult", 3.0)),
    }
    equity = float(cfg.get("initial_equity", 1_000_000))
    return params, limits, equity, extras


def _load_instruments(path: Path, symbols: Optional[list[str]] = None) -> Dict[str, InstrumentSpec]:
    raw = _load_yaml(path)
    table = raw.get("instruments", raw)
    out: Dict[str, InstrumentSpec] = {}
    for sym, spec in table.items():
        key = str(sym).upper()
        if symbols and key not in {s.upper() for s in symbols}:
            continue
        out[key] = instrument_from_mapping(key, spec)
    return out


def _prepare_market(args: argparse.Namespace, config_path: Path) -> tuple[dict, dict, CCT1Params, PortfolioRiskLimits, float, dict, dict]:
    instruments_path = Path(args.instruments)
    data_dir = Path(args.data_dir)
    cfg = _load_yaml(config_path)
    params, limits, equity, extras = _params_from_config(cfg)
    market = load_directory(data_dir, symbols=args.symbols)
    instruments = _load_instruments(instruments_path, symbols=list(market.keys()))
    missing = [s for s in market if s not in instruments]
    if missing:
        raise SystemExit(f"instruments.yaml missing specs for: {missing}")
    return market, instruments, params, limits, equity, extras, cfg


def cmd_run(args: argparse.Namespace) -> int:
    market, instruments, params, limits, equity, extras, _cfg = _prepare_market(args, Path(args.config))

    engine = FuturesBacktestEngine(
        instruments=instruments,
        params=params,
        risk_limits=limits,
        initial_equity=equity,
        use_ma_exit=extras["use_ma_exit"],
        use_channel_exit=extras["use_channel_exit"],
        chandelier_atr_mult=extras["chandelier_atr_mult"],
    )
    result = engine.run(market)
    print(summary_text(result))

    if args.out_dir:
        paths = write_reports(result, args.out_dir)
        print(f"Wrote reports to {args.out_dir}: {', '.join(p.name for p in paths.values())}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    market, instruments, params, limits, equity, extras, cfg = _prepare_market(args, Path(args.config))
    wf_cfg = walk_forward_config_from_mapping(cfg)
    if args.selection:
        from dataclasses import replace

        wf_cfg = replace(wf_cfg, selection=args.selection)

    result = run_walk_forward(
        market,
        instruments=instruments,
        params=params,
        risk_limits=limits,
        initial_equity=equity,
        extras=extras,
        config=wf_cfg,
    )
    print(walkforward_summary_text(result))

    out_dir = args.out_dir or str(DEFAULT_WALKFORWARD_OUT)
    paths = write_walkforward_reports(result, out_dir)
    print(f"Wrote walk-forward reports to {out_dir}: {', '.join(p.name for p in paths.values())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="futures-backtest", description="CCT-1 futures research backtest")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a day-bar backtest on continuous CSV data")
    run.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA), help="Directory of continuous CSVs")
    run.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Strategy/risk YAML")
    run.add_argument("--instruments", type=str, default=str(DEFAULT_INSTRUMENTS), help="Instrument specs YAML")
    run.add_argument("--symbols", nargs="*", default=None, help="Optional symbol filter")
    run.add_argument("--out-dir", type=str, default=None, help="Optional report output directory")
    run.set_defaults(func=cmd_run)

    wf = sub.add_parser("walkforward", help="Rolling train/test walk-forward evaluation (spec §7)")
    wf.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA), help="Directory of continuous CSVs")
    wf.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_WALKFORWARD_CONFIG),
        help="Walk-forward + strategy YAML (default: cct1_walkforward.yaml)",
    )
    wf.add_argument("--instruments", type=str, default=str(DEFAULT_INSTRUMENTS), help="Instrument specs YAML")
    wf.add_argument("--symbols", nargs="*", default=None, help="Optional symbol filter")
    wf.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_WALKFORWARD_OUT),
        help="Report output directory (default: artifacts/futures/walkforward)",
    )
    wf.add_argument(
        "--selection",
        type=str,
        choices=["default_params", "tiny_grid"],
        default=None,
        help="Override walkforward.selection from YAML",
    )
    wf.set_defaults(func=cmd_walkforward)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
