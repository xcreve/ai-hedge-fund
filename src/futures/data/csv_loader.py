"""Load continuous-contract OHLC CSVs for futures backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .schema import REQUIRED_COLUMNS


def load_continuous_csv(path: str | Path, symbol: Optional[str] = None) -> pd.DataFrame:
    """Load one continuous contract CSV into a normalized OHLCV frame.

    Expected columns (case-insensitive): date, open, high, low, close, volume.
    Optional: open_interest.
    Index is a DatetimeIndex named ``date``, sorted ascending, unique.
    """
    path = Path(path)
    if symbol is None:
        symbol = path.stem.upper()

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date")
    df.index.name = "date"
    df["symbol"] = symbol

    # Basic sanity: high >= low, high >= open/close, low <= open/close
    bad = (df["high"] < df["low"]) | (df["high"] < df[["open", "close"]].max(axis=1)) | (
        df["low"] > df[["open", "close"]].min(axis=1)
    )
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} rows violate OHLC consistency")

    return df


def load_directory(
    data_dir: str | Path,
    symbols: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all ``*.csv`` files under ``data_dir`` keyed by symbol (filename stem)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    wanted = {s.upper() for s in symbols} if symbols else None
    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        symbol = path.stem.upper()
        if wanted is not None and symbol not in wanted:
            continue
        out[symbol] = load_continuous_csv(path, symbol=symbol)
    if not out:
        raise FileNotFoundError(f"no CSV contracts found in {data_dir}")
    return out
