"""CCT-1 Contest-Common Trend signals (filter F, breakout S, volatility V)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.futures import indicators as ind


@dataclass(frozen=True)
class CCT1Params:
    n1: int = 20
    n2: int = 10
    sma_fast: int = 20
    sma_slow: int = 50
    atr_period: int = 20
    atr_pct_lookback: int = 100
    atr_pct_threshold: float = 40.0
    adx_period: int = 14
    adx_threshold: Optional[float] = 20.0  # None disables ADX filter
    stop_atr_mult: float = 2.0
    pyramid_atr_step: float = 0.5
    max_pyramid_adds: int = 3
    time_stop_bars: Optional[int] = 8  # research default; None disables


def compute_cct1_signals(ohlc: pd.DataFrame, params: Optional[CCT1Params] = None) -> pd.DataFrame:
    """Return a copy of ``ohlc`` with indicator and boolean signal columns.

    Entry booleans are evaluated on the **close** of day T and must be acted on
    at the **next open** by the execution engine (model A — no look-ahead).
    """
    params = params or CCT1Params()
    df = ohlc.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["sma_fast"] = ind.sma(close, params.sma_fast)
    df["sma_slow"] = ind.sma(close, params.sma_slow)
    df["atr"] = ind.atr_wilder(high, low, close, params.atr_period)
    df["donchian_high_n1"] = ind.donchian_high(high, params.n1)
    df["donchian_low_n1"] = ind.donchian_low(low, params.n1)
    df["donchian_high_n2"] = ind.donchian_high(high, params.n2)
    df["donchian_low_n2"] = ind.donchian_low(low, params.n2)
    df["atr_pctile"] = ind.atr_percentile(df["atr"], close, params.atr_pct_lookback)
    df["adx"] = ind.adx(high, low, close, params.adx_period)

    f_bull = (df["sma_fast"] > df["sma_slow"]) & (close > df["sma_slow"])
    f_bear = (df["sma_fast"] < df["sma_slow"]) & (close < df["sma_slow"])
    s_long = close > df["donchian_high_n1"]
    s_short = close < df["donchian_low_n1"]
    v_ok = df["atr_pctile"] >= params.atr_pct_threshold

    if params.adx_threshold is None:
        adx_ok = pd.Series(True, index=df.index)
    else:
        adx_ok = df["adx"] >= params.adx_threshold

    long_ok = f_bull & s_long & v_ok & adx_ok
    short_ok = f_bear & s_short & v_ok & adx_ok
    # Same-day long/short mutual exclusion
    conflict = long_ok & short_ok
    df["long_ok"] = long_ok & ~conflict
    df["short_ok"] = short_ok & ~conflict
    df["f_bull"] = f_bull
    df["f_bear"] = f_bear
    df["v_ok"] = v_ok
    df["adx_ok"] = adx_ok

    # Exit triggers evaluated on close (engine fills next open unless hard stop)
    df["exit_long_channel"] = close < df["donchian_low_n2"]
    df["exit_short_channel"] = close > df["donchian_high_n2"]
    df["exit_long_ma"] = close < df["sma_fast"]
    df["exit_short_ma"] = close > df["sma_fast"]

    return df
