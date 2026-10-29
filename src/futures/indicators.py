"""Technical indicators for CCT-1 (self-contained; does not touch agents/technicals)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Wilder ATR via ewm with alpha=1/period (adjust=False)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def donchian_high(high: pd.Series, window: int) -> pd.Series:
    """Highest high over the prior `window` bars (excludes current bar)."""
    return high.shift(1).rolling(window=window, min_periods=window).max()


def donchian_low(low: pd.Series, window: int) -> pd.Series:
    """Lowest low over the prior `window` bars (excludes current bar)."""
    return low.shift(1).rolling(window=window, min_periods=window).min()


def atr_percentile(atr: pd.Series, close: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling percentile of ATR/close over the past `lookback` observations."""
    ratio = atr / close.replace(0, np.nan)
    return ratio.rolling(window=lookback, min_periods=max(20, lookback // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0,
        raw=False,
    )


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=alpha, min_periods=period, adjust=False
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=alpha, min_periods=period, adjust=False
    ).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
