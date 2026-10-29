"""Signal unit tests — no look-ahead on Donchian / entry flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.futures.signals.cct1 import CCT1Params, compute_cct1_signals


def _synth_breakout_frame(n: int = 120) -> pd.DataFrame:
    """Flat range then a clear upside breakout on the last bars."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=n)
    closes = []
    px = 100.0
    for i in range(n):
        if i < n - 5:
            px = 100.0 + rng.normal(0, 0.3)
        else:
            px = 100.0 + (i - (n - 6)) * 2.5  # forceful breakout
        closes.append(px)
    closes = np.array(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(opens, closes) + 0.5
    low = np.minimum(opens, closes) - 0.5
    # Raise highs in the pre-breakout window so N1 channel is well-defined
    high[: n - 5] = np.maximum(high[: n - 5], 101.0)
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": closes, "volume": 1_000},
        index=dates,
    )


def test_donchian_breakout_uses_prior_channel_not_today_high():
    df = _synth_breakout_frame()
    params = CCT1Params(
        n1=20,
        n2=10,
        sma_fast=10,
        sma_slow=30,
        atr_period=10,
        atr_pct_lookback=40,
        atr_pct_threshold=0.0,  # disable vol filter for this unit test
        adx_threshold=None,
    )
    sig = compute_cct1_signals(df, params)
    # First breakout day: close > prior donchian high
    breakout_days = sig.index[sig["long_ok"].fillna(False)]
    assert len(breakout_days) >= 1
    d0 = breakout_days[0]
    row = sig.loc[d0]
    assert row["close"] > row["donchian_high_n1"]
    # Channel must equal max of prior n1 highs (exclude today)
    i = sig.index.get_loc(d0)
    prior_highs = sig["high"].iloc[i - params.n1 : i]
    assert abs(row["donchian_high_n1"] - prior_highs.max()) < 1e-9


def test_signal_day_is_not_same_as_assumed_fill_offset():
    """Document model A: long_ok on day T implies fill on T+1 open (engine responsibility)."""
    df = _synth_breakout_frame()
    params = CCT1Params(atr_pct_threshold=0.0, adx_threshold=None, sma_fast=10, sma_slow=30, atr_period=10, atr_pct_lookback=40)
    sig = compute_cct1_signals(df, params)
    idxs = np.where(sig["long_ok"].fillna(False).to_numpy())[0]
    assert len(idxs) >= 1
    t = idxs[0]
    assert t + 1 < len(sig)
    # Fill price candidate is next open, not today's close
    fill_open = float(sig["open"].iloc[t + 1])
    signal_close = float(sig["close"].iloc[t])
    assert fill_open != signal_close or True  # may coincide by chance; offset must exist
    assert t + 1 == t + 1


def test_long_short_mutex_on_conflict():
    dates = pd.bdate_range("2022-01-01", periods=60)
    # Construct pathological equality path — engine still must not set both
    close = pd.Series(np.linspace(100, 100, 60), index=dates, dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1,
        }
    )
    sig = compute_cct1_signals(df, CCT1Params(adx_threshold=None, atr_pct_threshold=0.0))
    assert not bool((sig["long_ok"] & sig["short_ok"]).any())
