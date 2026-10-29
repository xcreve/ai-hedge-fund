import numpy as np
import pandas as pd

from src.futures.indicators import atr_wilder, donchian_high, donchian_low, sma


def test_sma_min_periods():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_donchian_excludes_current_bar():
    high = pd.Series([10.0, 12.0, 11.0, 15.0, 14.0])
    # window=2 → prior 2 highs, excluding today
    dh = donchian_high(high, 2)
    assert np.isnan(dh.iloc[1])
    # at index 3 (value 15): prior highs are 12, 11 → 12
    assert dh.iloc[3] == 12.0
    # at index 4: prior 11, 15 → 15
    assert dh.iloc[4] == 15.0


def test_donchian_low_excludes_current():
    low = pd.Series([9.0, 8.0, 7.5, 8.5, 7.0])
    dl = donchian_low(low, 2)
    assert dl.iloc[3] == 7.5
    assert dl.iloc[4] == 7.5


def test_atr_positive_after_warmup():
    n = 40
    close = pd.Series(np.linspace(100, 120, n))
    high = close + 1.0
    low = close - 1.0
    atr = atr_wilder(high, low, close, period=14)
    assert atr.iloc[13:].notna().all()
    assert (atr.iloc[13:] > 0).all()
