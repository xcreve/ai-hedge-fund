"""ATR risk-based futures lot sizing."""

from __future__ import annotations

import math
from typing import Sequence

from src.futures.data.schema import InstrumentSpec

# Pyramid lot multipliers relative to Q0 (first unit + up to K adds)
DEFAULT_PYRAMID_WEIGHTS: Sequence[float] = (1.0, 0.75, 0.5, 0.5)


def size_initial_lots(
    *,
    equity: float,
    atr_points: float,
    instrument: InstrumentSpec,
    risk_per_trade: float,
) -> int:
    """Q0 = floor(equity * risk_per_trade / (ATR * multiplier))."""
    if equity <= 0 or atr_points <= 0 or risk_per_trade <= 0:
        return 0
    dollar_risk = atr_points * instrument.dollar_per_point()
    if dollar_risk <= 0:
        return 0
    q0 = math.floor(equity * risk_per_trade / dollar_risk)
    return max(int(q0), 0)


def lots_for_unit(q0: int, unit_index: int, weights: Sequence[float] = DEFAULT_PYRAMID_WEIGHTS) -> int:
    """Lots for pyramid unit ``unit_index`` (0 = initial entry).

    Initial unit is at least 1 when ``q0 >= 1``. Subsequent units may be 0
    if the scaled weight floors to zero.
    """
    if q0 <= 0 or unit_index < 0 or unit_index >= len(weights):
        return 0
    raw = int(math.floor(q0 * float(weights[unit_index])))
    if unit_index == 0:
        return max(raw, 1)
    return max(raw, 0)
