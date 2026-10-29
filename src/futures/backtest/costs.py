"""Futures commission and slippage cost model."""

from __future__ import annotations

from src.futures.data.schema import InstrumentSpec


def round_trip_cost_per_lot(instrument: InstrumentSpec, price: float) -> float:
    """Cash cost for opening *or* closing one lot (one-way).

    Slippage valued as ``slippage_ticks * tick_size * multiplier``.
    """
    slip = instrument.slippage_ticks * instrument.tick_size * instrument.dollar_per_point()
    return float(instrument.commission_per_lot + slip)


def apply_slippage_price(price: float, side: int, instrument: InstrumentSpec) -> float:
    """Adjust fill price for slippage. ``side``: +1 buy/cover, -1 sell/short."""
    slip = instrument.slippage_ticks * instrument.tick_size
    return float(price + side * slip)
