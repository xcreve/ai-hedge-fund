"""CSV schema and instrument metadata for continuous futures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS = ("open_interest",)


@dataclass(frozen=True)
class InstrumentSpec:
    """Per-symbol contract parameters used for sizing and costs."""

    symbol: str
    multiplier: float
    tick_size: float
    margin_rate: float
    sector: str = "unknown"
    currency_factor: float = 1.0
    commission_per_lot: float = 0.0
    slippage_ticks: float = 1.0

    def notional(self, price: float, lots: float) -> float:
        return abs(lots) * price * self.multiplier * self.currency_factor

    def margin(self, price: float, lots: float) -> float:
        return self.notional(price, lots) * self.margin_rate

    def dollar_per_point(self) -> float:
        return self.multiplier * self.currency_factor


def instrument_from_mapping(symbol: str, raw: Mapping) -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol,
        multiplier=float(raw["multiplier"]),
        tick_size=float(raw["tick_size"]),
        margin_rate=float(raw.get("margin_rate", 0.12)),
        sector=str(raw.get("sector", "unknown")),
        currency_factor=float(raw.get("currency_factor", 1.0)),
        commission_per_lot=float(raw.get("commission_per_lot", 0.0)),
        slippage_ticks=float(raw.get("slippage_ticks", 1.0)),
    )
