"""Portfolio-level risk budgets for research / contest modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class PortfolioRiskLimits:
    risk_per_trade: float = 0.0075
    max_symbol_margin_pct: float = 0.20
    max_portfolio_margin_pct: float = 0.50
    max_symbols: int = 6
    daily_loss_halt_pct: float = 0.03
    max_drawdown_delever_pct: float = 0.15
    sector_risk_mult: float = 1.5
    cooldown_bars: int = 1


@dataclass
class PortfolioRiskState:
    limits: PortfolioRiskLimits
    peak_equity: float
    day_start_equity: float
    risk_budget_scale: float = 1.0
    halt_new_entries: bool = False
    margin_by_symbol: Dict[str, float] = field(default_factory=dict)
    sector_risk_used: Dict[str, float] = field(default_factory=dict)
    symbol_sector: Dict[str, str] = field(default_factory=dict)

    def on_new_day(self, equity: float) -> None:
        self.day_start_equity = equity
        self.halt_new_entries = False
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd <= self.limits.max_drawdown_delever_pct * 0.5:
                self.risk_budget_scale = 1.0

    def update_equity(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.day_start_equity > 0:
            day_dd = (self.day_start_equity - equity) / self.day_start_equity
            if day_dd >= self.limits.daily_loss_halt_pct:
                self.halt_new_entries = True
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd >= self.limits.max_drawdown_delever_pct:
                self.risk_budget_scale = 0.5

    def effective_risk_per_trade(self) -> float:
        return self.limits.risk_per_trade * self.risk_budget_scale

    def open_symbol_count(self) -> int:
        return sum(1 for m in self.margin_by_symbol.values() if m > 0)

    def total_margin(self) -> float:
        return float(sum(self.margin_by_symbol.values()))

    def can_open(
        self,
        *,
        symbol: str,
        sector: str,
        new_margin: float,
        equity: float,
        trade_risk_dollars: float,
        is_new_symbol: bool,
    ) -> bool:
        if self.halt_new_entries or equity <= 0 or new_margin <= 0:
            return False
        if is_new_symbol and symbol not in self.margin_by_symbol and self.open_symbol_count() >= self.limits.max_symbols:
            return False
        symbol_margin = self.margin_by_symbol.get(symbol, 0.0) + new_margin
        if symbol_margin > equity * self.limits.max_symbol_margin_pct:
            return False
        if self.total_margin() + new_margin > equity * self.limits.max_portfolio_margin_pct:
            return False
        single = equity * self.effective_risk_per_trade()
        if single > 0:
            used = self.sector_risk_used.get(sector, 0.0) + trade_risk_dollars
            if used > single * self.limits.sector_risk_mult + 1e-9:
                return False
        return True

    def register_open(self, symbol: str, sector: str, margin: float, trade_risk: float) -> None:
        self.margin_by_symbol[symbol] = self.margin_by_symbol.get(symbol, 0.0) + margin
        self.sector_risk_used[sector] = self.sector_risk_used.get(sector, 0.0) + trade_risk
        self.symbol_sector[symbol] = sector

    def set_symbol_margin(self, symbol: str, margin: float) -> None:
        if margin <= 0:
            sector = self.symbol_sector.pop(symbol, None)
            self.margin_by_symbol.pop(symbol, None)
            if sector is not None:
                # Recompute sector risk from remaining open symbols in sector.
                remaining = [
                    s for s, sec in self.symbol_sector.items() if sec == sector
                ]
                if not remaining:
                    self.sector_risk_used[sector] = 0.0
        else:
            self.margin_by_symbol[symbol] = margin
