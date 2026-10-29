"""Day-bar futures backtest engine (execution model A).

Signal confirmed on close → fill at next open.
Hard stops fill when high/low penetrates the stop level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.futures.backtest import costs as costmod
from src.futures.data.schema import InstrumentSpec
from src.futures.execution.portfolio_risk import PortfolioRiskLimits, PortfolioRiskState
from src.futures.execution.position_sizer import lots_for_unit, size_initial_lots
from src.futures.signals.cct1 import CCT1Params, compute_cct1_signals


@dataclass
class TradeRecord:
    symbol: str
    dt: datetime
    side: str  # buy / sell / short / cover
    lots: int
    price: float
    reason: str
    pnl: float = 0.0


@dataclass
class Position:
    symbol: str
    direction: int  # +1 long, -1 short
    lots: int
    avg_price: float
    entry_atr: float
    stop: float
    units: int  # number of pyramid units filled (1..K+1)
    q0: int
    bars_held: int = 0
    extreme: float = 0.0  # highest high (long) / lowest low (short) since entry
    last_add_price: float = 0.0
    cooldown_until_i: int = -1


@dataclass
class PendingOrder:
    symbol: str
    action: str  # enter_long / enter_short / exit / add
    lots: int
    reason: str
    unit_index: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[TradeRecord]
    metrics: dict
    final_equity: float
    initial_equity: float


@dataclass
class FuturesBacktestEngine:
    instruments: Dict[str, InstrumentSpec]
    params: CCT1Params = field(default_factory=CCT1Params)
    risk_limits: PortfolioRiskLimits = field(default_factory=PortfolioRiskLimits)
    initial_equity: float = 1_000_000.0
    use_ma_exit: bool = True
    use_channel_exit: bool = True
    chandelier_atr_mult: float = 3.0

    def run(self, market: Dict[str, pd.DataFrame]) -> BacktestResult:
        if not market:
            raise ValueError("market data is empty")

        signals: Dict[str, pd.DataFrame] = {
            sym: compute_cct1_signals(df, self.params) for sym, df in market.items()
        }
        # Align on union of trading days
        all_dates = sorted({d for df in signals.values() for d in df.index})
        cash = float(self.initial_equity)
        positions: Dict[str, Position] = {}
        pending: List[PendingOrder] = []
        trades: List[TradeRecord] = []
        equity_rows: List[dict] = []
        last_exit_i: Dict[str, int] = {}

        risk = PortfolioRiskState(
            limits=self.risk_limits,
            peak_equity=self.initial_equity,
            day_start_equity=self.initial_equity,
        )

        for i, dt in enumerate(all_dates):
            # Mark day start equity before fills
            mtm0 = cash + self._unrealized(positions, signals, dt)
            risk.on_new_day(mtm0)

            # 1) Process pending orders at open
            still_pending: List[PendingOrder] = []
            for order in pending:
                df = signals.get(order.symbol)
                if df is None or dt not in df.index:
                    still_pending.append(order)
                    continue
                row = df.loc[dt]
                open_px = float(row["open"])
                atr = float(row["atr"]) if pd.notna(row["atr"]) else None
                cash = self._fill_order(
                    order=order,
                    open_px=open_px,
                    atr=atr,
                    dt=dt,
                    cash=cash,
                    positions=positions,
                    trades=trades,
                    risk=risk,
                    signals=signals,
                )
            pending = still_pending

            # 2) Intraday hard stops via high/low penetration
            for sym in list(positions.keys()):
                df = signals[sym]
                if dt not in df.index:
                    continue
                row = df.loc[dt]
                pos = positions[sym]
                hit, fill_px = self._stop_hit(pos, float(row["open"]), float(row["high"]), float(row["low"]))
                if hit:
                    cash = self._close_position(
                        pos=pos,
                        fill_px=fill_px,
                        dt=dt,
                        cash=cash,
                        positions=positions,
                        trades=trades,
                        risk=risk,
                        reason="stop",
                    )
                    last_exit_i[sym] = i

            pending_syms = {o.symbol for o in pending}

            # 3) Update trailing extremes / chandelier-tightened stops; pyramid checks; soft exits
            for sym, df in signals.items():
                if dt not in df.index:
                    continue
                row = df.loc[dt]
                if sym in positions:
                    if sym in pending_syms:
                        continue  # already have exit/add scheduled for next open
                    pos = positions[sym]
                    pos.bars_held += 1
                    hi, lo = float(row["high"]), float(row["low"])
                    atr = float(row["atr"]) if pd.notna(row["atr"]) else pos.entry_atr
                    if pos.direction > 0:
                        pos.extreme = max(pos.extreme, hi)
                        avg_stop = pos.avg_price - self.params.stop_atr_mult * atr
                        chandelier = pos.extreme - self.chandelier_atr_mult * atr
                        pos.stop = max(pos.stop, avg_stop, chandelier)
                    else:
                        pos.extreme = min(pos.extreme, lo) if pos.extreme != 0 else lo
                        avg_stop = pos.avg_price + self.params.stop_atr_mult * atr
                        chandelier = pos.extreme + self.chandelier_atr_mult * atr
                        pos.stop = min(pos.stop, avg_stop, chandelier)

                    # Soft exits on close → next open
                    exit_reason = self._soft_exit_reason(pos, row)
                    if exit_reason:
                        pending.append(
                            PendingOrder(symbol=sym, action="exit", lots=pos.lots, reason=exit_reason)
                        )
                        pending_syms.add(sym)
                    else:
                        add = self._pyramid_pending(pos, row)
                        if add is not None:
                            pending.append(add)
                            pending_syms.add(sym)
                else:
                    if sym in pending_syms:
                        continue
                    cooldown = self.risk_limits.cooldown_bars
                    if last_exit_i.get(sym, -10**9) + cooldown >= i:
                        continue
                    if bool(row.get("long_ok", False)):
                        pending.append(
                            PendingOrder(
                                symbol=sym, action="enter_long", lots=0, reason="breakout_long", unit_index=0
                            )
                        )
                        pending_syms.add(sym)
                    elif bool(row.get("short_ok", False)):
                        pending.append(
                            PendingOrder(
                                symbol=sym, action="enter_short", lots=0, reason="breakout_short", unit_index=0
                            )
                        )
                        pending_syms.add(sym)

            # 4) Mark-to-market equity
            equity = cash + self._unrealized(positions, signals, dt)
            risk.update_equity(equity)
            # Refresh margins for open positions
            for sym, pos in positions.items():
                inst = self.instruments[sym]
                px = self._mark_price(signals, sym, dt)
                risk.set_symbol_margin(sym, inst.margin(px, pos.lots))

            long_exp = short_exp = 0.0
            for sym, pos in positions.items():
                inst = self.instruments[sym]
                px = self._mark_price(signals, sym, dt)
                notional = inst.notional(px, pos.lots)
                if pos.direction > 0:
                    long_exp += notional
                else:
                    short_exp += notional

            equity_rows.append(
                {
                    "Date": dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                    "Portfolio Value": float(equity),
                    "Cash": float(cash),
                    "Long Exposure": float(long_exp),
                    "Short Exposure": float(short_exp),
                    "Gross Exposure": float(long_exp + short_exp),
                    "Net Exposure": float(long_exp - short_exp),
                }
            )

        equity_curve = pd.DataFrame(equity_rows)
        metrics = self._compute_metrics(equity_curve)
        final_eq = float(equity_curve["Portfolio Value"].iloc[-1]) if len(equity_curve) else self.initial_equity
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            final_equity=final_eq,
            initial_equity=self.initial_equity,
        )

    def _compute_metrics(self, equity_curve: pd.DataFrame) -> dict:
        from src.backtesting.metrics import PerformanceMetricsCalculator

        if equity_curve.empty:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
        points = equity_curve.to_dict(orient="records")
        return dict(PerformanceMetricsCalculator().compute_metrics(points))

    def _mark_price(self, signals: Dict[str, pd.DataFrame], sym: str, dt) -> float:
        df = signals[sym]
        if dt in df.index:
            return float(df.loc[dt, "close"])
        # last available
        hist = df.loc[:dt]
        return float(hist["close"].iloc[-1])

    def _unrealized(self, positions: Dict[str, Position], signals: Dict[str, pd.DataFrame], dt) -> float:
        total = 0.0
        for sym, pos in positions.items():
            if sym not in signals:
                continue
            df = signals[sym]
            if dt not in df.index:
                continue
            px = float(df.loc[dt, "close"])
            inst = self.instruments[sym]
            total += pos.direction * (px - pos.avg_price) * pos.lots * inst.dollar_per_point()
        return total

    def _stop_hit(self, pos: Position, open_px: float, high: float, low: float) -> tuple[bool, float]:
        if pos.direction > 0:
            if low <= pos.stop:
                # Gap through: fill at open if open already through stop
                fill = open_px if open_px < pos.stop else pos.stop
                return True, float(fill)
        else:
            if high >= pos.stop:
                fill = open_px if open_px > pos.stop else pos.stop
                return True, float(fill)
        return False, 0.0

    def _soft_exit_reason(self, pos: Position, row: pd.Series) -> Optional[str]:
        if pos.direction > 0:
            if self.use_channel_exit and bool(row.get("exit_long_channel", False)):
                return "channel_exit"
            if self.use_ma_exit and bool(row.get("exit_long_ma", False)):
                return "ma_exit"
        else:
            if self.use_channel_exit and bool(row.get("exit_short_channel", False)):
                return "channel_exit"
            if self.use_ma_exit and bool(row.get("exit_short_ma", False)):
                return "ma_exit"
        t_stop = self.params.time_stop_bars
        if t_stop is not None and pos.bars_held >= t_stop:
            atr = float(row["atr"]) if pd.notna(row["atr"]) else pos.entry_atr
            close = float(row["close"])
            float_pts = pos.direction * (close - pos.avg_price)
            if float_pts < 0.5 * atr:
                return "time_stop"
        return None

    def _pyramid_pending(self, pos: Position, row: pd.Series) -> Optional[PendingOrder]:
        max_units = 1 + self.params.max_pyramid_adds
        if pos.units >= max_units:
            return None
        atr = float(row["atr"]) if pd.notna(row["atr"]) else pos.entry_atr
        close = float(row["close"])
        float_pts = pos.direction * (close - pos.avg_price)
        if float_pts < 0.5 * atr:
            return None
        # Filter must still hold
        if pos.direction > 0 and not bool(row.get("f_bull", False)):
            return None
        if pos.direction < 0 and not bool(row.get("f_bear", False)):
            return None
        # Spacing from last add / entry
        step = self.params.pyramid_atr_step * atr
        move_from_last = pos.direction * (close - pos.last_add_price)
        if move_from_last < step:
            return None
        unit_index = pos.units  # next unit
        lots = lots_for_unit(pos.q0, unit_index)
        if lots <= 0:
            return None
        action = "add"
        return PendingOrder(symbol=pos.symbol, action=action, lots=lots, reason="pyramid", unit_index=unit_index)

    def _fill_order(
        self,
        *,
        order: PendingOrder,
        open_px: float,
        atr: Optional[float],
        dt,
        cash: float,
        positions: Dict[str, Position],
        trades: List[TradeRecord],
        risk: PortfolioRiskState,
        signals: Dict[str, pd.DataFrame],
    ) -> float:
        sym = order.symbol
        inst = self.instruments[sym]

        if order.action == "exit":
            if sym not in positions:
                return cash
            return self._close_position(
                pos=positions[sym],
                fill_px=open_px,
                dt=dt,
                cash=cash,
                positions=positions,
                trades=trades,
                risk=risk,
                reason=order.reason,
            )

        if order.action in ("enter_long", "enter_short"):
            if sym in positions:
                return cash  # already in a position
            if atr is None or atr <= 0 or pd.isna(atr):
                return cash
            direction = 1 if order.action == "enter_long" else -1
            equity = cash + self._unrealized(positions, signals, dt)
            q0 = size_initial_lots(
                equity=equity,
                atr_points=atr,
                instrument=inst,
                risk_per_trade=risk.effective_risk_per_trade(),
            )
            lots = lots_for_unit(q0, 0)
            if lots <= 0:
                return cash
            side = 1 if direction > 0 else -1
            fill_px = costmod.apply_slippage_price(open_px, side, inst)
            margin = inst.margin(fill_px, lots)
            trade_risk = atr * inst.dollar_per_point() * lots
            if not risk.can_open(
                symbol=sym,
                sector=inst.sector,
                new_margin=margin,
                equity=equity,
                trade_risk_dollars=trade_risk,
                is_new_symbol=True,
            ):
                return cash
            one_way = costmod.round_trip_cost_per_lot(inst, fill_px) * lots
            cash -= one_way
            stop = fill_px - direction * self.params.stop_atr_mult * atr
            positions[sym] = Position(
                symbol=sym,
                direction=direction,
                lots=lots,
                avg_price=fill_px,
                entry_atr=atr,
                stop=stop,
                units=1,
                q0=q0,
                bars_held=0,
                extreme=fill_px,
                last_add_price=fill_px,
            )
            risk.register_open(sym, inst.sector, margin, trade_risk)
            trades.append(
                TradeRecord(
                    symbol=sym,
                    dt=dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                    side="buy" if direction > 0 else "short",
                    lots=lots,
                    price=fill_px,
                    reason=order.reason,
                )
            )
            return cash

        if order.action == "add":
            if sym not in positions:
                return cash
            pos = positions[sym]
            lots = order.lots
            if lots <= 0:
                return cash
            side = 1 if pos.direction > 0 else -1
            fill_px = costmod.apply_slippage_price(open_px, side, inst)
            equity = cash + self._unrealized(positions, signals, dt)
            margin = inst.margin(fill_px, lots)
            atr_use = atr if atr and atr > 0 else pos.entry_atr
            trade_risk = atr_use * inst.dollar_per_point() * lots
            if not risk.can_open(
                symbol=sym,
                sector=inst.sector,
                new_margin=margin,
                equity=equity,
                trade_risk_dollars=trade_risk,
                is_new_symbol=False,
            ):
                return cash
            one_way = costmod.round_trip_cost_per_lot(inst, fill_px) * lots
            cash -= one_way
            new_lots = pos.lots + lots
            pos.avg_price = (pos.avg_price * pos.lots + fill_px * lots) / new_lots
            pos.lots = new_lots
            pos.units += 1
            pos.last_add_price = fill_px
            # Tighten stop toward avg ± 2 ATR vs existing
            if pos.direction > 0:
                pos.stop = max(pos.stop, pos.avg_price - self.params.stop_atr_mult * atr_use)
            else:
                pos.stop = min(pos.stop, pos.avg_price + self.params.stop_atr_mult * atr_use)
            risk.register_open(sym, inst.sector, margin, trade_risk)
            trades.append(
                TradeRecord(
                    symbol=sym,
                    dt=dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                    side="buy" if pos.direction > 0 else "short",
                    lots=lots,
                    price=fill_px,
                    reason="pyramid",
                )
            )
            return cash

        return cash

    def _close_position(
        self,
        *,
        pos: Position,
        fill_px: float,
        dt,
        cash: float,
        positions: Dict[str, Position],
        trades: List[TradeRecord],
        risk: PortfolioRiskState,
        reason: str,
    ) -> float:
        inst = self.instruments[pos.symbol]
        # Closing side opposite to position
        close_side = -1 if pos.direction > 0 else 1
        px = costmod.apply_slippage_price(fill_px, close_side, inst)
        pnl = pos.direction * (px - pos.avg_price) * pos.lots * inst.dollar_per_point()
        one_way = costmod.round_trip_cost_per_lot(inst, px) * pos.lots
        cash += pnl - one_way
        trades.append(
            TradeRecord(
                symbol=pos.symbol,
                dt=dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                side="sell" if pos.direction > 0 else "cover",
                lots=pos.lots,
                price=px,
                reason=reason,
                pnl=pnl - one_way,
            )
        )
        risk.set_symbol_margin(pos.symbol, 0.0)
        positions.pop(pos.symbol, None)
        return cash
