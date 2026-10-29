from src.futures.data.schema import InstrumentSpec
from src.futures.execution.position_sizer import lots_for_unit, size_initial_lots


def test_size_initial_lots_basic():
    inst = InstrumentSpec(symbol="RB", multiplier=10, tick_size=1.0, margin_rate=0.1)
    # equity=1e6, risk=1%, ATR=20 → dollar_risk=200 → Q0=floor(10000/200)=50
    q0 = size_initial_lots(equity=1_000_000, atr_points=20.0, instrument=inst, risk_per_trade=0.01)
    assert q0 == 50


def test_size_initial_lots_zero_atr():
    inst = InstrumentSpec(symbol="RB", multiplier=10, tick_size=1.0, margin_rate=0.1)
    assert size_initial_lots(equity=1_000_000, atr_points=0.0, instrument=inst, risk_per_trade=0.01) == 0


def test_pyramid_weights():
    assert lots_for_unit(40, 0) == 40
    assert lots_for_unit(40, 1) == 30
    assert lots_for_unit(40, 2) == 20
    assert lots_for_unit(40, 3) == 20
    assert lots_for_unit(40, 4) == 0
