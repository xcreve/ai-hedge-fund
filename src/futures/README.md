# CCT-1 Futures Backtest (research scaffold)

Bypass package for the **Contest-Common Trend (CCT-1)** rule strategy. It does **not** modify `src/agents/`, `src/main.py`, or the stock LLM `src/backtesting/` CLI.

Strategy rules live in the Project store spec (`docs/futures-quant-strategy-spec.md`). This folder is research-only; futures leverage can exceed capital — not investment advice.

## CSV schema (continuous contracts)

One file per symbol: `data/futures/sample/RB.csv`, etc.

| Column | Required | Notes |
|--------|----------|--------|
| `date` | yes | Daily bar date (`YYYY-MM-DD`) |
| `open` | yes | Session open |
| `high` | yes | Session high |
| `low` | yes | Session low |
| `close` | yes | Session close (signals evaluated here) |
| `volume` | yes | Numeric; may be synthetic in samples |
| `open_interest` | no | Optional |

Files must be sorted by date, one row per trading day. Use **main continuous** (or front-adjusted) prices for signals. Roll/mapping to actual front-month fills is out of scope for this scaffold.

## Execution model A

1. Filters + Donchian breakout confirmed on **close** of day T.  
2. Entries / soft exits / pyramid adds fill at **next open** (T+1).  
3. Hard ATR / chandelier stops fill when **high/low** penetrates the stop (gap → open).

## Run

```bash
poetry run futures-backtest run \
  --data-dir data/futures/sample \
  --config src/futures/configs/cct1_research.yaml \
  --out-dir artifacts/futures

# equivalent
poetry run python -m src.futures.cli run --data-dir data/futures/sample
```

Instrument multipliers / margins: `src/futures/configs/instruments.yaml`.  
Contest risk stub (optional, not default): `src/futures/configs/cct1_contest.yaml`.

## Tests

```bash
poetry run pytest tests/futures -q
```
