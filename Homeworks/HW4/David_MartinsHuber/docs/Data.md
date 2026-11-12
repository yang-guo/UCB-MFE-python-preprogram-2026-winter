# Market data: prices, volume, and point-in-time financials

`backtesting/market_data.py` provides one `MarketData` object for both
`Research.ipynb` and the backtesting engine. The database readers are reused
verbatim from `Lectures/modeling/course_data.py` through
`data_loading.py` (course data helpers, UCB MFE Python Pre-Program). The panel
construction below is specific to this submission.

## Load

```python
from backtesting import MarketData

data = MarketData.load(
    database="data/equities_top2000.duckdb",
    end="2026-09-30",
)
print(data.coverage)
```

`load` calls the verbatim `load_data` and `load_membership`, then adds the
remaining numeric fundamental columns with a separate read-only query. It does
not modify the loader or its assertions and warnings.

## Currency policy

All remaining names are US-listed and prices are USD. Tickers whose
`reported_currency` is present and not `USD` are removed at load time. In the
current database this drops `GFL` (CAD) and `SMMT` (GBP). A missing
`reported_currency` is treated as USD for these US listings.

Consequences:

- Raw financial levels are treated as USD after the filter.
- `market_cap_usd` uses the raw USD `close`, not `adj_close`:
  `shares_outstanding × close`.
- The vendor `market_cap` column is sparse (~4% of rows), so the derived
  `market_cap_usd` is the reliable field.

## Volume

`prices.volume` is raw shares and is loaded by `load_data`. The class exposes:

```python
data.volume_panel(start=None, end=None)   # dates × symbols
```

Volume is not split-adjusted, matching the vendor field. Zero-volume rows are
preserved; research should apply its own liquidity filter if needed.

## Point-in-time financials

A filing has no intraday timestamp, so it becomes usable at the close of the
following calendar day:

```text
available_date = filing_date + 1 day
```

`financial_panel` and `financials_at` use a backward as-of join from price
dates to `available_date`:

- The left side is always the price history, so no price row is lost.
- A value repeats on every subsequent price row until the next filing
  (forward fill).
- Rows before the first filing stay missing; they are never back-filled.
- There is no staleness cap by default. `max_age_days=` is available if a
  research experiment wants one.
- A filing whose `available_date` is after the query date is never visible.
- Filings with `filing_date <= fiscal_date` are treated as untrusted and
  excluded; the vendor snapshot contains a small number of these.

### Last-statement policy

- A later restatement of the same fiscal period is visible only from its own
  `available_date`; the original filing remains visible before that date.
- An old fiscal period filed late cannot replace a newer fiscal period that is
  already known.
- Identical `(symbol, available_date, fiscal_date)` keys keep the latest
  filing.

The implementation mirrors the lecture's conservative `financial_events`
logic, but it is not copied verbatim.

## Fields

`data.available_financials` lists the numeric columns a strategy may request:

- the eight loader fields (`revenue`, `net_income`, `operating_income`,
  `total_assets`, `total_debt`, `cash_and_equivalents`, `free_cash_flow`,
  `operating_cash_flow`);
- the six ratios (`net_margin`, `operating_roa`, `debt_to_assets`,
  `cash_to_assets`, `fcf_to_assets`, `accruals_to_assets`);
- the remaining numeric vendor columns (`market_cap`, `shares_outstanding`,
  `eps`, `ebitda`, `total_equity`, `capex`, ...);
- the derived `market_cap_usd`.

`fields=None` returns all of them. Research may subset freely.

## Strategy-declared columns

A strategy declares exactly what it needs:

```python
class MyStrategy(Strategy):
    required_financials = ("net_margin", "market_cap_usd")

    def propose_buys(self, context):
        row = context.financials.loc["AAPL"]
        ...
```

The engine validates the declaration at construction time. Unknown columns
raise `ValueError`; a strategy that declares fields without a `MarketData`
source also fails at construction. `context.financials` is indexed by symbol
and contains only the declared fields plus `available_date`, `fiscal_date`,
`filing_date`, and `filing_age_days`.

`context.volume` is a wide date-by-symbol frame cut at the signal close.
`context.fundamentals` remains the event-level filing history for backward
compatibility.

## Research

```python
panel = data.training_panel(
    feature_columns=("momentum_63", "net_margin", "market_cap_usd"),
    start="2010-01-01",
    end="2026-09-30",
)
```

`training_panel` returns `date, symbol, available_date, label_end, target` plus
price features, forward-filled financial columns, `filing_age_days`, and
optionally `sector`. Labels use the next session's entry and a 90-calendar-day
exit with a five-session delay cap. The engine's refit path accepts this frame
directly.

## Backtest

```python
engine = Backtester(
    market_data=data,
    strategy=MyStrategy(),
    training_panel=panel,
    feature_columns=("momentum_63", "net_margin", "market_cap_usd"),
)
result = engine.run("2026-01-02", "2026-09-30")
```

Passing `market_data` supplies prices, benchmark, membership, volume, and the
point-in-time financial view. Passing explicit `prices`, `benchmark`, and
`membership` still works for tests and synthetic data.

## Limitations

- The source is a vendor snapshot; it cannot recover original historical
  vintages if values were later restated.
- Missing `reported_currency` is assumed to be USD. This is reasonable for the
  US-listed universe but is an assumption, not a vendor guarantee.
- Financial levels are not converted with an FX time series; non-USD reporters
  are removed instead.
- `market_cap_usd` uses the filing's shares outstanding and the current raw
  close, so it moves with price between filings.
- The engine cannot detect a strategy that reads future files independently.
  The data contract protects the supplied context, not arbitrary Python I/O.

## Checks

```bash
../.venv/bin/python -m unittest tests.test_market_data -v
```

The tests cover forward fill, no back-fill, restatements, late old periods,
price-history preservation, non-USD removal, volume, market cap, engine
delivery of declared columns, and the training-panel schema.
