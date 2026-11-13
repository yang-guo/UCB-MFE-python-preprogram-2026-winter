# Fama-French five-factor construction

`backtesting/famafrench.py` builds an **FF5-inspired** model from the supplied
`MarketData`. It is not an exact CRSP/Compustat replication. The approximations
are listed below and encoded in the class docstring.

## Inputs

The builder requires `MarketData` loaded with the wider vendor fundamentals
(`extra_fundamental_columns=True`, the default). The required fields are:

| Field | Use |
|---|---|
| `shares_outstanding` | Market equity with the raw USD close |
| `total_equity` | Book equity |
| `total_assets` | Investment (year-over-year growth) |
| `operating_income` | Profitability numerator |
| `interest_expense` | Profitability numerator; falls back to operating income when missing |
| `close`, `adj_close` | Market equity and forward returns |
| `sector` | Financial Services exclusion |

## Point-in-time rules

- A filing is usable on `filing_date + 1 day`, the `MarketData` convention.
- At formation date `t`, book equity, profitability and investment use the
  value carried by the month-end **`lag_months` rows earlier**. The default is
  six monthly observations (the source date is within a few days of
  `t - 6 months`). This is a monthly-grid lag, not an exact calendar offset.
- Market equity is measured at `t` with the raw USD close.
- The forward return is from the `t` close to the next month-end close.
- Future filings and future prices therefore cannot change a formation-date
  observation. This is covered by `tests/test_famafrench.py`.

## Characteristics

```text
ME   = shares_outstanding * close
BE   = total_equity                       (available by t - lag)
BM   = BE / ME
OP   = (operating_income - interest_expense) / BE
INV  = total_assets(t - lag) / total_assets(t - lag - 12m) - 1
```

A name is eligible when it is a point-in-time execution-quarter member, has a
positive price and market equity, has positive book equity (default), has
finite BM/OP/INV, has a known sector, and is not in an excluded sector.
Same-day filings (`filing_date == fiscal_date`) and filings older than their
fiscal period are excluded upstream by the lecture's conservative
availability rule; the count is visible in the raw `fundamentals` table. A
filing missing book equity, total assets, operating income, or shares is
dropped from the FF5 event stream so it cannot shadow the last complete
filing.

## Sorts and factors

- Size breakpoint: eligible median ME.
- BM, OP, INV breakpoints: 30th and 70th percentiles.
- 2x3 portfolios: Small/Big x Low/Neutral/High, Weak/Neutral/Robust,
  Conservative/Neutral/Aggressive.
- Portfolio returns are value-weighted by ME within each cell. `weighting="equal"`
  is available for tests and robustness checks.
- `SMB` is the average of the three size legs (BM, OP, INV).
- `HML` = high minus low BM.
- `RMW` = robust minus weak OP.
- `CMA` = conservative minus aggressive INV.
- `MKT` is the value-weighted return of the eligible universe. **No risk-free
  rate is subtracted.**

## Approximations

| Item | Used here | Fama-French reference |
|---|---|---|
| Book equity | `total_equity` | Stockholders' equity + deferred taxes - preferred stock |
| Profitability | `(operating_income - interest_expense) / BE` | `(revenue - COGS - SG&A - interest) / BE` |
| Investment | Year-over-year `total_assets` growth | Growth in total assets |
| Market factor | Value-weighted eligible universe | Value-weighted all CRSP firms |
| Risk-free rate | Not used | One-month T-bill |
| Breakpoints | All-market median / 30-70 | NYSE median / 30-70 |
| Formation | Monthly, six-month accounting lag | June annual formation |
| Delisting | Not handled here | CRSP delisting returns |
| Financials | Excluded by default | Often excluded from HML/RMW/CMA |
| Negative BE | Excluded by default | Excluded |

## Output

```python
from backtesting import FamaFrench5, MarketData

data = MarketData.load("data/equities_top2000.duckdb", end="2026-09-30")
model = FamaFrench5(lag_months=6, exclude_sectors=("Financial Services",),
                    exclude_negative_be=True, weighting="cap")
result = model.build(data, start="2010-01-01", end="2025-12-31")

result.factor_returns   # date x MKT, SMB, HML, RMW, CMA
result.characteristics  # date, symbol, ME, BE, BM, OP, INV, FWD_RET, groups
result.portfolios       # date, characteristic, portfolio, return, n, weight_sum
result.diagnostics      # universe, eligible, exclusion counts
result.coverage         # universe / eligible / forward-return counts
```

`model.exposures(data, "2020-06-30")` returns one formation-date cross-section
with group assignments.

## Validation

`python -m unittest tests.test_famafrench -v` covers:

- a hand-calculated 2x3 cross-section with one name in every cell,
- the five factor formulas,
- Financial Services and negative-book-equity exclusions,
- year-over-year investment,
- future filings not changing prior characteristics,
- future prices not changing prior factor returns,
- missing required columns raising a clear error.

The builder was also smoke-tested on the full database: about 1,430 eligible
names per month in 2010 after membership, sector, book-equity and missing-data
filters, and a two-year build completes in a few seconds after loading.

## Limitations

- The model cannot recover original financial vintages from a restated vendor
  snapshot.
- Delisting returns must be supplied by the backtester; the factor builder
  simply drops a name with no forward price.
- A single market factor built from the eligible universe is not the CRSP
  market portfolio.
- A leg is `NaN` when any of its 2x3 cells is empty; the factor is not silently
  reweighted over the available cells. Tied breakpoints make that
  characteristic Neutral rather than putting every name in the High bucket.
- Trailing formation dates with no next-month return are dropped from
  `factor_returns`; interior short-history gaps remain.
- The six-month lag and monthly formation are research choices, not the exact
  Fama-French calendar.
