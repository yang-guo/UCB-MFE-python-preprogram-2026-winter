# Modeling data and shared course code

Lectures 7 and 8 follow one real equity research problem: predict a gross 90-calendar-day holding return, then account for a long-only portfolio. The classroom database and HW4 database share a schema, including quarterly universe membership.

## Setup and distribution

Run `uv sync --locked` from the repository root, select `.venv` as the notebook kernel, and download these files from bCourses to `Lectures/modeling/data/`:

| File | Use | Distinct securities | Price rows | Financial rows | Size |
|---|---|---:|---:|---:|---:|
| `equities_class.duckdb` | Five lecture notebooks | 30 | 59,286 | 1,059 | 4.3 MiB |
| `equities_top2000.duckdb` | HW4 quarterly universe | 3,849 | 15,839,788 | 284,198 | 314.8 MiB |

The full delivery covers **January 1, 2000–September 30, 2026** (first market session January 3, 2000). “Top2000” names the maximum selected in one quarter, not the size of the union. All available in-window price and financial histories for the union are retained, including observations outside a security's membership periods. This supports trailing features and marking/exiting existing positions. Selection provenance for 2000-Q1 includes pre-2000 dates; those inputs initialize membership, not a pre-2000 price-history export.

The class extract has 2008–2015 prices and 2007–2015 financial periods. Its fixed teaching pool is chosen from membership known before 2010: AAPL/MSFT anchors for worked arithmetic, then sector round robin by the 2010-Q1 rank. No future return, survival, or complete-history screen is used. The sample still applies quarterly membership; pre-2010 observations are warm-up. It is a teaching subset, not a representative full-market performance study. Four class names are marked inactive in the source snapshot.

Databases are ignored by Git and distributed through bCourses.

[manifest.json](data/manifest.json) records the delivered files' checksums, sizes, table counts, and quality findings. Inspect `universe` for the stock list and `universe_rebalances` for quarterly selection counts and coverage shortfalls.

## Quarterly selection and information timing

There are **107 quarterly selections**, 2000-Q1 through 2026-Q3. For each quarter:

1. Set `selection_date` to the last supplied market session before the calendar quarter starts. Membership becomes usable for entries from the first market session of the quarter.
2. Consider securities with a positive finite close on that selection date. The source labels them US-domiciled stocks on NYSE, NASDAQ, AMEX, or OTC. **Do not filter on current active status or future price coverage.** OTC is retained because a later venue migration must not automatically erase earlier history.
3. Use the latest fiscal quarter with positive finite reported shares, a valid filing date after fiscal end, and **filing date + one calendar day ≤ selection date**. Its fiscal period must be no more than 18 months old. A late old fiscal period does not replace a newer known period.
4. Calculate **capitalization proxy = selection-date close × known quarterly shares**. Rank descending, ticker breaking ties. Select the first 2,000, or all rankable names when fewer exist. Do not use later filings to complete an old quarter.

The first **46 quarters**, through 2011-Q2, have fewer than 2,000 rankable names; 2000-Q1 has **1,206**. The remaining quarters reach 2,000. These counts describe source coverage, not the number of stocks that actually existed. The union contains **1,355 securities currently marked inactive**. That status is audit information, not a predictor or historical filter.

`universe_candidates` retains every rankable candidate and the exact price/shares/dates used. `universe_membership` retains the selected rows. `universe_rebalances` records positive-price candidates, rankable candidates, selected counts, shortfalls, and currently inactive counts. For the class extract, `selected_count` remains the full-universe selection count and `delivered_members` is the subset included in class.

| Example | Membership used | Why |
|---|---|---|
| Signal 2000-03-30 for entry 2000-03-31 | 2000-Q1 | Entry is in Q1 |
| Signal after the 2000-03-31 close for entry 2000-04-03 | 2000-Q2 | The Q2 list is now known and entry is in Q2 |
| A stock exits the Q2 list while a Q1 lot remains open | No new buys; continue marking and normal maturity exit | Membership is not a liquidation instruction |

The helper joins membership by **the quarter of the next-session entry**, after constructing per-ticker history. It retains all rows and sets `is_universe_member` and `eligible`; it never deletes old price history. Use the dated mask for training rows, same-date peer features, and new-entry scores. The union itself is not an investable universe at every historical date. An old training row keeps its own historical membership, not the fit date's list.

## What this fixes, and what the source cannot establish

The quarterly reconstruction removes the deliberate selection of today's active winners. It is **recorded-date, availability-aware research data**, not a certified point-in-time constituent or market-cap feed.

- The security master lacks dated country, exchange, security-type, and sector histories. Those static fields still affect scope and classification. Historical ticker/issuer mappings and delisted coverage are incomplete; an inactive flag can also reflect a rename or reorganization. Do not equate 1,355 inactive flags with 1,355 verified delisting events.
- The source has only 13 explicit delisting dates among 5,264 inactive stock records overall. Only two union names have such dates, both after the study cutoff. These fields are retained as snapshot audit metadata. There is no reliable delisting-proceeds table. Never infer a cash payout, zero recovery, or a tradeable exit from a final price row or inactive flag.
- Financial records are latest vendor values for a fiscal period, not original filing vintages. Recorded filing dates restrict the join but cannot undo restatements or errors.
- Close and reported shares can have inconsistent split adjustments, units, or share-class bases. The capitalization ranking is approximate. For example, the recorded 2010-Q1 inputs put APDN's close above $2 million and TSYS's reported shares above 46 billion. These are investigation flags, not evidence of those firms' true historical capitalization. The candidate table exposes them rather than silently inventing split corrections. Exact rankings need verified contemporaneous prices/shares or historical market caps and a corporate-action/security master.
- Multiple share classes count separately. Current sectors are snapshot labels. Source missingness and inaccurate filing dates can exclude stocks unequally across time.

Students must report these residual limitations, audit implausible ranking inputs, and distinguish source-supported findings from assumptions. Dated selection reduces a major avoidable bias; it does not justify a claim of fully survivorship-free returns.

## Tables and units

| Table | Key / contents | Role |
|---|---|---|
| `universe` | Symbol; name, snapshot sector/industry/country/exchange, `is_active_snapshot`, `delisted_at_snapshot`, first/last selected quarter, selected-quarter count | Union inventory and audit; full-history aggregates/status are never historical inputs |
| `universe_candidates` | `(quarter_start,symbol)`; quarter/end/effective dates, selection close, shares/fiscal/filing/availability dates, capitalization proxy, rank | Reproduce each quarter's ranking, including candidates never selected; their full histories need not be in the union export |
| `universe_membership` | Selected rows from candidates, rank ≤2,000 | Dated new-entry and training membership |
| `universe_rebalances` | One row per quarter; availability/selection dates and coverage counts | Coverage and shortfall audit |
| `prices` | `(symbol,date)`; supplied OHLC, adjusted close, volume, source | Complete in-window history for the union; features and adjusted-unit accounting |
| `fundamentals` | `(symbol,fiscal_date,period)`; 46 financial and provenance fields | Quarterly availability-aware features; all source columns except maintenance timestamp |
| `benchmark` / `calendar` | `^SP500TR` total-return series / one date per benchmark session | Benchmark and shared market calendar |
| `dividends` | `(symbol,ex_date)`; dates and vendor per-share amounts | Optional actual-share research; never add to adjusted-price returns |
| `dataset_metadata` | key/value pairs | Variant, date bounds, source fingerprint, selection rule |

Returns, ratios, and volatility are decimals. Monetary statement fields use `reported_currency` where supplied; currency and vendor units require checking. Same-statement ratios reduce currency mismatches but do not prove accounting comparability. `shares_outstanding` may be a weighted-average quantity, not contemporaneous shares or float. Do not infer historical dollar liquidity from future-adjusted prices without validating the basis.

## Known quality findings and baseline policies

- Full financial records: **23,664** missing/nonpositive filing lags, **269,164** unknown currencies, and **273,351** missing market caps. Class equivalents: **14**, **1,058**, and **981**. Preserve raw data and audit counts; exclude invalid filing events from baseline features.
- **275 zero adjusted-price rows for BHRB**, 2000-01-03 through 2001-03-13, are retained in the delivered `prices` table. The loader warns and quarantines them before division; it never substitutes raw closes or fabricated returns.
- **28 price rows**, across nine tickers, have dates absent from the supplied `calendar` table. The loader warns and quarantines these rows. The class extract has no invalid or off-calendar prices.
- Financial features use valid quarterly filings from filing date + one calendar day. Blank ratios older than 550 days from fiscal end; never backfill from future reports. Same-date peer calculations use only historically eligible members.
- Compute historical features by ticker on the supplied market calendar without price filling. Class eligibility requires 63 complete trailing daily returns plus dated membership; longer features can remain missing for the training-fitted imputer.
- Positive finite prices can still be economically wrong. Investigate internal gaps, corporate actions, and delisting recoveries. Preserve unresolved held-stock incidents and label censoring.

## Conventions used in the class examples

The examples calculate features and scores after close on `t`, then enter at the next market session's observed close. Each purchase is held for 90 **calendar** days and exits at the first quoted close on/after maturity. Labels retain actual entry/exit dates; missing outcomes remain missing. Training uses only outcomes completed before the fit date, and validation keeps all stocks on a date together.

The classroom backtester uses fractional **adjusted-price units** to represent total-return exposure, so dividends are not added separately. It charges costs on each purchase and sale, with entry fees included in dollar budgets. Sales add cash immediately, and the strategy can use that balance when it next plans orders. Orders planned at the previous close keep their original budgets. Open positions are valued daily, including at the end. If a holding has no quote, we carry its last observed mark for at most five market sessions and flag it as stale; longer gaps stop the example for investigation.

The daily history includes initial capital before the first entry, so entry fees affect both the first return and maximum drawdown. Strategy and benchmark first invest at the same close. Cash earns zero; the Sharpe hurdle is a 2% annual rate converted to its daily equivalent. CAGR uses actual elapsed days / 365.25; volatility and Sharpe use 252 sessions. The fixed-horizon labels overlap and cannot be compounded into a portfolio return.

## Code used by the notebooks

- [course_data.py](course_data.py): read-only SQL/ticker batching, causal features, filing joins, dated membership, calendar labels, purged folds, preprocessing, and metrics.
- [backtest.py](backtest.py): the Lecture 8 framework. `Strategy.generate_orders` receives a portfolio snapshot and a separate copy of adjusted-price history through the signal close, including warm-up sessions and unfilled price gaps. `MonthlyTopNStrategy` shares monthly scheduling and allocation. `MomentumStrategy` calculates trailing returns from that history and applies dated quarterly membership; `CappedMomentumStrategy` adds concentration limits. `PredictedScoreStrategy` uses the research notebook's audited model predictions. `Backtester` runs the daily loop, `OrderManagementSystem` books fills and calculates PnL, and `BacktestAnalytics` reports performance. Each run returns a `BacktestResult` with daily history, fills, open lots, diagnostics, and PnL. HW4 students design their own strategy and implement their own backtester.

The research notebook saves its configuration, predictions, and accounting results under `Lectures/modeling/outputs/`, which is ignored by Git.

Use `load_data(..., symbols=batch)` for bounded-memory reads. Construct per-stock features in batches, then combine the same-date eligible rows **before** computing peer-relative features. Keep full quote history for holdings, and do not select stocks by future label completeness. The final homework evaluation extends through September 30, 2026 while classroom runs remain pre-2016.

## Primary references

The examples are checked against the installed course versions. These official references explain the underlying APIs:

- [sklearn estimator interfaces and conventions](https://scikit-learn.org/stable/developers/develop.html)
- [BaseEstimator hyperparameter inspection](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)
- [TransformerMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)
- [sklearn leakage and consistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html)
- [Pipelines and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [TimeSeriesSplit: samples and row-based gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [SimpleImputer and empty features](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Histogram gradient boosting and early stopping](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [DuckDB Python client](https://duckdb.org/docs/stable/clients/python/overview)
