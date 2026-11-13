# HW4 project structure

`Research.ipynb` is the research and modeling workspace. It is where we will explore the data, engineer features, train and evaluate prediction models, and develop the trading signals used by the strategy.

The goal is to make the research components reusable by the backtesting engine. Classes developed on the research side should follow clear, consistent interfaces so they can be imported and used by the backtester without duplicating their logic. In particular, model and signal components should make it clear what data they require, how they are fit using information available at the time, and how they produce predictions or scores for a strategy to act on.

The backtesting code will handle portfolio decisions and execution, cash and positions, fees, and performance tracking. Keeping those responsibilities behind stable interfaces lets us change or compare research models and strategies while using the same backtesting engine.

`data_loading.py` contains the reusable DuckDB loading functions used by both the research and backtesting code. These functions are copied from `Lectures/modeling/course_data.py`. We will use the Lecture 8 backtesting notebook as a structural reference, but will implement our own backtester rather than copy that notebook.

`load_data` and `load_membership` are verbatim copies of the lecture functions, including their assertions, warnings, and default end date of `2015-12-31`. Only their required imports and an attribution header are included in our module. Always pass the database path and end date explicitly. For research and model fitting, use `end="2015-12-31"`; the 2016-01-01–2016-03-31 buffer is an embargo period excluded from research/fitting. Use `end="2026-09-30"` only for the separate frozen final-test evaluation. The default database path remains relative to the loader module, exactly as in the original.

The lecture loader quarantines nonfinite or nonpositive stock adjusted closes and off-calendar stock quotes with warnings. Their other valid observations remain available; the raw database is unchanged. It does not interpolate prices or replace them with raw close. This leaves gaps for the engine: missing entry quotes reject orders, while held-position gaps use bounded stale marks and may stop the run. The original benchmark assertions remain intact. Inspect quarantined rows in the raw database and document their relevance to the experiment; do not exclude entire stocks based on future data problems.

## Backtester v1

The shadow-portfolio extension is described in [Portfolio.md](Portfolio.md). It adds independent component accounts behind a shared clock and a portfolio adapter with the same strategy output interface. A prepare/advance barrier ensures every decision precedes the next execution quote. Existing standalone usage remains supported.

Portfolio analysis is implemented separately in `backtesting/analytics.py`. See [Analysis.md](Analysis.md) for formulas and units, including annualized alpha, beta, historical daily VaR at 95% and 99%, and all required homework metrics. `result.performance()` delegates to that module.

`backtesting/contracts.py` defines the strategy and research interfaces, `oms.py` owns pending orders and accounting, and `engine.py` advances sessions, supplies quotes, controls refits, and collects results. This engine is independently implemented using the lecture's separation of responsibilities as a reference.

`Backtesting.ipynb` contains the executed April 1, 2016–September 30, 2026 full-universe run, comparison, and plots. The result reports 234,681 stale marks and 16,328 terminal open lots with a 10,000-session stale-mark limit; these carried marks materially weaken the NAV, so the notebook labels its performance exploratory. The notebook loads `data/equities_top2000.duckdb` from this submission; download it from bCourses. The indexed pending-sale implementation is in `backtesting/indexed_sales.py`.

Requests beyond the supplied benchmark date range raise `ValueError` by default. `allow_partial=True` is an explicit opt-in on standalone and coordinated runs: it emits a warning and records the requested/effective dates and clipping flag in `result.coverage` (or `bundle.portfolio.coverage`). It is not appropriate for claiming completion of the final homework period. The engine cannot infer exchange holidays beyond the supplied calendar; provide coverage around a weekend/holiday end or request the intended final session explicitly. Incremental methods and history properties require `begin()` before use and reject earlier access deliberately.

The OMS supports fees from 0 through 10,000 basis points per side. It checks all computed fill amounts and resulting cash before committing a fill, and rejects numerical overflow/underflow without altering pending orders or account state. Valuation must also remain finite. Any negative post-fill cash is rejected; purchases are clamped to actual cash so sequential rounding cannot create credit.

For full-budget purchases, representable quantity is rounded downward if reconstructed notional plus fee exceeds the reserved budget. This preserves the budget cap at large capital amounts without relaxing borrowing checks. Benchmark normalization divides by the initial benchmark quote before multiplying by capital; unrepresentable growth fails the run rather than returning a completed result with infinite values. Partial-run warnings occur before initialization is committed, so promoting warnings to errors does not corrupt the engine lifecycle.

Implement a research strategy by inheriting `Strategy`, choosing a stable `strategy_id`, and implementing `propose_buys(context)`. Return a mapping from symbol to fee-inclusive purchase budget. Symbols must be exact Python strings; subclasses and duplicate normalized tickers are rejected so one name cannot occupy several cap budgets. Context contains past adjusted closes, raw share volume, known execution-quarter membership, the event-level filing history, a point-in-time forward-filled financial row per symbol for the columns declared in `required_financials`, a detached planning copy of the previous account snapshot, and dates. The engine freezes the membership universe before strategy code runs and rebuilds the account snapshot for risk review, so edits to the planning copy cannot expand the tradable universe or hide exposure. The engine enforces membership even if the strategy forgets to filter candidates. One strategy owns one account in v1; the identifier is attribution, not a multi-strategy cash allocation system.

Every order, fill, and lot records `strategy_id`. Purchases have trigger `strategy`; mandatory sales keep the lot owner's strategy ID and have trigger `maturity`. The OMS reserves cash when accepting buys, then replaces reservations with actual debits on fills. Invalid dated quotes and repeated fills are rejected. Missing entry quotes reject orders and release reservations; held quotes can be carried for at most five sessions with diagnostics. Missing maturity quotes delay sales and are logged. Open lots remain valued at test end, and idle cash earns no interest. Next-close execution uses fractional adjusted units and defaults to five basis points per side.

For model refitting, override `needs_refit(context)` and `refit(data, context)`. Supply `training_panel` and explicit `feature_columns` when constructing the engine. The panel needs `date`, `symbol`, `available_date`, `label_end`, and `target`; each feature row must already be valid at its own historical date. V1 supplies expanding history of completed targets with `label_end < signal_date`; unfinished rows never enter the fit. `FitData.X` excludes target/audit columns. Fit preprocessing using those training rows only. A successful fit activates a copied strategy; access the current trained instance through `engine.strategy`. Failure stops the run without activating the candidate. Models must support Python deep copying. Refit audit counts describe supplied training rows; any extra selection inside a model must be documented by research.

`MarketData.training_panel(...)` builds that frame from the same point-in-time panels used by the engine. A row whose `available_date` is missing has no filing known yet; it is allowed in the panel and excluded by the refit cutoff rather than rejected at construction. Strategies declare financial columns with the class attribute `required_financials`; the engine validates the declaration and supplies exactly those columns in `context.financials`. See [Data.md](Data.md) for the forward-fill, restatement, currency, and volume rules.

`backtesting/famafrench.py` builds monthly FF5 factor returns and characteristic sorts from `MarketData`. Book equity, operating profitability and investment use a six-month accounting lag; market equity uses the formation-date raw close. The implementation is FF5-inspired rather than an exact CRSP/Compustat replication. See [FamaFrench.md](FamaFrench.md) for the formulas, the full approximation list, exclusions, and the output schema.

`strategies/factor.py` contains the FF5 regression tilt and the cap-weighted market baseline. The tilt fits a rolling five-factor regression on matured 90-calendar-day returns, ranks the cross-section, and targets the top quintile while excluding the bottom quintile. Existing lots are never sold early; the engine's 90-day maturity exits create three staggered sleeves. The strategy starts in the Q1 2016 buffer and reinvests maturity proceeds on the next session. See [FactorStrategy.md](FactorStrategy.md) for the model, target weights, sleeve schedule, and limitations.

`research/pairs.py` and `strategies/pairs.py` add the `Pairs_Tilt` research model: 10x cap-weighted volume bars, Ward clusters, ADF/Kalman pair signals, and long-only nine-sleeve execution with a cap-weighted fallback. See [PairsTilt.md](PairsTilt.md).

Strategies that fit themselves from the point-in-time price and factor history set `self_fitting = True`; the engine still owns the refit phase and audit, ignores any supplied `training_panel`, and does not require one. Only `delisting="error"` is supported. Held quotes may be carried for valuation within `max_stale_sessions`; exceeding that limit stops the run for investigation. A carried mark is never an executable quote, and missing prices cannot bypass 90-day maturity.

Financial filings are exposed only after a one-calendar-day lag and only when filing date follows fiscal date. This cannot undo later restatements in the source data. Training feature provenance and externally precomputed prediction correctness remain research responsibilities: Python interfaces are not a sandbox against a strategy reading future data independently. Do not load full future datasets into strategy state.

Run the hand-verifiable accounting and timing checks from this folder:

For interactive checks, open `audits/Testing.ipynb` with the Python (HW4) kernel. It includes visible order/fill tables, a flat-price round trip, next-session timing, refit inspection, and the regression suite. No database is needed.

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

Implementation and checks were developed with Codex assistance. The completed database run and its stale-mark limitation are documented in the submission notebook and audit report.
