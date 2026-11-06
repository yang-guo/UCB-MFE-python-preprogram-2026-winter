# HW4: Jing Jian — stock-return modeling and backtesting

The full-data experiment is complete. Read [REPORT.md](REPORT.md) for strategy
selection, final results, accounting evidence, and limitations. The final
strategy underperformed the S&P 500 total-return benchmark; outperforming it
is not required for the assignment.

## Setup

From the repository root:

```bash
uv sync --locked
```

Use Python 3.12 or later and the locked course dependencies. Place the bCourses
files `equities_class.duckdb` and `equities_top2000.duckdb` in
`Lectures/modeling/data/`. They are ignored by Git. The final experiment uses
the full database, not the classroom extract, and its SHA-256 is recorded in
`results/final_summary.json`.

## Run and reproduce

All commands below run from the repository root:

```bash
# Twenty hand-checkable accounting, timing, and data-handling tests.
uv run --locked python -m unittest discover -s Homeworks/HW4/Jing_Jian -v

# Reproduce the full development comparison, frozen final strategy, and report.
uv run --locked python Homeworks/HW4/Jing_Jian/reproduce.py

# Regenerate the submitted chart and report from the submitted daily results.
uv run --locked python Homeworks/HW4/Jing_Jian/make_report.py
```

Reproduction writes to ignored `outputs/reproduction/`, preserving submitted
results and their original pre-test freeze. A repeat reproduction refuses to
replace that directory; preserve or move it before starting a fresh run. The
runner uses all historical candidates, ticker batches, and monthly feature
rows to bound memory. Allow several minutes for a full run. No market-data API,
credentials, or external downloads are needed after obtaining the databases;
verified corporate-action terms and their sources are included locally.

`research.py development` and `research.py final` were the original staged
commands. The development command intentionally refuses to overwrite the
submitted freeze after a final attempt. Use `reproduce.py` for a fresh run.
`run_development.py` retains the earlier classroom prototype; it is not the
final model or final evaluation.

## Files and included evidence

- `research.py`: monthly causal features, chronological training, four
  development candidates, frozen selection, final evaluation and ledger audit.
- `backtester.py`: independently implemented cash/lot engine; no lecture
  backtester imports.
- `corporate_actions.py` and `.json`: source-supported conversions, default
  cash consideration, and ticker mapping. These resolve observed gaps without
  removing stocks based on future survival.
- `test_backtester.py` and `test_research.py`: twenty tests with artificial,
  hand-computable fixtures; actual experiments use supplied market data.
- `make_report.py`: exported cumulative-return/drawdown chart and findings.
- `results/`: compact metrics, daily curves, chart, development comparisons,
  frozen specification, accounting checks, action ledger, and captured tests.
- `outputs/`: ignored feature caches, detailed trades and experimental artifacts.

## Final strategy and timing

Predict next-session entry to 90-calendar-day maturity return with Ridge
(alpha 100) using five price features: 21/63-session momentum, momentum from
252 to 21 sessions ago, 63-session annualized volatility, and 252-session
maximum-relative drawdown. Fit feature clipping, imputation, scaling, and
1%/99% target clipping on training data only. The final price-only model won
the limited pre-2016 comparison against adding financial ratios and buying
only ten stocks. See REPORT.md for all four development results.

Use historical top-200 quarterly ranks within the supplied top-2000 universe,
252 complete trailing returns, raw close $5–$10,000, volatility at most 60%,
and no absolute daily return above 25% in the last 63 sessions. Screens use
only signal-date information. Do not filter current inactive stocks or future
missing outcomes. Keep existing holdings when membership changes.

Buy up to 20 positive-score names at the first market close each month using
previous-close predictions. Split the budget equally, capped at the smaller
of prior-close cash and one third of prior-close NAV. Start with $1 million,
long only, without leverage. Charge 10 bp on each purchase and voluntary sale,
use fractional adjusted-price units, and give cash zero interest. Each lot
exits at its own 90-calendar-day maturity or the next market session. Open
positions count in final NAV; dividends are not added again.

Training signals are from 2010–2015 only, with each outcome completed before
the fit. Development validation is 2013–2015. The final fit occurs before the
March 31, 2016 signal, uses only completed old labels, and is not retrained.
The final test runs April 1, 2016–September 30, 2026. The initial freeze and
execution-only amendments are recorded separately; no model/strategy retuning
was done against final results.

Missing entries are cancelled. Carry a missing held-stock mark for at most
five sessions and flag it; stop on a longer gap or a missing maturity exit.
Documented mandatory actions are booked separately, preserving original lot
maturities. Default merger cash is reserved until original maturity. The report
explains four source-supported corrections and remaining security-master,
corporate-action, execution, and point-in-time limitations.

## Attribution and AI assistance

Data loading, price/financial features, publication-aware financial joins,
quarterly membership, labels, and training preprocessing reuse
`Lectures/modeling/course_data.py` from Lectures 7–8. Model selection and
portfolio accounting are implemented here. Conventions for adjusted-price
units and metrics follow `Lectures/modeling/README.md`. The independent engine
does not import `Lectures/modeling/backtest.py`.

Codex generated the implementation, tests, experiment execution, report and
chart under Jing Jian's direction, including investigating and documenting
corporate-action gaps. The work is AI-assisted. Jing Jian should review and
be able to explain the submitted code and research decisions.

Submit this folder only, keeping databases, environments, caches and large
trade logs out of Git. PR title: `HW4: Jing Jian`; base: course `main`; assign
`kevinramlal` and `raghuramkowdeed`.
