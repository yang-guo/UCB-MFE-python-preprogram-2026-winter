# HW4: 90-day return model and long-only backtest


A long-only 90-calendar-day strategy on the full homework database. The trading rule is frozen from 2010–2015 development only, then evaluated on **1 April 2016–30 September 2026**. Beating the S&P 500 is not required to pass; this run does not beat it.

## How to run

From the repository root, use the course environment, then run from this folder:

```bash
uv sync --locked
cd Homeworks/HW4/Suge_Li
python test_accounting.py          # hand-checkable cash/lot/timing tests
python hw4.py --develop            # 2013–2015 comparison; writes results/freeze.json
python hw4.py --final              # frozen rule, Apr 2016–Sep 2026
```

`python hw4.py --smoke` uses `equities_class.duckdb` and writes `results/freeze_smoke.json`. It does **not** overwrite the full-database freeze.

Place `equities_top2000.duckdb` (and optionally `equities_class.duckdb`) in `Lectures/modeling/data/`. Do not commit database files.

## What was frozen (development only)

Candidates were scored with chronological folds on 2013–2015. Every label used in that comparison finishes before **2016-04-01**, so the buffer is used and the test period is not. New buys are restricted to names with `universe_rank ≤ 400`, unadjusted close ≥ $5, and signal-day dollar volume ≥ $1m. Extreme one-session adjusted-price jumps (4×) are not eligible for new buys.

| Candidate | Rank IC | CAGR | Sharpe |
|---|---:|---:|---:|
| Ridge on 90-day rank target | 0.031 | 14.61% | 1.16 |
| **12–1 momentum + operating ROA** | **0.050** | **17.34%** | 1.07 |
| Momentum + ROA + low volatility | 0.059 | 13.56% | 1.12 |
| Momentum + ROA + FCF − accruals | 0.057 | 16.03% | 1.10 |
| S&P 500 total return (same window) | — | 14.20% | — |

**Frozen rule:** among candidates with positive IC, take the highest development CAGR. That is **`mom_roa`**: equal-weight same-date ranks of 12–1 momentum (`momentum_252_21`) and operating ROA. Missing legs are not filled with 0.5. Ridge is walk-forward and is fit only on labels that have already ended; the factor rule is not fitted.

`results/development_returns_dev.png` is this 2013–2015 paper portfolio. `results/cumulative_returns_hw4.png` is the same frozen rule on 2016-04-01–2026-09-30. They are different periods, so the curves are not supposed to match.

| Choice | Rule |
|---|---|
| Target | Next-session entry to first quoted close on/after 90 calendar days |
| Portfolio | Long only, **$1,000,000**, no borrowing. First session of each month: buy the **top 15** scores, equal dollar budgets, spending `min(cash, NAV/3)` known at the **previous close**. Today's sale proceeds fund a later entry, not the same close |
| Costs | **5 bps** of notional on each actual buy and sell, inclusive of fees. Cash earns **0%**. Adjusted closes only |
| Missing quotes | No invented fills. A buy is skipped if that session has no quote. Open lots are marked at the last observed adjusted close. Gaps longer than five sessions are logged as `unresolved_held_gap` and still marked, not force-sold |

Signals use the previous session’s close; fills use the next session’s close. The first test close is **2016-04-01**. The benchmark invests $1,000,000 in `^SP500TR` at that close.

## Performance (final test: 2016-04-01 to 2026-09-30)

Metrics use daily NAV from the first test close through 2026-09-30, including cash and open lots. Sharpe uses daily excess returns over a 2% annual rate. CAGR uses elapsed calendar days / 365.25.

| | Strategy | S&P 500 total return |
|---|---:|---:|
| Final value | **$2,421,692** | **$4,402,747** |
| Total return | +142.2% | +340.3% |
| CAGR | 8.79% | 15.17% |
| Max drawdown | −27.5% | −33.8% |
| Sharpe | 0.42 | 0.77 |

The strategy **does not beat** the benchmark (about $1.98M lower final wealth). Chart: `results/cumulative_returns_hw4.png`.

Accounting at the end: cash $766,924 + holdings $1,654,768 = NAV $2,421,692. Realized PnL +$1,381,259; fees $49,806; 3,384 fills; 36 lots still open on 2026-09-30 (later 90-day exits are marked, not forced).

## Evidence that timing and cash are correct

`python test_accounting.py` checks, with numbers you can recompute by hand:

1. A $100,000 inclusive budget at $50 and 5 bps buys `100000 / (50 × 1.0005)` units and leaves NAV lower by the fee.
2. A sale credits `units × price × (1 − fee)`; spending more than remaining cash raises an error.
3. Two lots in the same ticker keep separate 90-day maturities.
4. Signal date must be strictly before the fill date.
5. On a tiny calendar, the April 1 buy is planned from March 31, the sale is on/after day 90, and the benchmark is $1e6 at the first test close.
6. Open lots still contribute to terminal NAV when the test ends before maturity.
7. A name that stops quoting is **not** sold at a made-up last price; it stays marked.
8. A lot that matures on an entry day is sold at that close, but the sale cash does not fund a buy on the same close.
9. A same-day price jump does not change the buy budget. The budget uses the previous close's NAV.

On the full run, the first fills are 2016-04-01 buys whose `signal_date` is 2016-03-31 and whose `due_date` is 2016-06-30. Each of the 15 names receives $22,222.22, which is `1e6 / 3 / 15`. See `results/fills_head_hw4.csv`. Daily `cash + holdings_value = nav`.

## Findings and limitations

- **The ranking is weak.** Development rank IC is only about 0.05. That is enough to beat the index in 2013–2015 paper trading, but not in 2016–2026, when large-cap beta dominated.
- **Cash drag.** One-third of NAV is reserved each month so three 90-day cohorts can overlap. Cash also sits after a cohort matures until the next month-start. The book never borrows.
- **Stale marks.** The test log has many `unresolved_held_gap` session events. Those names are carried at the last real quote. Terminal NAV for them is not a verified exit, and no delisting payout is invented.
- **No look-ahead in the signal.** Price features use trailing adjusted closes through the signal date. Financial ratios use the filing date plus one day. Membership is the entry quarter already known at the signal. Labels used to compare rules finish before 2016-04-01. The current sector field is not point-in-time, so it is not used as a feature.
- **Survivorship is reduced, not eliminated.** Quarterly membership is recorded-date and availability-aware. Inactive flags are not delisting proceeds.

## What I would try next

1. Name and sector caps from Lecture 8 `capped_budgets`.
2. Annual refits of Ridge using only labels that have already completed (the code path exists; it was not frozen because `mom_roa` won on development CAGR).
3. A simple 12–1 momentum-only book on the same backtester, to see how much ROA is contributing.
4. Require a recent complete quote streak before a new buy, to cut stale-mark risk.

## Code map and citations

- `backtester.py` — own OMS, monthly allocator, daily loop, and performance metrics. Accounting follows `Lectures/modeling/README.md` and the Lecture 8 OMS: inclusive fees, independent lots, and orders sized from the previous close before today's prices are seen.
- `features.py` — month-end panel, peer features, screens, factor scores.
- `hw4.py` — development comparison, freeze, test run, plots.
- `test_accounting.py` — the checks above.
- `results/` — freeze file, development table, performance table, chart, fill sample, daily NAV.

Feature construction, membership, labels, `QuantileClipper`, and `make_pipeline` are reused from `Lectures/modeling/course_data.py`. An AI coding assistant helped organize the files and implement the backtester. The strategy was frozen from the development comparison only, and every number below comes from running this code on `equities_top2000.duckdb`.
