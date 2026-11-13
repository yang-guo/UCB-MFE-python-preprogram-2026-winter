# Research plan (fixed on 2026-11-13, before any of these experiments was run)

This plan was drafted with an AI assistant and approved by Ian before the batch
was launched.  `scripts/run_study.py` executes it in one unattended run and
applies the rules below in code, so no choice is made after looking at results.
Everything here uses development data only (signals 2010-01-01 to 2015-12-31,
prices to 2016-03-31); research mode cannot load anything later.

## What is already known, and what is excluded because of it

An earlier demonstration run (`results/demo_run/`) used the final test period
once.  Its result (a low-volatility / value / quality composite lagged the
S&P 500 after 2016) is known to both Ian and the assistant.  To keep that
knowledge out of this study:

- No candidate, feature, sign or rule below was chosen by reference to the
  demonstration's test-period numbers.
- Two ideas that came up only after seeing those numbers (controlling market
  beta, restricting to the 1,000 largest names) are deliberately **not** tested.
- The development-period results of the demonstration are fair to use.  They
  showed gradient-boosted trees ranking worst, so trees are not re-tested here.

## Fixed settings

Weekly signals, 13 overlapping cohorts, 90-calendar-day hold, 5 bps fees,
5% name cap, 30% sector cap, training on signals from 2010 whose outcomes were
complete before each fit, quarterly refits.  A held stock that stops quoting is
settled at its last mark at maturity (`on_long_gap = settle_at_maturity`).

## Stage 0: how much of a result is luck or universe?

- 20 portfolios that pick 30 eligible names at random each week, run through
  the same backtester.  Their average shows what the universe and equal
  weighting deliver without any signal; their spread is the noise level any
  strategy difference must exceed.
- Decomposition for every candidate: (random average minus S&P 500) is the
  universe effect, (candidate minus random average) is the selection effect.

## Stage 1: feature study (descriptive only)

Rank IC of every registered feature against the next 90-day return, by signal
year 2010 to 2015.  This table is reported for understanding.  It is **not**
used to pick features or signs: composite signs come from the published
anomaly direction listed in `plan_alpha.json`, written before the table existed.

Five new features, each registered with its timing rule and covered by the
prefix-invariance leakage test: `roa_yoy_change`, `revenue_growth_yoy`,
`asset_growth_yoy`, `net_issuance_yoy`, `earnings_yield_ttm`.

## Stage 2: alpha candidates (one change per step)

| Step | Candidate | Change from the previous step |
|---|---|---|
| R1 | fixed composite, 7 core features | starting point |
| R2 | fixed composite, core + 5 new features | features only |
| R3 | Ridge on the same 12 features | model only (fitted weights instead of fixed signs) |
| R4 | Ridge on all 21 features | features only |
| R5 | Ridge on 21 features, penalty tuned inside each fit's training rows | model only |

**Selection rule.**  Measure each candidate by its walk-forward rank IC on
signal dates in 2012 to 2015.  A candidate is *stable* if its yearly mean IC is
positive in at least 3 of the 4 years.  Walk through R1 to R5 in order.  The
first stable candidate becomes the incumbent.  A later (more complex) candidate
replaces the incumbent only if it is stable and the paired difference in
per-date IC is positive beyond noise: the 5th percentile of a circular block
bootstrap of the mean difference (13-week blocks, 5,000 draws, fixed seed) must
be above zero.  Otherwise the simpler incumbent stays.

## Stage 3: portfolio variants of the selected alpha

Base: top 30, equal weight.  Variants, each a single change: top 50, top 100,
inverse-volatility weights.

**Adoption rule.**  A variant is adopted only if (a) its development CAGR
exceeds the base's by more than the standard deviation of CAGR across the 20
random portfolios, and (b) its maximum drawdown is no worse than the S&P 500's
over the same window.  If several pass, the one with the highest CAGR is taken;
combinations are not adopted because they were not tested.  If none passes, the
base stays.

## What happens after

The batch stops after writing `results/research/STUDY_SUMMARY.md`.  Ian reviews
it and decides what to freeze.  No stage is added without telling him first.
