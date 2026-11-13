# Selection rules of the DEMONSTRATION run (written before its experiments ran)

> These rules, the plans next to this file, and the strategy they selected were written by the AI
> assistant to exercise the pipeline end to end. They are an example of a pre-registered plan,
> not the submitted strategy and not rules Ian chose.

All experiments use development data only (signals 2010-01-01 .. 2015-12-31,
labels completed by 2016-03-31). The final test is run once, after freezing.

## Stage A: which alpha (features + model)?

Fixed strategy for all of Stage A: weekly cohorts, top 30 names, equal weight,
5% name cap, 30% sector cap, 5 bps per side, quarterly expanding-window refits.

1. Primary metric: **mean same-date rank IC across the 2012, 2013, 2014, 2015
   validation folds** (one fit per fold at the fold's first signal; training
   labels completed before that fit).
2. Eligible only if the IC is positive in **at least 3 of the 4 years**
   (stability matters more than a high average from one year).
3. If a simpler candidate is within **0.005 IC** of the best, choose the
   simpler one (fixed composite < Ridge < gradient boosting).
4. The development backtest (2012-2015) is reported for every candidate but
   used only as a sanity check: four years of overlapping 90-day cohorts is
   a small sample.

## Stage B: portfolio construction for the chosen alpha

One change at a time from the Stage A base (weekly, top 30, equal weight):
top 50; inverse-volatility sizing; monthly cohorts. Keep the base unless a
variant improves the development Sharpe by more than 0.10 without a worse
maximum drawdown (avoid chasing noise).

## Not allowed

- Looking at any final-test metric before freezing.
- Changing the protocol dates (they are hash-locked in `quantlab/protocol.py`).
- Re-selecting after the final run without `--acknowledge-reselection`
  (recorded in `results/final_runs.jsonl`).
