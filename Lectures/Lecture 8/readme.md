# October 29: Building a backtesting framework

Yang · 9am–noon Pacific · second Modeling session. Plan for **150 minutes including a ten-minute break**, leaving the remaining scheduled time for questions. [HW4](hw4.md) is due November 12, 2026.

We'll build a backtesting framework with separate classes for strategy decisions, order management, simulation, and analytics. Then we'll use it to compare a model fitted once with one refitted quarterly. Both notebooks use the small class database, run independently, and end with exercises.

## Class schedule

| Time | Notebook | Focus |
|---|---|---|
| 9:00–10:20 | [1. Build a backtesting framework](1_backtesting.ipynb) | Strategy interface, OMS accounting, daily simulation, analytics, and strategy subclasses |
| 10:20–10:30 | Break | Ten minutes |
| 10:30–11:20 | [2. Use the framework to compare research ideas](2_research_workflow.ipynb) | Hyperparameter selection, quarterly fits, completed labels, portfolio comparisons, and saved results |
| 11:20–11:30 | HW4 questions | Strategy choices, results, and submission |

Exercises can be completed after class.

## Preparation

Use `uv sync --locked`, select `.venv`, and place `equities_class.duckdb` from bCourses at `Lectures/modeling/data/`. Read the [data guide](../modeling/README.md). The class database is under 5 MiB, with pre-2016 history; HW4 uses the full union of quarterly top-2,000 selections and the April 2016–September 2026 final period.

Both notebooks import the classes in [backtest.py](../modeling/backtest.py). The first walks through the implementation: the backtester supplies prices through the signal close, and `MomentumStrategy` calculates its own signals. The second compares that strategy with `PredictedScoreStrategy`, which uses the audited Ridge predictions. All share the monthly schedule and allocation rules in `MonthlyTopNStrategy`. The research notebook saves its results under `Lectures/modeling/outputs/`, which is ignored by Git.

## Preparing for HW4

Choose your own features, model, and trading strategy. The [assignment](hw4.md) defines the common trading rules and deliverables; the lecture strategies are examples you can learn from and adapt.

| Task | Helpful examples |
|---|---|
| Engineer features | 7.2: trailing prices, financial statements, and sector comparisons |
| Train and validate a model | 7.1: sklearn interfaces and pipelines; 7.3: chronological validation and model comparison |
| Build a backtester | 8.1: separate strategy, order management, daily loop, and analytics classes |
| Evaluate the strategy | 8.1: performance metrics; 8.2: benchmark comparisons and diagnostics |
| Explore retraining | 8.2: fixed and quarterly-refitted models; retraining is optional in HW4 |

Write your own backtester. You may reuse lecture loading and preprocessing code, and should cite anything you adapt.

## Discussion questions

- Does a $100,000 inclusive budget purchase $100,000 of exposure after fees?
- How does cash from a sale reach the strategy's next order decision?
- Which prices can a strategy see when it plans tomorrow's purchases?
- Can three individually small purchases collectively breach a sector cap?
- What happens to two lots in the same ticker with different maturities?
- Why is a missing last row not evidence of a delisting?
- Why do terminal open positions affect NAV but not necessarily the predictive metric sample?
- What is frozen in a walk-forward experiment, and what is refitted?
- If a model improves rank IC but reduces portfolio returns, which tables would help explain the difference?
