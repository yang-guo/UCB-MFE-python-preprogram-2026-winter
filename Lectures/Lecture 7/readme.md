# October 26: Building stock-return models with sklearn

Yang · 9am–noon Pacific · first Modeling session. Plan for **150 minutes including a ten-minute break**, with the remaining scheduled time for questions and setup.

Use price and financial data to forecast 90-calendar-day stock returns. The notebooks cover sklearn's interfaces, feature construction, and model comparison across time. Each runs independently and ends with exercises.

## Class schedule

| Time | Notebook | Topics |
|---|---|---|
| 9:00–9:50 | [1. Estimators, transformers, and pipelines](1_sklearn_fundamentals.ipynb) | `fit`, `predict`, `transform`, learned attributes, cloning, and custom transformers |
| 9:50–10:35 | [2. Build features from prices and financial statements](2_transformers.ipynb) | Trailing returns, filing dates, quarterly membership, sector comparisons, and column preprocessing |
| 10:35–10:45 | Break | Ten minutes |
| 10:45–11:25 | [3. Fit a return model and evaluate its predictions](3_models_and_evaluation.ipynb) | Fitting and prediction, chronological validation, model comparison, and hyperparameter search |
| 11:25–11:30 | Preview of Lecture 8 | Using forecasts to select and trade stocks |

Exercises can be completed after class.

## Preparation and data

Run `uv sync --locked` from the repository root. Select `.venv` and obtain `equities_class.duckdb` from bCourses at `Lectures/modeling/data/`. It is a 4.3 MiB real-data extract of 30 stocks selected before 2010, with quarterly eligibility applied. No API or full-source database is needed for the lesson.

The [data guide](../modeling/README.md) explains the quarterly ranking, table definitions, and source limitations. HW4 uses `equities_top2000.duckdb`: 3,849 securities across 107 quarterly selections, history from 2000 through 2026-09-30, about 315 MiB. The class extract has 2008–2015 prices, so it loads quickly and excludes the homework's final evaluation period. Both databases are distributed through bCourses and ignored by Git.

## Discussion questions

- What does `StandardScaler.fit` learn, and what does `transform` do with those values?
- Should a scaled validation column have mean zero?
- How does `clone(pipe)` differ from `another = pipe`?
- Why should predicting one feature row alone give the same result as predicting it in a batch?
- When can a December financial statement first affect a forecast?
- Why can a December signal be too recent to train a model fitted in January?
- Can a model have a negative R² and still rank stocks usefully?

The [HW4 reference table](../Lecture%208/readme.md#preparing-for-hw4) points to examples that may help with your project.
