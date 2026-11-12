"""HW4: Ridge model of 90-day returns and a monthly top-N long-only strategy.

Adapted from Lectures/modeling/course_data.py and
Lectures/Lecture 8/2_research_workflow.ipynb.
"""

import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Lectures/modeling"))
from course_data import (FEATURES, apply_membership, date_folds, financial_events,  # noqa: E402
                         join_financials, load_data, load_membership, make_labels,
                         make_pipeline, prediction_metrics, price_features)
from backtester import Backtester, performance  # noqa: E402

DATABASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Lectures/modeling/data/equities_top2000.duckdb"
OUTPUT = Path(__file__).parent / "outputs"
RESULTS = Path(__file__).parent / "results"
END = "2026-09-30"
DEV_START, DEV_END = "2010-01-01", "2016-01-01"
DEV_BACKTEST = ("2014-01-01", "2015-12-31")
TEST_START = "2016-04-01"
BATCH = 250
SETTINGS = dict(initial_cash=1_000_000, fee_bps=5, top_n=20, cohort_fraction=1 / 3, holding_days=90)


def build_panel():
    """Course feature/label pipeline in ticker batches; keeps month-end signal rows."""
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        symbols = [s for (s,) in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    membership = load_membership(DATABASE, end=END)
    panels, test_prices = [], []
    for start in range(0, len(symbols), BATCH):
        prices, fundamentals, universe, benchmark = load_data(DATABASE, end=END, symbols=symbols[start:start + BATCH])
        if prices.empty:
            continue
        calendar = pd.DatetimeIndex(benchmark.date)
        month_ends = calendar.to_series().groupby(calendar.to_period("M")).max()
        panel = price_features(prices, calendar)
        panel = join_financials(panel, financial_events(fundamentals), universe)
        panel = apply_membership(panel, membership, calendar)
        panel = make_labels(panel, prices, calendar)
        panels.append(panel.loc[panel.date.isin(month_ends) & panel.date.ge("2009-12-01")])
        test_prices.append(prices.loc[prices.date.ge("2013-12-01"), ["date", "symbol", "adj_close"]])
        print(f"processed {min(start + BATCH, len(symbols))}/{len(symbols)} symbols")
    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    quotes = (pd.concat(test_prices).pivot(index="date", columns="symbol", values="adj_close")
              .reindex(calendar[calendar >= "2013-12-01"]))
    benchmark = benchmark.set_index("date").adj_close
    return panel, quotes, benchmark


def clip_target(target, bounds_from):
    """Clip targets at the 1%/99% quantiles of the training targets."""
    return target.clip(bounds_from.quantile(0.01), bounds_from.quantile(0.99))


def choose_alpha(panel):
    """Choose alpha on 2013-2015 validation folds."""
    development = panel.loc[panel.eligible & panel.date.ge(DEV_START) & panel.date.lt(DEV_END)
                            & panel.label_end.lt(DEV_END)].reset_index(drop=True)
    folds = date_folds(development, validation_years=(2013, 2014, 2015))
    search = GridSearchCV(make_pipeline(Ridge()), {"model__alpha": [1, 100, 1000]},
                          cv=folds, scoring="neg_root_mean_squared_error", error_score="raise")
    first_train = development.target.iloc[folds[0][0]]
    search.fit(development[FEATURES], clip_target(development.target, first_train))
    print(pd.DataFrame(search.cv_results_)[["param_model__alpha", "mean_test_score"]])
    return search.best_estimator_


def quarterly_scores(panel, model, first_fit, end):
    """Refit each quarter on labels that ended before the fit date."""
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    fit_signals = dates[(dates >= first_fit) & (dates < end) & dates.month.isin([3, 6, 9, 12])]
    parts = []
    for i, asof in enumerate(fit_signals):
        next_asof = fit_signals[i + 1] if i + 1 < len(fit_signals) else pd.Timestamp(end)
        train = panel.loc[panel.eligible & panel.date.ge(DEV_START) & panel.date.lt(asof) & panel.label_end.lt(asof)]
        to_score = panel.loc[panel.eligible & panel.date.ge(asof) & panel.date.lt(next_asof)]
        assert train.label_end.max() < to_score.date.min()
        fitted = clone(model).fit(train[FEATURES], clip_target(train.target, train.target))
        scored = to_score[["date", "symbol", "target"]].copy()
        scored["score"] = fitted.predict(to_score[FEATURES])
        parts.append(scored)
    return pd.concat(parts, ignore_index=True)


def add_benchmark(history, benchmark):
    """$1M in ^SP500TR at the first backtest close."""
    days = history.index[1:]
    history["benchmark_nav"] = float(SETTINGS["initial_cash"])
    history.loc[days, "benchmark_nav"] = SETTINGS["initial_cash"] * benchmark.loc[days] / benchmark.loc[days[0]]
    return history


def choose_top_n(panel, model, quotes, benchmark):
    """Pick the number of names by Sharpe on the 2014-2015 backtest."""
    scores = quarterly_scores(panel, model, "2013-12-31", DEV_BACKTEST[1])
    score_map = {date: g.set_index("symbol").score for date, g in scores.groupby("date")}
    rows = {}
    for top_n in (10, 20, 50):
        history, fills, _, _ = Backtester(quotes, score_map, **{**SETTINGS, "top_n": top_n}).run(*DEV_BACKTEST)
        check_accounting(history, fills, quotes)
        rows[f"top {top_n}"] = performance(history.nav)
    rows["S&P 500 TR"] = performance(add_benchmark(history, benchmark).benchmark_nav)
    table = pd.DataFrame(rows)
    print("Development backtest 2014-2015:\n", table)
    best = table.drop(columns="S&P 500 TR").loc["Sharpe"].idxmax()
    return int(best.split()[1]), table


def check_accounting(history, fills, quotes):
    """Check cash, NAV and trade timing."""
    cash_change = fills.groupby("date").cash_change.sum().reindex(history.index, fill_value=0)
    np.testing.assert_allclose(SETTINGS["initial_cash"] + cash_change.cumsum(), history.cash, atol=1e-6)
    np.testing.assert_allclose(history.cash + history.holdings_value, history.nav)
    buys, sells = fills[fills.side.eq("BUY")], fills[fills.side.eq("SELL")]
    assert (buys.signal_date < buys.date).all()
    assert (buys.date.dt.to_period("M") != buys.signal_date.dt.to_period("M")).all()
    assert (sells.date >= sells.due_date).all()
    assert quotes.index.max() <= pd.Timestamp(END)
    print("Accounting checks passed.")


def main():
    OUTPUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    panel, quotes, benchmark = build_panel()
    model = choose_alpha(panel)
    SETTINGS["top_n"], dev_table = choose_top_n(panel, model, quotes, benchmark)
    print("Chosen top_n:", SETTINGS["top_n"])
    dev_table.to_csv(RESULTS / "development_backtest.csv")
    scores = quarterly_scores(panel, model, "2016-03-31", END)

    tested = scores.loc[scores.date.ge("2016-03-31") & scores.target.notna()]
    print("Test prediction metrics:\n", prediction_metrics(tested, tested.score))

    score_map = {date: g.set_index("symbol").score for date, g in scores.groupby("date")}
    history, fills, open_lots, missed = Backtester(quotes, score_map, **SETTINGS).run(TEST_START, END)
    check_accounting(history, fills, quotes)

    history = add_benchmark(history, benchmark)

    results = pd.DataFrame({"strategy": performance(history.nav),
                            "S&P 500 TR": performance(history.benchmark_nav)})
    print(results)

    stale = open_lots.loc[open_lots.last_quote_date < history.index[-1]]
    stale_value = (stale.units * stale.last_mark).sum()
    print(f"Open lots at end: {len(open_lots)} | stale: {len(stale)} worth ${stale_value:,.0f} "
          f"| final value if stale lots = 0: ${history.nav.iloc[-1] - stale_value:,.0f} "
          f"| missed buys (no quote): {len(missed)}")

    ax = (history[["nav", "benchmark_nav"]] / SETTINGS["initial_cash"] - 1).plot(figsize=(10, 5))
    ax.legend(["Strategy", "S&P 500 TR"])
    ax.set(ylabel="Cumulative return", title="Strategy vs S&P 500 total return")
    plt.tight_layout()
    plt.savefig(RESULTS / "cumulative_returns.png")

    results.to_csv(RESULTS / "performance.csv")
    history.to_csv(OUTPUT / "history.csv")
    fills.to_csv(OUTPUT / "fills.csv", index=False)
    open_lots.to_csv(OUTPUT / "open_lots.csv", index=False)
    missed.to_csv(OUTPUT / "missed_buys.csv", index=False)


if __name__ == "__main__":
    main()
