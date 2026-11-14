"""HW4: 90-day return model and development-period validation.

Builds monthly cohorts (signal at the close before the first session of each
month, entry at that session's close, exit at the first quote on or after
entry + 90 calendar days), fits the forecasting model, and reproduces the
validation grid in README.md. See README.md for details.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

import backtest_riskadj_momentum as bt


OUTPUT = Path(__file__).resolve().parent / "results_validation"
DEV_START = pd.Timestamp("2010-01-01")
DEV_END = pd.Timestamp("2015-12-31")
VALID_START = pd.Timestamp("2013-01-01")
BUFFER_END = pd.Timestamp("2016-03-31")
SIZES = [300, 400, 500, 750]
NAMES = [10, 20, 30, 50]


def load_ranked_members(db):
    """Same issuer/instrument filter as the backtester, keeping the rank."""
    members = db.execute(
        r"""
        SELECT m.quarter_start, m.symbol, m.universe_rank
        FROM universe_membership AS m JOIN universe AS u USING (symbol)
        WHERE NOT regexp_matches(u.name,
              '(?i)\bnotes?\b|debenture|warrant|acquisition corp|%')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY m.quarter_start, lower(trim(u.name))
            ORDER BY m.universe_rank
        ) = 1
        """
    ).df()
    members["quarter_start"] = pd.to_datetime(members["quarter_start"])
    return members


def make_cohorts(db, calendar, prices, scores):
    """One row per (monthly entry, member): score, 90-day target, exit date."""
    monthly = calendar.to_series().groupby(calendar.to_period("M")).first()
    entries = pd.DatetimeIndex(monthly[monthly >= DEV_START])
    members = load_ranked_members(db)

    benchmark = db.execute(
        "SELECT date, adj_close FROM benchmark WHERE symbol = '^SP500TR'"
    ).df()
    benchmark = benchmark.set_index(pd.to_datetime(benchmark["date"]))["adj_close"]
    benchmark = benchmark.reindex(calendar)

    # For every session and stock, the index of the next observed quote.
    quoted = prices.notna().to_numpy()
    positions = np.where(quoted, np.arange(len(calendar))[:, None], np.nan)
    next_quote = pd.DataFrame(positions).bfill().to_numpy()
    values = prices.to_numpy()

    rows = []
    for entry in entries:
        entry_idx = calendar.get_loc(entry)
        maturity_idx = calendar.searchsorted(entry + pd.Timedelta(days=90))
        if maturity_idx >= len(calendar):
            break
        signal = calendar[entry_idx - 1]
        cohort = members[members["quarter_start"] == entry.to_period("Q").start_time]
        cols = prices.columns.get_indexer(cohort["symbol"])
        keep = cols >= 0
        cohort, cols = cohort[keep], cols[keep]

        exit_idx = next_quote[maturity_idx, cols]
        has_exit = np.isfinite(exit_idx)
        safe_idx = np.where(has_exit, exit_idx, 0).astype(int)
        exit_price = np.where(has_exit, values[safe_idx, cols], np.nan)

        rows.append(pd.DataFrame({
            "entry": entry,
            "signal": signal,
            "symbol": cohort["symbol"].to_numpy(),
            "universe_rank": cohort["universe_rank"].to_numpy(),
            "score": scores.loc[signal].to_numpy()[cols],
            "target": exit_price / values[entry_idx, cols] - 1,
            "exit_date": np.where(
                has_exit, calendar.to_numpy()[safe_idx], np.datetime64("NaT")
            ),
            "benchmark_90d": benchmark.iloc[maturity_idx] / benchmark.iloc[entry_idx] - 1,
        }))

    cohorts = pd.concat(rows, ignore_index=True)
    # Candidates need a score at the signal close (same screens as the backtest).
    return cohorts[cohorts["score"].notna()].copy()


def add_score_rank(cohorts):
    """Percentile of the score within each entry date among the top 500."""
    cohorts = cohorts[cohorts["universe_rank"] <= bt.MAX_RANK].copy()
    cohorts["score_rank"] = cohorts.groupby("entry")["score"].rank(pct=True)
    return cohorts


def fit_model(training):
    """Linear regression of the 90-day return on the score percentile.

    Targets are clipped at the training 1st/99th percentiles so a few extreme
    outcomes do not drive the slope.
    """
    low, high = training["target"].quantile([0.01, 0.99])
    model = LinearRegression()
    model.fit(training[["score_rank"]], training["target"].clip(low, high))
    return model


def evaluate(cohorts, start, end, fixed_model=None):
    """Monthly out-of-sample forecasts, using only completed training targets."""
    rows = []
    evaluated = cohorts[cohorts["entry"].between(start, end)]
    for entry, cohort in evaluated.groupby("entry"):
        if fixed_model is None:
            training = cohorts[
                (cohorts["exit_date"] <= cohort["signal"].iloc[0])
                & cohorts["target"].notna()
            ]
            model = fit_model(training)
        else:
            model = fixed_model
        scored = cohort[cohort["target"].notna()].assign(
            forecast=model.predict(cohort.loc[cohort["target"].notna(), ["score_rank"]])
        )
        top = scored.nlargest(bt.TOP_N, "score")
        rows.append({
            "entry": entry,
            "slope": model.coef_[0],
            "rank_ic": spearmanr(scored["forecast"], scored["target"])[0],
            "top30_forecast": top["forecast"].mean(),
            "top30_realized": top["target"].mean(),
            "universe_realized": scored["target"].mean(),
            "benchmark_90d": cohort["benchmark_90d"].iloc[0],
            "rmse_model": np.sqrt(((scored["forecast"] - scored["target"]) ** 2).mean()),
            "rmse_mean": np.sqrt(
                ((model.intercept_ + model.coef_[0] * 0.5 - scored["target"]) ** 2).mean()
            ),
        })
    return pd.DataFrame(rows)


def summarize(label, monthly):
    ic = monthly["rank_ic"]
    excess = monthly["top30_realized"] - monthly["benchmark_90d"]
    summary = pd.Series({
        "months": len(monthly),
        "mean_rank_ic": ic.mean(),
        "ic_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic) / 3),  # 3: overlapping 90-day targets
        "share_positive_ic": (ic > 0).mean(),
        "top30_forecast": monthly["top30_forecast"].mean(),
        "top30_realized": monthly["top30_realized"].mean(),
        "top30_minus_sp500": excess.mean(),
        "rmse_model": monthly["rmse_model"].mean(),
        "rmse_constant": monthly["rmse_mean"].mean(),
    }, name=label)
    print(f"\n{label}\n{summary.round(4).to_string()}")
    return summary


def validation_grid(cohorts):
    """Average 90-day excess over the S&P 500 TR of the top-N, development only."""
    dev = cohorts[cohorts["entry"].ge(DEV_START) & cohorts["exit_date"].le(DEV_END)]
    grid = pd.DataFrame(index=pd.Index(SIZES, name="universe"), columns=NAMES, dtype=float)
    for size in SIZES:
        pool = dev[dev["universe_rank"] <= size]
        for n in NAMES:
            top = pool.sort_values("score", ascending=False).groupby("entry").head(n)
            by_entry = top.groupby("entry").agg(
                realized=("target", "mean"), benchmark=("benchmark_90d", "first")
            )
            grid.loc[size, n] = (by_entry["realized"] - by_entry["benchmark"]).mean()
    print("\nDevelopment grid: average 90-day excess return of the top-N vs. S&P 500 TR")
    print(grid.map(lambda x: f"{x:+.2%}").to_string())
    return grid


def main():
    with duckdb.connect(str(bt.DATABASE), read_only=True) as db:
        calendar, prices, _ = bt.load_inputs(db)
        scores = bt.make_scores(prices)
        cohorts = make_cohorts(db, calendar, prices, scores)

    grid = validation_grid(cohorts)
    cohorts = add_score_rank(cohorts)

    # Validation: refit every month on targets completed before the signal.
    valid = evaluate(cohorts, VALID_START, DEV_END - pd.Timedelta(days=92))
    summaries = [summarize("Validation 2013-2015 (expanding refits)", valid)]

    # Final model: fixed, fitted on targets completed by the end of the buffer.
    training = cohorts[
        cohorts["entry"].ge(DEV_START)
        & cohorts["exit_date"].le(BUFFER_END)
        & cohorts["target"].notna()
    ]
    final_model = fit_model(training)
    print(f"\nFinal model: forecast_90d = {final_model.intercept_:.4f} "
          f"+ {final_model.coef_[0]:.4f} x score_percentile "
          f"({len(training):,} training rows)")
    test = evaluate(cohorts, bt.FINAL_START, bt.FINAL_END, fixed_model=final_model)
    summaries.append(summarize("Final test 2016-2026 (fixed model)", test))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT / "development_grid.csv")
    valid.to_csv(OUTPUT / "validation_monthly.csv", index=False)
    test.to_csv(OUTPUT / "test_monthly.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "model_summary.csv")
    print(f"\nResults: {OUTPUT}")


if __name__ == "__main__":
    main()
