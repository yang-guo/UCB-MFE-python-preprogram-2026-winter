"""Step 3: fit on the full universe through the 2016 buffer, then run the test.

The pipeline is the lecture Ridge pipeline. Training rows are eligible signals
from 2010-01-01 through 2015-12-31 whose 90-day labels end before 2016-04-01.
Scores after that are out of sample. The backtest buys the top 10 predictions
at the first close of each month and holds for 90 calendar days.

Benchmark: $1,000,000 in ^SP500TR at the first test close, held to the end.
"""

import gc
import sys
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import run_backtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODELING = REPO / "Lectures" / "modeling"
sys.path.insert(0, str(MODELING))

import course_data as cd  # noqa: E402

DB = MODELING / "data" / "equities_top2000.duckdb"
ARTIFACTS = HERE / "artifacts"
TEST_START = "2016-04-01"
TEST_END = "2026-09-30"
TRAIN_START = "2010-01-01"
LAST_DEV_SIGNAL = "2015-12-31"


def performance(nav, annual_rf=0.02):
    nav = nav.sort_index()
    returns = nav.pct_change().dropna()
    excess = returns - ((1 + annual_rf) ** (1 / 252) - 1)
    volatility = returns.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).total_seconds() / (365.25 * 24 * 3600)
    return pd.Series({
        "final_value": nav.iloc[-1],
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "Sharpe": np.sqrt(252) * excess.mean() / volatility,
    })


SCORE_START = "2016-03-01"
BATCH_SIZE = 200
TRAIN_COLUMNS = ["date", "symbol", "target", *cd.FEATURES]
SCORE_COLUMNS = ["date", "symbol", *cd.FEATURES]


def batched_frames(symbols):
    """Build one symbol batch and keep only the rows the fit and backtest use."""
    prices, fundamentals, universe, benchmark = cd.load_data(DB, end=TEST_END, symbols=symbols)
    if prices.empty:
        return None
    calendar = pd.DatetimeIndex(benchmark.date)
    membership = cd.load_membership(DB, end=TEST_END)
    membership = membership.loc[membership.symbol.isin(symbols)]
    panel = cd.price_features(prices, calendar)
    panel = cd.join_financials(panel, cd.financial_events(fundamentals), universe)
    panel = cd.apply_membership(panel, membership, calendar)
    panel = cd.make_labels(panel, prices, calendar)

    trainable = (
        panel.eligible & panel.target.notna()
        & panel.date.between(TRAIN_START, LAST_DEV_SIGNAL)
        & panel.label_end.lt(TEST_START)
    )
    to_score = panel.eligible & panel.date.between(SCORE_START, TEST_END)
    in_window = panel.date.ge(SCORE_START) & panel.adj_close.notna()
    kept = (
        panel.loc[trainable, TRAIN_COLUMNS].copy(),
        panel.loc[to_score, SCORE_COLUMNS].copy(),
        panel.loc[in_window, ["date", "symbol", "adj_close"]].copy(),
    )
    del panel, prices, fundamentals, universe, benchmark, membership
    return kept


def write_parts(limit):
    parts = ARTIFACTS / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for old in parts.glob("*.parquet"):
        old.unlink()
    with duckdb.connect(str(DB), read_only=True) as con:
        symbols = [row[0] for row in con.execute("SELECT DISTINCT symbol FROM universe ORDER BY symbol").fetchall()]
    if limit:
        symbols = symbols[:limit]
    print(f"symbols={len(symbols)} batch_size={BATCH_SIZE}", flush=True)
    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]
        kept = batched_frames(batch)
        tag = f"{start:05d}"
        if kept is not None:
            train, score, closes = kept
            train.to_parquet(parts / f"train_{tag}.parquet", index=False)
            score.to_parquet(parts / f"score_{tag}.parquet", index=False)
            closes.to_parquet(parts / f"close_{tag}.parquet", index=False)
            print(f"batch {tag} train={len(train)} score={len(score)} closes={len(closes)}", flush=True)
            del train, score, closes
        else:
            print(f"batch {tag} empty", flush=True)
        gc.collect()


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    write_parts(limit)
    parts = ARTIFACTS / "parts"
    train_paths = sorted(parts.glob("train_*.parquet"))
    if not train_paths:
        raise RuntimeError("no training rows were written")
    train = pd.concat([pd.read_parquet(path) for path in train_paths], ignore_index=True)
    print(f"training rows={len(train)}", flush=True)
    pipeline = cd.make_pipeline()
    pipeline.fit(train[cd.FEATURES], train["target"])
    del train
    gc.collect()
    model_path = ARTIFACTS / "ridge_90d_full.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": list(cd.FEATURES),
            "model_version": "ridge-alpha100-top2000-2010-2015",
            "train_start": TRAIN_START,
            "last_signal": LAST_DEV_SIGNAL,
            "label_end_before": TEST_START,
        },
        model_path,
    )
    print("wrote", model_path, flush=True)

    # March 2016 is the signal for the first April fill. Later scores stay out of sample.
    score = pd.concat([pd.read_parquet(path) for path in sorted(parts.glob("score_*.parquet"))], ignore_index=True)
    scored = score[["date", "symbol"]].copy()
    scored["score"] = pipeline.predict(score[cd.FEATURES])
    del score
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"test scores={len(scored)}", flush=True)

    closes = pd.concat([pd.read_parquet(path) for path in sorted(parts.glob("close_*.parquet"))], ignore_index=True)
    prices = closes.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    del closes
    result = run_backtest(prices, scored, TEST_START, TEST_END)
    history = result["history"].set_index("date")

    with duckdb.connect(str(DB), read_only=True) as con:
        benchmark = con.execute(
            "SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date",
            [TEST_END],
        ).fetchdf()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.set_index("date").adj_close.sort_index()
    first_test = history.index[history.index >= TEST_START][0]
    history["benchmark_nav"] = np.nan
    history.loc[history.index < first_test, "benchmark_nav"] = 1_000_000.0
    history.loc[history.index >= first_test, "benchmark_nav"] = (
        1_000_000.0 * benchmark.reindex(history.index).loc[first_test:] / benchmark.loc[first_test]
    )

    summary = pd.DataFrame({
        "strategy": performance(history.nav),
        "SP500TR": performance(history.benchmark_nav),
    })
    print(summary.to_string(), flush=True)
    print(f"open_lots={len(result['open_lots'])} fills={len(result['fills'])}", flush=True)
    summary.to_csv(ARTIFACTS / "test_summary.csv")
    history.to_csv(ARTIFACTS / "test_history.csv")

    growth = history[["nav", "benchmark_nav"]].div(1_000_000.0)
    ax = growth.plot(title="Growth of $1,000,000", figsize=(9, 4.5))
    ax.set_ylabel("Portfolio value / starting capital")
    ax.legend(["Ridge top 10", "S&P 500 total return"])
    plt.tight_layout()
    chart = ARTIFACTS / "cumulative_return.png"
    plt.savefig(chart, dpi=120)
    print("wrote", chart, flush=True)


if __name__ == "__main__":
    main()
