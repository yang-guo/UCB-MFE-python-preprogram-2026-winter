"""Reproducible pre-2016 baseline; intentionally does not run the final test.

Reuses the permitted data/preprocessing helpers in Lectures/modeling/course_data.py.
Trading and portfolio accounting are implemented independently in backtester.py.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

from backtester import StrategyConfig, entry_schedule, performance, run_backtest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Lectures/modeling"))
from course_data import (  # noqa: E402
    NUMERIC_FEATURES, apply_membership, financial_events, join_financials,
    load_data, load_membership, make_labels, make_pipeline, price_features,
)


def prepare_development(database, batch_size=50):
    """Keep reads/features bounded; never load post-2015 research observations."""
    with duckdb.connect(str(database), read_only=True) as con:
        symbols = [row[0] for row in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    membership = load_membership(database, end="2015-12-31")
    panels, price_parts = [], []
    benchmark = None
    for offset in range(0, len(symbols), batch_size):
        prices, fundamentals, universe, benchmark = load_data(
            database, end="2015-12-31", symbols=symbols[offset:offset + batch_size]
        )
        if prices.empty:
            continue
        panel = price_features(prices, benchmark.date)
        panel = join_financials(panel, financial_events(fundamentals), universe)
        panel = apply_membership(panel, membership, benchmark.date)
        # Unlike the lecture's optional delayed exits, the initial HW4 baseline
        # labels require the exact maturity session. Gaps remain censored.
        panel = make_labels(panel, prices, benchmark.date, max_exit_delay=0)
        panels.append(panel)
        price_parts.append(prices)
    if not panels:
        raise ValueError("Database contains no usable development prices")
    return (pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]),
            pd.concat(price_parts, ignore_index=True), benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path,
                        default=ROOT / "Lectures/modeling/data/equities_class.duckdb")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/development")
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"Missing database: {args.database}")
    print("Preparing development data through 2015 only...", flush=True)
    panel, prices, benchmark = prepare_development(args.database)
    calendar = pd.DatetimeIndex(benchmark.date)
    entries = entry_schedule(calendar, "2013-01-01", "2015-12-31")
    signal_dates = calendar[calendar.get_indexer(entries) - 1]
    fit_before = signal_dates[0]
    train = panel.loc[
        panel.eligible & panel.date.ge("2010-01-01") & panel.date.lt(fit_before)
        & panel.label_end.lt(fit_before) & panel.target.notna()
    ].copy()
    if train.empty:
        raise ValueError("No completed training outcomes before the first signal")
    # No test-label completeness filter here: scores must not depend on future
    # survival, future quotes, or whether the outcome can eventually be measured.
    decision_rows = panel.loc[panel.eligible & panel.date.isin(signal_dates)].copy()
    model = make_pipeline(numeric_features=NUMERIC_FEATURES, include_sector=False)
    model.fit(train[NUMERIC_FEATURES], train.target)
    scores = decision_rows[["date", "symbol"]].copy()
    scores["prediction"] = model.predict(decision_rows[NUMERIC_FEATURES])
    config = StrategyConfig()
    print(f"Training rows: {len(train):,}; monthly scores: {len(scores):,}", flush=True)
    result = run_backtest(prices, calendar, scores, entries[0], "2015-12-31", config)
    bench_prices = benchmark.set_index("date").adj_close.reindex(result.history.index)
    result.history["benchmark_nav"] = config.initial_cash * bench_prices / bench_prices.iloc[0]
    metrics = {
        "strategy": performance(result.history.nav, config.initial_cash),
        "benchmark": performance(result.history.benchmark_nav, config.initial_cash),
    }
    # Prediction quality is reported separately from investable portfolio returns.
    scored = decision_rows[["date", "symbol", "target", "label_end"]].merge(scores)
    measurable = scored.loc[scored.target.notna()]
    prediction_audit = {
        "scored_rows": len(scored), "completed_scored_labels": len(measurable),
        "censored_scored_labels": int(scored.target.isna().sum()),
        "rmse": float(np.sqrt(np.mean((measurable.prediction - measurable.target) ** 2)))
        if len(measurable) else None,
    }
    with args.database.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    run = {
        "stage": "development_only_not_final_results",
        "database": args.database.name, "database_sha256": checksum,
        "training_start": "2010-01-01", "fit_before_signal": str(fit_before.date()),
        "latest_training_outcome": str(train.label_end.max().date()),
        "validation_start": str(entries[0].date()), "validation_end": "2015-12-31",
        "training_rows": len(train), "features": NUMERIC_FEATURES,
        "model": "Ridge(alpha=100), fitted once; training-only preprocessing",
        "strategy": asdict(config), "prediction_audit": prediction_audit,
        "metrics": metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(args.output / "daily_portfolio.csv")
    result.fills.to_csv(args.output / "fills.csv", index=False)
    result.open_lots.to_csv(args.output / "open_lots.csv", index=False)
    result.diagnostics.to_csv(args.output / "diagnostics.csv", index=False)
    scored.to_csv(args.output / "prediction_audit.csv", index=False)
    (args.output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    print(f"Development run complete. Files saved under {args.output}")
    print("Final test period (2016–2026) has not been evaluated.")


if __name__ == "__main__":
    main()
