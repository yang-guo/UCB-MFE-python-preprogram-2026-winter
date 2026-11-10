"""Evaluate a frozen model once; this script does not fit any estimator."""

from __future__ import annotations

import argparse
import hashlib
import json

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import UnresolvedPriceGap, performance, simulate
from research import (
    DATABASE, HERE, OUTPUT, KalmanRegression, build_panel,
    predict_candidate, save_json,
)


def save_partial(error, directory):
    """Keep the failed path for investigation, without calling it a result."""
    if error.partial is not None:
        for name, frame in error.partial.items():
            frame.to_csv(directory / f"partial_{name}.csv", index=False)


def load_quotes(start, end, symbols):
    """Read actual adjusted quotes; do not fill unavailable rows."""
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        data = connection.execute(
            """SELECT date,symbol,adj_close FROM prices
               WHERE date BETWEEN ? AND ?
               AND symbol IN (SELECT unnest(?)) ORDER BY date,symbol""",
            [str(start), str(end), list(symbols)],
        ).fetchdf()
        benchmark = connection.execute(
            "SELECT date,adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
            [str(start), str(end)],
        ).fetchdf()
    data["date"] = data.date.astype("datetime64[ns]")
    benchmark["date"] = benchmark.date.astype("datetime64[ns]")
    data.loc[~np.isfinite(data.adj_close) | data.adj_close.le(0), "adj_close"] = np.nan
    return data.pivot(index="date", columns="symbol", values="adj_close"), benchmark


def save_results(result, benchmark, config, start, end, directory):
    directory.mkdir(exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    initial = config["initial_capital"]
    benchmark = benchmark.copy()
    benchmark["nav"] = initial * benchmark.adj_close / benchmark.adj_close.iloc[0]
    benchmark.to_csv(directory / "benchmark.csv", index=False)
    strategy_metrics = performance(result["history"], initial, start, end)
    benchmark_metrics = performance(benchmark, initial, start, end)
    comparison = pd.DataFrame({"strategy": strategy_metrics, "benchmark": benchmark_metrics})
    comparison.to_csv(directory / "performance.csv")
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(result["history"].date, result["history"].nav / initial - 1,
              label="Frozen model strategy, net of costs")
    axis.plot(benchmark.date, benchmark.nav / initial - 1, label="S&P 500 total return")
    axis.set(ylabel="Cumulative return", xlabel="Date", title="Fixed-rule portfolio comparison")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "cumulative_returns.png", dpi=140)
    plt.close(fig)
    save_json(directory / "evaluation_status.json", {
        "status": "completed", "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "stale_valuation_days": int(result["history"].stale_lots.gt(0).sum()),
        "data_incidents": len(result["incidents"]),
        "final_open_lots": len(result["open_lots"]),
    })
    return comparison


def development_backtest():
    """Use only the locked winner's 2015 predictions; do not tune the rules."""
    config = json.loads((OUTPUT / "selected_before_confirmation.json").read_text())
    predictions = pd.read_csv(OUTPUT / "confirmation_predictions.csv", parse_dates=["date"])
    predictions = predictions[["date", "symbol", "prediction"]]
    start, end = "2015-02-02", "2015-12-31"
    prices, benchmark = load_quotes("2015-01-30", end, predictions.symbol.unique())
    calendar = pd.DatetimeIndex(benchmark.date)
    target = OUTPUT / "development_backtest"
    target.mkdir(exist_ok=True)
    try:
        result = simulate(prices, calendar, predictions, start, end, config["strategy"])
    except UnresolvedPriceGap as error:
        save_partial(error, target)
        save_json(target / "evaluation_status.json", {"status": "unresolved_data_gap", "reason": str(error)})
        print(str(error), flush=True)
        return
    print(save_results(result, benchmark.loc[benchmark.date.ge(start)],
                       config["strategy"], start, end, target).to_string(), flush=True)


def final_test():
    """Require an unchanged source/model fingerprint and a fresh test log."""
    frozen_path = OUTPUT / "frozen_experiment.json"
    if not frozen_path.exists():
        raise RuntimeError("Develop and freeze an experiment first")
    frozen = json.loads(frozen_path.read_text())
    for file, expected in frozen["source_sha256"].items():
        if hashlib.sha256((HERE / file).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed after freeze: {file}")
    model_path = OUTPUT / "frozen_model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != frozen["model_sha256"]:
        raise RuntimeError("Frozen model changed")
    target = OUTPUT / "final_test"
    target.mkdir(exist_ok=True)
    log = target / "test_started.json"
    if log.exists():
        raise RuntimeError("The final test has already started; retain its result without retuning")
    save_json(log, {"started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "freeze_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest()})
    config = frozen["config"]
    start, end = config["test_start"], config["test_end"]
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        previous = connection.execute(
            "SELECT max(date) FROM calendar WHERE date < ?", [start]
        ).fetchone()[0]
    panel, calendar = build_panel(end, str(previous), labels=False)
    trained = joblib.load(model_path)
    predictions = panel[["date", "symbol"]].copy()
    predictions["prediction"] = predict_candidate(
        trained["model"], trained["features"], panel
    )
    predictions.to_csv(target / "predictions.csv", index=False)
    prices, benchmark = load_quotes(previous, end, predictions.symbol.unique())
    try:
        result = simulate(prices, calendar, predictions, start, end, config["strategy"])
    except UnresolvedPriceGap as error:
        save_partial(error, target)
        save_json(target / "evaluation_status.json", {
            "status": "unresolved_data_gap", "reason": str(error),
            "model": frozen["model"], "no_test_period_retraining": True,
        })
        print(f"Final evaluation unresolved: {error}", flush=True)
        return
    comparison = save_results(
        result, benchmark.loc[benchmark.date.ge(start)],
        config["strategy"], start, end, target,
    )
    print(comparison.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["development", "final"])
    arguments = parser.parse_args()
    if arguments.command == "development":
        development_backtest()
    else:
        final_test()
