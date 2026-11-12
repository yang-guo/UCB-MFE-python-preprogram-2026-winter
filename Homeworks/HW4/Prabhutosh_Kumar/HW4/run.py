"""Run from any directory: python /path/to/run.py --config /path/to/config.json."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from strategy.data import build_dataset, ROOT
from strategy.model import select, fit
from strategy.backtest import run, metrics

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if config["holding_days"] != 90:
        raise ValueError("HW4 requires a 90-calendar-day holding period")
    if not (0 <= config["fee"] < 1 and 0 < config["tranche_fraction"] <= 1
            and 0 < config["max_stock_weight"] <= 1 and config["top_n"] >= 1):
        raise ValueError("Invalid allocation or fee configuration")
    if config["end"] != "2026-09-30":
        raise ValueError("Final HW4 evaluation must end on 2026-09-30")
    panel, prices, benchmark, extremes, ranking = build_dataset(config)
    missing = set(config["features"]) - set(panel.columns)
    if missing:
        raise ValueError(f"Unknown feature names: {missing}")
    # Only the explicit causal feature registry may be used as predictors.
    from strategy.data import course
    allowed = set(course.NUMERIC_FEATURES + ["reversal_5", "volatility_21"])
    if not set(config["features"]) <= allowed:
        raise ValueError("Register new causal features in run.py; target/audit columns are forbidden")
    winner, validation, audits = select(panel, config)
    validation.to_csv(output / "validation.csv", index=False)
    # Save the settled choice before generating any final-test predictions.
    locked = {**config, "selected_alpha": winner, "retraining": "none", "selection": "mean 2013-2015 rank IC"}
    (output / "locked_config.json").write_text(json.dumps(locked, indent=2))
    print(f"Development selected Ridge alpha={winner}; final test configuration frozen", flush=True)
    model, audit = fit(panel, pd.Timestamp("2016-04-01"), winner, config["features"])
    audits.append(audit)
    test = panel.loc[panel.date.ge("2016-03-31") & panel.date.lt(config["end"])].copy()
    test["prediction"] = model.predict(test[config["features"]])
    scores = test.pivot(index="date", columns="symbol", values="prediction")
    daily, trades, lots, incidents = run(prices, scores, "2016-04-01", config["end"], config)
    b = benchmark.reindex(daily.index)
    if b.isna().any() or (b <= 0).any():
        raise ValueError("Incomplete benchmark")
    daily["benchmark"] = config["capital"] * b / b.iloc[0]
    performance = pd.DataFrame({"strategy": metrics(daily.value, config["capital"]),
                                "SP500TR": metrics(daily.benchmark, config["capital"])}).T
    performance.to_csv(output / "performance.csv", index_label="portfolio")
    # Larger row-level files stay local and out of Git.
    artifacts = HERE / "outputs" / output.name
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, frame in [("daily", daily), ("trades", trades), ("open_lots", lots),
                        ("incidents", incidents), ("predictions", test),
                        ("extreme_price_moves", extremes), ("ranking_flags", ranking)]:
        frame.to_csv(artifacts / f"{name}.csv", index=name == "daily")
    pd.DataFrame(audits).to_csv(output / "fit_audit.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label in [("value", "Strategy (provisional marks)"), ("benchmark", "S&P 500 total return")]:
        ax.plot(daily.index, daily[column] / config["capital"] - 1, label=label)
    ax.set(title="HW4: April 2016 – September 2026", ylabel="Cumulative return")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cumulative_returns.png", dpi=150)
    plt.close(fig)
    stale = daily.unresolved_value.iloc[-1]
    gap = performance.loc["strategy", "final_value"] - performance.loc["SP500TR", "final_value"]
    cagr_gap = performance.loc["strategy", "CAGR"] - performance.loc["SP500TR", "CAGR"]
    lines = ["# Findings", "", "## Measured result", "",
             "These are research results using supplied adjusted prices and provisional stale marks.", "",
             "| Portfolio | Final value | CAGR | Maximum drawdown | Sharpe |", "|---|---:|---:|---:|---:|"]
    for name, row in performance.iterrows():
        lines.append(f"| {name} | ${row.final_value:,.2f} | {row.CAGR:.2%} | {row.max_drawdown:.2%} | {row.Sharpe:.3f} |")
    lines += ["", "![Cumulative returns](cumulative_returns.png)", "",
              f"Strategy minus benchmark: ${gap:+,.2f}; CAGR difference {100*cagr_gap:+.2f} percentage points.",
              f"The computed strategy {'beats' if gap > 0 else 'does not beat'} the benchmark under these assumptions.",
              "", "## Reliability and accounting", "",
              f"Final cash ${daily.cash.iloc[-1]:,.2f}; open-position value ${daily.holdings.iloc[-1]:,.2f}. "
              f"There are {len(lots)} open lots. Fees paid: ${trades.fee.sum():,.2f}.",
              f"Final exposure without a quote for more than {config['stale_limit']} sessions: ${stale:,.2f} "
              f"({stale/daily.value.iloc[-1]:.2%} of NAV). Maximum such exposure: "
              f"{(daily.unresolved_value/daily.value).max():.2%}.",
              "Stale lots are frozen at their last observed mark and cannot release cash until an actual quote returns. "
              "This is a provisional accounting convention, not an assumed executable exit or verified recovery. "
              "Any unresolved exposure prevents a reliable economic performance claim. Set strict_prices=true to stop after five missing sessions.",
              f"The price audit found {len(extremes):,} daily moves exceeding 50% in absolute size; "
              f"the ranking audit flagged {len(ranking):,} unusually large price/share inputs. "
              "These are investigation flags, not verified errors. Raw adjusted prices were not heuristically rewritten.",
              "Cash reconciliation, nonnegative cash, prior-session signals, and final cash plus holdings were checked during the run. "
              "Synthetic unit tests separately check fees, exact maturity, weekend maturity, missing entries/exits, and first-day loss measurement.",
              "", "## Model, validation and strategy", "",
              f"Ridge alpha {winner:g} was selected by mean cross-sectional rank correlation in 2013, 2014 and 2015. "
              "Each annual fit uses signals from 2010 onward and only outcomes completed before January 1. "
              "Validation outcomes must finish by December 31, 2015. No final-test return selects a parameter. "
              "Validation is predictive, not a portfolio return; overlapping 90-day labels are not compounded.",
              f"The selected model's mean validation rank IC was {validation.loc[validation.alpha.eq(winner), 'rank_IC'].mean():.4f}. "
              "A low or negative rank IC is evidence of weak prediction, even for the best candidate. "
              "This is a completed modeling experiment, not evidence that the fitted model is useful for investing.",
              "Final fit is April 1, 2016, using development signals through December 2015 whose outcomes have completed. "
              "There is no test-period retraining. Imputation, clipping, scaling and target clipping use each training set only.",
              f"Each month, buy the top {config['top_n']} scores using up to {config['tranche_fraction']:.0%} of NAV, "
              f"limited by previously available cash and a {config['max_stock_weight']:.0%} per-stock planned weight. "
              "The cap is measured at planning prices; overnight moves may breach it. Multiple lots can coexist. "
              "No position is sold early to rebalance. Each lot exits after 90 calendar days at the next quoted market close.",
              "", "## Interpretation and next experiments", "",
              "The model trades estimated gross 90-day returns; its modest linear structure is easy to inspect and regularize. "
              "Momentum, reversal, volatility, drawdown, profitability, leverage and cash-flow ratios describe different risks. "
              "The equal-weight selected-stock portfolio differs from a capitalization-weighted index and can have size, industry and style concentration. "
              "Cash waiting for new monthly entries and unresolved exits can materially change returns.",
              "All regularization candidates, including unsuccessful ones, are recorded in validation.csv. "
              "No post-test strategy search was performed. Next experiments: verified corporate-action and delisting proceeds, "
              "sector caps, alternative feature groups, and fee sensitivity evaluated on development data before a new untouched test.",
              "The database uses quarterly capitalization proxies and snapshot metadata, incomplete delisting histories, "
              "and restated financial values. Filing-date joins do not restore original filing vintages. "
              "Costs are a fixed proportional approximation; market impact, spread variation, taxes and interest on cash are omitted. "
              "Adjusted-price units already include the supplied dividend adjustments; dividends are not added again.",
              "", "## Provenance", "",
              "The attached hw4.ipynb was read as a reference, not copied. Data loading, base features, filing joins, "
              "membership, labels and the preprocessing pipeline reuse Lectures/modeling/course_data.py. "
              "The lot backtester and research runner were written for this submission with OpenAI Codex assistance. "
              "The student should review the code and explain these assumptions."]
    (output / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    fingerprint = hashlib.sha256()
    with (ROOT / config["database"]).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            fingerprint.update(block)
    (output / "run_metadata.json").write_text(json.dumps({"database_sha256": fingerprint.hexdigest(),
        "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "full_ledger_directory": str(artifacts.relative_to(HERE))}, indent=2))
    print(performance.to_string(), flush=True)
    print(f"Saved results in {output}; full ledgers in {artifacts}", flush=True)


if __name__ == "__main__":
    main()
