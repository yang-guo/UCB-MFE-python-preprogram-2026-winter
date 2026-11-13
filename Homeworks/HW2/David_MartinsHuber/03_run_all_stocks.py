"""Run parameterized stock reports and collect a portfolio risk summary.

Run from this homework folder:
    python 03_run_all_stocks.py

The requested period is the six complete calendar months before the month of
latest cleaned data. The dated output folder records the run date, not the
historical analysis dates.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import logging
import math
from pathlib import Path
import re
import sqlite3

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import nbformat
import pandas as pd
import papermill as pm
from tqdm import tqdm

HOMEWORK_DIR = Path(__file__).resolve().parent
CLEAN_TABLE = "ohlc_hw_cleaned"
SUMMARY_COLUMNS = [
    "ticker",
    "avg_daily_return",
    "avg_volatility",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "actual_start_date",
    "actual_end_date",
    "trading_days",
    "report_path",
]


def parse_args() -> argparse.Namespace:
    """Parse optional paths and execution-kernel settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=HOMEWORK_DIR / "data" / "data.db",
        help="Working SQLite database containing ohlc_hw_cleaned.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=HOMEWORK_DIR / "02_stock_report_template.ipynb",
        help="Parameterized single-stock notebook.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HOMEWORK_DIR / "reports",
        help="Root folder for reports/YYYY-MM-DD output.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Jupyter kernel used by Papermill.",
    )
    return parser.parse_args()


def read_universe_and_latest(db_path: Path) -> tuple[list[str], pd.Timestamp]:
    """Read normalized ticker names and the latest cleaned observation date."""
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Working database not found: {db_path}. Run Part 1 first or pass --db-path."
        )
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        table_names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", connection
        )["name"].tolist()
        if CLEAN_TABLE not in table_names:
            raise RuntimeError(
                f"Table {CLEAN_TABLE} is missing from {db_path}. Run 01_data_cleaning.ipynb first."
            )
        rows = pd.read_sql_query(
            f"SELECT ticker, MAX(ts) AS latest_ts FROM {CLEAN_TABLE} "
            "GROUP BY ticker ORDER BY ticker",
            connection,
        )
    rows["ticker"] = rows["ticker"].astype("string").str.strip().str.upper()
    rows = rows.dropna(subset=["ticker", "latest_ts"])
    if rows.empty:
        raise RuntimeError(f"{CLEAN_TABLE} has no usable ticker/date records.")
    tickers = rows["ticker"].drop_duplicates().sort_values().tolist()
    latest = pd.to_datetime(rows["latest_ts"], errors="coerce").max()
    if pd.isna(latest):
        raise RuntimeError(f"Could not parse dates from {CLEAN_TABLE}.ts.")
    return tickers, latest


def six_complete_month_window(latest_observation: str | pd.Timestamp) -> tuple[str, str]:
    """Return the six full calendar months before latest_observation's month."""
    latest = pd.Timestamp(latest_observation).normalize()
    containing_month_start = latest.to_period("M").start_time
    end = containing_month_start - pd.Timedelta(days=1)
    start = containing_month_start - pd.DateOffset(months=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def finite_number(value: object) -> float:
    """Return a finite float, or NaN for absent/non-numeric metrics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def safe_report_stem(ticker: str) -> str:
    """Keep report filenames portable while preserving ordinary ticker symbols."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker.strip())
    if not stem:
        raise ValueError(f"Ticker {ticker!r} cannot form a report filename.")
    return stem


def collect_metrics(report_path: Path) -> dict[str, object]:
    """Read the hidden, named metrics output from one successful notebook."""
    executed = nbformat.read(report_path, as_version=4)
    records = [
        output.metadata["hw2_report_metrics"]
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if "hw2_report_metrics" in output.get("metadata", {})
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected one hw2_report_metrics output in {report_path}; found {len(records)}."
        )
    metrics = records[0]
    if not isinstance(metrics, dict):
        raise TypeError(f"hw2_report_metrics must be a dictionary in {report_path}")
    return metrics


def generate_one_report(
    ticker: str,
    start_date: str,
    end_date: str,
    db_path: Path,
    template_path: Path,
    report_path: Path,
    kernel_name: str,
) -> dict[str, object]:
    """Execute one ticker report and return its named metrics."""
    pm.execute_notebook(
        input_path=str(template_path),
        output_path=str(report_path),
        parameters={
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "db_path": str(db_path.resolve()),
        },
        kernel_name=kernel_name,
        log_output=False,
        progress_bar=False,
    )
    return collect_metrics(report_path)


def create_summary_chart(summary: pd.DataFrame, output_path: Path) -> bool:
    """Save the portfolio risk-return scatter; return False if no plottable rows."""
    required = ["avg_volatility", "total_return", "sharpe_ratio"]
    plot_data = summary.dropna(subset=required).copy()
    finite_mask = np.isfinite(plot_data[required].to_numpy(dtype=float)).all(axis=1)
    plot_data = plot_data.loc[finite_mask]
    if plot_data.empty:
        return False

    fig, ax = plt.subplots(figsize=(12, 8))
    sharpe_min = float(plot_data["sharpe_ratio"].min())
    sharpe_max = float(plot_data["sharpe_ratio"].max())
    if sharpe_min == sharpe_max:
        padding = max(abs(sharpe_min) * 0.01, 0.01)
        color_limits = (sharpe_min - padding, sharpe_max + padding)
    else:
        color_limits = (sharpe_min, sharpe_max)
    scatter = ax.scatter(
        plot_data["avg_volatility"],
        plot_data["total_return"],
        c=plot_data["sharpe_ratio"],
        cmap="viridis",
        vmin=color_limits[0],
        vmax=color_limits[1],
        s=90,
        edgecolors="#0f172a",
        linewidths=0.5,
        alpha=0.88,
    )
    for _, row in plot_data.nlargest(min(5, len(plot_data)), "total_return").iterrows():
        ax.annotate(
            row["ticker"],
            (row["avg_volatility"], row["total_return"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Sharpe Ratio")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=15, fontweight="bold")
    ax.set_xlabel("Average annualized rolling volatility")
    ax.set_ylabel("Total return")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def run_batch(args: argparse.Namespace) -> int:
    """Run every ticker, log failures, and write current-run summary artifacts."""
    db_path = args.db_path.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Report template not found: {template_path}")

    tickers, latest_date = read_universe_and_latest(db_path)
    start_date, end_date = six_complete_month_window(latest_date)
    run_date = date.today().strftime("%Y-%m-%d")
    output_dir = output_root / run_date
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Cleaned tickers: %d", len(tickers))
    logging.info("Latest cleaned observation: %s", latest_date.strftime("%Y-%m-%d"))
    logging.info("Common analysis window: %s through %s", start_date, end_date)
    logging.info("Run output directory: %s", output_dir)

    successful_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    current_run_reports: list[str] = []

    for ticker in tqdm(tickers, desc="Generating reports"):
        stem = safe_report_stem(ticker)
        report_path = output_dir / f"{stem}_report.ipynb"
        try:
            metrics = generate_one_report(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                db_path=db_path,
                template_path=template_path,
                report_path=report_path,
                kernel_name=args.kernel_name,
            )
            successful_rows.append({
                "ticker": ticker,
                "avg_daily_return": finite_number(metrics.get("avg_daily_return")),
                "avg_volatility": finite_number(metrics.get("avg_volatility")),
                "max_drawdown": finite_number(metrics.get("max_drawdown")),
                "sharpe_ratio": finite_number(metrics.get("sharpe_ratio")),
                "total_return": finite_number(metrics.get("total_return")),
                "actual_start_date": metrics.get("actual_start_date"),
                "actual_end_date": metrics.get("actual_end_date"),
                "trading_days": metrics.get("trading_days"),
                "report_path": (
                    str(report_path.relative_to(HOMEWORK_DIR))
                    if report_path.is_relative_to(HOMEWORK_DIR)
                    else str(report_path.resolve())
                ),
            })
            current_run_reports.append(str(report_path))
        except Exception as error:  # Keep the batch running if one report fails.
            logging.exception("Report failed for %s", ticker)
            failures.append({"ticker": ticker, "error": f"{type(error).__name__}: {error}"})

    summary = pd.DataFrame(successful_rows, columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(
            "sharpe_ratio", ascending=False, na_position="last"
        ).reset_index(drop=True)
    summary_path = output_dir / "portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)

    failed_path = output_dir / "failed_stocks.txt"
    failed_path.write_text(
        "".join(f"{entry['ticker']}\t{entry['error']}\n" for entry in failures),
        encoding="utf-8",
    )
    manifest = {
        "run_date": run_date,
        "latest_cleaned_observation": latest_date.strftime("%Y-%m-%d"),
        "analysis_start_date": start_date,
        "analysis_end_date": end_date,
        "requested_ticker_count": len(tickers),
        "successful_tickers": [row["ticker"] for row in successful_rows],
        "failed_tickers": failures,
        "current_run_reports": current_run_reports,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Successfully generated {len(successful_rows)}/{len(tickers)} reports")
    if not summary.empty:
        ranked = summary.loc[np.isfinite(summary["sharpe_ratio"].to_numpy(dtype=float))]
        if ranked.empty:
            print("No report has an estimable Sharpe ratio; rankings are unavailable.")
        else:
            ranked = ranked.sort_values("sharpe_ratio", ascending=False)
            print("\nTop 10 stocks by Sharpe ratio:")
            print(ranked.head(10).to_string(index=False))
            print("\nBottom 10 stocks by Sharpe ratio:")
            print(ranked.tail(10).to_string(index=False))
        chart_path = output_dir / "risk_return_scatter.png"
        if create_summary_chart(summary, chart_path):
            print(f"\nSaved risk-return chart: {chart_path}")
        else:
            print("No finite volatility/return/Sharpe rows; no portfolio chart was created.")
    else:
        print("No reports succeeded. The CSV contains headers only; no portfolio summary or chart is available.")

    print(f"Portfolio CSV: {summary_path}")
    print(f"Failure log: {failed_path}")
    return 0 if not failures else 1


def main() -> int:
    """Run the report batch from command-line arguments."""
    return run_batch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
