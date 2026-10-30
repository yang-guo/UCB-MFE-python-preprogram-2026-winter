"""Generate one Papermill risk report per ticker and a portfolio summary.

Run from this submission directory, for example:
    python 03_run_all_stocks.py --db-path data/data.db

The analysis window is the six complete calendar months before the month
containing the latest cleaned observation. Output date directories use the
actual run date and are independent of that historical analysis window.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import traceback

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import papermill as pm
from tqdm import tqdm


HOMEWORK_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = HOMEWORK_DIR / "02_stock_report_template.ipynb"
CLEAN_TABLE = "ohlc_hw_cleaned"
SUMMARY_COLUMNS = [
    "ticker",
    "avg_daily_return",
    "avg_volatility",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "report_path",
]


def prepare_papermill_template(template_path: Path, staging_dir: Path) -> Path:
    """Make a Papermill-ready temporary copy without modifying the source.

    The assignment requires a cell tagged ``parameters``. If the source notebook
    was edited with a cell editor that cannot modify cell metadata, add the tag
    to a temporary copy. Ensure a standard Python kernelspec is present too.
    """
    notebook = nbformat.read(template_path, as_version=4)
    parameter_cells = [
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
    ]
    if not parameter_cells:
        candidates = [
            cell for cell in notebook.cells
            if cell.cell_type == "code" and "# Parameters" in cell.source
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Template must contain exactly one '# Parameters' code cell "
                "or a code cell tagged 'parameters'."
            )
        candidates[0].metadata["tags"] = ["parameters"]
    if not notebook.metadata.get("kernelspec"):
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    nbformat.validate(notebook)
    staged_path = staging_dir / "02_stock_report_template_papermill.ipynb"
    nbformat.write(notebook, staged_path)
    return staged_path


def parse_args() -> argparse.Namespace:
    """Parse paths supplied by the user while keeping defaults portable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=HOMEWORK_DIR / "data" / "data.db",
        help="Local working database containing ohlc_hw_cleaned.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HOMEWORK_DIR / "reports",
        help="Root directory for dated report runs.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE_PATH,
        help="Parameterized stock-report notebook.",
    )
    return parser.parse_args()


def analysis_window(latest_observation: str | pd.Timestamp) -> tuple[str, str]:
    """Return the six complete months before latest_observation's month.

    For example, if the latest cleaned observation is 2024-12-06, the window
    is 2024-06-01 through 2024-11-30. The latest day does not certify that its
    month has complete data.
    """
    latest = pd.Timestamp(latest_observation).normalize()
    current_month_start = latest.to_period("M").start_time
    end = current_month_start - pd.Timedelta(days=1)
    start = (current_month_start - pd.DateOffset(months=6)).normalize()
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def load_tickers_and_latest(db_path: Path) -> tuple[list[str], pd.Timestamp]:
    """Read ticker coverage and the latest date without modifying the DB."""
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Download data.db from bCourses and "
            "place a working copy at the path supplied with --db-path."
        )
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", connection
        )["name"].tolist()
        if CLEAN_TABLE not in tables:
            raise RuntimeError(
                f"Table {CLEAN_TABLE!r} is missing from {db_path}. Run "
                "01_data_cleaning.ipynb first."
            )
        frame = pd.read_sql_query(
            f"SELECT ticker, MAX(ts) AS latest_ts FROM {CLEAN_TABLE} "
            "GROUP BY ticker ORDER BY ticker",
            connection,
        )
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame = frame.dropna(subset=["ticker", "latest_ts"])
    if frame.empty:
        raise RuntimeError(f"{CLEAN_TABLE} contains no usable tickers/dates.")
    tickers = frame["ticker"].drop_duplicates().sort_values().tolist()
    latest = pd.to_datetime(frame["latest_ts"], errors="coerce").max()
    if pd.isna(latest):
        raise RuntimeError(f"Could not parse any dates in {CLEAN_TABLE}.ts.")
    return tickers, latest


def finite_or_nan(value: object) -> float:
    """Normalize scalar JSON metrics, preserving undefined metrics as null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def execute_reports(
    tickers: list[str],
    start_date: str,
    end_date: str,
    db_path: Path,
    template_path: Path,
    run_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]], list[Path]]:
    """Execute each ticker independently and collect only this run's successes."""
    successful_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    successful_paths: list[Path] = []

    for ticker in tqdm(tickers, desc="Generating reports"):
        safe_ticker = "".join(
            character for character in ticker if character.isalnum() or character in "._-"
        )
        report_path = run_dir / f"{safe_ticker}_report.ipynb"
        metrics_path = run_dir / f"{safe_ticker}_metrics.json"
        try:
            pm.execute_notebook(
                input_path=str(template_path),
                output_path=str(report_path),
                parameters={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "db_path": str(db_path.resolve()),
                    "metrics_path": str(metrics_path.resolve()),
                },
                progress_bar=False,
                log_output=True,
                kernel_name="python3",
            )
            if not metrics_path.is_file():
                raise RuntimeError("Report completed without its metrics sidecar.")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            required = {
                "ticker",
                "actual_start_date",
                "actual_end_date",
                "trading_days",
                "avg_daily_return",
                "avg_volatility",
                "max_drawdown",
                "sharpe_ratio",
                "total_return",
            }
            missing = required.difference(metrics)
            if missing:
                raise RuntimeError(f"Metrics sidecar missing keys: {sorted(missing)}")
            successful_rows.append(
                {
                    "ticker": ticker,
                    "avg_daily_return": finite_or_nan(metrics["avg_daily_return"]),
                    "avg_volatility": finite_or_nan(metrics["avg_volatility"]),
                    "max_drawdown": finite_or_nan(metrics["max_drawdown"]),
                    "sharpe_ratio": finite_or_nan(metrics["sharpe_ratio"]),
                    "total_return": finite_or_nan(metrics["total_return"]),
                    "report_path": str(report_path.relative_to(HOMEWORK_DIR)),
                }
            )
            successful_paths.append(report_path)
        except Exception as error:  # Continue the portfolio run after any ticker failure.
            failures.append(
                {
                    "ticker": ticker,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"FAILED {ticker}: {type(error).__name__}: {error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    summary = pd.DataFrame(successful_rows, columns=SUMMARY_COLUMNS)
    return summary, failures, successful_paths


def save_dashboard(summary: pd.DataFrame, run_dir: Path) -> None:
    """Save rankings and risk/return scatter for successful reports only."""
    csv_path = run_dir / "portfolio_summary.csv"
    # Keep the persisted dashboard in the same ranking order shown below.
    summary = summary.sort_values(
        "sharpe_ratio", ascending=False, na_position="last"
    ).reset_index(drop=True)
    summary.to_csv(csv_path, index=False)
    if summary.empty:
        print("No successful reports; saved an empty portfolio_summary.csv.")
        return

    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values(
        "sharpe_ratio", ascending=False
    )
    print("\nTop stocks by Sharpe ratio (undefined values excluded):")
    print(ranked.head(10).to_string(index=False))
    print("\nBottom stocks by Sharpe ratio (undefined values excluded):")
    print(ranked.tail(10).sort_values("sharpe_ratio").to_string(index=False))

    chart_data = summary.dropna(subset=["avg_volatility", "total_return"])
    if chart_data.empty:
        print("No finite volatility/return pairs; dashboard chart was not created.")
        return

    figure, axis = plt.subplots(figsize=(12, 8))
    finite_sharpe = chart_data["sharpe_ratio"].replace([np.inf, -np.inf], np.nan)
    color_values = finite_sharpe.fillna(0.0)
    points = axis.scatter(
        chart_data["avg_volatility"],
        chart_data["total_return"],
        c=color_values,
        cmap="RdYlGn",
        edgecolor="black",
        alpha=0.82,
    )
    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label("Annualized Sharpe ratio (undefined shown as 0 for color only)")
    top_by_return = chart_data.nlargest(min(5, len(chart_data)), "total_return")
    for _, row in top_by_return.iterrows():
        axis.annotate(
            row["ticker"],
            (row["avg_volatility"], row["total_return"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_title("Risk-Return Profile of Portfolio")
    axis.set_xlabel("Mean annualized 20-day rolling volatility (fraction)")
    axis.set_ylabel("Total adjusted-close return (fraction)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "portfolio_summary.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run reports, persist an exact run manifest and summarize successes."""
    args = parse_args()
    db_path = args.db_path.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Report template not found: {template_path}")

    tickers, latest = load_tickers_and_latest(db_path)
    start_date, end_date = analysis_window(latest)
    run_date = date.today().isoformat()
    run_dir = output_root / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Latest cleaned observation: {latest:%Y-%m-%d}; common analysis window: "
        f"{start_date} through {end_date}; run directory: {run_dir}"
    )

    with tempfile.TemporaryDirectory(prefix="papermill-template-", dir=run_dir) as staging:
        staged_template = prepare_papermill_template(template_path, Path(staging))
        summary, failures, successful_paths = execute_reports(
            tickers, start_date, end_date, db_path, staged_template, run_dir
        )
    save_dashboard(summary, run_dir)
    (run_dir / "failed_stocks.txt").write_text(
        "".join(f"{item['ticker']}\t{item['error']}\n" for item in failures),
        encoding="utf-8",
    )
    manifest = {
        "run_date": run_date,
        "db_path": str(db_path),
        "latest_cleaned_observation": latest.strftime("%Y-%m-%d"),
        "analysis_start_date": start_date,
        "analysis_end_date": end_date,
        "requested_tickers": tickers,
        "successful_reports": [str(path.relative_to(HOMEWORK_DIR)) for path in successful_paths],
        "failures": failures,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Successfully generated {len(successful_paths)}/{len(tickers)} reports")
    print(f"Summary: {run_dir / 'portfolio_summary.csv'}")
    if failures:
        print(f"Failures: {run_dir / 'failed_stocks.txt'}")


if __name__ == "__main__":
    main()
