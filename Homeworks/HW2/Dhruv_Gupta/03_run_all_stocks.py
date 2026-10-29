"""Generate one Papermill risk report for every ticker in ohlc_hw_cleaned.

Homework 2, sections 3.1 through 3.3. The analysis window follows Lecture 4,
module 4 (``six_complete_months`` in ``4_running_analysis_at_scale.ipynb``):
the six calendar months before the month that contains the latest cleaned
observation. December 6, 2024 therefore uses June 1 through November 30, 2024.
A latest observation on a month-end uses the same rule. That date is the last
row in the delivery, and the month that contains it is left out of the window.
Every ticker receives this same requested window.

The output folder is ``reports/{YYYY-MM-DD}/``. That date is the day the
script runs. It is chosen from the computer clock and is separate from the
historical window passed into each notebook.

Run this file with the same interpreter that can execute the template, from
any working directory:

    python 03_run_all_stocks.py

That interpreter needs ``papermill``, ``scrapbook``, ``tqdm``, and
``ipykernel``, plus the template's packages. ``DataFrame.style`` in the
template also needs ``jinja2``. Papermill launches this same interpreter for
every report, using the kernel setup from ``Lectures/Lecture 4/report_runtime.py``.

The template is ``02_stock_report_template.ipynb``. Papermill passes
``ticker``, ``start_date``, ``end_date``, and an absolute ``db_path``. The
notebook's parameter cell asks the batch runner for that absolute path so
every report opens the working copy written by ``01_data_cleaning.ipynb``.
The template saves named metrics with ``scrapbook.glue``, as Lecture 4 module
3 does. This script reads those names only from notebooks whose execution
returned. It does not scan the reports folder: a failed Papermill run can
still leave a notebook on disk, and an older run can leave a stale file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm


HOMEWORK_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = HOMEWORK_DIR / "02_stock_report_template.ipynb"
DB_PATH = HOMEWORK_DIR / "data" / "data.db"
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
METRIC_SCRAPS = [
    "avg_daily_return",
    "avg_volatility",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
]


def six_complete_months(latest_date) -> tuple[str, str]:
    """Return the six complete months before ``latest_date``'s month.

    This is the Lecture 4 function. ``latest_date`` is an observation from
    ``ohlc_hw_cleaned``, parsed as a calendar date. The start is the first
    day of the month six months earlier. The end is the last day of the
    month before ``latest_date``.

    December 6, 2024 returns ``2024-06-01`` and ``2024-11-30``. November 30,
    2024 returns ``2024-05-01`` and ``2024-10-31``. The lecture's January 15,
    2025 check returns ``2024-07-01`` and ``2024-12-31``.
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError("Cannot choose a window from an empty date range.")
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def check_window_rule() -> None:
    """Stop if the date rule drifts from the homework and lecture examples."""
    examples = {
        "2024-12-06": ("2024-06-01", "2024-11-30"),
        "2024-11-30": ("2024-05-01", "2024-10-31"),
        "2025-01-15": ("2024-07-01", "2024-12-31"),
    }
    for latest, expected in examples.items():
        found = six_complete_months(latest)
        if found != expected:
            raise AssertionError(
                f"six_complete_months({latest}) returned {found}; "
                f"the homework rule requires {expected}."
            )


def load_tickers_and_latest(db_path: Path) -> tuple[list[str], str]:
    """Read distinct tickers and the latest ``ts`` from the cleaned table.

    Lecture 4, module 4 reads these two facts from ``ohlc`` for the classroom
    demo. Homework 2 reads both from ``ohlc_hw_cleaned`` in the local working
    copy. The connection is read-only, matching ``02_stock_report_template.ipynb``.
    """
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run 01_data_cleaning.ipynb to create the local working copy."
        )
    ticker_query = f"SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker"
    latest_query = f"SELECT MAX(ts) AS latest FROM {CLEAN_TABLE}"
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            conn,
            params=(CLEAN_TABLE,),
        )
        if tables.empty:
            raise ValueError(
                f"{db_path} has no {CLEAN_TABLE} table. "
                "Run 01_data_cleaning.ipynb before generating reports."
            )
        tickers = pd.read_sql_query(ticker_query, conn)["ticker"].tolist()
        latest = pd.read_sql_query(latest_query, conn)["latest"].iloc[0]
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    if not tickers:
        raise ValueError(f"{CLEAN_TABLE} has no tickers to report.")
    if latest is None:
        raise ValueError("Cannot choose a window from an empty date range.")
    return tickers, str(latest)


def interpreter_kernel(staging: Path) -> str:
    """Register this interpreter as the kernel Papermill will launch.

    Adapted from ``prepare_report_run`` in ``Lectures/Lecture 4/report_runtime.py``.
    The kernel spec's argv is ``sys.executable``, so each report runs in the
    environment that started this script. The spec lives in a temporary
    directory and is removed when the batch finishes.
    """
    kernel_name = "course-report"
    kernel_dir = staging / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True)
    kernel_spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Course report",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel_spec), encoding="utf-8")
    prior_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(staging) + (os.pathsep + prior_path if prior_path else "")
    return kernel_name


def run_report(
    ticker: str,
    start: str,
    end: str,
    folder: Path,
    db_path: Path,
    kernel_name: str,
) -> Path:
    """Execute the template once and return the saved notebook path.

    Errors propagate. The caller records them and continues with the next
    ticker, which is the loop in Lecture 4, module 4. Papermill may still
    leave a partial notebook on disk after a failed cell. A path is recorded
    as successful only when this function returns.
    """
    output = folder / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE_PATH),
        str(output),
        parameters={
            "ticker": ticker,
            "start_date": start,
            "end_date": end,
            "db_path": str(db_path.resolve()),
        },
        kernel_name=kernel_name,
        cwd=str(HOMEWORK_DIR),
        progress_bar=False,
    )
    return output


def write_failure_logs(run_dir: Path, failures: list[dict[str, str]]) -> None:
    """Save the lecture's two failure logs for this run.

    ``failed_stocks.txt`` is one ticker per line. ``failed_stocks.json`` keeps
    the exception text Papermill raised, including the notebook cell traceback.
    """
    (run_dir / "failed_stocks.txt").write_text(
        "".join(item["ticker"] + "\n" for item in failures),
        encoding="utf-8",
    )
    (run_dir / "failed_stocks.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )


def record_failure(failures: list[dict[str, str]], ticker: str, exc: BaseException) -> None:
    """Remember one ticker's error and print a short line for the progress log."""
    failures.append(
        {
            "ticker": ticker,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    )
    print(f"FAILED: {ticker}. The full error is saved in failed_stocks.json.")


def scrap_number(value) -> float:
    """Return a metric glued by the template, using NaN when it was undefined."""
    if value is None:
        return float("nan")
    number = float(value)
    if pd.isna(number):
        return float("nan")
    return number


def read_report_row(path: Path, start_date: str, end_date: str) -> dict[str, object]:
    """Read one successful report's glued metrics.

    Lecture 4 module 4 reads ``sb.read_notebook(path).scraps``. The names come
    from the template's Scrapbook cell. A notebook that finished without those
    names is rejected here, so a partial file cannot enter the summary.
    """
    scraps = sb.read_notebook(str(path)).scraps
    required = [
        "ticker",
        "start_date",
        "end_date",
        "first_observation",
        "last_observation",
        "trading_days",
        *METRIC_SCRAPS,
    ]
    missing = [name for name in required if name not in scraps]
    if missing:
        raise KeyError(f"{path.name} is missing named outputs: {', '.join(missing)}")

    ticker = str(scraps["ticker"].data)
    expected_ticker = path.name.removesuffix("_report.ipynb")
    if ticker != expected_ticker:
        raise ValueError(
            f"{path.name} glued ticker {ticker!r}, which does not match the filename."
        )
    glued_start = str(scraps["start_date"].data)
    glued_end = str(scraps["end_date"].data)
    if glued_start != start_date or glued_end != end_date:
        raise ValueError(
            f"{ticker} was requested for {start_date} to {end_date}, "
            f"but the notebook recorded {glued_start} to {glued_end}."
        )

    row: dict[str, object] = {
        "ticker": ticker,
        "start_date": glued_start,
        "end_date": glued_end,
        "first_observation": str(scraps["first_observation"].data),
        "last_observation": str(scraps["last_observation"].data),
        "trading_days": int(scraps["trading_days"].data),
        "report_path": path.resolve().relative_to(HOMEWORK_DIR).as_posix(),
    }
    for name in METRIC_SCRAPS:
        row[name] = scrap_number(scraps[name].data)
    return row


def collect_successful_reports(
    paths: list[Path],
    start_date: str,
    end_date: str,
    failures: list[dict[str, str]],
) -> pd.DataFrame:
    """Collect named outputs from this run's completed notebooks.

    ``paths`` is the list ``run_report`` returned. Notebooks left behind by a
    failed cell, or written by an earlier run, are not included.
    """
    rows = []
    for path in paths:
        ticker = path.name.removesuffix("_report.ipynb")
        try:
            rows.append(read_report_row(path, start_date, end_date))
        except Exception as exc:
            record_failure(failures, ticker, exc)
    if not rows:
        return pd.DataFrame(columns=[*SUMMARY_COLUMNS, "first_observation", "last_observation", "trading_days"])
    summary = pd.DataFrame(rows)
    if summary["ticker"].duplicated().any():
        duplicated = summary.loc[summary["ticker"].duplicated(keep=False), "ticker"].unique()
        raise ValueError(f"Duplicate tickers in this run's successful reports: {list(duplicated)}")
    return summary


def portfolio_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the homework summary, ranked by Sharpe ratio.

    Undefined Sharpe ratios stay in the table and sort after every defined
    value. Rankings that display the top and bottom stocks drop those rows.
    """
    frame = summary.loc[:, SUMMARY_COLUMNS].copy()
    return frame.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)


def coverage_report(summary: pd.DataFrame, start_date: str, end_date: str) -> str:
    """Explain the common request and each report's observed dates.

    The cleaned histories stop on different days. Each metric uses the rows
    that were loaded for that ticker. A report that ends earlier than another
    describes a shorter sample, and the two are not the same set of sessions.
    """
    lines = [
        f"Requested window for every ticker: {start_date} to {end_date}.",
        "Actual coverage is the first and last cleaned session inside that window.",
        "Each price series stops on its last session. Later dates stay empty.",
    ]
    grouped = (
        summary.groupby(["first_observation", "last_observation"], as_index=False)
        .agg(stocks=("ticker", "size"), trading_days=("trading_days", "min"))
        .sort_values(["last_observation", "first_observation"])
    )
    for row in grouped.itertuples(index=False):
        stock_word = "report" if row.stocks == 1 else "reports"
        verb = "covers" if row.stocks == 1 else "cover"
        lines.append(
            f"{row.stocks} {stock_word} {verb} {row.first_observation} to "
            f"{row.last_observation} ({int(row.trading_days)} trading days)."
        )
    longest = summary["last_observation"].max()
    shorter = summary.loc[summary["last_observation"] < longest].sort_values("ticker")
    if shorter.empty:
        lines.append(
            "Every successful report ends on the same session, so the rankings share that observed history."
        )
    else:
        lines.append(
            f"{len(shorter)} reports end before {longest}: {', '.join(shorter['ticker'].tolist())}."
        )
        lines.append(
            "Their return, volatility, drawdown, and Sharpe ratio describe that shorter sample. "
            "A comparison with a report that continues through "
            f"{longest} places two different sets of sessions side by side."
        )
    return "\n".join(lines)


def print_sharpe_rankings(summary_df: pd.DataFrame) -> None:
    """Print the top and bottom Sharpe ratios, skipping undefined values.

    With fewer than 20 defined ratios the two groups overlap. With fewer than
    10 they contain the same reports.
    """
    ranked = summary_df.dropna(subset=["sharpe_ratio"]).reset_index(drop=True)
    view_columns = [column for column in SUMMARY_COLUMNS if column != "report_path"]
    print(f"Sharpe rankings use {len(ranked)} reports with a defined Sharpe ratio.")
    if ranked.empty:
        print("No Sharpe ratio is defined, so there is no top or bottom ranking.")
        return
    if len(ranked) < 20:
        print("Fewer than 20 reports have a defined Sharpe ratio. The top and bottom groups overlap.")
    top = ranked.head(10).loc[:, view_columns]
    bottom = ranked.tail(10).sort_values("sharpe_ratio", ascending=True).loc[:, view_columns]
    with pd.option_context("display.float_format", lambda value: f"{value:.6f}", "display.width", 120):
        print("Top 10 by Sharpe ratio:")
        print(top.to_string(index=False))
        print("Bottom 10 by Sharpe ratio:")
        print(bottom.to_string(index=False))


def save_risk_return_plot(summary: pd.DataFrame, destination: Path) -> None:
    """Save the portfolio scatter plot for this run.

    The x-axis is mean rolling volatility and the y-axis is total return.
    Point color is the Sharpe ratio. The five highest total returns are labeled.
    Display units are percent; the CSV keeps the original fractions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotable = summary.dropna(subset=["avg_volatility", "total_return"]).copy()
    if plotable.empty:
        raise ValueError("Successful reports have no volatility and return values to plot.")

    figure, axes = plt.subplots(figsize=(10, 6.5), layout="constrained")
    defined = plotable["sharpe_ratio"].notna()
    if defined.any():
        points = axes.scatter(
            plotable.loc[defined, "avg_volatility"] * 100,
            plotable.loc[defined, "total_return"] * 100,
            c=plotable.loc[defined, "sharpe_ratio"],
            cmap="viridis",
            s=46,
            zorder=3,
        )
        figure.colorbar(points, ax=axes, label="Sharpe ratio")
    if (~defined).any():
        axes.scatter(
            plotable.loc[~defined, "avg_volatility"] * 100,
            plotable.loc[~defined, "total_return"] * 100,
            color="#9ca3af",
            s=46,
            label="Sharpe ratio undefined",
            zorder=2,
        )
        axes.legend(frameon=False)

    labels = plotable.nlargest(min(5, len(plotable)), "total_return")
    for row in labels.itertuples(index=False):
        axes.annotate(
            row.ticker,
            (row.avg_volatility * 100, row.total_return * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes.margins(x=0.12, y=0.12)
    axes.set_xlabel("Average rolling volatility (%)")
    axes.set_ylabel("Total return (%)")
    axes.set_title("Risk-Return Profile of Portfolio")
    axes.grid(True, color="#e5e7eb", linewidth=0.8)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_portfolio_summary(summary: pd.DataFrame, run_dir: Path, start_date: str, end_date: str) -> Path:
    """Write the ranked CSV, the scatter plot, and the coverage explanation."""
    coverage = coverage_report(summary, start_date, end_date)
    print(coverage)
    summary_df = portfolio_summary(summary)
    csv_path = run_dir / "portfolio_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print_sharpe_rankings(summary_df)
    chart_path = run_dir / "portfolio_risk_return.png"
    save_risk_return_plot(summary, chart_path)
    print(f"Summary: {csv_path}")
    print(f"Risk-return chart: {chart_path}")
    return csv_path


def main() -> None:
    """Generate every report, then summarize the ones that finished."""
    check_window_rule()
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Report template not found: {TEMPLATE_PATH}")
    if importlib.util.find_spec("ipykernel") is None:
        raise RuntimeError(
            "ipykernel is not installed for this interpreter. "
            "Papermill uses it to start each report."
        )
    if importlib.util.find_spec("scrapbook") is None:
        raise RuntimeError(
            "scrapbook is not installed for this interpreter. "
            "The summary reads the named outputs glued by the template."
        )
    if importlib.util.find_spec("jinja2") is None:
        print(
            "jinja2 is not installed in this interpreter. The template styles "
            "tables with DataFrame.style, so those reports will fail until jinja2 is installed."
        )

    tickers, latest_observation = load_tickers_and_latest(DB_PATH)
    start_date, end_date = six_complete_months(latest_observation)
    run_date = date.today().isoformat()
    run_dir = HOMEWORK_DIR / "reports" / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("portfolio_summary.csv", "portfolio_risk_return.png"):
        stale = run_dir / stale_name
        if stale.is_file():
            stale.unlink()

    print(f"Latest cleaned observation: {pd.Timestamp(latest_observation):%Y-%m-%d}")
    print(f"Analysis window: {start_date} to {end_date}")
    print(f"Run directory: {run_dir}")
    print(f"Tickers: {len(tickers)}")

    successful_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    collected = pd.DataFrame()
    staging = Path(tempfile.mkdtemp(prefix="report-kernel-"))
    try:
        kernel_name = interpreter_kernel(staging)
        for ticker in tqdm(tickers, desc="Generating reports"):
            try:
                successful_paths.append(
                    run_report(ticker, start_date, end_date, run_dir, DB_PATH, kernel_name)
                )
            except Exception as exc:
                record_failure(failures, ticker, exc)
        collected = collect_successful_reports(successful_paths, start_date, end_date, failures)
    finally:
        write_failure_logs(run_dir, failures)
        coverage_rows = []
        report_names = []
        if not collected.empty:
            report_names = [
                f"{ticker}_report.ipynb" for ticker in collected["ticker"].tolist()
            ]
            coverage_rows = collected.loc[
                :, ["ticker", "first_observation", "last_observation", "trading_days"]
            ].to_dict(orient="records")
        manifest = {
            "run_date": run_date,
            "latest_cleaned_observation": str(pd.Timestamp(latest_observation).date()),
            "start_date": start_date,
            "end_date": end_date,
            "reports": report_names,
            "coverage": coverage_rows,
            "failures": [{"ticker": item["ticker"], "error": item["error"]} for item in failures],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        shutil.rmtree(staging, ignore_errors=True)

    completed = 0 if collected.empty else len(collected)
    print(f"Successfully generated {completed}/{len(tickers)} reports")
    if failures:
        print(f"Failures: {run_dir / 'failed_stocks.txt'}")
    if collected.empty:
        print("Every ticker failed. Inspect failed_stocks.json. There is no portfolio summary.")
        return
    write_portfolio_summary(collected, run_dir, start_date, end_date)


if __name__ == "__main__":
    main()
