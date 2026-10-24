"""Execute the HW2 stock template and summarize successful reports from this run."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sqlite3
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm


HOME = Path(__file__).resolve().parent
SUMMARY_COLUMNS = [
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
    "sharpe_ratio", "total_return", "report_path", "requested_start_date",
    "requested_end_date", "actual_start_date", "actual_end_date",
    "n_prices", "n_returns", "coverage_notes",
]
METRIC_COLUMNS = [
    "avg_daily_return", "avg_volatility", "max_drawdown",
    "sharpe_ratio", "total_return",
]


def last_complete_months(latest_date: str, months: int = 6) -> tuple[str, str]:
    """Use calendar months before the latest observation's month, even at month-end."""
    if not isinstance(months, int) or months < 1:
        raise ValueError("months must be a positive integer.")
    latest = pd.Timestamp(date.fromisoformat(latest_date))
    first_of_month = latest.replace(day=1)
    start = first_of_month - pd.DateOffset(months=months)
    end = first_of_month - pd.Timedelta(days=1)
    return start.date().isoformat(), end.date().isoformat()


def load_universe(database: Path) -> tuple[list[str], str]:
    """Read the cleaned ticker universe and latest date through a read-only connection."""
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ohlc_hw_cleaned'"
        ).fetchone():
            raise ValueError("Run 01_data_cleaning.ipynb before this script.")
        tickers = [
            row[0] for row in connection.execute(
                "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker"
            )
        ]
        latest = connection.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    finally:
        connection.close()
    if not tickers or latest is None:
        raise ValueError("The cleaned table is empty.")
    if any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]*", symbol)
           for symbol in tickers):
        raise ValueError("The cleaned ticker universe contains an invalid identifier.")
    date.fromisoformat(latest)
    return tickers, latest


def portable_path(path: Path, base: Path = HOME) -> str:
    """Prefer a homework-relative path; keep external custom locations absolute."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def collect_report_metrics(path: Path, symbol: str, start: str, end: str) -> dict:
    """Collect and validate the named output of one successfully executed report."""
    scraps = sb.read_notebook(str(path)).scraps.data_dict
    if "report_metrics" not in scraps:
        raise ValueError("Executed notebook has no report_metrics named output.")
    metrics = scraps["report_metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("report_metrics must be a dictionary.")
    if (metrics.get("ticker"), metrics.get("start_date"), metrics.get("end_date")) != (
        symbol, start, end
    ):
        raise ValueError("Report identity/window differs from this run's parameters.")
    actual_start = date.fromisoformat(metrics["actual_start_date"])
    actual_end = date.fromisoformat(metrics["actual_end_date"])
    if not date.fromisoformat(start) <= actual_start <= actual_end <= date.fromisoformat(end):
        raise ValueError("Actual report coverage is outside the requested window.")
    n_prices, n_returns = metrics["n_prices"], metrics["n_returns"]
    if not isinstance(n_prices, int) or n_prices < 1 or n_returns != n_prices - 1:
        raise ValueError("Inconsistent price/return counts.")
    row = {"ticker": symbol}
    for name in METRIC_COLUMNS:
        value = metrics[name]
        if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value)):
            raise ValueError(f"Invalid numeric output: {name}")
        row[name] = np.nan if value is None else float(value)
    notes = metrics.get("coverage_notes", [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise ValueError("coverage_notes must be a list of strings.")
    row.update({
        "report_path": portable_path(path),
        "requested_start_date": start,
        "requested_end_date": end,
        "actual_start_date": actual_start.isoformat(),
        "actual_end_date": actual_end.isoformat(),
        "n_prices": n_prices,
        "n_returns": n_returns,
        "coverage_notes": " ".join(notes),
    })
    return row


def ranked_stocks(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank finite Sharpe ratios only, with ticker as a reproducible tie-breaker."""
    return summary.loc[np.isfinite(summary["sharpe_ratio"])].sort_values(
        ["sharpe_ratio", "ticker"], ascending=[False, True]
    )


def print_rankings(summary: pd.DataFrame) -> None:
    """Display top and bottom ten stocks, allowing overlap in a small valid universe."""
    ranked = ranked_stocks(summary)
    columns = ["ticker", "sharpe_ratio", "total_return", "actual_end_date", "n_prices"]
    if ranked.empty:
        print("No defined Sharpe ratios are available for ranking.")
        return
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10)[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10).sort_values(["sharpe_ratio", "ticker"])[columns].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    if len(ranked) < 20:
        print("Top and bottom groups overlap because fewer than 20 valid Sharpe ratios exist.")
    print("Returns are fractions. Actual coverage varies; these are individual-stock comparisons,")
    print("not a weighted portfolio return or an identical-coverage performance comparison.")


def plot_portfolio(summary: pd.DataFrame, output: Path, start: str, end: str) -> None:
    """Plot mean rolling volatility versus total return, colored by annualized Sharpe."""
    plotted = summary.loc[
        np.isfinite(summary["avg_volatility"]) & np.isfinite(summary["total_return"])
    ].copy()
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(11, 7))
        colored = plotted.loc[np.isfinite(plotted["sharpe_ratio"])]
        undefined = plotted.loc[~np.isfinite(plotted["sharpe_ratio"])]
        if len(colored):
            points = ax.scatter(
                colored["avg_volatility"], colored["total_return"],
                c=colored["sharpe_ratio"], cmap="RdYlGn", s=85,
                edgecolors="#344454", linewidths=.6, alpha=.9, zorder=3,
            )
            fig.colorbar(points, ax=ax, label="Annualized Sharpe ratio (course approximation)")
        if len(undefined):
            ax.scatter(undefined["avg_volatility"], undefined["total_return"],
                       color="#89929b", s=85, label="Sharpe unavailable", zorder=3)
            ax.legend()
        if plotted.empty:
            ax.text(.5, .5, "No successful report has both finite volatility and return.",
                    ha="center", transform=ax.transAxes)
        label_offsets = [(8, 12), (8, -16), (8, 16), (10, -19), (-36, -13)]
        for number, row in enumerate(plotted.nlargest(5, "total_return").itertuples()):
            ax.annotate(
                row.ticker, (row.avg_volatility, row.total_return),
                xytext=label_offsets[number], textcoords="offset points",
                fontsize=10, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "#89929b", "linewidth": .6},
            )
        ax.axhline(0, color="#89929b", linewidth=.8)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Mean 20-return rolling annualized volatility")
        ax.set_ylabel("Total adjusted-price return over actual coverage")
        ax.set_title("Risk-Return Profile of Portfolio", fontsize=18, fontweight="bold", pad=28)
        ax.text(.5, 1.025, f"Common requested window: {start} to {end} | {len(summary)} successful reports",
                ha="center", transform=ax.transAxes, fontsize=10, color="#65717d")
        ax.margins(x=.17, y=.2)
        fig.text(.08, .035,
                 "Actual histories may end early: see coverage columns in portfolio_summary.csv.\n"
                 "X uses mean rolling volatility; it differs from whole-period volatility. Top five plotted returns are labeled.",
                 fontsize=9, color="#65717d")
        fig.tight_layout(rect=(0, .09, 1, .98))
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)


def write_run_logs(directory: Path, manifest: dict, failures: list[dict]) -> None:
    """Persist run provenance and the failures recorded in this run, including partial progress."""
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (directory / "failed_stocks.txt").write_text(
        "\n\n".join(f"{item['ticker']}: {item['error']}\n{item['traceback']}" for item in failures),
        encoding="utf-8",
    )


def run_batch(
    database: Path,
    template: Path,
    output_root: Path,
    run_date: str,
    kernel_name: str = "python3",
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Execute every ticker, isolate failures and collect only this run's successful notebooks."""
    database, template = database.resolve(), template.resolve()
    if not template.is_file():
        raise FileNotFoundError(f"Template not found: {template}")
    run_date = date.fromisoformat(run_date).isoformat()
    tickers, latest = load_universe(database)
    start, end = last_complete_months(latest)
    directory = output_root.resolve() / run_date
    directory.mkdir(parents=True, exist_ok=True)
    # Remove only this driver's old dashboard artifacts, never arbitrary report notebooks.
    # An all-failed rerun must not leave a previous summary pretending to be current.
    summary_path = directory / "portfolio_summary.csv"
    chart_path = directory / "portfolio_risk_return.png"
    summary_path.unlink(missing_ok=True)
    chart_path.unlink(missing_ok=True)
    successes, failures = [], []
    manifest = {
        "state": "running", "run_date": run_date, "latest_cleaned_observation": latest,
        "requested_start_date": start, "requested_end_date": end,
        "tickers": tickers, "successful_reports": [], "failed_tickers": [],
    }
    write_run_logs(directory, manifest, failures)
    print(f"Latest cleaned observation: {latest}; common requested window: {start} to {end}.")
    for symbol in tqdm(tickers, desc="Generating reports", disable=not show_progress):
        report = directory / f"{symbol}_report.ipynb"
        try:
            pm.execute_notebook(
                str(template), str(report),
                parameters={"ticker": symbol, "start_date": start, "end_date": end,
                            "db_path": portable_path(database)},
                kernel_name=kernel_name, cwd=str(HOME),
                start_timeout=60, execution_timeout=180,
                progress_bar=False, log_output=False,
            )
            row = collect_report_metrics(report, symbol, start, end)
            successes.append(row)
            manifest["successful_reports"].append(portable_path(report))
        except Exception as error:
            failures.append({"ticker": symbol, "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc()})
            manifest["failed_tickers"].append(symbol)
            tqdm.write(f"{symbol} failed: {type(error).__name__}: {error}")
        write_run_logs(directory, manifest, failures)
    print(f"Successfully generated {len(successes)}/{len(tickers)} reports")
    summary = pd.DataFrame(successes, columns=SUMMARY_COLUMNS)
    if summary.empty:
        manifest["state"] = "all_failed"
        write_run_logs(directory, manifest, failures)
        print("All reports failed. No current summary or dashboard was generated; see failed_stocks.txt.")
        return summary, failures
    summary = summary.sort_values(
        ["sharpe_ratio", "ticker"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    # Fractions remain numeric. Missing metrics are blank CSV cells, not invented zeros.
    summary.to_csv(summary_path, index=False, na_rep="")
    plot_portfolio(summary, chart_path, start, end)
    manifest["state"] = "completed" if not failures else "completed_with_failures"
    manifest["successful_count"] = len(summary)
    manifest["failed_count"] = len(failures)
    write_run_logs(directory, manifest, failures)
    print_rankings(summary)
    print(f"\nSummary: {summary_path}\nChart: {chart_path}")
    return summary, failures


def main() -> int:
    """Parse local-run options and return a nonzero exit code if any report fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HOME / "data/data.db")
    parser.add_argument("--template", type=Path, default=HOME / "02_stock_report_template.ipynb")
    parser.add_argument("--output-root", type=Path, default=HOME / "reports")
    parser.add_argument("--run-date", default=date.today().isoformat(),
                        help="Output folder's local run date (YYYY-MM-DD), unrelated to the historical window.")
    parser.add_argument("--kernel", default="python3", help="Installed Jupyter kernel name.")
    args = parser.parse_args()
    summary, failures = run_batch(args.db, args.template, args.output_root, args.run_date, args.kernel)
    return 0 if len(summary) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
