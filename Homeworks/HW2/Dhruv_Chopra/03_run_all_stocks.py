"""Generate a risk report for every ticker in ohlc_hw_cleaned and summarize the portfolio.

Usage (from this folder, after running 01_data_cleaning.ipynb):
    python 03_run_all_stocks.py
    python 03_run_all_stocks.py --tickers AAPL MSFT      # quick test on a subset

Outputs go to reports/<run date>/:
    <TICKER>_report.ipynb    executed copy of 02_stock_report_template.ipynb per ticker
    failed_stocks.txt        tickers that failed, with the error message (empty if none)
    portfolio_summary.csv    one row per successful report, sorted by Sharpe ratio
    risk_return.png          risk-return scatter plot
    run.log                  full log of the run
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_DB = HERE.parent / "data" / "data.db"
CLEAN_TABLE = "ohlc_hw_cleaned"

SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                   "total_return", "report_path"]
COVERAGE_COLUMNS = ["first_date", "last_date", "trading_days", "coverage_note"]

log = logging.getLogger("run_all_stocks")


def parse_args() -> argparse.Namespace:
    """Command-line options; every default works when run from this folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB, help="SQLite database with ohlc_hw_cleaned")
    parser.add_argument("--output-root", type=Path, default=HERE / "reports", help="parent folder for run folders")
    parser.add_argument("--run-date", default=date.today().isoformat(), help="run folder name (default: today)")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel used to execute the reports")
    parser.add_argument("--tickers", nargs="+", help="run only these tickers (default: all in the table)")
    return parser.parse_args()


def query(db_path: Path, sql: str) -> pd.DataFrame:
    """Run a read-only query so the batch can never modify the database."""
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(sql, conn)


def load_tickers(db_path: Path) -> list[str]:
    """All tickers in the cleaned table."""
    return query(db_path, f"SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker")["ticker"].tolist()


def six_complete_months(latest_date: str) -> tuple[str, str]:
    """Six calendar months before the month containing the latest observation.

    Anchored to the data, not the computer clock: a latest date of 2024-12-06 gives
    2024-06-01 to 2024-11-30. The latest month is excluded even when the data ends on its last day,
    because a final trading date alone does not prove the month's delivery is complete.
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError("Cannot choose a window: the cleaned table is empty.")
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(ticker: str, start: str, end: str, run_dir: Path, db_path: Path, kernel: str) -> Path:
    """Execute the template for one ticker; raises if the notebook fails."""
    output = run_dir / f"{ticker}_report.ipynb"
    output.unlink(missing_ok=True)  # never leave a stale report from an earlier run in place
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": str(db_path.resolve())},
        kernel_name=kernel, cwd=str(HERE), progress_bar=False,
    )
    return output


def describe_error(exc: Exception) -> str:
    """One-line error message; papermill errors carry the notebook's exception name and value."""
    if isinstance(exc, pm.PapermillExecutionError):
        return f"{exc.ename}: {exc.evalue}"
    return f"{type(exc).__name__}: {exc}"


def collect_results(paths: dict[str, Path], start: str, end: str) -> pd.DataFrame:
    """Read the named scrapbook outputs from this run's successful reports only."""
    rows = []
    for ticker, path in paths.items():
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in SUMMARY_COLUMNS[:-1] + COVERAGE_COLUMNS + ["start_date", "end_date"]}
        if (row["ticker"], row["start_date"], row["end_date"]) != (ticker, start, end):
            raise ValueError(f"{path} does not match {ticker} {start}..{end}")
        # Relative to this folder when possible, so the committed CSV has no machine-specific paths.
        row["report_path"] = str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)
        rows.append(row)
    summary = pd.DataFrame(rows)[SUMMARY_COLUMNS + COVERAGE_COLUMNS]
    return summary.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)


def print_rankings(summary: pd.DataFrame) -> None:
    """Top and bottom 10 by Sharpe; undefined Sharpe ratios are excluded from the ranking."""
    ranked = summary.dropna(subset=["sharpe_ratio"])
    excluded = sorted(set(summary["ticker"]) - set(ranked["ticker"]))
    if excluded:
        print(f"Excluded from ranking (undefined Sharpe): {', '.join(excluded)}")
    if len(ranked) < 20:
        print(f"Note: only {len(ranked)} ranked stocks, so the top and bottom 10 overlap.")
    display_cols = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "last_date"]
    formats = {"sharpe_ratio": "{:.2f}".format, "total_return": "{:+.2%}".format,
               "avg_volatility": "{:.2%}".format, "max_drawdown": "{:.2%}".format}
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10)[display_cols].to_string(index=False, formatters=formats))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10).iloc[::-1][display_cols].to_string(index=False, formatters=formats))


def plot_risk_return(summary: pd.DataFrame, start: str, end: str, path: Path) -> None:
    """Scatter of mean rolling volatility vs total return, coloured by Sharpe, top 5 returns labelled."""
    fig, ax = plt.subplots(figsize=(12, 8))
    defined = summary["sharpe_ratio"].notna()
    full = summary["last_date"] >= summary["last_date"].max()
    scatter = None
    for mask, marker, label in [(full, "o", f"data through {summary['last_date'].max()}"),
                                (~full, "^", "shorter history (ends earlier)")]:
        part = summary[mask & defined]
        if len(part):
            scatter = ax.scatter(part["avg_volatility"], part["total_return"], c=part["sharpe_ratio"],
                                 cmap="RdYlGn", vmin=summary["sharpe_ratio"].min(), vmax=summary["sharpe_ratio"].max(),
                                 s=110, marker=marker, edgecolors="black", linewidths=0.6, label=label)
    undefined = summary[~defined]
    if len(undefined):
        ax.scatter(undefined["avg_volatility"], undefined["total_return"], color="lightgray", s=110,
                   edgecolors="black", label="Sharpe undefined")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    for _, row in summary.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]), xytext=(7, 5),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set(title="Risk-Return Profile of Portfolio", xlabel="Average 20-day rolling volatility (annualized)",
           ylabel=f"Total return ({start} to {end}, or last available date)")
    ax.grid(alpha=0.3)
    # Neutral legend markers: colour encodes Sharpe (colorbar), marker shape encodes coverage.
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_facecolor("lightgray")
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Run every report, record failures, then summarize the successes. Returns the exit code."""
    args = parse_args()
    run_dir = args.output_root / args.run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path, plot_path = run_dir / "portfolio_summary.csv", run_dir / "risk_return.png"
    # A rerun reuses the dated folder: remove the previous run's dashboard up front so that a failed or
    # interrupted run can never leave an old summary or plot next to the new results.
    for stale in (summary_path, plot_path):
        stale.unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "run.log", mode="w")])
    logging.getLogger("papermill").setLevel(logging.WARNING)

    if not args.db_path.exists():
        print(f"Database not found: {args.db_path.resolve()}", file=sys.stderr)
        return 1
    tickers = args.tickers or load_tickers(args.db_path)
    latest = query(args.db_path, f"SELECT MAX(ts) AS latest FROM {CLEAN_TABLE}")["latest"].iloc[0]
    start, end = six_complete_months(latest)
    print(f"Latest cleaned observation: {latest} -> analysis window {start} to {end}")
    print(f"Writing reports to {run_dir}")
    log.info("window %s..%s, %d tickers", start, end, len(tickers))

    successes: dict[str, Path] = {}
    failures: dict[str, str] = {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successes[ticker] = run_report(ticker, start, end, run_dir, args.db_path, args.kernel)
            log.info("%s: ok", ticker)
        except Exception as exc:  # one bad stock must not stop the batch
            failures[ticker] = describe_error(exc)
            log.error("%s: %s", ticker, failures[ticker])

    print(f"\nSuccessfully generated {len(successes)}/{len(tickers)} reports")
    (run_dir / "failed_stocks.txt").write_text("".join(f"{t}\t{msg}\n" for t, msg in failures.items()))
    if failures:
        print(f"Failed: {', '.join(failures)} (details in {run_dir / 'failed_stocks.txt'})")

    if not successes:
        print("No successful reports, so there is no portfolio summary.")
        return 1

    summary = collect_results(successes, start, end)
    summary.to_csv(summary_path, index=False)
    short = summary[summary["last_date"] < summary["last_date"].max()]
    if len(short):
        print(f"\n{len(short)} of {len(summary)} stocks have data ending before {summary['last_date'].max()} "
              f"(e.g. {', '.join(short['ticker'].head(5))}); their metrics cover a shorter period.")
    print_rankings(summary)
    plot_risk_return(summary, start, end, plot_path)
    print(f"\nSaved {summary_path} and {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
