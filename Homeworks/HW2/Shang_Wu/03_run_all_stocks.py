"""Generate a risk report for every stock in ohlc_hw_cleaned and summarize the portfolio.

For each ticker, this script executes ``02_stock_report_template.ipynb`` with papermill over one
common window: the six complete calendar months before the month of the latest cleaned observation.
Results are collected from the successful reports' scrapbook outputs, ranked by Sharpe ratio, and
saved with a risk-return chart.

Usage (from this folder):
    python 03_run_all_stocks.py                      # all tickers
    python 03_run_all_stocks.py --tickers AAPL MSFT  # a subset, e.g. for testing

Outputs, in reports/{run date}/:
    {TICKER}_report.ipynb     executed report per ticker
    failed_stocks.txt         tickers whose report failed (empty file if none)
    portfolio_summary.csv     one row per successful report, values as fractions
    risk_return_profile.png   scatter of average volatility vs. total return
    run.log                   full log, including error tracebacks
"""

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render charts to files; no display needed

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_DB = HERE / "data" / "data.db"
DEFAULT_OUTPUT_ROOT = HERE / "reports"

# Scraps each report must provide (see the last section of the template).
METRICS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
COVERAGE = ["first_date", "last_date", "trading_days"]

log = logging.getLogger("run_all_stocks")


def parse_args(argv=None):
    """Command-line options; defaults reproduce the homework run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB, help="database with ohlc_hw_cleaned")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="report notebook to execute")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="parent folder for run folders")
    parser.add_argument("--tickers", nargs="+", help="run only these tickers (default: all in the table)")
    parser.add_argument("--kernel", default=None, help="Jupyter kernel name (default: the notebook's own)")
    return parser.parse_args(argv)


def setup_logging(log_file: Path) -> None:
    """Log INFO to the console (through tqdm, so the progress bar stays intact) and DEBUG to a file."""
    class TqdmHandler(logging.Handler):
        def emit(self, record):
            tqdm.write(self.format(record))

    log.setLevel(logging.DEBUG)
    console = TqdmHandler(level=logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [console, file]


def read_only(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only, so the batch run cannot modify it."""
    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path} (run 01_data_cleaning.ipynb first)")
    return sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)


def load_tickers(db_path: Path) -> list[str]:
    """All distinct tickers in ohlc_hw_cleaned, sorted."""
    with closing(read_only(db_path)) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def latest_observation(db_path: Path) -> pd.Timestamp:
    """The latest date in ohlc_hw_cleaned (across all tickers)."""
    with closing(read_only(db_path)) as conn:
        (latest,) = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()
    if latest is None:
        raise ValueError("ohlc_hw_cleaned is empty")
    return pd.Timestamp(latest)


def window_trading_days(db_path: Path, start: str, end: str) -> tuple[str, str]:
    """First and last dates in the window that any ticker traded (the window's actual trading days)."""
    with closing(read_only(db_path)) as conn:
        first, last = conn.execute("SELECT MIN(ts), MAX(ts) FROM ohlc_hw_cleaned WHERE ts BETWEEN ? AND ?",
                                   (start, end)).fetchone()
    if first is None:
        raise ValueError(f"no data in ohlc_hw_cleaned between {start} and {end}")
    return first, last


def six_complete_months(latest: pd.Timestamp) -> tuple[str, str]:
    """The six calendar months before the month containing ``latest``, as (start, end) ISO dates.

    The month containing the latest observation is always excluded, even when ``latest`` falls on
    the last day of a month: a final trading date alone does not prove the month's data is complete.
    """
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(ticker: str, start: str, end: str, args, run_dir: Path) -> Path:
    """Execute the template for one ticker; papermill errors propagate to the caller."""
    output = run_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(args.template),
        str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": str(args.db_path.resolve())},
        cwd=str(args.template.parent),
        kernel_name=args.kernel,
        progress_bar=False,
    )
    return output


def collect_results(paths: dict[str, Path]) -> pd.DataFrame:
    """Read the named scraps of the given (successful) reports into one row per ticker."""
    rows = []
    for ticker, path in paths.items():
        scraps = sb.read_notebook(str(path)).scraps
        row = {"ticker": ticker}
        row.update({name: scraps[name].data for name in METRICS + COVERAGE})
        row["report_path"] = str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)
        rows.append(row)
    columns = ["ticker"] + METRICS + COVERAGE + ["report_path"]
    return pd.DataFrame(rows, columns=columns)


def show_rankings(summary: pd.DataFrame, n: int = 10) -> None:
    """Print the top and bottom ``n`` tickers by Sharpe ratio (undefined Sharpe ratios excluded)."""
    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False)
    excluded = summary.loc[summary["sharpe_ratio"].isna(), "ticker"].tolist()
    if excluded:
        log.info("Excluded from ranking (undefined Sharpe ratio): %s", ", ".join(excluded))

    display = ranked.assign(  # percentages for display only; the CSV keeps fractions
        avg_daily_return=ranked["avg_daily_return"].map("{:+.3%}".format),
        avg_volatility=ranked["avg_volatility"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.1%}"),
        max_drawdown=ranked["max_drawdown"].map("{:.2%}".format),
        total_return=ranked["total_return"].map("{:+.2%}".format),
        sharpe_ratio=ranked["sharpe_ratio"].map("{:.2f}".format),
    )[["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "avg_daily_return", "last_date"]]

    if len(ranked) < 2 * n:
        log.info("Only %d ranked reports, so the top and bottom groups overlap.", len(ranked))
    log.info("\nTop %d by Sharpe ratio:\n%s", min(n, len(ranked)), display.head(n).to_string(index=False))
    log.info("\nBottom %d by Sharpe ratio:\n%s", min(n, len(ranked)), display.tail(n).to_string(index=False))


def plot_risk_return(summary: pd.DataFrame, start: str, end: str, path: Path) -> None:
    """Scatter of mean rolling volatility vs. total return, colored by Sharpe ratio; top 5 by return labelled."""
    data = summary.dropna(subset=["avg_volatility", "total_return"])
    fig, ax = plt.subplots(figsize=(11, 7))
    points = ax.scatter(data["avg_volatility"], data["total_return"], c=data["sharpe_ratio"], cmap="RdYlGn",
                        s=90, edgecolors="black", linewidths=0.6)
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in data.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set(xlabel="Average 20-day rolling volatility (annualized)", ylabel="Total return",
           title="Risk-Return Profile of Portfolio")
    ax.text(0.01, 0.01, f"Requested window {start} to {end}; each point covers that ticker's available data.",
            transform=ax.transAxes, fontsize=8, color="gray")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.output_root / date.today().strftime("%Y-%m-%d")  # run date, not the analysis window
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    latest = latest_observation(args.db_path)
    start, end = six_complete_months(latest)
    tickers = args.tickers or load_tickers(args.db_path)
    log.info("Latest cleaned observation: %s -> report window %s to %s", latest.date(), start, end)
    log.info("Generating %d reports into %s", len(tickers), run_dir)

    succeeded: dict[str, Path] = {}
    failed: dict[str, str] = {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            succeeded[ticker] = run_report(ticker, start, end, args, run_dir)
        except Exception as exc:  # one bad ticker must not stop the batch
            reason = f"{exc.ename}: {exc.evalue}" if isinstance(exc, pm.PapermillExecutionError) else repr(exc)
            failed[ticker] = reason
            log.warning("FAILED %s: %s", ticker, reason)
            log.debug("Traceback for %s", ticker, exc_info=True)

    (run_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failed), encoding="utf-8")
    log.info("\nSuccessfully generated %d/%d reports", len(succeeded), len(tickers))
    if failed:
        log.info("Failed tickers saved to %s", run_dir / "failed_stocks.txt")

    if not succeeded:
        log.error("No reports succeeded, so there is no portfolio summary. See %s", run_dir / "run.log")
        return 1

    summary_df = collect_results(succeeded)
    summary_df = summary_df.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)

    # Coverage: compare each report with the window's actual first/last trading days, not calendar dates.
    window_first, window_last = window_trading_days(args.db_path, start, end)
    summary_df["full_coverage"] = (summary_df["first_date"] == window_first) & (summary_df["last_date"] == window_last)
    summary_df.to_csv(run_dir / "portfolio_summary.csv", index=False)
    partial = summary_df[~summary_df["full_coverage"]]
    log.info("\nCoverage: window trading days run %s to %s; %d of %d reports cover all of it.",
             window_first, window_last, len(summary_df) - len(partial), len(summary_df))
    if len(partial):
        ends = partial.groupby("last_date")["ticker"].apply(", ".join)
        for last_date, names in ends.items():
            log.info("  data ends %s: %s", last_date, names)
        log.info("Metrics for these tickers describe a shorter period; compare them with full-coverage tickers "
                 "with care (e.g. total return over ~4.5 vs ~6 months).")
    show_rankings(summary_df)
    plot_risk_return(summary_df, start, end, run_dir / "risk_return_profile.png")
    log.info("\nSaved %s and %s", run_dir / "portfolio_summary.csv", run_dir / "risk_return_profile.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
