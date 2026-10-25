"""Generate a risk report for every stock with papermill and build a portfolio summary.

Run from this folder with the HW2 environment (after 01_data_cleaning.ipynb has created
the ohlc_hw_cleaned table):

    HW2/Scripts/python.exe 03_run_all_stocks.py      (Windows)
    HW2/bin/python 03_run_all_stocks.py              (macOS / Linux)

Outputs, all inside reports/{run date}/:
    {TICKER}_report.ipynb    one executed report per successful ticker
    failed_stocks.txt        tickers whose report failed (empty file if none)
    run.log                  log of the whole run, including full error messages
    portfolio_summary.csv    one row per successful ticker, sorted by Sharpe ratio
    risk_return.png          risk-return scatter plot of the portfolio
"""

import logging
import os
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures to files only (no window needed)
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

# All paths are relative to this script, so it works from any current directory
HOMEWORK_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = HOMEWORK_DIR / "02_stock_report_template.ipynb"
DB_PATH = HOMEWORK_DIR / "data" / "data.db"
REPORTS_ROOT = HOMEWORK_DIR / "reports"
CLEAN_TABLE = "ohlc_hw_cleaned"

SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                   "sharpe_ratio", "total_return", "report_path"]

logger = logging.getLogger("run_all_stocks")


def setup_logging(log_file: Path) -> None:
    """Short messages on the console; everything (including full tracebacks) in the log file."""
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()                       # avoid duplicate lines if main() runs twice
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    log_file_handler = logging.FileHandler(log_file, encoding="utf-8")
    log_file_handler.setLevel(logging.DEBUG)
    for handler in (console, log_file_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def load_tickers_and_latest_date(db_path: Path) -> tuple[list[str], str]:
    """Return the sorted list of tickers and the latest observation date in the cleaned table."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}. Run 01_data_cleaning.ipynb first.")
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
        tickers = pd.read_sql_query(f"SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker", conn)
        latest = conn.execute(f"SELECT MAX(ts) FROM {CLEAN_TABLE}").fetchone()[0]
    if latest is None:
        raise ValueError(f"{CLEAN_TABLE} is empty: nothing to report on.")
    return tickers["ticker"].tolist(), latest


def last_six_complete_months(latest_date: str) -> tuple[str, str]:
    """Six calendar months before the month that contains the latest observation.

    The latest month is always excluded (even when the latest date is a month-end),
    because a final trading date alone does not prove that the month is complete.
    Example: latest 2024-12-06 -> ("2024-06-01", "2024-11-30").
    """
    first_day_of_latest_month = pd.Timestamp(latest_date).to_period("M").start_time
    start = first_day_of_latest_month - pd.DateOffset(months=6)
    end = first_day_of_latest_month - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(ticker: str, start_date: str, end_date: str, output_dir: Path) -> Path:
    """Execute the template for one ticker; any error is raised to the caller."""
    output_path = output_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        os.path.relpath(TEMPLATE_PATH),   # relative paths: papermill stores them in the report's metadata
        os.path.relpath(output_path),
        # db_path is relative to cwd below, so the saved reports contain no machine-specific paths
        parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                    "db_path": DB_PATH.relative_to(HOMEWORK_DIR).as_posix()},
        kernel_name="python3",      # the Python kernel of the environment running this script
        cwd=str(HOMEWORK_DIR),
        progress_bar=False,
    )
    return output_path


def collect_results(report_paths: list[Path]) -> pd.DataFrame:
    """Read the 'report_results' scrap from each SUCCESSFUL report of this run (never the whole folder)."""
    rows = []
    for path in report_paths:
        results = sb.read_notebook(str(path)).scraps["report_results"].data
        results["report_path"] = path.relative_to(HOMEWORK_DIR).as_posix()
        rows.append(results)
    summary = pd.DataFrame(rows)
    # null in the scrap means "undefined"; make it a proper NaN number
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    summary[numeric] = summary[numeric].astype(float)
    return summary


def plot_risk_return(summary: pd.DataFrame, output_path: Path, start_date: str, end_date: str) -> None:
    """Scatter of average volatility vs total return, coloured by Sharpe ratio, top 5 returns annotated."""
    plottable = summary.dropna(subset=["avg_volatility", "total_return"])
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    points = ax.scatter(plottable["avg_volatility"], plottable["total_return"], c=plottable["sharpe_ratio"],
                        cmap="RdYlGn", s=110, edgecolors="black", linewidths=0.6)
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in plottable.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(7, 5),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Average 20-day rolling volatility (annualized)")
    ax.set_ylabel("Total return over the window")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=16, fontweight="bold")
    ax.text(0.01, 0.01, f"Requested window {start_date} to {end_date}; top 5 by total return labelled. "
                        "Some tickers have shorter coverage (see portfolio_summary.csv).",
            transform=ax.transAxes, fontsize=9, color="dimgray")
    ax.grid(True, alpha=0.4)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Run every report, record failures, and build the portfolio summary. Returns the exit code."""
    # 1. Tickers and the analysis window, both taken from the cleaned table (not from the computer clock)
    tickers, latest_date = load_tickers_and_latest_date(DB_PATH)
    start_date, end_date = last_six_complete_months(latest_date)

    # 2. Output folder named after the RUN date (independent of the historical window)
    run_dir = REPORTS_ROOT / date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    logger.info("Latest observation in %s: %s", CLEAN_TABLE, latest_date)
    logger.info("Analysis window (last 6 complete months): %s to %s", start_date, end_date)
    logger.info("Generating %d reports into %s", len(tickers), run_dir)

    # 3. One report per ticker; a failure is logged and the loop continues
    successful_paths, failed = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successful_paths.append(run_report(ticker, start_date, end_date, run_dir))
        except Exception as error:
            failed.append(ticker)
            last_line = str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__
            logger.error("Report for %s failed: %s", ticker, last_line)
            logger.debug("Full error for %s:\n%s", ticker, error)

    (run_dir / "failed_stocks.txt").write_text("".join(f"{ticker}\n" for ticker in failed), encoding="utf-8")
    print(f"Successfully generated {len(successful_paths)}/{len(tickers)} reports")
    if failed:
        logger.warning("Failed tickers (also in failed_stocks.txt): %s", ", ".join(failed))

    if not successful_paths:
        logger.error("No report succeeded, so there is no portfolio summary. See run.log for the errors.")
        return 1

    # 4. Portfolio summary from the successful reports of THIS run only
    results = collect_results(successful_paths)
    summary_df = (results[SUMMARY_COLUMNS + ["first_observation", "last_observation", "trading_days"]]
                  .sort_values("sharpe_ratio", ascending=False, na_position="last")
                  .reset_index(drop=True))
    summary_df.to_csv(run_dir / "portfolio_summary.csv", index=False)

    # Disclose shorter histories: same requested window for everyone, but not the same coverage
    window_end = pd.Timestamp(end_date)
    short = summary_df[(window_end - pd.to_datetime(summary_df["last_observation"])).dt.days > 7]
    if not short.empty:
        logger.info("%d of %d tickers have data only until %s (before the window end %s); their metrics "
                    "cover a shorter period and are not fully comparable: %s", len(short), len(summary_df),
                    ", ".join(sorted(short["last_observation"].unique())), end_date, ", ".join(short["ticker"]))

    # Rankings exclude undefined Sharpe ratios; with fewer than 20 tickers the two lists can overlap
    ranked = summary_df.dropna(subset=["sharpe_ratio"])
    display_columns = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown",
                       "avg_daily_return", "last_observation"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print(f"\nTop 10 by Sharpe ratio ({start_date} to {end_date}):")
        print(ranked.head(10)[display_columns].to_string(index=False))
        print(f"\nBottom 10 by Sharpe ratio ({start_date} to {end_date}):")
        print(ranked.tail(10)[display_columns].to_string(index=False))
    excluded = summary_df["sharpe_ratio"].isna().sum()
    if excluded:
        logger.info("%d ticker(s) excluded from the ranking because their Sharpe ratio is undefined", excluded)

    plot_risk_return(summary_df, run_dir / "risk_return.png", start_date, end_date)
    logger.info("Saved portfolio_summary.csv and risk_return.png in %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
