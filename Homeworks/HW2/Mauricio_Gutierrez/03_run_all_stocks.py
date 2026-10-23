"""HW2 Part 3: generate a risk report for every stock and summarize the portfolio.

What this script does:
    1. Reads the list of tickers and the latest observation date from `ohlc_hw_cleaned`.
    2. Chooses the analysis window: the six complete calendar months before the month
       of the latest observation (e.g. latest 2024-12-06 -> 2024-06-01 to 2024-11-30).
    3. Runs `02_stock_report_template.ipynb` once per ticker with papermill and saves
       each executed notebook to `reports/{run date}/{ticker}_report.ipynb`.
    4. Collects the metrics each successful report saved with scrapbook (`sb.glue`)
       into a portfolio summary, prints the top/bottom 10 by Sharpe ratio, draws a
       risk-return scatter plot, and saves `portfolio_summary.csv`.

Run it from the repository root (or anywhere) with the course environment:
    uv run python Homeworks/HW2/Mauricio_Gutierrez/03_run_all_stocks.py

Run `01_data_cleaning.ipynb` first: it creates the `ohlc_hw_cleaned` table this script reads.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts to files only; no window is needed when running a script

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import pandas as pd  # noqa: E402
import papermill as pm  # noqa: E402
import scrapbook as sb  # noqa: E402
from tqdm import tqdm  # noqa: E402

# --- Paths ---------------------------------------------------------------------
# Everything is located relative to this file, so the script works no matter which
# folder it is launched from.
HW_DIR = Path(__file__).resolve().parent
TEMPLATE = HW_DIR / "02_stock_report_template.ipynb"
DB_PATH = HW_DIR / "data" / "data.db"
REPORTS_ROOT = HW_DIR / "reports"

# "python3" tells papermill to run each report with the SAME Python that runs this script
# (jupyter_client swaps in the current interpreter). With `uv run` that is the course
# environment, whatever kernel name the template notebook was last saved with.
KERNEL_NAME = "python3"

# Values every report saves with sb.glue (see the last cell of the template)
SCRAP_NAMES = [
    "ticker", "start_date", "end_date", "first_date", "last_date", "trading_days",
    "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return",
]

# Column order of the portfolio summary: the assignment's columns first, then the actual
# data coverage of each report (histories end on different dates, so it must be disclosed)
SUMMARY_COLUMNS = [
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
    "total_return", "report_path", "first_date", "last_date", "trading_days",
]

log = logging.getLogger("run_all_stocks")


# --- Step 1: tickers and analysis window ---------------------------------------
def load_tickers_and_latest_date(db_path: Path) -> tuple[list[str], str]:
    """Return (sorted list of tickers, latest observation date) from ohlc_hw_cleaned.

    Raises FileNotFoundError if the database is missing, and ValueError if the cleaned
    table is missing or empty.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    # Read-only connection: this script never changes the data
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type = 'table'", conn)["name"]
        if "ohlc_hw_cleaned" not in set(tables):
            raise ValueError("Table ohlc_hw_cleaned not found: run 01_data_cleaning.ipynb first")
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn
        )["ticker"].tolist()
        latest = pd.read_sql_query("SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn)["latest"].iloc[0]
    if not tickers or latest is None:
        raise ValueError("ohlc_hw_cleaned is empty")
    return tickers, latest


def six_complete_months(latest_date: str) -> tuple[str, str]:
    """Return (start, end) of the six calendar months before the latest date's month.

    The month containing the latest observation is always excluded, even when the latest
    date is a month-end, because a final trading date alone does not prove the month's
    data is complete.

    >>> six_complete_months("2024-12-06")
    ('2024-06-01', '2024-11-30')
    >>> six_complete_months("2025-01-15")   # crosses a year boundary
    ('2024-07-01', '2024-12-31')
    """
    latest = pd.Timestamp(latest_date)
    month_start = latest.to_period("M").start_time          # first day of the latest month
    start = month_start - pd.DateOffset(months=6)            # six months earlier
    end = month_start - pd.Timedelta(days=1)                 # last day of the previous month
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# --- Step 2: run one report per ticker -----------------------------------------
def run_report(ticker: str, start_date: str, end_date: str, out_dir: Path) -> Path:
    """Execute the template for one ticker and return the path of the executed notebook.

    Any error raised inside the notebook (e.g. too little history) is raised here as a
    papermill exception, so the caller can record the ticker as failed.
    """
    output = out_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE),
        str(output),
        parameters={
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "db_path": str(DB_PATH),   # absolute path, so it works from any working directory
        },
        kernel_name=KERNEL_NAME,
        cwd=str(HW_DIR),               # the report runs inside the homework folder
        progress_bar=False,            # one overall tqdm bar instead of one bar per notebook
    )
    return output


def generate_reports(tickers: list[str], start_date: str, end_date: str,
                     out_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """Run every report, continuing after failures.

    Returns (successes, failures): ticker -> report path for reports that finished, and
    ticker -> error message for those that did not. Papermill still writes a notebook for a
    failed run (useful to see the error), so only the `successes` paths may be summarized.
    """
    successes: dict[str, Path] = {}
    failures: dict[str, str] = {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successes[ticker] = run_report(ticker, start_date, end_date, out_dir)
        except Exception as exc:  # any failure for one ticker must not stop the others
            # papermill's message ends with the notebook's own error, e.g. "ValueError: ..."
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
            failures[ticker] = message
            log.error("%s failed: %s", ticker, message)
    return successes, failures


# --- Step 3: portfolio summary --------------------------------------------------
def collect_summary(successes: dict[str, Path]) -> pd.DataFrame:
    """Build the portfolio summary from the scrapbook values of successful reports only.

    Raises ValueError if there are no successful reports (an empty summary would be misleading).
    """
    if not successes:
        raise ValueError("No successful reports to summarize")
    rows = []
    for ticker, path in successes.items():
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in SCRAP_NAMES}
        # Short path with forward slashes, so the CSV reads the same on Windows, macOS and Linux
        row["report_path"] = path.relative_to(HW_DIR).as_posix()
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Safety check: every report must answer the same requested window
    if summary["start_date"].nunique() != 1 or summary["end_date"].nunique() != 1:
        raise ValueError("Reports were generated for different windows")
    return (
        summary[SUMMARY_COLUMNS]
        .sort_values("sharpe_ratio", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with fractions shown as percentages (the CSV keeps the raw fractions)."""
    shown = df[["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown",
                "avg_daily_return", "first_date", "last_date"]].copy()
    for col in ["total_return", "max_drawdown"]:
        shown[col] = shown[col].map("{:+.2%}".format)              # signed: gains and losses
    shown["avg_volatility"] = shown["avg_volatility"].map("{:.2%}".format)   # always >= 0
    shown["avg_daily_return"] = shown["avg_daily_return"].map("{:+.3%}".format)
    shown["sharpe_ratio"] = shown["sharpe_ratio"].map("{:.2f}".format)
    return shown


def print_rankings(summary: pd.DataFrame, n: int = 10) -> None:
    """Print the top and bottom `n` stocks by Sharpe ratio, skipping undefined (NaN) values.

    With fewer than 2*n ranked stocks, the two lists overlap; that is expected.
    """
    ranked = summary.dropna(subset=["sharpe_ratio"])          # already sorted, best first
    excluded = len(summary) - len(ranked)
    if excluded:
        print(f"\n{excluded} stock(s) left out of the rankings: Sharpe ratio undefined")
    print(f"\nTop {n} stocks by Sharpe ratio")
    print(format_for_display(ranked.head(n)).to_string(index=False))
    print(f"\nBottom {n} stocks by Sharpe ratio")
    print(format_for_display(ranked.tail(n).iloc[::-1]).to_string(index=False))   # worst first


def report_coverage(summary: pd.DataFrame) -> None:
    """Point out reports whose data ends before the rest of the portfolio's does.

    The reference is the latest last_date among all reports, i.e. the last trading day in
    the window that has data at all (the window's calendar end can be a weekend or holiday,
    e.g. 2024-11-30 is a Saturday). Prices are never extended past a stock's last
    observation, so the shorter reports cover a shorter period: compare them with care.
    """
    last_trading_day = summary["last_date"].max()
    short = summary[summary["last_date"] < last_trading_day]
    if short.empty:
        print(f"\nEvery report has data up to {last_trading_day}, the last trading day in the window.")
        return
    print(f"\n{len(short)} of {len(summary)} reports end before {last_trading_day}, "
          "the last trading day in the window (their data stops earlier):")
    for last_date, group in short.groupby("last_date"):
        print(f"  data ends {last_date}: {', '.join(group['ticker'])}")


def plot_risk_return(summary: pd.DataFrame, start_date: str, end_date: str, path: Path) -> None:
    """Scatter plot of mean rolling volatility vs total return, colored by Sharpe ratio.

    The top 5 stocks by total return are labeled. Stocks with an undefined Sharpe ratio are
    drawn as gray points (they have no color on the Sharpe scale).
    """
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    x = summary["avg_volatility"] * 100       # percent only for display
    y = summary["total_return"] * 100
    defined = summary["sharpe_ratio"].notna()

    points = ax.scatter(x[defined], y[defined], c=summary.loc[defined, "sharpe_ratio"],
                        cmap="RdYlGn", s=80, edgecolor="black", linewidth=0.5)
    if (~defined).any():
        ax.scatter(x[~defined], y[~defined], color="gray", s=80, label="Sharpe undefined")
        ax.legend(loc="lower right")
    fig.colorbar(points, ax=ax, label="Sharpe ratio")

    for row in summary.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)   # above the line = made money over the window
    ax.margins(0.12)                              # keep labels inside the chart
    ax.grid(True, alpha=0.4)
    ax.set(xlabel="Mean rolling volatility, annualized (%)", ylabel="Total return (%)")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=16, fontweight="bold")
    # Caption inside the bottom-left corner of the plot, where it cannot overlap the axis labels
    ax.text(0.01, 0.01, f"Window {start_date} to {end_date}; color = Sharpe ratio; labels = top 5 by return",
            transform=ax.transAxes, fontsize=9, color="dimgray")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Main -----------------------------------------------------------------------
def main() -> int:
    """Run the whole pipeline. Returns the exit code: 0 if at least one report succeeded."""
    tickers, latest = load_tickers_and_latest_date(DB_PATH)
    start_date, end_date = six_complete_months(latest)

    # The folder is named after the day the script runs, not after the analysis window
    out_dir = REPORTS_ROOT / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Errors are shown on screen and also written to run.log in the output folder
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log", mode="w")],
    )
    logging.getLogger("papermill").setLevel(logging.WARNING)   # hide papermill's per-notebook progress lines
    log.info("Latest cleaned observation: %s", latest)
    log.info("Analysis window: %s to %s (%d tickers)", start_date, end_date, len(tickers))
    log.info("Output folder: %s", out_dir)

    successes, failures = generate_reports(tickers, start_date, end_date, out_dir)

    # One ticker per line; the file is written (empty) even when nothing failed
    (out_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failures), encoding="utf-8")
    print(f"\nSuccessfully generated {len(successes)}/{len(tickers)} reports")
    if failures:
        print(f"Failed tickers ({len(failures)}), saved to failed_stocks.txt:")
        for ticker, message in failures.items():
            print(f"  {ticker}: {message}")

    if not successes:
        print("No report succeeded, so there is no portfolio summary. See run.log for the errors.")
        return 1

    summary = collect_summary(successes)
    summary_path = out_dir / "portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)          # raw fractions, not percentages

    print_rankings(summary)
    report_coverage(summary)
    plot_path = out_dir / "risk_return_profile.png"
    plot_risk_return(summary, start_date, end_date, plot_path)

    print(f"\nPortfolio summary saved to {summary_path}")
    print(f"Risk-return chart saved to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
