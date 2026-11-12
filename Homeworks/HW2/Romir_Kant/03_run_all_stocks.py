"""Generate a risk report for every stock in ohlc_hw_cleaned and summarize the portfolio.

Run from this folder after 01_data_cleaning.ipynb has created ohlc_hw_cleaned:

    python 03_run_all_stocks.py

For each ticker, papermill executes 02_stock_report_template.ipynb over the last six complete
months in the database and saves reports/{run date}/{ticker}_report.ipynb. Failures are logged
and skipped. Metrics are collected (with Scrapbook) only from reports that succeeded in this run,
then ranked, plotted and saved to reports/{run date}/portfolio_summary.csv.
"""

import argparse
import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HOMEWORK_DIR = Path(__file__).resolve().parent
TEMPLATE = HOMEWORK_DIR / "02_stock_report_template.ipynb"
SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                   "total_return", "report_path", "first_date", "last_date", "trading_days"]

log = logging.getLogger("run_all_stocks")


def load_tickers_and_latest_date(db_path: Path) -> tuple[list[str], pd.Timestamp]:
    """Return the sorted tickers in ohlc_hw_cleaned and the latest observation date."""
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"]
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    return tickers.tolist(), pd.Timestamp(latest)


def six_complete_months(latest: pd.Timestamp) -> tuple[str, str]:
    """The six calendar months before the month containing `latest`, as ISO dates.

    The month of the latest observation is excluded even if it ends on a month-end, because a final
    trading date alone does not prove that month's delivery is complete.
    e.g. latest 2024-12-06 -> ("2024-06-01", "2024-11-30").
    """
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_reports(tickers: list[str], start_date: str, end_date: str, db_path: Path,
                output_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """Execute the template once per ticker. Returns ({ticker: report path}, {ticker: error})."""
    successes, failures = {}, {}
    # Reports run with cwd=HOMEWORK_DIR, so a relative db_path keeps them portable to other machines.
    db = db_path.resolve()
    db_param = db.relative_to(HOMEWORK_DIR).as_posix() if db.is_relative_to(HOMEWORK_DIR) else str(db)
    for ticker in tqdm(tickers, desc="Generating reports"):
        report_path = output_dir / f"{ticker}_report.ipynb"
        try:
            pm.execute_notebook(
                TEMPLATE, report_path,
                parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                            "db_path": db_param},
                cwd=HOMEWORK_DIR, progress_bar=False,
            )
            successes[ticker] = report_path
        except Exception as exc:  # one bad stock must not stop the batch
            # A failed run still leaves a partial notebook on disk; it is never collected below.
            message = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', exc)}"
            failures[ticker] = message.strip().splitlines()[0]
            tqdm.write(f"FAILED {ticker}: {failures[ticker]}")
            log.warning("Report failed for %s: %s", ticker, failures[ticker])
    return successes, failures


def collect_results(successes: dict[str, Path]) -> pd.DataFrame:
    """Read the named Scrapbook outputs from each successful report into one DataFrame."""
    rows = []
    for ticker, path in successes.items():
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in SUMMARY_COLUMNS if name in scraps}
        # Relative to this folder when possible, so the CSV doesn't contain machine-specific paths.
        shown = path.relative_to(HOMEWORK_DIR) if path.is_relative_to(HOMEWORK_DIR) else path
        row["ticker"], row["report_path"] = ticker, shown.as_posix()
        rows.append(row)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    summary[numeric] = summary[numeric].astype(float)  # None (undefined) -> NaN
    return summary.sort_values("sharpe_ratio", ascending=False, na_position="last", ignore_index=True)


def print_rankings(summary: pd.DataFrame, end_date: str) -> None:
    """Print the top and bottom 10 by Sharpe ratio and flag shorter coverage."""
    ranked = summary.dropna(subset=["sharpe_ratio"])  # undefined Sharpe cannot be ranked
    show = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "last_date"]
    formats = {"sharpe_ratio": "{:.2f}".format, "total_return": "{:+.2%}".format,
               "avg_volatility": "{:.2%}".format, "max_drawdown": "{:.2%}".format}
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10)[show].to_string(index=False, formatters=formats))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10)[show].to_string(index=False, formatters=formats))
    if len(ranked) < len(summary):
        print(f"\nNot ranked (undefined Sharpe): {summary.loc[summary['sharpe_ratio'].isna(), 'ticker'].tolist()}")
    if len(ranked) < 20:
        print("Fewer than 20 ranked stocks: the top and bottom lists overlap.")

    short = summary[pd.to_datetime(summary["last_date"]) < pd.Timestamp(end_date) - pd.offsets.BDay(5)]
    if len(short):
        print(f"\nNote: {len(short)} of {len(summary)} stocks have data ending before {end_date} "
              f"(e.g. {', '.join(short['ticker'].head(5))} end {short['last_date'].min()}). Their metrics cover a "
              "shorter period, so compare them with the others with care.")


def plot_risk_return(summary: pd.DataFrame, end_date: str, path: Path) -> None:
    """Scatter average volatility vs total return, coloured by Sharpe, top 5 returns labelled."""
    fig, ax = plt.subplots(figsize=(11, 7))
    data = summary.dropna(subset=["avg_volatility", "total_return"])
    full = pd.to_datetime(data["last_date"]) >= pd.Timestamp(end_date) - pd.offsets.BDay(5)
    defined = data["sharpe_ratio"].notna()
    norm = plt.Normalize(data["sharpe_ratio"].min(), data["sharpe_ratio"].max())
    for mask, marker in ((full, "o"), (~full, "^")):
        subset = data[mask & defined]
        points = ax.scatter(subset["avg_volatility"], subset["total_return"], c=subset["sharpe_ratio"],
                            cmap="RdYlGn", norm=norm, marker=marker, s=90, edgecolors="black")
    undefined = data[~defined]
    if len(undefined):
        ax.scatter(undefined["avg_volatility"], undefined["total_return"], color="lightgray",
                   edgecolors="black", s=90)
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for _, row in data.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                    xytext=(6, 6), textcoords="offset points", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(title="Risk-Return Profile of Portfolio", xlabel="Average 20-day rolling volatility (annualized)",
           ylabel="Total return over the report period")
    ax.grid(True)
    # Neutral legend markers: colour encodes Sharpe only, marker shape encodes coverage.
    handles = [Line2D([], [], marker=m, linestyle="", markerfacecolor=c, markeredgecolor="black",
                      markersize=9, label=label)
               for m, c, label in (("o", "white", "Full window"), ("^", "white", "Data end early"))]
    if len(undefined):
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="lightgray",
                              markeredgecolor="black", markersize=9, label="Sharpe undefined"))
    ax.legend(handles=handles, loc="best", title="Marker = coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Run the batch; return a process exit code (1 if no report succeeded)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=HOMEWORK_DIR / "data" / "data.db")
    parser.add_argument("--output-root", type=Path, default=HOMEWORK_DIR / "reports")
    parser.add_argument("--tickers", nargs="+", help="run only these tickers (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("papermill").setLevel(logging.WARNING)  # keep the progress bar readable

    if not args.db_path.is_file():
        log.error("Database not found: %s. Run 01_data_cleaning.ipynb first.", args.db_path)
        return 1
    all_tickers, latest = load_tickers_and_latest_date(args.db_path)
    tickers = args.tickers or all_tickers
    start_date, end_date = six_complete_months(latest)
    output_dir = args.output_root / date.today().isoformat()  # run date, not the analysis window
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Latest observation %s -> analysis window %s to %s", latest.date(), start_date, end_date)
    log.info("Writing %d reports to %s", len(tickers), output_dir)

    successes, failures = run_reports(tickers, start_date, end_date, args.db_path, output_dir)
    print(f"\nSuccessfully generated {len(successes)}/{len(tickers)} reports")
    (output_dir / "failed_stocks.txt").write_text(
        "".join(f"{ticker}: {error}\n" for ticker, error in failures.items()), encoding="utf-8")
    if failures:
        print(f"Failed tickers saved to {output_dir / 'failed_stocks.txt'}: {', '.join(failures)}")
    if not successes:
        print("No report succeeded, so there is no portfolio summary.")
        return 1

    summary = collect_results(successes)
    print_rankings(summary, end_date)
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    plot_risk_return(summary, end_date, output_dir / "risk_return_profile.png")
    print(f"\nSaved {output_dir / 'portfolio_summary.csv'} and {output_dir / 'risk_return_profile.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
