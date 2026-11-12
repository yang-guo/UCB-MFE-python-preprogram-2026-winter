"""Run the single-stock risk report for every ticker and build a portfolio summary.

Steps
1. Read the list of tickers from the table ohlc_hw_cleaned.
2. Find the report window: the 6 complete calendar months before the month of the
   latest cleaned observation (not the computer clock).
3. For each ticker, run 02_stock_report_template.ipynb with papermill and save the
   executed notebook to reports/<run date>/<TICKER>_report.ipynb.
4. Log failures, write failed_stocks.txt, and print "Successfully generated N/M reports".
5. Collect the named results (Scrapbook) of the successful reports only, then save
   reports/<run date>/portfolio_summary.csv and a risk-return scatter plot.

Run from anywhere:  python 03_run_all_stocks.py
Needs: data/data.db (working copy with the table ohlc_hw_cleaned) next to this file.
"""
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw without a window, so the script also runs on a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "data.db"
TEMPLATE = HERE / "02_stock_report_template.ipynb"
REPORTS_DIR = HERE / "reports"
FAILED_FILE = HERE / "failed_stocks.txt"

# Names saved by the template with sb.glue(). Values are fractions (not percentages).
METRICS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
COVERAGE = ["first_observation", "last_observation", "trading_days"]


def load_tickers_and_latest_date(db_path):
    """Return the sorted ticker list and the latest observation date (read-only)."""
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
        tickers = [row[0] for row in conn.execute(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker")]
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    return tickers, pd.Timestamp(latest)


def last_six_complete_months(latest):
    """Window = the 6 calendar months before the month that holds the latest date.

    Example: latest date 2024-12-06 -> 2024-06-01 to 2024-11-30.
    """
    month_start = latest.replace(day=1)
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_one_report(ticker, start_date, end_date, out_dir):
    """Execute the template for one ticker. Raises an exception if the report fails."""
    output = out_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE),
        str(output),
        parameters={
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "db_path": str(DB_PATH),  # full path, so it works from any folder
        },
        progress_bar=False,
        cwd=str(HERE),
    )
    return output


def collect_results(report_path):
    """Read the named results saved inside one executed report."""
    scraps = sb.read_notebook(str(report_path)).scraps
    row = {"ticker": scraps["ticker"].data}
    for name in METRICS:
        value = scraps[name].data
        row[name] = np.nan if value is None else float(value)  # None means undefined
    row["report_path"] = str(report_path.relative_to(HERE))
    for name in COVERAGE:
        row[name] = scraps[name].data
    return row


def make_scatter(summary, window, out_path):
    """Risk-return scatter: x = average volatility, y = total return, color = Sharpe."""
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    ranked = summary.dropna(subset=["sharpe_ratio"])
    undefined = summary[summary["sharpe_ratio"].isna()]
    points = ax.scatter(ranked["avg_volatility"] * 100, ranked["total_return"] * 100,
                        c=ranked["sharpe_ratio"], cmap="RdYlGn", s=90,
                        edgecolor="black", linewidth=0.5)
    if not undefined.empty:
        ax.scatter(undefined["avg_volatility"] * 100, undefined["total_return"] * 100,
                   color="gray", s=90, edgecolor="black", linewidth=0.5,
                   label="Sharpe ratio undefined")
        ax.legend()
    if not ranked.empty:
        fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for _, row in summary.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"] * 100, row["total_return"] * 100),
                    xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Risk-Return Profile of Portfolio",
           xlabel="Average rolling volatility (annualized, %)",
           ylabel=f"Total return {window[0]} to {window[1]} (%)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not DB_PATH.is_file():
        sys.exit(f"Database not found: {DB_PATH}. Put the working copy of data.db there.")
    if not TEMPLATE.is_file():
        sys.exit(f"Template notebook not found: {TEMPLATE}")

    tickers, latest = load_tickers_and_latest_date(DB_PATH)
    if not tickers:
        sys.exit("The table ohlc_hw_cleaned has no tickers. Run 01_data_cleaning.ipynb first.")
    start_date, end_date = last_six_complete_months(latest)
    run_date = date.today().isoformat()  # the run date, not the analysis window
    out_dir = REPORTS_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Latest observation: {latest.date()}  ->  window {start_date} to {end_date}")
    print(f"{len(tickers)} tickers. Reports go to {out_dir.relative_to(HERE)}/")

    succeeded, failed = [], {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            succeeded.append((ticker, run_one_report(ticker, start_date, end_date, out_dir)))
        except Exception as error:  # keep going: one bad stock must not stop the batch
            lines = str(error).strip().splitlines()
            failed[ticker] = f"{type(error).__name__}: {lines[-1] if lines else ''}"

    FAILED_FILE.write_text("".join(f"{t}\n" for t in failed))
    if failed:
        (out_dir / "errors.log").write_text(
            "".join(f"{t}\t{msg}\n" for t, msg in failed.items()))
        print(f"Failed tickers: {', '.join(failed)} (see failed_stocks.txt and errors.log)")
    print(f"Successfully generated {len(succeeded)}/{len(tickers)} reports")

    if not succeeded:
        sys.exit("No report succeeded, so there is no portfolio summary.")

    rows = []
    for ticker, path in succeeded:
        try:
            rows.append(collect_results(path))
        except Exception as error:
            print(f"Could not read results of {ticker}: {error}")
    summary = pd.DataFrame(rows)
    summary = summary.sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary = summary[["ticker"] + METRICS + ["report_path"] + COVERAGE].reset_index(drop=True)
    summary.to_csv(out_dir / "portfolio_summary.csv", index=False)

    ranked = summary.dropna(subset=["sharpe_ratio"])  # undefined Sharpe is not ranked
    show = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown"]
    print("\nTop 10 by Sharpe ratio")
    print(ranked.head(10)[show].to_string(index=False))
    print("\nBottom 10 by Sharpe ratio")
    print(ranked.tail(10)[show].to_string(index=False))

    # Supplied histories end on different dates. Say so when a report is shorter.
    window_last = summary["last_observation"].max()
    short = summary[summary["last_observation"] < window_last]
    if not short.empty:
        print(f"\nNote: the latest observation in the window is {window_last}, but "
              f"{len(short)} tickers end earlier ({', '.join(short['ticker'])}). "
              "Their reports cover a shorter period, so compare them with care.")
    make_scatter(summary, (start_date, end_date), out_dir / "risk_return_scatter.png")
    print(f"\nSaved {out_dir.relative_to(HERE)}/portfolio_summary.csv and risk_return_scatter.png")


if __name__ == "__main__":
    main()