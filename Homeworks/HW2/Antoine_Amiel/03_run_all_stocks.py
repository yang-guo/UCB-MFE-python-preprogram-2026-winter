"""Generate a risk report for every stock in ohlc_hw_cleaned, then a portfolio summary.

Usage (from this homework folder, after running 01_data_cleaning.ipynb):

    python 03_run_all_stocks.py
    python 03_run_all_stocks.py --tickers AAPL MSFT    # subset, e.g. for a quick test

For each ticker, papermill executes 02_stock_report_template.ipynb over a common window
(the last 6 complete calendar months of the database) and writes
reports/<run date>/<TICKER>_report.ipynb. A failing ticker is logged and skipped.
Metrics are collected with scrapbook from the successful reports of this run only.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                   "total_return", "report_path"]
COVERAGE_COLUMNS = ["start_date", "end_date", "first_date", "last_date", "trading_days"]

log = logging.getLogger("run_all_stocks")


def display_path(path: Path) -> str:
    """Path relative to the homework folder when possible, absolute otherwise."""
    try:
        return str(path.relative_to(HERE))
    except ValueError:
        return str(path)


def load_tickers(db_path: Path) -> list[str]:
    """Return the sorted list of tickers in ohlc_hw_cleaned."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def last_six_complete_months(db_path: Path) -> tuple[str, str]:
    """Return (start, end) of the six calendar months before the month of the latest observation.

    The window is anchored on the data, not on the computer clock. The month containing the latest
    cleaned observation is always excluded, even when that observation falls on a month-end, because
    a final trading date alone does not certify a complete delivery for that month.
    """
    with sqlite3.connect(db_path) as conn:
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    month_start = pd.Timestamp(latest).to_period("M").to_timestamp()
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(template: Path, output: Path, ticker: str, start: str, end: str, db_path: Path,
               kernel: str) -> Path:
    """Execute the template for one ticker with papermill; errors propagate to the caller."""
    pm.execute_notebook(
        str(template), str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": str(db_path)},
        kernel_name=kernel, cwd=str(HERE), progress_bar=False,
    )
    return output


def collect_metrics(paths: list[Path]) -> pd.DataFrame:
    """Read the 'metrics' scrap of each successful report into one DataFrame."""
    rows = []
    for path in paths:
        metrics = dict(sb.read_notebook(str(path)).scraps["metrics"].data)
        metrics["report_path"] = display_path(path)
        rows.append(metrics)
    summary = pd.DataFrame(rows)
    if summary["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers among the collected reports")
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    summary[numeric] = summary[numeric].astype(float)  # None (undefined) becomes NaN
    return summary


def plot_risk_return(summary: pd.DataFrame, path: Path) -> None:
    """Scatter of average volatility vs total return, colored by Sharpe ratio, top 5 returns annotated."""
    data = summary.dropna(subset=["avg_volatility", "total_return"])
    full = data["last_date"] >= data["end_date"].str[:7]  # observed into the last month of the window
    fig, ax = plt.subplots(figsize=(11, 7.5))
    vmax = np.nanmax(np.abs(data["sharpe_ratio"])) if data["sharpe_ratio"].notna().any() else 1.0
    common = dict(c=data["sharpe_ratio"], cmap="RdYlGn", vmin=-vmax, vmax=vmax, s=90, edgecolors="black",
                  linewidths=0.6)
    sc = ax.scatter(data.loc[full, "avg_volatility"], data.loc[full, "total_return"], marker="o",
                    **{**common, "c": data.loc[full, "sharpe_ratio"]})
    ax.scatter(data.loc[~full, "avg_volatility"], data.loc[~full, "total_return"], marker="^",
               **{**common, "c": data.loc[~full, "sharpe_ratio"]})
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    for _, row in data.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel="Average 20-day rolling volatility (annualized)", ylabel="Total return over observed coverage",
           title="Risk-Return Profile of Portfolio")
    ax.title.set_fontweight("bold")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#bbbbbb", markeredgecolor="black", markersize=9, label="Full window"),
        Line2D([], [], marker="^", ls="", color="#bbbbbb", markeredgecolor="black", markersize=9,
               label="History ends before the last month (shorter coverage)"),
    ], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/data.db", help="local working copy of the course database")
    parser.add_argument("--template", default="02_stock_report_template.ipynb")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel used to run the reports")
    parser.add_argument("--tickers", nargs="+", help="override the ticker list (default: all cleaned tickers)")
    args = parser.parse_args(argv)

    db_path = (HERE / args.db).resolve()
    template = (HERE / args.template).resolve()
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    tickers = args.tickers or load_tickers(db_path)
    start, end = last_six_complete_months(db_path)
    out_dir = HERE / args.reports_dir / date.today().isoformat()  # run date, independent of the window
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log", mode="w")])
    log.info("Window %s to %s, %d tickers, output %s", start, end, len(tickers), out_dir)
    print(f"Analysis window: {start} to {end} (last 6 complete months of the database)")
    print(f"Output directory: {display_path(out_dir)}")

    successes, failures = [], {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        output = out_dir / f"{ticker}_report.ipynb"
        try:
            successes.append(run_report(template, output, ticker, start, end, db_path, args.kernel))
            log.info("%s: report written to %s", ticker, output.name)
        except Exception as exc:  # one failing stock must not stop the batch
            reason = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            failures[ticker] = reason
            log.error("%s failed: %s", ticker, reason)
            if output.exists():  # keep the partial notebook for debugging, under a name that cannot be mistaken
                output.replace(output.with_name(f"{ticker}_report_FAILED.ipynb"))

    (out_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failures))
    print(f"Successfully generated {len(successes)}/{len(tickers)} reports")
    for ticker, reason in failures.items():
        print(f"  failed: {ticker}: {reason}")
    if not successes:
        print("No report succeeded, so there is no portfolio summary. See failed_stocks.txt and run.log.")
        return 1

    summary = collect_metrics(successes)
    summary = summary.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)
    summary[SUMMARY_COLUMNS + COVERAGE_COLUMNS].to_csv(out_dir / "portfolio_summary.csv", index=False)

    short = summary[summary["last_date"] < summary["end_date"].str[:7]]
    if len(short):
        print(f"\n{len(short)} of {len(summary)} histories end before the last month of the window "
              f"(last observation {short['last_date'].min()} to {short['last_date'].max()}); their metrics "
              "cover a shorter period and are not strictly comparable with full-window stocks.")

    ranked = summary.dropna(subset=["sharpe_ratio"])
    if len(ranked) < len(summary):
        print(f"{len(summary) - len(ranked)} report(s) with undefined Sharpe ratio excluded from the ranking.")
    view = ranked[SUMMARY_COLUMNS[:-1] + ["last_date"]]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print("\nTop 10 by Sharpe ratio:\n", view.head(10).to_string(index=False))
        print("\nBottom 10 by Sharpe ratio:\n", view.tail(10).to_string(index=False))

    plot_risk_return(summary, out_dir / "risk_return_profile.png")
    print(f"\nSaved {display_path(out_dir)}/portfolio_summary.csv and risk_return_profile.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
