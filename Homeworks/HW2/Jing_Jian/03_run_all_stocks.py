"""Generate stock reports and collect successful results.

Run after 01_data_cleaning.ipynb:
    python 03_run_all_stocks.py
All paths default to this script's directory, regardless of terminal location.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # Save charts without opening a desktop window.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


def analysis_window(latest_date: str) -> tuple[str, str]:
    """Return six calendar months preceding the latest observation's month."""
    month_start = pd.Timestamp(latest_date).normalize().replace(day=1)
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def load_universe(database: Path) -> tuple[list[str], str, str]:
    """Read all cleaned tickers and derive one common requested window."""
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    with sqlite3.connect(database) as conn:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn
        )["ticker"].tolist()
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    if not tickers or latest is None:
        raise ValueError("Cleaned table is empty. Run the cleaning notebook first.")
    start, end = analysis_window(latest)
    return tickers, start, end


def collect_metrics(report: Path, ticker: str, start: str, end: str) -> dict:
    """Read named Scrapbook outputs from this successful report only."""
    book = sb.read_notebook(str(report))
    names = ["ticker", "start_date", "end_date", "actual_start", "actual_end",
             "n_prices", "avg_daily_return", "avg_volatility", "max_drawdown",
             "sharpe_ratio", "total_return"]
    result = {name: book.scraps[name].data for name in names}
    if (result["ticker"], result["start_date"], result["end_date"]) != (ticker, start, end):
        raise ValueError("Report identity/window does not match this run.")
    result["report_path"] = str(report.relative_to(ROOT)) if report.is_relative_to(ROOT) else str(report)
    return result


def save_dashboard(summary: pd.DataFrame, destination: Path) -> None:
    """Plot risk versus return, color finite Sharpe values, and mark top returns."""
    valid = summary.loc[
        np.isfinite(summary["avg_volatility"])
        & np.isfinite(summary["total_return"])
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    finite_sharpe = np.isfinite(valid["sharpe_ratio"])
    colored = valid.loc[finite_sharpe]
    if not colored.empty:
        points = ax.scatter(colored["avg_volatility"], colored["total_return"],
                            c=colored["sharpe_ratio"], cmap="viridis", s=80,
                            edgecolor="white", linewidth=0.7)
        fig.colorbar(points, ax=ax, label="Annualized Sharpe ratio")
    undefined = valid.loc[~finite_sharpe]
    if not undefined.empty:
        ax.scatter(undefined["avg_volatility"], undefined["total_return"],
                   color="gray", marker="x", label="Undefined Sharpe")
        ax.legend()
    for rank, (_, row) in enumerate(valid.nlargest(5, "total_return").iterrows()):
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                    xytext=(8, -18 if rank % 2 else 8), textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5})
    if valid.empty:
        ax.text(0.5, 0.5, "No finite risk-return pairs available", transform=ax.transAxes,
                ha="center")
    ax.set(title="Risk-Return Profile of Portfolio",
           xlabel="Mean annualized 20-day rolling volatility", ylabel="Total return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.01, "Actual observation coverage differs by ticker; inspect CSV coverage columns.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main() -> int:
    """Execute every ticker independently, then summarize this run's successes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=ROOT / "data/data.db")
    args = parser.parse_args()
    database = args.db_path.resolve()
    tickers, start, end = load_universe(database)
    template = ROOT / "02_stock_report_template.ipynb"
    if not template.is_file():
        raise FileNotFoundError(template)

    # Run date is separate from the historical analysis window.
    run_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    destination = ROOT / "reports" / run_date
    destination.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=destination / "batch.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    print(f"Requested common window: {start} to {end}")
    print(f"Output folder: {destination}")
    print("Shorter actual histories are disclosed; prices are not extrapolated.")
    successes, failures = [], []

    # Each failure is isolated, including errors collecting the named outputs.
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            if not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z0-9.^_-]+", ticker):
                raise ValueError(f"Unsafe report filename for ticker: {ticker!r}")
            report = destination / f"{ticker}_report.ipynb"
            pm.execute_notebook(
                str(template), str(report),
                parameters={"ticker": ticker, "start_date": start, "end_date": end,
                            "db_path": str(database)},
                cwd=str(ROOT), kernel_name="hw2-jing-jian", progress_bar=False,
                execution_timeout=180,
            )
            # Never scan the output directory: failed/stale notebooks may exist.
            successes.append(collect_metrics(report, ticker, start, end))
            logging.info("Successfully generated %s", ticker)
        except Exception as exc:
            failures.append(ticker)
            logging.exception("Failed %s: %s", ticker, exc)
            tqdm.write(f"Failed {ticker}: {exc}")

    (destination / "failed_stocks.txt").write_text(
        "".join(f"{ticker}\n" for ticker in failures)
    )
    print(f"Successfully generated {len(successes)}/{len(tickers)} reports")
    if not successes:
        # Clear only stale aggregate outputs from an earlier run today.
        for name in ["portfolio_summary.csv", "portfolio_summary.png"]:
            (destination / name).unlink(missing_ok=True)
        print("All reports failed. No portfolio summary was generated. See batch.log.")
        return 1

    summary = pd.DataFrame(successes).sort_values(
        "sharpe_ratio", ascending=False, na_position="last"
    )
    summary.to_csv(destination / "portfolio_summary.csv", index=False)
    ranked = summary.loc[np.isfinite(summary["sharpe_ratio"])]
    columns = ["ticker", "sharpe_ratio", "total_return", "actual_start", "actual_end"]
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10)[columns].to_string(index=False))
    print("\nBottom 10 by Sharpe ratio (worst first):")
    print(ranked.tail(10).sort_values("sharpe_ratio")[columns].to_string(index=False))
    if len(ranked) < 20:
        print("Top/bottom groups may overlap because fewer than 20 finite Sharpes exist.")
    save_dashboard(summary, destination / "portfolio_summary.png")
    print("\nSaved portfolio_summary.csv and portfolio_summary.png.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
