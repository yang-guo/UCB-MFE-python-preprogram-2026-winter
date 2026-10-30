"""Generate a risk report for every ticker in ohlc_hw_cleaned with papermill.

Usage (from this folder, after running 01_data_cleaning.ipynb):

    uv run python 03_run_all_stocks.py
    uv run python 03_run_all_stocks.py --tickers AAPL MSFT   # quick test on a subset

Steps:
1. read the tickers and the latest observation date from ohlc_hw_cleaned;
2. choose one common window: the six calendar months before the month of the latest
   observation (latest 2024-12-06 -> 2024-06-01 to 2024-11-30);
3. execute 02_stock_report_template.ipynb once per ticker into reports/{run date}/;
4. log and continue on failures, then write failed_stocks.txt;
5. collect named outputs (scrapbook scraps) from this run's successful reports only,
   save portfolio_summary.csv, and plot the risk-return profile.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_DB = HERE / "data" / "data.db"
REPORTS_ROOT = HERE / "reports"
CLEAN_TABLE = "ohlc_hw_cleaned"

# Scraps glued by the template; every successful report must provide all of them.
SCRAPS = ["ticker", "start_date", "end_date", "first_obs", "last_obs", "trading_days", "n_returns",
          "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return",
          "annualized_return", "annualized_volatility"]
SUMMARY_COLS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                "total_return", "report_path"]
COVERAGE_COLS = ["first_obs", "last_obs", "trading_days"]

log = logging.getLogger("run_all_stocks")


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite database without write access."""
    return sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)


def load_universe(db_path: Path) -> tuple[list[str], str]:
    """Return (sorted tickers, latest observation date) from the cleaned table."""
    with closing(connect_read_only(db_path)) as conn:
        tickers = pd.read_sql_query(f"SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker", conn)["ticker"].tolist()
        latest = pd.read_sql_query(f"SELECT MAX(ts) AS latest FROM {CLEAN_TABLE}", conn)["latest"].iloc[0]
    if not tickers or latest is None:
        raise ValueError(f"{CLEAN_TABLE} is empty; run 01_data_cleaning.ipynb first.")
    return tickers, latest


def six_complete_months(latest_date: str) -> tuple[str, str]:
    """Six calendar months before the month that contains `latest_date`.

    Anchored to the data, not the computer clock. The month of the latest observation is
    always excluded, even if the latest date is a month-end, because a final trading date
    alone does not prove that month's delivery is complete.
    """
    latest = pd.Timestamp(latest_date)
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(ticker: str, start: str, end: str, out_dir: Path, db_path: Path) -> Path:
    """Execute the template for one ticker; exceptions propagate to the caller."""
    output = out_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": str(db_path.resolve())},
        kernel_name="python3", cwd=str(HERE), progress_bar=False,
    )
    return output


def collect_summary(paths: list[Path]) -> pd.DataFrame:
    """Build the portfolio summary from the given (successful, this-run) report paths only."""
    rows = []
    for path in paths:
        scraps = sb.read_notebook(str(path)).scraps
        missing = [name for name in SCRAPS if name not in scraps]
        if missing:
            raise ValueError(f"{path.name} is missing scraps {missing}")
        row = {name: scraps[name].data for name in SCRAPS}
        row["report_path"] = str(path.relative_to(HERE))
        rows.append(row)
    if not rows:
        raise ValueError("No successful reports to summarize.")
    summary = pd.DataFrame(rows)
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return",
               "annualized_return", "annualized_volatility"]
    summary[numeric] = summary[numeric].astype(float)  # None (undefined) -> NaN
    if summary["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers in this run's reports.")
    return summary.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)


def plot_risk_return(summary: pd.DataFrame, start: str, end: str, path: Path) -> None:
    """Scatter of mean rolling volatility vs. total return, colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(11, 7))
    defined = summary["sharpe_ratio"].notna()
    points = ax.scatter(summary.loc[defined, "avg_volatility"], summary.loc[defined, "total_return"],
                        c=summary.loc[defined, "sharpe_ratio"], cmap="RdYlGn", s=90, edgecolor="black", linewidth=0.5)
    if (~defined).any():
        ax.scatter(summary.loc[~defined, "avg_volatility"], summary.loc[~defined, "total_return"],
                   color="lightgray", s=90, edgecolor="black", linewidth=0.5, label="Sharpe undefined")
        ax.legend(loc="lower right")
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in summary.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(xlabel="Mean 20-day rolling volatility (annualized)", ylabel="Total return",
           title=f"Risk-Return Profile of Portfolio\n{start} to {end} (requested window)")
    ax.grid(True, alpha=0.4)
    ax.margins(0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt_table(frame: pd.DataFrame) -> str:
    """Display formatting for the console (fractions stay fractions in the CSV)."""
    shown = frame[["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "avg_daily_return", "last_obs"]].copy()
    for col in ["total_return", "avg_volatility", "max_drawdown"]:
        shown[col] = shown[col].map(lambda v: f"{v:+.2%}" if col != "avg_volatility" else f"{v:.2%}")
    shown["avg_daily_return"] = shown["avg_daily_return"].map(lambda v: f"{v:+.4%}")
    shown["sharpe_ratio"] = shown["sharpe_ratio"].map(lambda v: f"{v:.2f}")
    return shown.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--tickers", nargs="+", help="optional subset of tickers (default: all)")
    parser.add_argument("--run-date", default=date.today().isoformat(), help="output folder name (default: today)")
    args = parser.parse_args(argv)

    run_dir = REPORTS_ROOT / args.run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "run.log", mode="w"), logging.StreamHandler(sys.stderr)])

    if not args.db_path.is_file():
        log.error("Database not found at %s", args.db_path)
        return 1
    tickers, latest = load_universe(args.db_path)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers]
    start, end = six_complete_months(latest)
    log.info("Latest cleaned observation %s -> common window %s to %s; %d tickers; output %s",
             latest, start, end, len(tickers), run_dir)

    successes: list[Path] = []
    failures: dict[str, str] = {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successes.append(run_report(ticker, start, end, run_dir, args.db_path))
        except Exception as exc:  # keep going: one bad ticker must not stop the batch
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            failures[ticker] = message
            log.error("FAILED %s: %s", ticker, message)

    (run_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failures))
    print(f"\nSuccessfully generated {len(successes)}/{len(tickers)} reports")
    if failures:
        print(f"Failed tickers saved to {run_dir / 'failed_stocks.txt'}: {', '.join(failures)}")
    if not successes:
        print("No report succeeded, so there is no portfolio summary. Check run.log.")
        return 1

    summary = collect_summary(successes)
    summary[SUMMARY_COLS + ["start_date", "end_date"] + COVERAGE_COLS].to_csv(run_dir / "portfolio_summary.csv", index=False)

    # A gap of more than a week before the window end means the supplied history stops early
    # (a small gap is just weekends/holidays, e.g. a last trading day of Nov 29 for a Nov 30 end).
    cutoff = (pd.Timestamp(end) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    short = summary[summary["last_obs"] < cutoff]
    if len(short):
        print(f"\nCoverage note: {len(short)} tickers' supplied history ends before {end} "
              f"(last observation {', '.join(sorted(short['last_obs'].unique()))}). Their metrics cover fewer days "
              "and are not extrapolated; compare them with the full-window tickers with care.")

    ranked = summary[summary["sharpe_ratio"].notna()]
    excluded = summary.loc[summary["sharpe_ratio"].isna(), "ticker"].tolist()
    if excluded:
        print(f"Excluded from Sharpe rankings (undefined Sharpe): {', '.join(excluded)}")
    print(f"\nTop 10 by Sharpe ratio ({start} to {end}):\n{fmt_table(ranked.head(10))}")
    print(f"\nBottom 10 by Sharpe ratio:\n{fmt_table(ranked.tail(10).iloc[::-1])}")

    plot_risk_return(summary, start, end, run_dir / "risk_return_profile.png")
    print(f"\nSaved {run_dir / 'portfolio_summary.csv'} and {run_dir / 'risk_return_profile.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
