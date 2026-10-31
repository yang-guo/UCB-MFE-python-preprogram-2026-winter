"""Generate a risk report for every ticker in ohlc_hw_cleaned and build a portfolio summary.

Usage (from any directory):
    python 03_run_all_stocks.py
    python 03_run_all_stocks.py --tickers AAPL IBM NOT_A_TICKER
    python 03_run_all_stocks.py --db-path path/to/data.db
"""
import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from papermill.exceptions import PapermillExecutionError
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_DB = HERE / "data" / "data.db"
SUMMARY_COLUMNS = [
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
    "sharpe_ratio", "total_return", "report_path",
    "actual_start", "actual_end", "n_prices",
]


def connect(db_path):
    return closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True))


def load_tickers(db_path):
    """All tickers in the cleaned table, sorted."""
    with connect(db_path) as conn:
        return [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker")]


def latest_observation(db_path):
    """Latest date in the cleaned table."""
    with connect(db_path) as conn:
        (ts,) = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()
    if ts is None:
        raise RuntimeError("ohlc_hw_cleaned is empty")
    return date.fromisoformat(ts)


def last_six_complete_months(latest):
    """The six calendar months before the month containing `latest`, as (start, end)."""
    month_start = latest.replace(day=1)
    end = month_start - timedelta(days=1)
    year, month = month_start.year, month_start.month - 6
    if month < 1:
        year, month = year - 1, month + 12
    return date(year, month, 1), end


def run_report(ticker, start, end, db_path, out_dir):
    """Execute the template for one ticker and return the output notebook path."""
    out_path = out_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE),
        str(out_path),
        parameters={
            "ticker": ticker,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "db_path": str(db_path),
        },
        cwd=str(HERE),
        progress_bar=False,
    )
    return out_path


def describe_error(exc):
    if isinstance(exc, PapermillExecutionError):
        return f"{exc.ename}: {exc.evalue}"
    return f"{type(exc).__name__}: {exc}"


def collect_metrics(report_paths):
    """Read the glued metrics from this run's successful reports only."""
    rows = []
    for path in report_paths:
        metrics = sb.read_notebook(str(path)).scraps["metrics"].data
        rows.append({**metrics, "report_path": path.relative_to(HERE).as_posix()})
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def plot_risk_return(summary, end, out_path):
    """Scatter of mean rolling volatility vs total return, colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 8))
    defined = summary["sharpe_ratio"].notna()
    short = pd.to_datetime(summary["actual_end"]) < pd.Timestamp(end) - pd.Timedelta(days=5)
    vmin, vmax = summary.loc[defined, "sharpe_ratio"].min(), summary.loc[defined, "sharpe_ratio"].max()

    sc = None
    for mask, marker, label in [(defined & ~short, "o", "full window"), (defined & short, "s", "data ends early")]:
        if mask.any():
            sc = ax.scatter(summary.loc[mask, "avg_volatility"], summary.loc[mask, "total_return"],
                            c=summary.loc[mask, "sharpe_ratio"], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                            marker=marker, s=90, edgecolor="black", label=label)
    if (~defined).any():
        ax.scatter(summary.loc[~defined, "avg_volatility"], summary.loc[~defined, "total_return"],
                   color="lightgray", edgecolor="black", s=90, label="Sharpe undefined")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="Sharpe ratio")

    for _, row in summary.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                    xytext=(7, 5), textcoords="offset points", fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Average 20-day rolling volatility (annualized)")
    ax.set_ylabel("Total return")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_table(title, df):
    show = df[["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "avg_daily_return"]].copy()
    show["sharpe_ratio"] = show["sharpe_ratio"].map("{:.2f}".format)
    for col in ["total_return", "avg_volatility", "max_drawdown"]:
        show[col] = show[col].map("{:.2%}".format)
    show["avg_daily_return"] = show["avg_daily_return"].map("{:.4%}".format)
    print(f"\n{title}\n{show.to_string(index=False)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--tickers", nargs="+", help="run only these tickers (default: all)")
    args = parser.parse_args(argv)

    db_path = args.db_path.resolve()
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}. Run 01_data_cleaning.ipynb first.")

    tickers = args.tickers or load_tickers(db_path)
    latest = latest_observation(db_path)
    start, end = last_six_complete_months(latest)
    out_dir = HERE / "reports" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Latest cleaned observation: {latest}")
    print(f"Report window: {start} to {end} ({len(tickers)} tickers)")
    print(f"Output: {out_dir}")

    succeeded, failed = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            succeeded.append(run_report(ticker, start, end, db_path, out_dir))
        except Exception as exc:
            failed.append((ticker, describe_error(exc)))
            tqdm.write(f"  {ticker} failed -> {failed[-1][1]}")

    (out_dir / "failed_stocks.txt").write_text("".join(f"{t}\t{err}\n" for t, err in failed), encoding="utf-8")
    print(f"\nSuccessfully generated {len(succeeded)}/{len(tickers)} reports")
    if failed:
        print(f"Failed tickers saved to {out_dir / 'failed_stocks.txt'}")
    if not succeeded:
        print("No report succeeded, so there is no portfolio summary to build.")
        return 1

    summary = collect_metrics(succeeded).sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary.to_csv(out_dir / "portfolio_summary.csv", index=False)

    short = summary[pd.to_datetime(summary["actual_end"]) < pd.Timestamp(end) - pd.Timedelta(days=5)]
    if len(short):
        print(f"\n{len(short)} tickers have data ending before {end}; their metrics cover a shorter period:")
        print(", ".join(f"{t} (to {d})" for t, d in zip(short["ticker"], short["actual_end"])))

    ranked = summary.dropna(subset=["sharpe_ratio"])
    excluded = len(summary) - len(ranked)
    if excluded:
        print(f"\n{excluded} tickers excluded from the ranking (undefined Sharpe ratio)")
    print_table("Top 10 by Sharpe ratio", ranked.head(10))
    print_table("Bottom 10 by Sharpe ratio", ranked.tail(10).iloc[::-1])

    plot_risk_return(summary, end, out_dir / "portfolio_summary.png")
    print(f"\nSaved {out_dir / 'portfolio_summary.csv'} and portfolio_summary.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
