"""Generate a risk report for every ticker with papermill, then summarize the results.

Run from this folder:
    uv run python 03_run_all_stocks.py                      # all tickers
    uv run python 03_run_all_stocks.py --tickers AAPL MSFT  # quick test on a few
    uv run python 03_run_all_stocks.py --test-failure       # adds a fake ticker to show error handling
"""
import argparse
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw straight to files; no window needed
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "02_stock_report_template.ipynb"
DB_PATH = BASE / "data" / "data.db"
METRICS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
COVERAGE = ["first_date", "last_date", "n_days"]


def get_tickers_and_window(db_path):
    """Return (tickers, start_date, end_date) from ohlc_hw_cleaned.

    The window is the six calendar months BEFORE the month of the latest observation
    (latest = 2024-12-06 -> 2024-06-01 to 2024-11-30). It is anchored to the data, not the clock.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
        latest = pd.read_sql_query("SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn)["latest"].iloc[0]
    month_start = pd.Timestamp(latest).replace(day=1)
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(ticker, start, end, out_dir):
    """Execute the template notebook for one ticker; return (output_path, recorded values)."""
    out_path = out_dir / f"{ticker}_report.ipynb"
    out_path.unlink(missing_ok=True)        # never leave a stale report from an earlier run
    pm.execute_notebook(str(TEMPLATE), str(out_path),
                        parameters={"ticker": ticker, "start_date": start,
                                    "end_date": end, "db_path": str(DB_PATH)},
                        cwd=str(BASE), progress_bar=False)
    scraps = sb.read_notebook(str(out_path)).scraps
    return out_path, {k: scraps[k].data for k in METRICS + COVERAGE}


def make_scatter(summary_df, start, end, out_dir):
    """Risk-return scatter: x = avg volatility, y = total return, color = Sharpe."""
    plot_df = summary_df.dropna(subset=["avg_volatility", "total_return", "sharpe_ratio"])
    if plot_df.empty:
        print("No stocks with a defined Sharpe ratio; scatter plot skipped.")
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(plot_df["avg_volatility"], plot_df["total_return"], c=plot_df["sharpe_ratio"],
                    cmap="RdYlGn", s=90, edgecolor="black")
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    for _, row in plot_df.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=10, fontweight="bold")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=15, fontweight="bold")
    ax.set_xlabel("Average rolling volatility (annualized)")
    ax.set_ylabel("Total return over the window")
    ax.grid(True, alpha=0.3)
    ax.text(0.01, 0.01, f"Requested window {start} to {end}; observed coverage differs by ticker (see coverage.csv)",
            transform=ax.transAxes, fontsize=9)
    fig.savefig(out_dir / "risk_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tickers", nargs="+", help="only run these tickers")
    parser.add_argument("--test-failure", action="store_true", help="add a fake ticker to show error handling")
    args = parser.parse_args()

    tickers, start, end = get_tickers_and_window(DB_PATH)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    if args.test_failure:
        tickers.append("NOT_A_TICKER")

    out_dir = BASE / "reports" / date.today().isoformat()      # run date, not analysis date
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Window: {start} to {end} | {len(tickers)} tickers | output: {out_dir}")

    rows, failed = [], {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            path, values = run_report(ticker, start, end, out_dir)
            rows.append({"ticker": ticker, **values, "report_path": str(path.relative_to(BASE))})
        except Exception as exc:                                  # log it and keep going
            last_line = (str(exc).strip().splitlines() or [""])[-1]
            failed[ticker] = f"{type(exc).__name__}: {last_line[:200]}"

    print(f"Successfully generated {len(rows)}/{len(tickers)} reports")
    (BASE / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failed))
    if failed:
        (out_dir / "errors.log").write_text("".join(f"{t}\t{msg}\n" for t, msg in failed.items()))
        print("Failed:", ", ".join(failed), "(details in errors.log)")
    if not rows:
        print("No reports succeeded; no summary was created.")
        return 1

    summary_df = pd.DataFrame(rows)
    summary_df[METRICS] = summary_df[METRICS].apply(pd.to_numeric, errors="coerce")
    coverage = summary_df[["ticker"] + COVERAGE].copy()
    summary_df = (summary_df[["ticker"] + METRICS + ["report_path"]]
                  .sort_values("sharpe_ratio", ascending=False, na_position="last")
                  .reset_index(drop=True))
    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)
    coverage.to_csv(out_dir / "coverage.csv", index=False)

    ranked = summary_df.dropna(subset=["sharpe_ratio"])         # undefined Sharpe is excluded from rankings
    cols = ["ticker"] + METRICS
    print("\nTop 10 by Sharpe ratio:");    print(ranked[cols].head(10).round(4).to_string(index=False))
    print("\nBottom 10 by Sharpe ratio:"); print(ranked[cols].tail(10).round(4).to_string(index=False))
    print("\nObserved coverage (requested window differs by ticker):")
    print(coverage.to_string(index=False))

    make_scatter(summary_df, start, end, out_dir)
    print(f"\nSaved: {out_dir / 'portfolio_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())