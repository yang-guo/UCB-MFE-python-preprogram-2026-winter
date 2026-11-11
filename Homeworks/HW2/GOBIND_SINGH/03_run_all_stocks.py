"""HW2 Part 3: generate a risk report for every stock with papermill, then summarize the portfolio.

Run from the repository root with the course environment, after 01_data_cleaning.ipynb has
created ohlc_hw_cleaned:

    uv run python Homeworks/HW2/GOBIND_SINGH/03_run_all_stocks.py

Outputs go to reports/<run date>/ next to this script: one <TICKER>_report.ipynb per stock,
failed_stocks.txt, run.log, portfolio_summary.csv and portfolio_risk_return.png.
"""

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import papermill as pm  # noqa: E402
import scrapbook as sb  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_DB = HERE / "data" / "data.db"
CLEAN_TABLE = "ohlc_hw_cleaned"
WINDOW_MONTHS = 6
SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                   "sharpe_ratio", "total_return", "report_path"]
COVERAGE_COLUMNS = ["first_date", "last_date", "trading_days"]

log = logging.getLogger("run_all_stocks")


def load_tickers(db_path):
    """All distinct tickers in the cleaned table, sorted."""
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(f"SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker").fetchall()
    return [row[0] for row in rows]


def latest_observation(db_path):
    """Latest date in the cleaned table. The analysis window is anchored here, not to the computer clock."""
    with closing(sqlite3.connect(db_path)) as conn:
        (latest,) = conn.execute(f"SELECT MAX(ts) FROM {CLEAN_TABLE}").fetchone()
    if latest is None:
        raise ValueError(f"{CLEAN_TABLE} is empty; run 01_data_cleaning.ipynb first")
    return pd.Timestamp(latest)


def last_complete_months(latest, months=WINDOW_MONTHS):
    """The `months` calendar months before the month containing `latest`, as ISO (start, end).

    The latest month is always excluded, even when `latest` falls on its last day, because a final
    trading date alone doesn't prove that month's delivery is complete. Example: 2024-12-06 gives
    2024-06-01 to 2024-11-30.
    """
    month_start = latest.to_period("M").to_timestamp()
    start = month_start - pd.DateOffset(months=months)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_report(ticker, start_date, end_date, db_path, output_dir):
    """Execute the template for one ticker and return the output path.

    On failure the partial notebook is renamed to <TICKER>_report_FAILED.ipynb, so it can't be mistaken
    for a finished report, and the error propagates to the caller.
    """
    output = output_dir / f"{ticker}_report.ipynb"
    try:
        pm.execute_notebook(
            str(TEMPLATE), str(output),
            parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date, "db_path": str(db_path)},
            kernel_name="python3", cwd=str(HERE), progress_bar=False,
        )
    except Exception:
        if output.exists():
            output.replace(output_dir / f"{ticker}_report_FAILED.ipynb")
        raise
    return output


def display_path(path):
    """Path relative to this script's folder when possible, so the CSV stays portable."""
    try:
        return path.relative_to(HERE).as_posix()
    except ValueError:
        return str(path)


def collect_summaries(report_paths):
    """Read the `report_summary` scrap from each successful report of this run into one DataFrame."""
    rows = []
    for path in report_paths:
        row = sb.read_notebook(str(path)).scraps["report_summary"].data
        row["report_path"] = display_path(path)
        rows.append(row)
    summary = pd.DataFrame(rows)
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    summary[numeric] = summary[numeric].apply(pd.to_numeric, errors="coerce")
    return summary[SUMMARY_COLUMNS + COVERAGE_COLUMNS]


def plot_risk_return(summary, start_date, end_date, path):
    """Scatter of mean rolling volatility vs total return, colored by Sharpe, with the top 5 by return labeled."""
    data = summary.dropna(subset=["avg_volatility", "total_return"])
    sharpe = data["sharpe_ratio"]
    norm = None
    if sharpe.notna().any() and sharpe.min() < 0 < sharpe.max():
        norm = matplotlib.colors.TwoSlopeNorm(vmin=sharpe.min(), vcenter=0, vmax=sharpe.max())
    fig, ax = plt.subplots(figsize=(11, 7))
    points = ax.scatter(data["avg_volatility"], data["total_return"], c=sharpe, cmap="RdYlGn", norm=norm,
                        s=90, edgecolors="black", linewidths=0.5)
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in data.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 5),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Risk-Return Profile of Portfolio",
           xlabel="Average 20-day rolling volatility (annualized)", ylabel="Total return")
    ax.text(0.99, 0.01, f"Requested window {start_date} to {end_date}; top 5 by total return labeled",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="dimgray")
    ax.margins(x=0.08, y=0.12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_rankings(summary):
    """Print the top and bottom 10 by Sharpe ratio; tickers with an undefined Sharpe are excluded."""
    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False)
    excluded = sorted(set(summary["ticker"]) - set(ranked["ticker"]))
    view = ranked.drop(columns="report_path")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print("\nTop 10 by Sharpe ratio:")
        print(view.head(10).to_string(index=False))
        print("\nBottom 10 by Sharpe ratio:")
        print(view.tail(10).iloc[::-1].to_string(index=False))
    if len(ranked) < 20:
        print(f"\nNote: only {len(ranked)} ranked tickers, so the top and bottom lists overlap.")
    if excluded:
        print(f"Excluded from rankings (undefined Sharpe): {', '.join(excluded)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database with ohlc_hw_cleaned")
    parser.add_argument("--tickers", nargs="+", help="run only these tickers (default: all in the table)")
    parser.add_argument("--output-root", type=Path, default=HERE / "reports", help="parent of the run-date folder")
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = args.db.resolve()
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    tickers = [t.strip().upper() for t in args.tickers] if args.tickers else load_tickers(db_path)
    latest = latest_observation(db_path)
    start_date, end_date = last_complete_months(latest)

    output_dir = args.output_root.resolve() / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output_dir / "run.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    for stale in [*output_dir.glob("*_report*.ipynb"), *output_dir.glob("portfolio_*")]:
        stale.unlink()

    print(f"Latest observation {latest:%Y-%m-%d}; analysis window {start_date} to {end_date}")
    print(f"Writing {len(tickers)} reports to {output_dir}")
    log.info("window %s to %s, %d tickers, db %s", start_date, end_date, len(tickers), db_path)

    succeeded, failed = [], {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            succeeded.append(run_report(ticker, start_date, end_date, db_path, output_dir))
            log.info("%s: ok", ticker)
        except Exception as exc:  # one bad ticker must not stop the batch
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            failed[ticker] = message
            log.error("%s: %s", ticker, exc)

    print(f"\nSuccessfully generated {len(succeeded)}/{len(tickers)} reports")
    failed_path = output_dir / "failed_stocks.txt"
    failed_path.write_text("".join(f"{t}\t{msg}\n" for t, msg in failed.items()))
    if failed:
        print(f"Failed tickers ({len(failed)}), saved to {failed_path.name}:")
        for ticker, message in failed.items():
            print(f"  {ticker}: {message}")
    if not succeeded:
        print("No successful reports, so there is no portfolio summary. See run.log for details.")
        return 1

    summary = collect_summaries(succeeded).sort_values("sharpe_ratio", ascending=False, na_position="last")
    first, last = summary["first_date"].min(), summary["last_date"].max()
    short = summary[(summary["first_date"] > first) | (summary["last_date"] < last)]
    print(f"\nCoverage: the fullest histories run {first} to {last} (window edges that fall on weekends have no data).")
    if len(short):
        print(f"{len(short)} of {len(summary)} tickers have shorter histories in this window: "
              + ", ".join(f"{t} ({a} to {b})" for t, a, b in
                          short[["ticker", "first_date", "last_date"]].sort_values("ticker").itertuples(index=False)))
        print("Their metrics cover fewer trading days, so compare them with the others with care.")

    print_rankings(summary)
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    plot_risk_return(summary, start_date, end_date, output_dir / "portfolio_risk_return.png")
    print(f"\nSaved portfolio_summary.csv and portfolio_risk_return.png to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
