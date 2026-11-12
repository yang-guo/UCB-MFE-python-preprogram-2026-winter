"""Generate a risk report for every stock in ohlc_hw_cleaned, then summarize the portfolio.

For each ticker, this script executes 02_stock_report_template.ipynb with papermill over one common window
(the 6 calendar months before the month of the latest observation), collects the metrics each successful
report exports via scrapbook, and writes:

    reports/{YYYY-MM-DD}/{ticker}_report.ipynb   one executed report per ticker
    reports/{YYYY-MM-DD}/portfolio_summary.csv   one row of metrics per successful report
    reports/{YYYY-MM-DD}/risk_return.png         risk-return scatter plot of the portfolio
    reports/{YYYY-MM-DD}/failed_stocks.txt       tickers whose report failed (empty if none)
    reports/{YYYY-MM-DD}/run.log                 full log, including error details

{YYYY-MM-DD} is the date the script is run.

Usage (from this folder, with the homework environment active):
    python 03_run_all_stocks.py
    python 03_run_all_stocks.py --tickers AAPL MSFT    # run a subset, e.g. while testing
"""

import argparse
import logging
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render plots to files; no display needed

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "data.db"
DEFAULT_TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_OUTPUT_ROOT = HERE / "reports"
CLEAN_TABLE = "ohlc_hw_cleaned"

SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                   "total_return", "report_path"]
EXTRA_COLUMNS = ["annualized_return", "annualized_volatility", "first_day", "last_day", "trading_days"]

log = logging.getLogger("run_all_stocks")


def read_sql(db_path: Path, query: str) -> pd.DataFrame:
    """Run a read-only query against the database."""
    with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return pd.read_sql_query(query, conn)


def load_tickers(db_path: Path) -> list[str]:
    """All distinct tickers in the cleaned table, sorted."""
    return read_sql(db_path, f"SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker")["ticker"].tolist()


def last_complete_months(latest: pd.Timestamp, n_months: int = 6) -> tuple[str, str]:
    """Start and end date of the `n_months` calendar months before the month containing `latest`.

    The month containing the latest observation is always excluded, even if `latest` is its last day:
    a final trading date alone does not prove that month's delivery is complete.
    Example: latest = 2024-12-06 (or 2024-11-30) gives ("2024-06-01", "2024-11-30") (or ("2024-05-01", "2024-10-31")).
    """
    latest = pd.Timestamp(latest).normalize()
    end = latest.replace(day=1) - pd.Timedelta(days=1)
    start = (end - pd.DateOffset(months=n_months - 1)).replace(day=1)
    return f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"


def run_report(ticker: str, start_date: str, end_date: str, template: Path, db_path: Path,
               output_dir: Path) -> Path:
    """Execute the report template for one ticker and return the output notebook path.

    If the notebook raises an error, papermill still saves it, with the traceback in the failing cell.
    It is renamed to *_FAILED.ipynb (so it can be inspected) and the exception is re-raised for the caller.
    """
    output = output_dir / f"{ticker}_report.ipynb"
    try:
        # Relative paths keep the local folder layout out of the saved reports: papermill records the
        # input/output paths in the notebook metadata, and db_path appears in the injected parameters cell.
        pm.execute_notebook(
            os.path.relpath(template), os.path.relpath(output),
            parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                        "db_path": os.path.relpath(db_path.resolve(), template.parent.resolve())},
            cwd=str(template.parent),
            kernel_name="python3",  # the Python running this script, i.e. the homework environment
            progress_bar=False,
        )
    except Exception:
        if output.exists():
            output.replace(output.with_name(f"{ticker}_report_FAILED.ipynb"))
        raise
    return output


def collect_results(report_paths: list[Path]) -> pd.DataFrame:
    """Read the `results` scrap from each executed report into one row per ticker."""
    rows = []
    for path in report_paths:
        results = sb.read_notebook(str(path)).scraps["results"].data
        shown_path = path.relative_to(HERE) if path.is_relative_to(HERE) else path
        rows.append({**results, "report_path": str(shown_path)})
    return pd.DataFrame(rows)


def plot_risk_return(summary: pd.DataFrame, start_date: str, end_date: str, output: Path) -> None:
    """Scatter of average volatility vs total return, coloured by Sharpe ratio, top 5 by return labelled.

    Stocks without a defined Sharpe ratio are drawn in gray.
    """
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    undefined = summary[summary["sharpe_ratio"].isna()]
    if not undefined.empty:
        ax.scatter(undefined["avg_volatility"], undefined["total_return"], color="#9e9e9e", s=110,
                   edgecolor="black", linewidth=0.6, zorder=3, label="Sharpe ratio undefined")
        ax.legend(loc="lower right")
    summary = summary[summary["sharpe_ratio"].notna()]
    points = ax.scatter(summary["avg_volatility"], summary["total_return"], c=summary["sharpe_ratio"],
                        cmap="RdYlGn", s=110, edgecolor="black", linewidth=0.6, zorder=3)
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Sharpe ratio")

    # Label the top 5 by return. Try a few positions around each point and keep the first that
    # doesn't overlap a label already placed.
    renderer = fig.canvas.get_renderer()
    candidates = [(8, 6, "left"), (8, -14, "left"), (-8, 6, "right"), (-8, -14, "right")]
    placed = []
    for _, row in summary.nlargest(5, "total_return").iterrows():
        for i, (dx, dy, ha) in enumerate(candidates):
            label = ax.annotate(f"{row['ticker']} ({row['total_return']:+.1%})",
                                (row["avg_volatility"], row["total_return"]), xytext=(dx, dy),
                                textcoords="offset points", ha=ha, fontsize=10, fontweight="bold")
            box = label.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed) or i == len(candidates) - 1:
                placed.append(box)
                break
            label.remove()

    ax.axhline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set(xlabel="Average 20-day volatility (annualized)", ylabel="Total return")
    ax.set_title(f"Risk-Return Profile of Portfolio\n{start_date} to {end_date}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.35)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def format_table(df: pd.DataFrame) -> str:
    """Summary rows as an aligned text table with percentages."""
    pct = {col: "{:+.2%}".format for col in ["avg_daily_return", "max_drawdown", "total_return"]}
    return df.to_string(index=False, formatters={**pct, "avg_volatility": "{:.2%}".format,
                                                 "sharpe_ratio": "{:.2f}".format})


def setup_logging(log_file: Path) -> None:
    """Log INFO to the console and everything (including tracebacks) to log_file."""
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(log_file)
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(console)
    log.addHandler(file)


def parse_args() -> argparse.Namespace:
    """Command-line options; the defaults run every ticker with the files in this folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database with ohlc_hw_cleaned")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="report notebook to execute")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="parent folder for reports")
    parser.add_argument("--months", type=int, default=6, help="number of complete months to analyze")
    parser.add_argument("--tickers", nargs="+", help="only run these tickers (default: all)")
    return parser.parse_args()


def main() -> None:
    """Generate all reports, record failures, then build and save the portfolio summary."""
    args = parse_args()
    if not args.db.is_file():
        raise SystemExit(f"Database not found: {args.db.resolve()} (run 01_data_cleaning.ipynb first)")

    output_dir = args.output_root / f"{date.today():%Y-%m-%d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "run.log")

    # 1. Tickers and date range, anchored to the latest date in the data rather than today's date
    tickers = args.tickers or load_tickers(args.db)
    latest = pd.Timestamp(read_sql(args.db, f"SELECT MAX(ts) AS latest FROM {CLEAN_TABLE}")["latest"].iloc[0])
    start_date, end_date = last_complete_months(latest, args.months)
    log.info(f"Latest observation {latest:%Y-%m-%d} -> analysis window {start_date} to {end_date}")
    log.info(f"Writing {len(tickers)} reports to {output_dir}")

    # 2. One report per ticker; a failure is logged and the loop moves on
    successes, failures = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successes.append(run_report(ticker, start_date, end_date, args.template, args.db, output_dir))
        except Exception as exc:
            failures.append(ticker)
            log.error(f"{ticker}: report failed ({type(exc).__name__}). See run.log for details.")
            log.debug(f"{ticker} traceback", exc_info=True)

    log.info(f"Successfully generated {len(successes)}/{len(tickers)} reports")
    (output_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failures))
    if failures:
        log.info(f"Failed tickers saved to {output_dir / 'failed_stocks.txt'}: {', '.join(failures)}")
    if not successes:
        raise SystemExit("No reports succeeded, so there is nothing to summarize.")

    # 3. Portfolio summary from the metrics each report exported
    results = collect_results(successes)
    summary = results[SUMMARY_COLUMNS + EXTRA_COLUMNS].sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)

    # Rankings use only stocks with a defined Sharpe ratio; with fewer than 20, top and bottom overlap
    ranked = summary[summary["sharpe_ratio"].notna()]
    unranked = summary.loc[summary["sharpe_ratio"].isna(), "ticker"].tolist()
    shown = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    log.info("\nTop 10 by Sharpe ratio:\n" + format_table(ranked[shown].head(10)))
    log.info("\nBottom 10 by Sharpe ratio:\n" + format_table(ranked[shown].tail(10)))
    if unranked:
        log.info(f"Not ranked (undefined Sharpe ratio): {', '.join(unranked)}")

    cutoff = pd.Timestamp(end_date) - pd.offsets.BDay(5)  # allow for holidays at the end of the window
    short = summary[pd.to_datetime(summary["last_day"]) < cutoff]
    if not short.empty:
        log.info(f"\nNote: {len(short)} ticker(s) have no data after {short['last_day'].max()}, "
                 f"so their reports cover a shorter period than {start_date} to {end_date}.")

    plot_risk_return(summary, start_date, end_date, output_dir / "risk_return.png")
    log.info(f"\nSaved {output_dir / 'portfolio_summary.csv'} and {output_dir / 'risk_return.png'}")
    log.info("Each report's actual coverage is in the first_day, last_day and trading_days columns.")


if __name__ == "__main__":
    main()
