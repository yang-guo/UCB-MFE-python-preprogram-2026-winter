"""Generate a risk report for every stock in ohlc_hw_cleaned (HW2 Part 3).

For each ticker, this script runs 02_stock_report_template.ipynb with
papermill over the last 6 complete months of data, then builds a portfolio
summary table and a risk-return chart.

Usage (from this folder, after running 01_data_cleaning.ipynb):
    python 03_run_all_stocks.py

Output:
    reports/{YYYY-MM-DD}/{ticker}_report.ipynb   one report per ticker
    reports/{YYYY-MM-DD}/portfolio_summary.csv   metrics for every ticker
    reports/{YYYY-MM-DD}/risk_return.png         risk-return scatter plot
    reports/{YYYY-MM-DD}/run.log                 log of the run
    failed_stocks.txt                            tickers that failed (if any)
"""

import json
import logging
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import papermill as pm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

# Paths are relative to this script, so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "data.db"
TEMPLATE = BASE_DIR / "02_stock_report_template.ipynb"
REPORTS_DIR = BASE_DIR / "reports"
FAILED_FILE = BASE_DIR / "failed_stocks.txt"

N_MONTHS = 6

log = logging.getLogger("run_all_stocks")


def load_tickers(db_path: Path) -> list[str]:
    """Return every ticker in ohlc_hw_cleaned, sorted."""
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker"
        ).fetchall()
    return [row[0] for row in rows]


def latest_date(db_path: Path) -> pd.Timestamp:
    """Return the latest observation date in ohlc_hw_cleaned."""
    with closing(sqlite3.connect(db_path)) as conn:
        (value,) = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()
    return pd.Timestamp(value)


def last_complete_months(latest: pd.Timestamp, n_months: int = N_MONTHS) -> tuple[str, str]:
    """Return (start, end) for the last `n_months` complete calendar months.

    Anchored to the latest date in the data, not the computer clock. The month
    containing `latest` is never used, even when `latest` is its last day:
    a final trading date alone doesn't prove that month's data is complete.
    Examples: latest = 2024-12-06 -> ("2024-06-01", "2024-11-30")
              latest = 2024-11-30 -> ("2024-05-01", "2024-10-31")
    """
    end = latest.normalize().replace(day=1) - pd.Timedelta(days=1)  # end of previous month
    start = (end - pd.DateOffset(months=n_months - 1)).replace(day=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def setup_logging(output_dir: Path) -> None:
    """Log to run.log in the output folder; only warnings go to the console."""
    log.setLevel(logging.INFO)
    log.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)  # keep the console free for the progress bar
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(console)


def make_report_paths_relative(report_path: Path) -> None:
    """Rewrite papermill's recorded input/output paths relative to this folder.

    papermill stores the full paths it was given in the report's metadata;
    this keeps paths from this computer out of the saved reports.
    """
    nb = nbformat.read(report_path, as_version=4)
    run_info = nb.metadata.get("papermill", {})
    for key in ("input_path", "output_path"):
        if key in run_info:
            run_info[key] = Path(run_info[key]).resolve().relative_to(BASE_DIR).as_posix()
    nbformat.write(nb, report_path)


def run_report(ticker: str, start_date: str, end_date: str, output_dir: Path) -> dict:
    """Execute the template for one ticker and return its metrics."""
    report_path = output_dir / f"{ticker}_report.ipynb"
    metrics_path = output_dir / f"{ticker}_metrics.json"

    pm.execute_notebook(
        input_path=str(TEMPLATE),
        output_path=str(report_path),
        parameters={
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            # Relative to BASE_DIR (the notebook's working directory, see cwd),
            # so the saved reports don't contain paths from this computer
            "db_path": DB_PATH.relative_to(BASE_DIR).as_posix(),
            "metrics_path": metrics_path.relative_to(BASE_DIR).as_posix(),
        },
        cwd=str(BASE_DIR),
        progress_bar=False,
    )
    make_report_paths_relative(report_path)

    metrics = json.loads(metrics_path.read_text())
    metrics_path.unlink()  # the summary CSV holds these numbers from here on
    metrics["report_path"] = str(report_path.relative_to(BASE_DIR))
    return metrics


def build_summary(results: list[dict]) -> pd.DataFrame:
    """Build the portfolio summary table, sorted by Sharpe ratio (best first)."""
    columns = [
        "ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
        "sharpe_ratio", "total_return", "first_date", "last_date", "trading_days",
        "report_path",
    ]
    summary = pd.DataFrame(results, columns=columns)
    return summary.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def plot_risk_return(summary: pd.DataFrame, start_date: str, end_date: str, path: Path) -> None:
    """Scatter avg volatility vs total return, colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 8))

    points = ax.scatter(
        summary["avg_volatility"], summary["total_return"],
        c=summary["sharpe_ratio"], cmap="RdYlGn", s=110,
        # Center the colors on 0: negative Sharpe = red, positive = green
        norm=TwoSlopeNorm(vcenter=0, vmin=min(summary["sharpe_ratio"].min(), -0.5),
                          vmax=max(summary["sharpe_ratio"].max(), 0.5)),
        edgecolor="white", linewidth=1.5, zorder=3,
    )
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Sharpe ratio")

    # Label the 5 best performers by total return
    for _, row in summary.nlargest(5, "total_return").iterrows():
        ax.annotate(
            row["ticker"], (row["avg_volatility"], row["total_return"]),
            xytext=(8, 6), textcoords="offset points", fontsize=11, fontweight="bold",
        )

    ax.axhline(0, color="#52514e", linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Average 20-day volatility (annualized)")
    ax.set_ylabel("Total return")
    ax.set_title(f"Risk-Return Profile of Portfolio\n{start_date} to {end_date}",
                 fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_table(df: pd.DataFrame) -> str:
    """Format the summary for printing: percentages and a rounded Sharpe."""
    pct = lambda v: f"{v:+.2%}"
    return df.to_string(
        index=False,
        columns=["ticker", "total_return", "avg_daily_return", "avg_volatility",
                 "max_drawdown", "sharpe_ratio"],
        formatters={
            "total_return": pct,
            "avg_daily_return": lambda v: f"{v:+.3%}",
            "avg_volatility": lambda v: f"{v:.2%}",
            "max_drawdown": pct,
            "sharpe_ratio": lambda v: f"{v:.2f}",
        },
    )


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found at {DB_PATH}. Copy data.db into data/ first.")

    tickers = load_tickers(DB_PATH)
    start_date, end_date = last_complete_months(latest_date(DB_PATH))

    output_dir = REPORTS_DIR / date.today().strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    print(f"Analysis window: {start_date} to {end_date} (last {N_MONTHS} complete months)")
    print(f"Writing reports to {output_dir.relative_to(BASE_DIR)}/")
    log.info("Window %s to %s, %d tickers", start_date, end_date, len(tickers))

    results, failed = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            results.append(run_report(ticker, start_date, end_date, output_dir))
            log.info("%s: done", ticker)
        except Exception as exc:  # one bad stock must not stop the batch
            failed.append(ticker)
            # papermill errors carry the notebook's exception in .ename / .evalue
            reason = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', exc)}"
            log.error("%s failed: %s", ticker, reason)

    print(f"\nSuccessfully generated {len(results)}/{len(tickers)} reports")

    FAILED_FILE.write_text("".join(f"{t}\n" for t in failed))
    if failed:
        print(f"Failed tickers saved to {FAILED_FILE.name}: {', '.join(failed)}")

    if not results:
        return

    summary = build_summary(results)
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)

    # Rankings use only defined Sharpe ratios (the CSV keeps every ticker).
    # With fewer than 20 ranked stocks, the top and bottom 10 overlap.
    ranked = summary.dropna(subset=["sharpe_ratio"])
    unranked = summary.loc[summary["sharpe_ratio"].isna(), "ticker"]
    if not unranked.empty:
        print(f"\nExcluded from rankings (Sharpe undefined): {', '.join(unranked)}")

    print("\nTop 10 by Sharpe ratio:")
    print(format_table(ranked.head(10)))
    print("\nBottom 10 by Sharpe ratio:")
    print(format_table(ranked.tail(10)))

    plot_risk_return(summary, start_date, end_date, output_dir / "risk_return.png")
    print(f"\nSaved portfolio_summary.csv and risk_return.png to {output_dir.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
