"""Generate a risk report for every cleaned ticker with papermill and summarize the portfolio.

Run from anywhere after 01_data_cleaning.ipynb has created ohlc_hw_cleaned:
    python 03_run_all_stocks.py
"""
import logging
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "data.db"
TEMPLATE = HERE / "02_stock_report_template.ipynb"
METRICS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]

logging.basicConfig(format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def six_complete_months(latest: str) -> tuple[str, str]:
    """The six calendar months before the month containing `latest`."""
    month_start = pd.Timestamp(latest).to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"


def load_tickers_and_window() -> tuple[list[str], str, str]:
    """All tickers in ohlc_hw_cleaned and the common analysis window."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)
        latest = pd.read_sql_query("SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn)["latest"].iloc[0]
    return tickers["ticker"].tolist(), *six_complete_months(latest)


def run_reports(tickers: list[str], start: str, end: str, out_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Execute the template for each ticker; return successful report paths and failed tickers."""
    reports, failed = {}, []
    for ticker in tqdm(tickers, desc="Generating reports"):
        output = out_dir / f"{ticker}_report.ipynb"
        try:
            pm.execute_notebook(
                TEMPLATE, output, cwd=HERE, progress_bar=False,
                parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": str(DB_PATH)},
            )
            reports[ticker] = output
        except Exception as exc:  # one bad ticker must not stop the batch
            log.error("%s failed: %s", ticker, getattr(exc, "evalue", exc))
            failed.append(ticker)
    return reports, failed


def summarize(reports: dict[str, Path]) -> pd.DataFrame:
    """Portfolio summary built only from this run's successful reports, sorted by Sharpe ratio."""
    rows = []
    for ticker, path in reports.items():
        scraps = sb.read_notebook(str(path)).scraps
        rows.append({"ticker": ticker, **{m: scraps[m].data for m in METRICS},
                     "report_path": str(path.relative_to(HERE))})
    summary = pd.DataFrame(rows)
    return summary.sort_values("sharpe_ratio", ascending=False, na_position="last", ignore_index=True)


def plot_risk_return(summary: pd.DataFrame, path: Path) -> None:
    """Scatter of mean rolling volatility vs total return, colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.scatter(summary["avg_volatility"], summary["total_return"], c=summary["sharpe_ratio"],
                        cmap="RdYlGn", s=70, edgecolor="black")
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in summary.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 6), textcoords="offset points")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Average 20-day volatility (annualized)", ylabel="Total return",
           title="Risk-Return Profile of Portfolio")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tickers, start, end = load_tickers_and_window()
    out_dir = HERE / "reports" / f"{date.today():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Window {start} to {end} | {len(tickers)} tickers | output {out_dir.relative_to(HERE)}")

    reports, failed = run_reports(tickers, start, end, out_dir)
    print(f"Successfully generated {len(reports)}/{len(tickers)} reports")
    (out_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failed))

    if not reports:
        print("No successful reports; portfolio summary skipped.")
        return

    summary = summarize(reports)
    summary.to_csv(out_dir / "portfolio_summary.csv", index=False)
    plot_risk_return(summary, out_dir / "risk_return.png")

    ranked = summary.dropna(subset=["sharpe_ratio"])
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print("\nTop 10 by Sharpe ratio:\n", ranked.head(10).drop(columns="report_path").to_string(index=False))
        print("\nBottom 10 by Sharpe ratio:\n", ranked.tail(10).drop(columns="report_path").to_string(index=False))


if __name__ == "__main__":
    main()
