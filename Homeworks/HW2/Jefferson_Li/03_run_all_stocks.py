"""Generate the HW2 reports and portfolio summary. Jefferson Li."""

import argparse
from datetime import date, datetime
import json
import logging
import os
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SUMMARY_COLUMNS = [
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
    "total_return", "report_path", "requested_start", "requested_end",
    "actual_start", "actual_end", "n_observations",
]


def analysis_window(latest_date):
    """Six calendar months preceding the month containing the latest observation."""
    month_start = pd.Timestamp(latest_date).normalize().replace(day=1)
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def draw_portfolio(summary, output):
    """Plot volatility versus total return, colored by available Sharpe values."""
    points = summary.loc[np.isfinite(summary["avg_volatility"]) & np.isfinite(summary["total_return"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    defined = points["sharpe_ratio"].notna() & np.isfinite(points["sharpe_ratio"])
    if defined.any():
        scatter = ax.scatter(points.loc[defined, "avg_volatility"], points.loc[defined, "total_return"],
                             c=points.loc[defined, "sharpe_ratio"], cmap="viridis", s=50)
        fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    if (~defined).any():
        ax.scatter(points.loc[~defined, "avg_volatility"], points.loc[~defined, "total_return"],
                   color="gray", label="Sharpe unavailable")
        ax.legend()
    for row in points.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return),
                    xytext=(5, 5), textcoords="offset points")
    ax.set(title="Risk-Return Profile of Portfolio", xlabel="Mean rolling annualized volatility", ylabel="Total return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def run_batch(db_path="data/data.db", output_root="reports", run_date=None):
    """Execute every ticker independently and collect only this run's successes."""
    os.chdir(ROOT)
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    day = run_date or datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    date.fromisoformat(day)
    output = Path(output_root) / day
    output.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    finally:
        conn.close()
    if not tickers or latest is None:
        raise ValueError("No cleaned observations. Run 01_data_cleaning.ipynb first.")
    start, end = analysis_window(latest)
    print(f"Requested analysis window: {start} to {end}; latest cleaned observation: {latest}", flush=True)
    print("Each report discloses its actual coverage; no trailing prices are extrapolated.", flush=True)
    # Relative paths keep generated notebooks portable and avoid embedding a home folder.
    report_db_path = os.path.relpath(database, ROOT)
    records, failures = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        destination = output / f"{ticker}_report.ipynb"
        try:
            if Path(str(ticker)).name != ticker or ticker in {".", ".."}:
                raise ValueError("Unsafe ticker filename")
            pm.execute_notebook(
                "02_stock_report_template.ipynb", str(destination),
                parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": report_db_path},
                kernel_name="python3", cwd=str(ROOT), progress_bar=False,
                execution_timeout=180, log_output=False,
            )
            notebook = sb.read_notebook(str(destination))
            metrics = dict(notebook.scraps["report_metrics"].data)
            if metrics.get("ticker") != ticker or metrics.get("requested_start") != start or metrics.get("requested_end") != end:
                raise ValueError("Executed report metrics do not match requested parameters.")
            metrics["report_path"] = str(destination)
            records.append({column: metrics.get(column) for column in SUMMARY_COLUMNS})
        except Exception as error:
            message = f"{type(error).__name__}: {error}".replace("\n", " ")
            failures.append(f"{ticker}\t{message}")
            logging.error("%s failed: %s", ticker, message)

    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    summary[numeric] = summary[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    summary = summary.sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary.to_csv(output / "portfolio_summary.csv", index=False)
    (output / "failed_stocks.txt").write_text("\n".join(failures) + ("\n" if failures else ""))
    info = {"run_date": day, "latest_observation": latest, "start_date": start, "end_date": end,
            "attempted": len(tickers), "succeeded": len(summary), "failed": len(failures),
            "successful_tickers": summary["ticker"].tolist()}
    (output / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    chart = output / "risk_return_scatter.png"
    if summary.empty:
        if chart.exists():
            chart.unlink()  # Remove a stale chart from a previous run in the same folder.
        print("No successful reports; summary CSV has headers only and no dashboard was drawn.")
    else:
        ranked = summary.loc[summary["sharpe_ratio"].notna()]
        print("\nTop 10 by Sharpe:")
        print(ranked.head(10)[["ticker", "sharpe_ratio", "total_return", "actual_end"]].to_string(index=False))
        print("\nBottom 10 by Sharpe:")
        print(ranked.tail(10)[["ticker", "sharpe_ratio", "total_return", "actual_end"]].to_string(index=False))
        draw_portfolio(summary, chart)
    print(f"Successfully generated {len(summary)}/{len(tickers)} reports", flush=True)
    return summary, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/data.db")
    parser.add_argument("--output-root", default="reports")
    parser.add_argument("--run-date", default=None, help="Output folder date, default today in Pacific Time")
    arguments = parser.parse_args()
    run_batch(arguments.db_path, arguments.output_root, arguments.run_date)
