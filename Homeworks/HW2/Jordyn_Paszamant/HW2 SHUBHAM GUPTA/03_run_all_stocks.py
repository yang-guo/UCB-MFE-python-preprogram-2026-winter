"""Execute one stock notebook per cleaned ticker and collect its metrics."""
import argparse
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter

# Resolve these from the project root. Change db_path here for another database.
db_path = "data/data.db"
kernel_name = "python3"
COLUMNS = [
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
    "sharpe_ratio", "total_return", "report_path",
]


def reporting_window(latest_date):
    month_start = pd.Timestamp(latest_date).normalize().replace(day=1)
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def scatter(summary, output):
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    plottable = summary.loc[
        np.isfinite(summary.avg_volatility) & np.isfinite(summary.total_return)
    ]
    colored = plottable.loc[np.isfinite(plottable.sharpe_ratio)]
    undefined = plottable.loc[~np.isfinite(plottable.sharpe_ratio)]
    if len(colored):
        points = ax.scatter(colored.avg_volatility, colored.total_return,
                            c=colored.sharpe_ratio, cmap="viridis", s=70,
                            edgecolors="white", linewidths=0.6)
        fig.colorbar(points, ax=ax, label="Annualized Sharpe ratio (2% risk-free)")
    if len(undefined):
        ax.scatter(undefined.avg_volatility, undefined.total_return, c="grey",
                   marker="x", label="Undefined Sharpe")
        ax.legend(frameon=False)
    for offset, row in enumerate(plottable.nlargest(5, "total_return").itertuples()):
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return),
                    xytext=(7, 8 + (offset % 2) * 7), textcoords="offset points",
                    fontsize=9, arrowprops={"arrowstyle": "-", "color": "grey"})
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set(xlabel="Mean 20-day annualized volatility",
           ylabel="Total adjusted-close return",
           title="Six-month stock risk and return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-process", action="store_true",
                        help="execute through the optional Papermill IPython engine")
    # %run supplies the same script arguments as command-line execution.
    options = parser.parse_args()
    engine = None
    if options.in_process:
        from inprocess_engine import register
        register()
        engine = "inprocess"
    root = Path.cwd()
    database = Path(db_path).resolve()
    template = root / "02_stock_report_template.ipynb"
    if not database.is_file() or not template.is_file():
        raise FileNotFoundError("Run from the project root with data/data.db in place.")
    # Reporting folders use the execution date in the student's timezone.
    run_day = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    output = root / "reports" / run_day
    output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output / "batch_errors.log", mode="w")
    logger = logging.getLogger("stock_batch")
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    try:
        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as con:
            tickers = [row[0] for row in con.execute(
                "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker"
            )]
            latest = con.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
        if not tickers or latest is None:
            raise ValueError("Cleaned table is empty. Execute 01_data_cleaning.ipynb first.")
        start, end = reporting_window(latest)
        print(f"Latest observation: {latest}; reporting window: {start} to {end}", flush=True)
        metrics, failed = [], []
        for ticker in tickers:
            destination = output / f"{ticker}_report.ipynb"
            try:
                if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\^-]*", ticker):
                    raise ValueError("Ticker cannot be used as a safe report filename")
                pm.execute_notebook(
                    str(template), str(destination),
                    parameters={"ticker": ticker, "start_date": start,
                                "end_date": end, "db_path": db_path},
                    kernel_name=kernel_name, engine_name=engine, cwd=str(root),
                    progress_bar=False, log_output=False,
                )
                result = sb.read_notebook(str(destination)).scraps["report_metrics"].data
                if result.get("ticker") != ticker:
                    raise ValueError("Collected ticker does not match the requested report")
                row = {name: result[name] for name in COLUMNS if name != "report_path"}
                for name, value in row.items():
                    if name != "ticker" and value is not None and not np.isfinite(value):
                        raise ValueError(f"Non-finite report metric: {name}")
                row["report_path"] = destination.relative_to(root).as_posix()
                metrics.append(row)
                print(f"OK {ticker}", flush=True)
            except Exception:
                failed.append(ticker)
                logger.exception("Report failed for %s", ticker)
                print(f"FAILED {ticker}; see batch_errors.log", flush=True)
        summary = pd.DataFrame(metrics, columns=COLUMNS)
        for column in COLUMNS[1:-1]:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
        temporary = output / "portfolio_summary.csv.tmp"
        summary.to_csv(temporary, index=False)
        temporary.replace(output / "portfolio_summary.csv")
        (output / "failed_stocks.txt").write_text(
            "".join(f"{ticker}\n" for ticker in failed), encoding="utf-8"
        )
        scatter(summary, output / "risk_return_scatter.png")
        ranked = summary.loc[np.isfinite(summary.sharpe_ratio)].sort_values(
            ["sharpe_ratio", "ticker"], ascending=[False, True]
        )
        print("\nTop ten by Sharpe\n" + ranked.head(10).to_string(index=False))
        print("\nBottom ten by Sharpe\n" +
              ranked.sort_values(["sharpe_ratio", "ticker"]).head(10).to_string(index=False))
        print(f"\nCompleted {len(metrics)}/{len(tickers)}; failures {len(failed)}")
        return summary
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
