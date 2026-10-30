"""Generate stock reports and summarize successful executions.

Run after 01_data_cleaning.ipynb has created data/data.db.
Analysis dates come from the database; output folders use the run date.
"""

from pathlib import Path
from contextlib import closing
from datetime import datetime
import json
import os
import sqlite3
import sys
import traceback

import matplotlib
matplotlib.use("Agg")  # Save charts without opening interactive windows.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm


HOMEWORK_DIR = Path(__file__).resolve().parent
DATABASE = HOMEWORK_DIR / "data" / "data.db"
TEMPLATE = HOMEWORK_DIR / "02_stock_report_template.ipynb"


def six_complete_months(latest_date):
    """Return six calendar months before the latest observation's month."""
    latest = pd.Timestamp(latest_date)

    if pd.isna(latest):
        raise ValueError("Cannot select a window from an empty database.")

    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def load_universe(database):
    """Read the cleaned ticker universe and latest observation date."""
    database = Path(database)

    if not database.is_file():
        raise FileNotFoundError(f"Working database not found: {database}")

    with closing(sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True
    )) as conn:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker",
            conn,
        )["ticker"].tolist()

        latest_date = pd.read_sql_query(
            "SELECT MAX(ts) AS latest_date FROM ohlc_hw_cleaned",
            conn,
        ).loc[0, "latest_date"]

    if not tickers:
        raise ValueError("The cleaned ticker universe is empty.")

    return tickers, latest_date

def prepare_kernel(run_dir):
    """Make child reports use this interpreter without a user-level install."""
    kernel_root = run_dir / "_jupyter"
    kernel_dir = kernel_root / "kernels" / "hw2-batch"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    (kernel_dir / "kernel.json").write_text(
        json.dumps({
            "argv": [
                sys.executable, "-m", "ipykernel_launcher",
                "-f", "{connection_file}",
            ],
            "display_name": "HW2 batch",
            "language": "python",
        }),
        encoding="utf-8",
    )

    existing = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        part for part in [str(kernel_root), existing] if part
    )
    return "hw2-batch"


def run_report(symbol, start, end, run_dir, kernel_name):
    """Execute one report and collect its required named outputs."""
    output = run_dir / f"{symbol}_report.ipynb"

    pm.execute_notebook(
        input_path=str(TEMPLATE),
        output_path=str(output),
        parameters={
            "ticker": symbol,
            "start_date": start,
            "end_date": end,
            "db_path": str(DATABASE),
        },
        kernel_name=kernel_name,
        cwd=str(HOMEWORK_DIR),
        progress_bar=False,
    )

    scraps = sb.read_notebook(str(output)).scraps
    metric_names = [
        "ticker", "avg_daily_return", "avg_volatility",
        "max_drawdown", "sharpe_ratio", "total_return",
        "actual_start", "actual_end", "trading_days",
    ]
    row = {name: scraps[name].data for name in metric_names}

    if (
        row["ticker"] != symbol
        or scraps["start_date"].data != start
        or scraps["end_date"].data != end
    ):
        raise ValueError("Exported parameters do not match the request.")

    # Relative paths remain usable when the homework folder is moved.
    row["report_path"] = output.relative_to(HOMEWORK_DIR).as_posix()
    return row


def save_dashboard(summary, run_dir):
    """Save the risk-return scatter for successful report outputs."""
    plot_data = summary.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["avg_volatility", "total_return"]
    )
    if plot_data.empty:
        print("No finite risk-return pairs available for the scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    defined = plot_data.dropna(subset=["sharpe_ratio"])

    if not defined.empty:
        points = ax.scatter(
            defined["avg_volatility"],
            defined["total_return"],
            c=defined["sharpe_ratio"],
            cmap="viridis",
        )
        fig.colorbar(points, ax=ax, label="Sharpe ratio")

    undefined = plot_data.loc[plot_data["sharpe_ratio"].isna()]
    if not undefined.empty:
        ax.scatter(
            undefined["avg_volatility"],
            undefined["total_return"],
            color="gray", label="Undefined Sharpe",
        )
        ax.legend()

    for row in plot_data.nlargest(5, "total_return").itertuples():
        ax.annotate(
            row.ticker,
            (row.avg_volatility, row.total_return),
            xytext=(5, 5), textcoords="offset points",
        )

    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(
        title="Risk-Return Profile of Portfolio",
        xlabel="Mean rolling volatility (annualized)",
        ylabel="Total return",
    )
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "portfolio_summary.png", dpi=150)
    plt.close(fig)


def main():
    """Generate all reports; summarize only this run's successful results."""
    tickers, latest_date = load_universe(DATABASE)
    start, end = six_complete_months(latest_date)

    run_date = datetime.now().strftime("%Y-%m-%d")
    run_dir = HOMEWORK_DIR / "reports" / run_date

    # Protect earlier outputs from accidental same-day overwrites.
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {run_dir}. "
            "Preserve or rename it before rerunning."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    kernel_name = prepare_kernel(run_dir)

    print(f"Requested window: {start} to {end}")
    successful_rows = []
    failures = []

    for symbol in tqdm(tickers, desc="Generating reports"):
        try:
            successful_rows.append(
                run_report(symbol, start, end, run_dir, kernel_name)
            )
        except Exception:
            failures.append({
                "ticker": symbol,
                "error": traceback.format_exc(),
            })
            print(f"FAILED: {symbol}; continuing with remaining tickers.")

    (run_dir / "failed_stocks.txt").write_text(
        "".join(item["ticker"] + "\n" for item in failures),
        encoding="utf-8",
    )
    (run_dir / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "start_date": start,
            "end_date": end,
            "successful_reports": [
                row["report_path"] for row in successful_rows
            ],
            "failed_tickers": [item["ticker"] for item in failures],
        }, indent=2),
        encoding="utf-8",
    )

    print(
        f"Successfully generated {len(successful_rows)}/{len(tickers)} reports"
    )

    if not successful_rows:
        print("All reports failed; no portfolio summary was generated.")
        return

    summary = pd.DataFrame(successful_rows)
    numeric_columns = [
        "avg_daily_return", "avg_volatility",
        "max_drawdown", "sharpe_ratio", "total_return",
    ]
    summary[numeric_columns] = summary[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    summary = summary.sort_values(
        "sharpe_ratio", ascending=False, na_position="last"
    )

    required_columns = [
        "ticker", "avg_daily_return", "avg_volatility",
        "max_drawdown", "sharpe_ratio", "total_return", "report_path",
    ]
    summary[required_columns].to_csv(
        run_dir / "portfolio_summary.csv", index=False
    )
    summary[[
        "ticker", "actual_start", "actual_end", "trading_days"
    ]].to_csv(run_dir / "report_coverage.csv", index=False)

    ranked = summary.loc[np.isfinite(summary["sharpe_ratio"])]
    print("\nTop 10 by Sharpe:")
    print(ranked[required_columns].head(10).to_string(index=False))
    print("\nBottom 10 by Sharpe:")
    print(ranked[required_columns].tail(10).to_string(index=False))

    print(
        "\nComparisons use one requested window but potentially different "
        "observed histories. See report_coverage.csv."
    )
    save_dashboard(summary, run_dir)


if __name__ == "__main__":
    main()