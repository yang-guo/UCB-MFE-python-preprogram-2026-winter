"""Generate risk notebooks for every stock and a portfolio summary dashboard."""

from datetime import date
import json
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from IPython.display import display
from tqdm import tqdm

HOMEWORK_DIR = Path(__file__).resolve().parent
DATABASE = HOMEWORK_DIR / "data" / "data.db"
TEMPLATE = HOMEWORK_DIR / "02_stock_report_template.ipynb"


def get_six_complete_months(database):
    """Return the six full calendar months before the latest data month."""
    with sqlite3.connect(database) as connection:
        latest = pd.read_sql_query(
            "SELECT MAX(ts) AS latest_date FROM ohlc_hw_cleaned", connection
        ).iloc[0, 0]
    if pd.isna(latest):
        raise ValueError("ohlc_hw_cleaned has no observations.")
    latest_month_start = pd.Timestamp(latest).to_period("M").start_time
    start = latest_month_start - pd.DateOffset(months=6)
    end = latest_month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def read_named_metrics(report_path):
    """Read only the named summary payload saved by the successful report."""
    notebook = sb.read_notebook(str(report_path))
    payload = notebook.scraps["summary_json"].data
    return json.loads(payload)


def main():
    if not DATABASE.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE}")
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Report template not found: {TEMPLATE}")

    start_date, end_date = get_six_complete_months(DATABASE)
    run_date = date.today().isoformat()
    output_dir = HOMEWORK_DIR / "reports" / run_date
    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE) as connection:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", connection
        )["ticker"].tolist()

    successful = []
    failures = []
    for ticker in tqdm(tickers, desc="Generating reports"):
        report_path = output_dir / f"{ticker}_report.ipynb"
        try:
            pm.execute_notebook(
                input_path=str(TEMPLATE),
                output_path=str(report_path),
                parameters={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "db_path": str(DATABASE),
                },
                kernel_name="python3",
                cwd=str(HOMEWORK_DIR),
                log_output=True,
            )
            metrics = read_named_metrics(report_path)
            successful.append({
                "ticker": ticker,
                "avg_daily_return": metrics["avg_daily_return"],
                "avg_volatility": metrics["avg_volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "total_return": metrics["total_return"],
                "report_path": str(report_path.relative_to(HOMEWORK_DIR)),
                "actual_start": metrics["actual_start"],
                "actual_end": metrics["actual_end"],
            })
        except Exception as error:
            failures.append(f"{ticker}: {type(error).__name__}: {error}")
            report_path.unlink(missing_ok=True)

    (output_dir / "failed_stocks.txt").write_text(
        "\n".join(failures) + ("\n" if failures else ""), encoding="utf-8"
    )
    print(f"Successfully generated {len(successful)}/{len(tickers)} reports")
    print(f"Requested common window: {start_date} to {end_date}")

    if not successful:
        print("No reports succeeded; portfolio summary was not created.")
        return

    summary = pd.DataFrame(successful)
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False)
    print("Top 10 stocks by Sharpe ratio:")
    display(ranked.head(10))
    print("Bottom 10 stocks by Sharpe ratio:")
    display(ranked.tail(10).sort_values("sharpe_ratio"))

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_data = summary.dropna(subset=["avg_volatility", "total_return", "sharpe_ratio"])
    points = ax.scatter(
        plot_data["avg_volatility"], plot_data["total_return"],
        c=plot_data["sharpe_ratio"], cmap="viridis", s=65, alpha=0.85,
    )
    fig.colorbar(points, ax=ax, label="Annualized Sharpe ratio")
    top_return = plot_data.nlargest(5, "total_return")
    for row in top_return.itertuples(index=False):
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    ax.set_title("Risk-Return Profile of Portfolio")
    ax.set_xlabel("Mean 20-day annualized volatility")
    ax.set_ylabel("Total return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_return_profile.png", dpi=160)
    plt.close(fig)
    print("Saved portfolio_summary.csv and risk_return_profile.png in", output_dir)


if __name__ == "__main__":
    main()
