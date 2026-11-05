import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import papermill as pm
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
from datetime import date
import logging
import json
import nbformat

HW_DIR = Path(__file__).resolve().parent

DB_PATH = HW_DIR / "data" / "data.db"
TEMPLATE_PATH = HW_DIR / "02_stock_report_template.ipynb"
REPORTS_DIR = HW_DIR / "reports"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

if not TEMPLATE_PATH.exists():
    raise FileNotFoundError(f"Notebook template not found: {TEMPLATE_PATH}")


conn = sqlite3.connect(DB_PATH)

try:
    ticker_df = pd.read_sql_query(
        """
        SELECT DISTINCT ticker
        FROM ohlc_hw_cleaned
        WHERE ticker IS NOT NULL
        ORDER BY ticker
        """,
        conn
    )

    latest_date_df = pd.read_sql_query(
        """
        SELECT MAX(ts) AS latest_date
        FROM ohlc_hw_cleaned
        """,
        conn
    )

finally:
    conn.close()

tickers = ticker_df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()

if not tickers:
    raise ValueError("No tickers found in ohlc_hw_cleaned.")

latest_date = pd.to_datetime(
    latest_date_df.loc[0, "latest_date"],
    errors="coerce"
)

if pd.isna(latest_date):
    raise ValueError("Could not determine the latest observation date.")


latest_month_start = latest_date.to_period("M").start_time

end_date = latest_month_start - pd.Timedelta(days=1)

start_date = latest_month_start - pd.DateOffset(months=6)

start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")


print("Number of tickers:", len(tickers))
print("Latest database observation:", latest_date.date())
print("Analysis start date:", start_date)
print("Analysis end date:", end_date)

run_date = date.today().isoformat()

output_dir = REPORTS_DIR / run_date
output_dir.mkdir(parents=True, exist_ok=True)

print("Reports will be saved in:", output_dir)

log_file = output_dir / "run.log"

logger = logging.getLogger("stock_report_runner")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Starting multi-stock report generation.")
logger.info("Analysis period: %s to %s", start_date, end_date)
logger.info("Number of tickers: %d", len(tickers))

successful = []
failed = []
portfolio_rows = []

for ticker in tqdm(tickers, desc="Generating reports"):

    output_notebook = output_dir / f"{ticker}_report.ipynb"

    parameters = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date
    }

    try:
        pm.execute_notebook(
            input_path=str(TEMPLATE_PATH),
            output_path=str(output_notebook),
            parameters=parameters,
            cwd=str(HW_DIR),
            kernel_name="mfe-hw2",
            log_output=True
        )

        nb = nbformat.read(str(output_notebook), as_version=4)

        metrics = None

        for cell in nb.cells:
            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        output_text = output.get("text", "")

                        for line in output_text.splitlines():
                            if line.startswith("PORTFOLIO_METRICS:"):
                                metrics = json.loads(
                                    line.replace(
                                        "PORTFOLIO_METRICS:", "", 1
                                    )
                                )

        if metrics is None:
            raise ValueError(
                f"Portfolio metrics not found for {ticker}"
            )

        portfolio_rows.append({
            "ticker": ticker,
            "avg_daily_return": metrics["avg_daily_return"],
            "avg_volatility": metrics["avg_volatility"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_return": metrics["total_return"],
            "report_path": str(output_notebook)
        })

        successful.append(ticker)
        logger.info(f"Successfully generated report for {ticker}")

    except Exception:
        failed.append(ticker)
        logger.exception(f"Failed to generate report for {ticker}")
        continue


print(f"Successfully generated {len(successful)}/{len(tickers)} reports")

logger.info(
    f"Successfully generated {len(successful)}/{len(tickers)} reports"
)

failed_file = output_dir / "failed_stocks.txt"

with open(failed_file, "w") as f:
    for ticker in failed:
        f.write(ticker + "\n")

if failed:
    print("Failed tickers:", ", ".join(failed))

if len(successful) == 0:
    print("No reports were successfully generated.")

if len(portfolio_rows) == 0:
    print("No successful reports with usable metrics.")
    print("Portfolio summary will not be created.")

else:
    summary_df = pd.DataFrame(portfolio_rows)

    metric_columns = [
        "avg_daily_return",
        "avg_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "total_return"
    ]

    for column in metric_columns:
        summary_df[column] = pd.to_numeric(
            summary_df[column],
            errors="coerce"
        )

    summary_df[metric_columns] = summary_df[metric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    summary_df = summary_df.sort_values(
        "sharpe_ratio",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    summary_path = output_dir / "portfolio_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nPortfolio Summary:")
    print(summary_df.to_string(index=False))

    ranked_df = summary_df.dropna(subset=["sharpe_ratio"])

    print("\nTOP 10 STOCKS BY SHARPE RATIO")
    print(ranked_df.head(10).to_string(index=False))

    print("\nBOTTOM 10 STOCKS BY SHARPE RATIO")
    print(
        ranked_df.tail(10)
        .sort_values("sharpe_ratio", ascending=True)
        .to_string(index=False)
    )

    if ranked_df.empty:
        print("\nNo stocks have a defined Sharpe ratio.")

    fig, ax = plt.subplots(figsize=(12, 8))

    valid = summary_df[
        np.isfinite(summary_df["avg_volatility"])
        & np.isfinite(summary_df["total_return"])
        & np.isfinite(summary_df["sharpe_ratio"])
    ]

    undefined = summary_df[
        np.isfinite(summary_df["avg_volatility"])
        & np.isfinite(summary_df["total_return"])
        & ~np.isfinite(summary_df["sharpe_ratio"])
    ]

    if not valid.empty:
        scatter = ax.scatter(
            valid["avg_volatility"],
            valid["total_return"],
            c=valid["sharpe_ratio"],
            cmap="viridis",
            s=80,
            alpha=0.8
        )

        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Sharpe Ratio")

    if not undefined.empty:
        ax.scatter(
            undefined["avg_volatility"],
            undefined["total_return"],
            color="gray",
            marker="x",
            s=80,
            label="Undefined Sharpe Ratio"
        )
        ax.legend()

    top_5 = summary_df.dropna(
        subset=["avg_volatility", "total_return"]
    ).nlargest(5, "total_return")

    for _, row in top_5.iterrows():
        ax.annotate(
            row["ticker"],
            (row["avg_volatility"], row["total_return"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    ax.set_xlabel("Average Rolling Volatility")
    ax.set_ylabel("Total Return")
    ax.set_title("Risk-Return Profile of Portfolio")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / "portfolio_risk_return.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPortfolio summary saved to: {summary_path}")
    print(f"Portfolio plot saved to: {plot_path}")


logger.info("Multi-stock report generation completed.")
logger.info("Successful reports: %d", len(successful))
logger.info("Failed reports: %d", len(failed))

if failed:
    logger.warning("Tickers with failed reports: %s", ", ".join(failed))

print("\nReport generation complete!")
print("Successful:", len(successful))
print("Failed:", len(failed))
print("Output directory:", output_dir)
print("Log file:", log_file)

