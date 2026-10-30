import sqlite3
from pathlib import Path
from datetime import datetime
import calendar

import pandas as pd
import matplotlib.pyplot as plt
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "data.db"
TEMPLATE_PATH = BASE_DIR / "02_stock_report_template.ipynb"

# Create dated reports folder
REPORT_DIR = BASE_DIR / "reports" / datetime.now().strftime("%Y-%m-%d")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Read available stocks and latest observation date
with sqlite3.connect(DB_PATH) as conn:
    tickers = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker",
        conn
    )["ticker"].tolist()

    latest_date = pd.read_sql_query(
        "SELECT MAX(ts) AS latest_date FROM ohlc_hw_cleaned",
        conn
    )["latest_date"].iloc[0]

# Use the six complete calendar months before the latest observation month
latest = pd.Timestamp(latest_date)
end_date = latest.replace(day=1) - pd.Timedelta(days=1)
start_date = end_date.replace(day=1) - pd.DateOffset(months=5)

print(f"Reporting window: {start_date.date()} to {end_date.date()}")
print(f"Stocks found: {len(tickers)}")

results = []
failed = []

# Generate reports for each ticker
for ticker in tqdm(tickers, desc="Generating stock reports"):
    output_path = REPORT_DIR / f"{ticker}_report.ipynb"

    try:
        pm.execute_notebook(
            str(TEMPLATE_PATH),
            str(output_path),
            parameters={
                "ticker": ticker,
                "start_date": str(start_date.date()),
                "end_date": str(end_date.date()),
                "db_path": str(DB_PATH),
            },
            kernel_name="python3",
            progress_bar=False,
        )

        notebook = sb.read_notebook(str(output_path))
        metrics = notebook.scraps["report_metrics"].data
        metrics["report_path"] = str(output_path)

        results.append(metrics)

    except Exception as error:
        failed.append(f"{ticker}: {error}")
        print(f"Failed: {ticker} - {error}")

# Save portfolio summary
summary = pd.DataFrame(results)

if not summary.empty:
    summary.to_csv(REPORT_DIR / "portfolio_summary.csv", index=False)

    # Rank stocks by Sharpe ratio
    ranked = summary.dropna(subset=["sharpe_ratio"])
    ranked = ranked.sort_values("sharpe_ratio", ascending=False)

    print("\nTop 10 stocks by Sharpe ratio:")
    print(ranked[["ticker", "sharpe_ratio"]].head(10))

    print("\nBottom 10 stocks by Sharpe ratio:")
    print(ranked[["ticker", "sharpe_ratio"]].tail(10))

    # Risk-return scatter plot
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(
        summary["avg_volatility"],
        summary["total_return"],
        c=summary["sharpe_ratio"],
        cmap="coolwarm",
        alpha=0.8,
    )

    plt.colorbar(scatter, label="Sharpe Ratio")
    plt.xlabel("Average Annualized Rolling Volatility")
    plt.ylabel("Total Return")
    plt.title("Stock Risk vs. Return")

    # Label the five highest-return stocks
    top_returns = summary.nlargest(5, "total_return")

    for _, row in top_returns.iterrows():
        plt.annotate(
            row["ticker"],
            (row["avg_volatility"], row["total_return"]),
        )

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "risk_return_scatter.png", dpi=150)
    plt.close()

# Save any failures
(REPORT_DIR / "failed_stocks.txt").write_text(
    "\n".join(failed),
    encoding="utf-8",
)

print(f"\nSuccessful reports: {len(results)}")
print(f"Failed reports: {len(failed)}")
print(f"Output folder: {REPORT_DIR}")