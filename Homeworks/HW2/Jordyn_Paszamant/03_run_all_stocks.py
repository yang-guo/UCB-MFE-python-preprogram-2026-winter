
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import papermill as pm
import scrapbook as sb
from tqdm import tqdm


# 3.1 Load stocks and determine analysis dates

db_path = "data/data.db"
template_path = "02_stock_report_template.ipynb"

run_date = datetime.now().strftime("%Y-%m-%d")
output_dir = Path("reports") / run_date
output_dir.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(db_path) as conn:
    tickers_df = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker",
        conn
    )

    latest_df = pd.read_sql_query(
        "SELECT MAX(ts) AS latest_date FROM ohlc_hw_cleaned",
        conn
    )

tickers = tickers_df["ticker"].tolist()
latest_date = pd.to_datetime(latest_df.loc[0, "latest_date"])

if pd.isna(latest_date):
    raise ValueError("No valid dates found in cleaned data.")

# Last six complete months before latest observation month
latest_month = latest_date.to_period("M")
start_date = (latest_month - 6).start_time.strftime("%Y-%m-%d")
end_date = (
    latest_month.start_time - pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

print("Total stocks:", len(tickers))
print("Latest observation:", latest_date.date())
print("Analysis period:", start_date, "to", end_date)


# 3.2 Generate reports for all tickers

successful_reports = []
failed_stocks = []

for ticker in tqdm(tickers, desc="Generating reports"):
    output_path = output_dir / f"{ticker}_report.ipynb"

    try:
        pm.execute_notebook(
            input_path=template_path,
            output_path=str(output_path),
            parameters={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "db_path": db_path
            },
            kernel_name="python3",
            log_output=False,
            progress_bar=False
        )

        notebook = sb.read_notebook(str(output_path))
        metrics = notebook.scraps["report_metrics"].data

        successful_reports.append({
            "ticker": ticker,
            "avg_daily_return": metrics["avg_daily_return"],
            "avg_volatility": metrics["avg_volatility"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_return": metrics["total_return"],
            "report_path": str(output_path)
        })

    except Exception as e:
        failed_stocks.append(ticker)
        print(f"Failed: {ticker} - {e}")


print(
    f"\nSuccessfully generated "
    f"{len(successful_reports)}/{len(tickers)} reports"
)

with open(output_dir / "failed_stocks.txt", "w") as f:
    for ticker in failed_stocks:
        f.write(ticker + "\n")


# 3.3 Portfolio summary

summary_df = pd.DataFrame(
    successful_reports,
    columns=[
        "ticker",
        "avg_daily_return",
        "avg_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "total_return",
        "report_path"
    ]
)

summary_df.to_csv(
    output_dir / "portfolio_summary.csv",
    index=False
)

if summary_df.empty:
    print("No successful reports were generated.")

else:
    ranked = summary_df[
        np.isfinite(summary_df["sharpe_ratio"])
    ].sort_values("sharpe_ratio", ascending=False)

    print("\nTop 10 stocks by Sharpe ratio:")
    print(ranked.head(10).to_string(index=False))

    print("\nBottom 10 stocks by Sharpe ratio:")
    print(ranked.tail(10).to_string(index=False))

    # Risk-return scatter plot
    plot_data = summary_df[
        np.isfinite(summary_df["avg_volatility"]) &
        np.isfinite(summary_df["total_return"]) &
        np.isfinite(summary_df["sharpe_ratio"])
    ].copy()

    if not plot_data.empty:
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            plot_data["avg_volatility"],
            plot_data["total_return"],
            c=plot_data["sharpe_ratio"],
            cmap="viridis",
            s=90,
            edgecolors="black",
            alpha=0.8
        )

        top_five = plot_data.nlargest(5, "total_return")

        for _, row in top_five.iterrows():
            ax.annotate(
                row["ticker"],
                (
                    row["avg_volatility"],
                    row["total_return"]
                ),
                xytext=(5, 5),
                textcoords="offset points"
            )

        plt.colorbar(
            scatter,
            ax=ax,
            label="Sharpe Ratio"
        )

        ax.set_title("Risk-Return Profile of Portfolio")
        ax.set_xlabel("Average Annualized Rolling Volatility")
        ax.set_ylabel("Total Return")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_dir / "risk_return_scatter.png",
            dpi=300
        )

        plt.show()

print("\nFinished. Results saved in:", output_dir)
