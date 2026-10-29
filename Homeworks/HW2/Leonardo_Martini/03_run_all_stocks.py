"""Run the stock report template for every ticker and build a portfolio summary.

Run from the homework folder:
    uv run python 03_run_all_stocks.py
"""
import os
import sqlite3
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

DB_PATH = "data/data.db"
TEMPLATE = "02_stock_report_template.ipynb"


def get_tickers(conn):
    """Return the list of tickers in the cleaned table."""
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)
    return tickers["ticker"].tolist()


def get_date_range(conn):
    """Return the last 6 complete months before the month of the latest observation."""
    last_date = pd.read_sql_query("SELECT MAX(ts) AS ts FROM ohlc_hw_cleaned", conn)["ts"].iloc[0]
    first_of_last_month = pd.Timestamp(last_date).replace(day=1)
    start = first_of_last_month - pd.DateOffset(months=6)
    end = first_of_last_month - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), last_date


def run_reports(tickers, start_date, end_date, output_dir):
    """Run the template for each ticker. Return the successful reports and the failed tickers."""
    done = {}
    failed = []
    for ticker in tqdm(tickers, desc="Generating reports"):
        output_path = os.path.join(output_dir, f"{ticker}_report.ipynb")
        try:
            pm.execute_notebook(
                TEMPLATE,
                output_path,
                parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date, "db_path": DB_PATH},
                progress_bar=False,
            )
            done[ticker] = output_path
        except Exception as e:
            print(f"{ticker} failed: {str(e).splitlines()[-1]}")
            failed.append(ticker)
    return done, failed


def build_summary(done):
    """Read the metrics saved by each successful report."""
    rows = []
    for ticker, path in done.items():
        scraps = sb.read_notebook(path).scraps
        rows.append({
            "ticker": ticker,
            "first_date": scraps["first_date"].data,
            "last_date": scraps["last_date"].data,
            "avg_daily_return": scraps["avg_daily_return"].data,
            "avg_volatility": scraps["avg_volatility"].data,
            "max_drawdown": scraps["max_drawdown"].data,
            "sharpe_ratio": scraps["sharpe_ratio"].data,
            "total_return": scraps["total_return"].data,
            "report_path": path,
        })
    return pd.DataFrame(rows)


def plot_summary(summary, output_dir):
    """Scatter plot of average volatility vs total return, colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.scatter(summary["avg_volatility"], summary["total_return"], c=summary["sharpe_ratio"], cmap="RdYlGn", s=80)
    fig.colorbar(points, label="Sharpe ratio")
    for _, row in summary.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Average rolling volatility (annualized)")
    ax.set_ylabel("Total return")
    ax.set_title("Risk-Return Profile of Portfolio")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "risk_return_profile.png"), dpi=120, bbox_inches="tight")
    plt.close()


def main():
    conn = sqlite3.connect(DB_PATH)
    tickers = get_tickers(conn)
    start_date, end_date, last_date = get_date_range(conn)
    conn.close()

    output_dir = os.path.join("reports", date.today().strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Latest observation in the database: {last_date}")
    print(f"Report window for all tickers: {start_date} to {end_date}")

    done, failed = run_reports(tickers, start_date, end_date, output_dir)
    print(f"Successfully generated {len(done)}/{len(tickers)} reports")

    with open(os.path.join(output_dir, "failed_stocks.txt"), "w") as f:
        f.write("\n".join(failed))

    if len(done) == 0:
        print("No report was generated, so there is no summary.")
        return

    summary = build_summary(done)
    summary.to_csv(os.path.join(output_dir, "portfolio_summary.csv"), index=False)

    short = summary[summary["last_date"] < summary["last_date"].max()]
    if len(short) > 0:
        print("\nTickers whose data ends earlier than the others (shorter coverage, compare with care):")
        print(short[["ticker", "first_date", "last_date"]].to_string(index=False))

    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False)
    cols = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown"]
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10)[cols].to_string(index=False))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10)[cols].to_string(index=False))

    plot_summary(summary, output_dir)


if __name__ == "__main__":
    main()
