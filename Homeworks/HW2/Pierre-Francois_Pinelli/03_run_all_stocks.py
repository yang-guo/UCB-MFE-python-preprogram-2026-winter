"""Run 02_stock_report_template.ipynb for each stock with papermill, then make the portfolio summary.

How to use it (after 01_data_cleaning.ipynb has made the table ohlc_hw_cleaned):
    python 03_run_all_stocks.py

Results in reports/{date of the run}/: one {TICKER}_report.ipynb for each stock, failed_stocks.txt,
portfolio_summary.csv and risk_return_profile.png.
"""
import os
import sqlite3
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the chart in a file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import papermill as pm  # noqa: E402
import scrapbook as sb  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent          # the homework folder, from wherever the script is started
DB_PATH = Path("data") / "data.db"             # the paths below start from HERE
TEMPLATE = Path("02_stock_report_template.ipynb")


def six_complete_months(latest_date):
    """The six calendar months before the month of the last date in the data.

    Example: last date 2024-12-06 -> from 2024-06-01 to 2024-11-30.
    """
    month_start = pd.Timestamp(latest_date).to_period("M").start_time   # first day of the last month
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def main():
    """Run the template for each stock, then save failed_stocks.txt, portfolio_summary.csv and the chart."""
    os.chdir(HERE)   # work from the homework folder, so the paths written in the reports are short
    if not DB_PATH.exists():
        raise SystemExit("data/data.db not found: copy the course database there, then run 01_data_cleaning.ipynb.")

    # 1. List of stocks and dates of the analysis, both from the clean table
    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
    latest = pd.read_sql_query("SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn)["latest"].iloc[0]
    conn.close()
    start_date, end_date = six_complete_months(latest)
    print(f"{len(tickers)} tickers | latest date {latest} | window {start_date} to {end_date}")

    # 2. Folder named with the date of the run (not the dates of the analysis)
    out_dir = Path("reports") / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. One report for each stock. An error on one stock must not stop the others.
    successes, failures = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        output = out_dir / f"{ticker}_report.ipynb"
        try:
            pm.execute_notebook(
                str(TEMPLATE), str(output),
                parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                            "db_path": DB_PATH.as_posix()},
                kernel_name="python3", progress_bar=False,
            )
            successes.append(output)
        except Exception as error:
            failures.append(ticker)
            reason = getattr(error, "evalue", str(error))   # papermill gives the error message of the notebook
            print(f"FAILED {ticker}: {reason}")
            if output.exists():                      # papermill saves the notebook until the cell with the error
                output.replace(out_dir / f"{ticker}_report_FAILED.ipynb")   # replace() also works if the file already exists

    print(f"Successfully generated {len(successes)}/{len(tickers)} reports")
    (out_dir / "failed_stocks.txt").write_text("".join(t + "\n" for t in failures))
    if not successes:
        print("No report succeeded, so there is no summary. See the errors above.")
        return

    # 4. Read the saved results of the successful reports only
    rows = []
    for path in successes:
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in ["ticker", "avg_daily_return", "avg_volatility",
                                                    "max_drawdown", "sharpe_ratio", "total_return",
                                                    "last_obs", "trading_days"]}
        row["report_path"] = path.as_posix()
        rows.append(row)
    results = pd.DataFrame(rows)
    metrics = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    results[metrics] = results[metrics].apply(pd.to_numeric, errors="coerce")   # an undefined Sharpe comes back as "nan"
    summary_df = results[["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                          "total_return", "report_path"]].sort_values("sharpe_ratio", ascending=False)
    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)

    ranked = summary_df.dropna(subset=["sharpe_ratio"])           # a Sharpe that cannot be computed is not ranked
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10).drop(columns="report_path").to_string(index=False))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10).drop(columns="report_path").to_string(index=False))

    short = results[results["last_obs"] < results["last_obs"].max()]     # stocks whose data ends earlier
    if len(short) > 0:
        print(f"\nNote: {len(short)} stocks have data only until {short['last_obs'].max()} "
              f"({short['trading_days'].min()} days instead of {results['trading_days'].max()}). "
              "Their total return and max drawdown cover a shorter period, so compare them with care. "
              "Average return, volatility and Sharpe are per day or per year, so they stay comparable.")

    # 5. Risk-return chart
    fig, ax = plt.subplots(figsize=(11, 7))
    points = ax.scatter(summary_df["avg_volatility"] * 100, summary_df["total_return"] * 100,
                        c=summary_df["sharpe_ratio"], cmap="RdYlGn", s=80, edgecolors="black")
    fig.colorbar(points, label="Sharpe ratio")
    for _, row in summary_df.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"] * 100, row["total_return"] * 100),
                    xytext=(6, 6), textcoords="offset points", fontweight="bold")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True)
    ax.set(xlabel="Average 20-day volatility, annualized (%)", ylabel="Total return (%)",
           title="Risk-Return Profile of Portfolio")
    fig.savefig(out_dir / "risk_return_profile.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_dir / 'portfolio_summary.csv'} and risk_return_profile.png")


if __name__ == "__main__":
    main()
