"""
Part 3: generate a risk report for every stock with papermill, then summarize them.

Steps
1. Read the ticker list and latest date from ohlc_hw_cleaned
2. Use six calendar months before the month of the latest
   cleaned observation (e.g. latest 2024-12-06 -> 2024-06-01 to 2024-11-30).
3. Run 02_stock_report_template.ipynb once per ticker into reports/{run date}/.
   Each successful report also writes its key metrics to a JSON file.
4. Log failures, then build portfolio_summary.csv and a summary visualization.
"""
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
from tqdm import tqdm

# set up paths for reading
hw_dir = Path(__file__).resolve().parent
template = hw_dir / "02_stock_report_template.ipynb"
db_path = hw_dir / "data" / "data.db"


def six_complete_months(latest_date):
    """Six calendar months before the latest observation's month 
    (e.g. latest 2024-12-06 -> 2024-06-01 to 2024-11-30)"""
    month_start = pd.Timestamp(latest_date).to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def make_report_kernel():
    """Create a temporary kernel that runs this same Python, so reports use this script's packages."""
    # make a disposable folder to hold kernal definition
    kernel_root = Path(tempfile.mkdtemp(prefix="hw2_kernel_"))
    # jupyter looks for kernals in the directory
    kernel_dir = kernel_root / "kernels" / "hw2-report"
    kernel_dir.mkdir(parents=True)

    # "kernel.json" signals how to start the kernal
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "HW2 report", "language": "python"}))
    # also search temp folder for kernals
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    return "hw2-report"


def run_report(ticker, start, end, folder):
    """Run one report. errors are recorded"""
    # directory of execution and metrics JSON saved
    output = folder / f"{ticker}_report.ipynb"
    metrics_file = folder / "metrics" / f"{ticker}_metrics.json"
    # execute part2 notebook 
    pm.execute_notebook(
        str(template), str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end,
                    # absolute path for compatibility
                    "db_path": str(db_path), "metrics_path": str(metrics_file)},
        # use tqdm to show overall progress. Hide it here
        kernel_name=kernel_name, cwd=str(hw_dir), progress_bar=False,
    )
    return output, metrics_file

# Execution steps
# 1. Read tickers and latest date from the cleaned table
with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
    latest_date = pd.read_sql_query("SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn)["latest"].iloc[0]

# 2. 6-month window for every ticker. Anchored to the data
start_date, end_date = six_complete_months(latest_date)
print("Database latest:", latest_date, "| Six-month window:", start_date, "to", end_date)

# 3. Output folder named by the run date
run_dir = hw_dir / "reports" / date.today().isoformat()
(run_dir / "metrics").mkdir(parents=True, exist_ok=True)
print("Reports are saved in:", run_dir)

# 4. One report per ticker. Record failures and continue the loop
kernel_name = make_report_kernel()
# (ticker, report path, metrics file)
successes = []
failures = []
for ticker in tqdm(tickers, desc="Generating reports"):
    try:
        successes.append((ticker, *run_report(ticker, start_date, end_date, run_dir)))
    # Log exceptions/ failures
    except Exception as exc:
        failures.append(ticker)
        tqdm.write(f"FAILED: {ticker}: {str(exc).strip().splitlines()[-1]}")

print(f"Successfully generated {len(successes)}/{len(tickers)} reports")
(run_dir / "failed_stocks.txt").write_text("".join(t + "\n" for t in failures))

# Stop if a run failed all. Nothing to summarize
if not successes:
    sys.exit("No successful reports, so no portfolio summary was created. See failed_stocks.txt.")

# Summary: collect metrics from this run's successful reports only
rows = []
for ticker, report_path, metrics_file in successes:
    row = json.loads(metrics_file.read_text())
    row["report_path"] = str(report_path.relative_to(hw_dir))
    rows.append(row)

# build the table and chooses the columns
summary_df = pd.DataFrame(rows)[[
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return",
    "report_path", "first_obs", "last_obs", "trading_days"]]
# missing value -> NaN
summary_df[["avg_volatility", "sharpe_ratio"]] = summary_df[["avg_volatility", "sharpe_ratio"]].astype(float)
# Sort by best Sharpe first
summary_df = summary_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
# save to csv file
summary_df.to_csv(run_dir / "portfolio_summary.csv", index=False)

# Top/bottom 10 by Sharpe. Undefined Sharpe values are excluded from the ranking
ranked = summary_df.dropna(subset=["sharpe_ratio"])
cols = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown"]
# print top/bottom 10 Sharpe
print("\nTop 10 by Sharpe ratio:\n", ranked[cols].head(10).round(4).to_string(index=False))
print("\nBottom 10 by Sharpe ratio:\n", ranked[cols].tail(10).round(4).to_string(index=False))
# warns when lists overlap
if len(ranked) < 20:
    print("Fewer than 20 ranked stocks, so the top and bottom lists overlap.")

# Same 6-month window for every ticker, but handle ones that the histories end earlier
full_end = summary_df["last_obs"].max()
short = summary_df[summary_df["last_obs"] < full_end]
if not short.empty:
    print(f"\nNote: {len(short)} histories end before {full_end}, so their metrics cover fewer days.")
    # One line per end date and list the stocks that stop there
    for last_obs, group in short.groupby("last_obs"):
        print(f"  ends {last_obs}: {', '.join(group['ticker'])}")

# Risk-return scatter - volatility vs total return, colored by Sharpe, label top 5 returns
plot_df = summary_df.dropna(subset=["avg_volatility", "sharpe_ratio"])
# constrained to avoid plots overlapping
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
# convert to %
points = ax.scatter(plot_df["avg_volatility"] * 100, plot_df["total_return"] * 100,
                    c=plot_df["sharpe_ratio"], cmap="viridis")
fig.colorbar(points, ax=ax, label="Sharpe ratio")

# label top 5 stocks with highest return 
for row in plot_df.nlargest(5, "total_return").itertuples():
    ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                xytext=(4, 4), textcoords="offset points")

# formatting label and title
ax.set(xlabel="Mean rolling volatility, annualized (%)", ylabel="Total return (%)",
       title="Risk-Return Profile of Portfolio")
ax.grid(True)
fig.savefig(run_dir / "risk_return_profile.png", dpi=150, bbox_inches="tight")
# Completion message
print("\nSaved portfolio_summary.csv and risk_return_profile.png in", run_dir)
