import os
import sqlite3
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import papermill as pm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Determine analysis window
conn = sqlite3.connect("data/data.db")
max_date_str = pd.read_sql_query("SELECT MAX(ts) AS max_ts FROM ohlc_hw_cleaned", conn).iloc[0]["max_ts"]
max_date = pd.to_datetime(max_date_str)

first_of_max_month = max_date.replace(day=1)
end_date_dt = first_of_max_month - pd.Timedelta(days=1)
start_date_dt = (end_date_dt.replace(day=1) - pd.DateOffset(months=5)).replace(day=1)

start_date = start_date_dt.strftime("%Y-%m-%d")
end_date = end_date_dt.strftime("%Y-%m-%d")

print(f"Analysis window: {start_date} to {end_date}")

# 2. Get STRICTLY all tickers as requested by HW2 instructions
tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
conn.close()

print(f"Total tickers to process: {len(tickers)}")

# 3. Create run-date output directory
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
output_dir = f"reports/{today_str}"
os.makedirs(output_dir, exist_ok=True)

# 4. Helper function for Papermill
def process_ticker(ticker):
    out_notebook = os.path.join(output_dir, f"{ticker}_report.ipynb")
    try:
        pm.execute_notebook(
            input_path="02_stock_report_template.ipynb",
            output_path=out_notebook,
            parameters={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "db_path": "data/data.db"
            },
            log_output=False
        )
        return ticker, out_notebook, None
    except Exception as e:
        return ticker, None, str(e)

# 5. Parallel Execution (4 workers to avoid Jupyter memory overload)
successful_reports = []
failed_tickers = []

print("Processing stocks using 4 parallel workers...")
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_ticker, t): t for t in tickers}
    for future in tqdm(as_completed(futures), total=len(tickers), desc="Generating reports"):
        t, out_path, err = future.result()
        if err is None:
            successful_reports.append((t, out_path))
        else:
            failed_tickers.append((t, err))

print(f"\nExecution finished: {len(successful_reports)}/{len(tickers)} reports generated successfully.")

# Log failed tickers as requested
if failed_tickers:
    failed_path = os.path.join(output_dir, "failed_stocks.txt")
    with open(failed_path, "w") as f:
        for t, err in failed_tickers:
            f.write(f"{t}: {err}\n")
    print(f"Failed tickers logged to {failed_path}")

# 6. Build portfolio summary dashboard
print("\nBuilding portfolio summary dashboard...")
summary_data = []
conn = sqlite3.connect("data/data.db")

for ticker, path in successful_reports:
    df_t = pd.read_sql_query(
        "SELECT ts, adj_close FROM ohlc_hw_cleaned WHERE ticker = ? AND ts BETWEEN ? AND ? ORDER BY ts ASC",
        conn,
        params=(ticker, start_date, end_date)
    )
    if len(df_t) < 2:
        continue
    
    df_t["log_ret"] = np.log(df_t["adj_close"] / df_t["adj_close"].shift(1))
    clean_rets = df_t["log_ret"].dropna()
    if len(clean_rets) < 2:
        continue
        
    avg_daily_ret = clean_rets.mean()
    rolling_vol = clean_rets.rolling(20).std() * np.sqrt(252)
    avg_vol = rolling_vol.mean()
    
    cum_max = df_t["adj_close"].cummax()
    max_dd = ((df_t["adj_close"] - cum_max) / cum_max).min()
    
    tot_ret = (df_t["adj_close"].iloc[-1] / df_t["adj_close"].iloc[0]) - 1
    
    ann_mean = avg_daily_ret * 252
    ann_std = clean_rets.std() * np.sqrt(252)
    sharpe = (ann_mean - 0.02) / ann_std if ann_std > 0 else np.nan
    
    summary_data.append({
        "ticker": ticker,
        "avg_daily_return": avg_daily_ret,
        "avg_volatility": avg_vol,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "total_return": tot_ret,
        "report_path": path
    })

conn.close()

summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(output_dir, "portfolio_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Portfolio summary saved to {summary_csv_path}")

# Display Rankings
valid_sharpe = summary_df.dropna(subset=["sharpe_ratio"]).sort_values(by="sharpe_ratio", ascending=False)
print("\n--- Top 10 Stocks by Sharpe Ratio ---")
print(valid_sharpe[["ticker", "total_return", "avg_volatility", "sharpe_ratio"]].head(10))

print("\n--- Bottom 10 Stocks by Sharpe Ratio ---")
print(valid_sharpe[["ticker", "total_return", "avg_volatility", "sharpe_ratio"]].tail(10))

# 7. Scatter plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    summary_df["avg_volatility"],
    summary_df["total_return"],
    c=summary_df["sharpe_ratio"],
    cmap="viridis",
    alpha=0.7,
    edgecolors="none"
)
plt.colorbar(sc, label="Sharpe Ratio")
plt.xlabel("Average Rolling Volatility")
plt.ylabel("Total Return")
plt.title("Risk-Return Profile of Portfolio")
plt.grid(True, linestyle="--", alpha=0.5)

top5 = summary_df.sort_values(by="total_return", ascending=False).head(5)
for _, row in top5.iterrows():
    plt.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                 xytext=(5, 5), textcoords="offset points", fontweight="bold")

plt.savefig(os.path.join(output_dir, "portfolio_risk_return.png"), bbox_inches="tight")
plt.show()
