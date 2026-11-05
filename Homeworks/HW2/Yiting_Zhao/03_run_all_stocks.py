#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import asyncio


# In[2]:


import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import papermill as pm
from tqdm import tqdm

db_path = "data/data.db"
template = "02_stock_report_template.ipynb"


# In[3]:


conn = sqlite3.connect(db_path)

tickers = pd.read_sql("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()

print(len(tickers), "tickers")


# In[4]:


latest = pd.read_sql("SELECT MAX(ts) AS ts FROM ohlc_hw_cleaned", conn)["ts"][0]
latest = pd.to_datetime(latest)

month_start = latest.replace(day=1)
end = month_start - pd.Timedelta(days=1)
start = month_start - pd.DateOffset(months=6)

start_date = start.strftime("%Y-%m-%d")
end_date = end.strftime("%Y-%m-%d")

print("Latest date:", latest.date())
print("Window:", start_date, "to", end_date)


# In[5]:


run_date = date.today().isoformat()
out_dir = f"reports/{run_date}"
os.makedirs(out_dir, exist_ok=True)

print("Saving reports to", out_dir)





# In[7]:


succeeded = []
failed = []

for ticker in tqdm(tickers, desc="Generating reports"):
    out_path = f"{out_dir}/{ticker}_report.ipynb"
    try:
        pm.execute_notebook(
            template,
            out_path,
            parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date, "db_path": db_path},
            kernel_name="python3",
            start_timeout=180,
            progress_bar=False,
        )
        succeeded.append([ticker, out_path])
    except Exception as e:
        failed.append(ticker)
        print("FAILED", ticker, type(e).__name__, e)

print(f"Successfully generated {len(succeeded)}/{len(tickers)} reports")


# In[8]:


with open(f"{out_dir}/failed_stocks.txt", "w") as f:
    for t in failed:
        f.write(t + "\n")

if len(succeeded) == 0:
    raise SystemExit("All reports failed, so no summary was created.")


# In[9]:


rows = []

for ticker, path in succeeded:
    query = "SELECT ts, adj_close FROM ohlc_hw_cleaned WHERE ticker = ? AND ts BETWEEN ? AND ? ORDER BY ts"
    data = pd.read_sql(query, conn, params=(ticker, start_date, end_date))

    if data.empty:
        print("No data in window for", ticker)
        continue

    p = data["adj_close"]

    r = np.log(p / p.shift(1)).dropna()
    vol = r.rolling(20).std() * np.sqrt(252)
    drawdown = (p - p.cummax()) / p.cummax()

    if len(r) < 2 or r.std() == 0:
        sharpe = np.nan
    else:
        sharpe = (r.mean() * 252 - 0.02) / (r.std() * np.sqrt(252))

    rows.append({
        "ticker": ticker,
        "avg_daily_return": r.mean(),
        "avg_volatility": vol.mean(),
        "max_drawdown": drawdown.min(),
        "sharpe_ratio": sharpe,
        "total_return": p.iloc[-1] / p.iloc[0] - 1,
        "first_date": data["ts"].iloc[0],
        "last_date": data["ts"].iloc[-1],
        "n_days": len(data),
        "report_path": path,
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(f"{out_dir}/portfolio_summary.csv", index=False)
print(summary_df.head())


# In[10]:


ranked = summary_df.dropna(subset=["sharpe_ratio"])
ranked = ranked.sort_values("sharpe_ratio", ascending=False)

short = summary_df["last_date"] < end_date
print(short.sum(), "stocks have data ending before", end_date)

print("Top 10:")
print(ranked.head(10))
print("Bottom 10:")
print(ranked.tail(10))


# In[11]:


plot_df = summary_df.dropna(subset=["avg_volatility", "total_return", "sharpe_ratio"])

fig, ax = plt.subplots(figsize=(10, 7))
points = ax.scatter(plot_df["avg_volatility"], plot_df["total_return"], c=plot_df["sharpe_ratio"], cmap="RdYlGn")
fig.colorbar(points, label="Sharpe ratio")

top5 = plot_df.sort_values("total_return", ascending=False).head(5)
for i in range(len(top5)):
    ax.annotate(top5["ticker"].iloc[i], (top5["avg_volatility"].iloc[i], top5["total_return"].iloc[i]))

ax.set_title("Risk-Return Profile of Portfolio")
ax.set_xlabel("Average volatility")
ax.set_ylabel("Total return")
ax.grid(alpha=0.3)

fig.savefig(f"{out_dir}/risk_return.png")
plt.show()

conn.close()






