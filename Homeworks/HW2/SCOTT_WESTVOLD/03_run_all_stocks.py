import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import papermill as pm
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import logging
import scrapbook as sb
from matplotlib.ticker import PercentFormatter



db_path = "./data/data.db"
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM ohlc_hw_cleaned",
                       conn)
conn.close()


validTickers = df['ticker'].unique()
startDate = pd.to_datetime(pd.to_datetime(df['ts'].max()).strftime('%Y-%m-%d')) - timedelta(365//2)
endDate = pd.to_datetime(pd.to_datetime(df['ts'].max()).strftime('%Y-%m-%d'))

report_dir = os.path.join("reports", datetime.today().strftime("%Y-%m-%d"))
os.makedirs(report_dir, exist_ok=True)


# validTickers = ['AAPL', 'NVDA']


## Logging: messages go to the console and to run_log.txt in today's report folder ##
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(os.path.join(report_dir, "run_log.txt")),
                              logging.StreamHandler()])

## Write Failed Tickers, One Per Line ##
def write_failed_tickers(failed_tickers, report_dir):
    with open(os.path.join(report_dir, "failed_stocks.txt"), "w") as f:
        f.write("\n".join(failed_tickers))


succeeded = 0
failed_tickers = []
for tx in tqdm(validTickers, desc="Generating reports"):
    # Execute a notebook with parameters; one failing stock doesn't stop the rest
    try:
        pm.execute_notebook(
            input_path='02_stock_report_template.ipynb',
            output_path= report_dir + "/" + tx + "_report.ipynb",
            parameters={
                'ticker': tx,
                'start_date': startDate.strftime('%Y-%m-%d'),
                'end_date': endDate.strftime('%Y-%m-%d'),
                'db_path': db_path
            }
        )
        succeeded += 1
        logging.info(f"{tx}: report written")
    except Exception as e:
        failed_tickers.append(tx)
        # Papermill errors carry the notebook's own error name/message (e.g. ValueError: Insufficient history ...)
        logging.error(f"{tx}: report FAILED - {getattr(e, 'ename', type(e).__name__)}: {getattr(e, 'evalue', e)}")

## Run Summary ##
write_failed_tickers(failed_tickers, report_dir)
logging.info(f"Done: {succeeded}/{len(validTickers)} reports succeeded"
             + (f"; failed: {', '.join(failed_tickers)}" if failed_tickers else ""))





## Collect Summary Tables from Every Successful Report ##
book = sb.read_notebooks(report_dir)
tables = pd.DataFrame()
for name, nb in book.items():
    if "summary_table" not in nb.scraps:      # failed reports never reached the glue cell
        continue
    table = nb.scraps["scrap_book_summary_table"].data
    tables = pd.concat([tables, table], axis = 0, ignore_index = True)

all_summaries = tables
for i in range(len(all_summaries.index)):
    all_summaries['report_path'] = os.path.join(".", report_dir, all_summaries['ticker'].iloc[i] + "_report.ipynb")


portfolio_summary = all_summaries.dropna(subset=['sharpe_ratio'])
portfolio_summary_top_tx = portfolio_summary.sort_values('sharpe_ratio', ascending = False)['ticker'].iloc[:10]
portfolio_summary_bottom_tx = portfolio_summary.sort_values('sharpe_ratio', ascending = True)['ticker'].iloc[:10]
portfolio_summary_tx = pd.concat([portfolio_summary_top_tx,portfolio_summary_bottom_tx]).drop_duplicates().values
portfolio_summary = portfolio_summary[portfolio_summary['ticker'].isin(portfolio_summary_tx)].sort_values('sharpe_ratio', ascending = False)
portfolio_summary.to_csv(os.path.join(report_dir, "portfolio_summary.csv"))








## Portfolio Risk-Return Plot ##

def plot_risk_return(summary_df, ax=None, top_n=5):
    """Scatter avg volatility vs total return per stock, colored by Sharpe ratio, labeling the top_n by return."""
    ax = ax or plt.subplots(figsize=(10, 7))[1]

    ## Scatter Colored by Sharpe Ratio ##
    points = ax.scatter(summary_df['avg_volatility'], summary_df['total_return'],
                        c=summary_df['sharpe_ratio'], cmap='viridis', s=80, edgecolor='k')
    ax.figure.colorbar(points, ax=ax, label='Sharpe Ratio')

    ## Label Top Stocks by Return ##
    for _, row in summary_df.nlargest(top_n, 'total_return').iterrows():
        ax.annotate(row['ticker'], (row['avg_volatility'], row['total_return']),
                    xytext=(6, 6), textcoords='offset points', fontsize=9, fontweight='bold')

    ## Formatting ##
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Average Volatility (annualized)', ylabel='Total Return')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    return ax

plot_risk_return(portfolio_summary)
plt.show()


