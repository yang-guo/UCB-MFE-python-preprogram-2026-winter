"""HW2 Part 3: run the single stock report for every ticker and summarize the results.

Run it with the course environment active:

    python 03_run_all_stocks.py

1. Read the tickers and the latest date from the ohlc_hw_cleaned table.
2. Use the last 6 complete calendar months before the month of the latest date.
3. Run 02_stock_report_template.ipynb with papermill once per ticker.
4. Collect the results of the reports that worked and make the portfolio summary.
"""

import sqlite3
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # only save the figure to a file, do not open a window
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

# paths are relative to the folder of this script
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'data' / 'data.db'
TEMPLATE = BASE_DIR / '02_stock_report_template.ipynb'


def six_complete_months(latest_date):
    """Start and end date of the six calendar months before the month of latest_date.

    Example: 2024-12-06 -> ('2024-06-01', '2024-11-30')
    """
    latest = pd.Timestamp(latest_date)
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def run_one_report(ticker, start_date, end_date, output_dir):
    """Run the template notebook for one ticker and return the path of the new notebook."""
    output_path = output_dir / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(TEMPLATE),
        str(output_path),
        parameters={'ticker': ticker, 'start_date': start_date, 'end_date': end_date, 'db_path': str(DB_PATH)},
        kernel_name='python3',
        cwd=str(BASE_DIR),
        progress_bar=False,
    )
    return output_path


def collect_results(successful_paths):
    """Read the values saved with scrapbook in the reports that worked."""
    rows = []
    for path in successful_paths:
        scraps = sb.read_notebook(str(path)).scraps
        rows.append({
            'ticker': scraps['ticker'].data,
            'avg_daily_return': scraps['avg_daily_return'].data,
            'avg_volatility': scraps['avg_volatility'].data,
            'max_drawdown': scraps['max_drawdown'].data,
            'sharpe_ratio': scraps['sharpe_ratio'].data,
            'total_return': scraps['total_return'].data,
            'report_path': str(path.relative_to(BASE_DIR)),
            # the real coverage of each report (not every ticker has data until the end of the window)
            'first_date': scraps['first_date'].data,
            'last_date': scraps['last_date'].data,
            'trading_days': scraps['trading_days'].data,
        })
    return pd.DataFrame(rows)


def make_scatter_plot(summary, output_path):
    """Average volatility (x) vs total return (y), colored by the Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.scatter(summary['avg_volatility'], summary['total_return'],
                        c=summary['sharpe_ratio'], cmap='viridis', s=70, edgecolor='black')
    fig.colorbar(points, ax=ax, label='Sharpe ratio')

    # label the 5 stocks with the highest total return
    top5 = summary.sort_values('total_return', ascending=False).head(5)
    for i in range(len(top5)):
        row = top5.iloc[i]
        ax.annotate(row['ticker'], (row['avg_volatility'], row['total_return']),
                    xytext=(5, 5), textcoords='offset points')

    ax.set_title('Risk-Return Profile of Portfolio')
    ax.set_xlabel('Average rolling volatility (annualized)')
    ax.set_ylabel('Total return over the period')
    ax.grid(True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # tickers and date range come from the cleaned table
    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
    latest_date = pd.read_sql_query('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    conn.close()

    start_date, end_date = six_complete_months(latest_date)
    print(f'{len(tickers)} tickers | latest date in the database: {latest_date}')
    print(f'Analysis window (last 6 complete months): {start_date} to {end_date}')

    # the folder is named after the day we run the script
    output_dir = BASE_DIR / 'reports' / date.today().strftime('%Y-%m-%d')
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_paths = []
    failed_tickers = []
    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            successful_paths.append(run_one_report(ticker, start_date, end_date, output_dir))
        except Exception as error:
            print(f'FAILED: {ticker}: {error}')
            failed_tickers.append(ticker)

    print(f'Successfully generated {len(successful_paths)}/{len(tickers)} reports')

    with open(output_dir / 'failed_stocks.txt', 'w') as file:
        for ticker in failed_tickers:
            file.write(ticker + '\n')

    if len(successful_paths) == 0:
        print('No report worked, so there is no summary. See failed_stocks.txt.')
        return

    # summary sorted by Sharpe ratio (stocks without a Sharpe ratio are left out of the rankings)
    summary_df = collect_results(successful_paths)
    summary_df = summary_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    ranked = summary_df.dropna(subset=['sharpe_ratio'])
    print('\nTop 10 by Sharpe ratio:')
    print(ranked.head(10)[['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility']].to_string(index=False))
    print('\nBottom 10 by Sharpe ratio:')
    print(ranked.tail(10)[['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility']].to_string(index=False))

    summary_df.to_csv(output_dir / 'portfolio_summary.csv', index=False)
    make_scatter_plot(summary_df, output_dir / 'risk_return_profile.png')
    print(f'\nSaved portfolio_summary.csv and risk_return_profile.png in {output_dir}')


if __name__ == '__main__':
    main()
