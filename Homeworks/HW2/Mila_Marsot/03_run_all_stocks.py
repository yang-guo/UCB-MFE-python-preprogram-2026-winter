"""HW2 Part 3: run the stock report for every ticker and build a portfolio summary.

Run 01_data_cleaning.ipynb first (it creates ohlc_hw_cleaned), then, with the same
Python environment as the notebooks:

    python 03_run_all_stocks.py

Outputs go to reports/{run date}/: one {ticker}_report.ipynb per stock,
portfolio_summary.csv, risk_return_profile.png, failed_stocks.txt and failed_stocks.json.
"""
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

# Paths are relative to this script, so it works from any folder
HOMEWORK_DIR = Path(__file__).resolve().parent
TEMPLATE = HOMEWORK_DIR / '02_stock_report_template.ipynb'
DB_PATH = HOMEWORK_DIR / 'data' / 'data.db'


def six_complete_months(latest_date):
    """Six calendar months before the month of the latest observation."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose a window from an empty date range.')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def run_report(ticker, start, end, folder):
    """Run the report template for one ticker and return the path of the executed notebook."""
    output = folder / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={'ticker': ticker, 'start_date': start, 'end_date': end,
                    'db_path': str(DB_PATH), 'volatility_window': 20},
        cwd=str(HOMEWORK_DIR), progress_bar=False,
    )
    return output


def collect_reports(paths):
    """Read the saved results of the successful reports into one table."""
    names = ['ticker', 'first_date', 'last_date', 'trading_days', 'avg_daily_return',
             'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']
    rows = []
    for path in paths:
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in names}
        row['report_path'] = str(path.relative_to(HOMEWORK_DIR))  # e.g. reports/2026-10-26/AAPL_report.ipynb
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary['ticker'].duplicated().any():
        raise ValueError('The same ticker appears twice in this run.')
    return summary


def plot_risk_return(summary, path):
    """Scatter plot of volatility against total return, colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    points = ax.scatter(summary['avg_volatility'] * 100, summary['total_return'] * 100,
                        c=summary['sharpe_ratio'], cmap='viridis', s=60)
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    # Label the five stocks with the highest total return
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.margins(x=0.15, y=0.15)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Average rolling volatility, annualized (%)',
           ylabel='Total return (%)')
    ax.grid(True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f'Database not found at {DB_PATH}: run 01_data_cleaning.ipynb first')

    # 1. Tickers and date window, both from the cleaned table
    with closing(sqlite3.connect(DB_PATH.as_uri() + '?mode=ro', uri=True)) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
        latest_date = pd.read_sql_query('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    start_date, end_date = six_complete_months(latest_date)
    print(f'{len(tickers)} tickers | latest date in the database: {latest_date}')
    print(f'Analysis window (last 6 complete months): {start_date} to {end_date}')

    # 2. Output folder named after today's date (the run date, not the analysis dates)
    run_dir = HOMEWORK_DIR / 'reports' / date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Reports will be saved in: {run_dir}')

    # 3. One report per ticker; a failure is logged and the loop continues
    successful_paths = []
    failures = []
    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            successful_paths.append(run_report(ticker, start_date, end_date, run_dir))
        except Exception as exc:
            failures.append({'ticker': ticker, 'error': str(exc)})
            print(f'FAILED: {ticker}. The full error is saved in failed_stocks.json.')
    (run_dir / 'failed_stocks.txt').write_text(''.join(item['ticker'] + '\n' for item in failures))
    (run_dir / 'failed_stocks.json').write_text(json.dumps(failures, indent=2))
    print(f'Successfully generated {len(successful_paths)}/{len(tickers)} reports')

    if not successful_paths:
        print('All reports failed, so there is no summary to build. See failed_stocks.json.')
        return

    # 4. Portfolio summary from the successful reports only, sorted by Sharpe ratio
    summary = collect_reports(successful_paths)
    summary_df = summary[['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
                          'sharpe_ratio', 'total_return', 'report_path']]
    summary_df = summary_df.sort_values('sharpe_ratio', ascending=False)
    summary_df.to_csv(run_dir / 'portfolio_summary.csv', index=False)

    ranked = summary_df.dropna(subset=['sharpe_ratio']).drop(columns='report_path')
    print('\nTop 10 stocks by Sharpe ratio:')
    print(ranked.head(10).to_string(index=False))
    print('\nBottom 10 stocks by Sharpe ratio:')
    print(ranked.tail(10).to_string(index=False))

    # Each stock's actual dates: some histories end before the window, so their metrics cover a shorter period
    print(f'\nActual coverage of each report (requested window: {start_date} to {end_date}):')
    print(summary[['ticker', 'first_date', 'last_date', 'trading_days']].to_string(index=False))

    # 5. Risk-return chart
    plot_risk_return(summary, run_dir / 'risk_return_profile.png')
    print(f'\nSaved portfolio_summary.csv and risk_return_profile.png in {run_dir}')


if __name__ == '__main__':
    main()
