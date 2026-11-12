"""HW2 part 3: run 02_stock_report_template.ipynb for every ticker and build a portfolio summary.

Run 01_data_cleaning.ipynb first (it creates ohlc_hw_cleaned), then: python 03_run_all_stocks.py
Everything is written to reports/<today's date>/.
"""
import logging
import os
import sqlite3
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

FOLDER = Path(__file__).resolve().parent
TEMPLATE = FOLDER / '02_stock_report_template.ipynb'
DB_PATH = FOLDER / 'data' / 'data.db'


def six_complete_months(latest_date):
    """The six calendar months before the month of latest_date, e.g. 2024-12-06 gives 2024-06-01 to 2024-11-30."""
    month_start = pd.Timestamp(latest_date).to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def run_report(ticker, start, end, out_dir):
    """Execute the template for one ticker and return the path of the saved report notebook."""
    output = out_dir / f'{ticker}_report.ipynb'
    # run it from out_dir with a relative db path, so the saved report can be re-run where it is
    pm.execute_notebook(str(TEMPLATE), str(output), cwd=str(out_dir), progress_bar=False,
                        parameters={'ticker': ticker, 'start_date': start, 'end_date': end,
                                    'db_path': os.path.relpath(DB_PATH, out_dir)})
    return output


def plot_risk_return(summary, path):
    """Scatter of average volatility vs total return, colored by Sharpe, with the top 5 returns labeled."""
    fig, ax = plt.subplots(figsize=(12, 7.5), layout='constrained')
    points = ax.scatter(summary['avg_volatility'] * 100, summary['total_return'] * 100, c=summary['sharpe_ratio'],
                        cmap='RdYlGn', s=110, edgecolor='black')
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100), xytext=(7, 5),
                    textcoords='offset points', fontweight='bold')
    ax.axhline(0, color='gray', lw=0.8)
    ax.grid(True, alpha=0.5)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Average 20-day volatility, annualized (%)',
           ylabel='Total return (%)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run every ticker, then write failed_stocks.txt, portfolio_summary.csv and the risk-return plot."""
    if not DB_PATH.exists():
        raise SystemExit('data/data.db not found: copy it there and run 01_data_cleaning.ipynb first')

    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
    latest = pd.read_sql('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    conn.close()

    start, end = six_complete_months(latest)
    out_dir = FOLDER / 'reports' / date.today().strftime('%Y-%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out_dir / 'run.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    print(f'last date in ohlc_hw_cleaned: {latest}, so the window is {start} to {end} ({len(tickers)} tickers)')

    reports, failed = {}, []
    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            reports[ticker] = run_report(ticker, start, end, out_dir)
        except Exception:
            logging.exception('%s failed', ticker)
            tqdm.write(f'{ticker} failed, see run.log')
            failed.append(ticker)

    print(f'Successfully generated {len(reports)}/{len(tickers)} reports')
    (out_dir / 'failed_stocks.txt').write_text(''.join(t + '\n' for t in failed))
    if not reports:
        print('every report failed, so there is nothing to summarize')
        return

    # only the reports that succeeded in this run, not whatever notebooks are in the folder
    rows = []
    for ticker, path in reports.items():
        m = sb.read_notebook(str(path)).scraps['report_metrics'].data
        rows.append({'ticker': ticker, 'avg_daily_return': m['avg_daily_return'],
                     'avg_volatility': m['avg_volatility'], 'max_drawdown': m['max_drawdown'],
                     'sharpe_ratio': m['sharpe_ratio'], 'total_return': m['total_return'],
                     'report_path': str(path.relative_to(FOLDER)),
                     # histories end on different dates, so keep what each report actually covers
                     'first_date': m['first_date'], 'last_date': m['last_date'], 'trading_days': m['trading_days']})
    summary_df = pd.DataFrame(rows).sort_values('sharpe_ratio', ascending=False)
    summary_df.to_csv(out_dir / 'portfolio_summary.csv', index=False)

    ranked = summary_df.dropna(subset=['sharpe_ratio'])
    shown = ranked[['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown', 'trading_days']].round(3)
    print('\nTop 10 by Sharpe ratio:')
    print(shown.head(10).to_string(index=False))
    print('\nBottom 10 by Sharpe ratio:')
    print(shown.tail(10).to_string(index=False))
    if len(ranked) < len(summary_df):
        print('left out of the ranking (Sharpe undefined):', ', '.join(summary_df.loc[summary_df['sharpe_ratio'].isna(), 'ticker']))

    short = summary_df[summary_df['last_date'] < summary_df['last_date'].max()]
    if len(short):
        print(f'\n{len(short)} tickers stop before {summary_df["last_date"].max()}, so their numbers cover fewer days:',
              ', '.join(short['ticker']))

    plot_risk_return(ranked, out_dir / 'risk_return_profile.png')
    print(f'\nsaved portfolio_summary.csv and risk_return_profile.png in {out_dir.relative_to(FOLDER)}')


if __name__ == '__main__':
    main()
