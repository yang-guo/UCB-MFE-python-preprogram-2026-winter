"""HW2 Part 3: run the risk report for every ticker and summarize the results."""
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HW_DIR = Path(__file__).resolve().parent
TEMPLATE = HW_DIR / '02_stock_report_template.ipynb'
DB_PATH = HW_DIR / 'data' / 'data.db'
REPORTS_ROOT = HW_DIR / 'reports'

SUMMARY_COLUMNS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio',
                   'total_return', 'report_path']
COVERAGE_COLUMNS = ['first_date', 'last_date', 'trading_days']


def load_tickers_and_latest(db_path):
    """Return (sorted tickers, latest ts) from ohlc_hw_cleaned."""
    if not db_path.is_file():
        raise FileNotFoundError(f'{db_path} not found. Copy data.db there and run 01_data_cleaning.ipynb first.')
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
        latest = pd.read_sql_query('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    return tickers, latest


def six_complete_months(latest_date):
    """Six calendar months before the month containing latest_date."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose a window from an empty table.')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def run_report(ticker, start, end, folder):
    """Execute the template for one ticker and return the output path."""
    output = folder / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={'ticker': ticker, 'start_date': start, 'end_date': end, 'db_path': str(DB_PATH)},
        kernel_name='python3', cwd=str(HW_DIR), progress_bar=False,
    )
    return output


def collect_reports(paths):
    """Build the summary from successful report paths only."""
    rows = []
    for path in paths:
        scrap = sb.read_notebook(str(path)).scraps['summary'].data
        row = {name: scrap[name] for name in SUMMARY_COLUMNS[:-1] + COVERAGE_COLUMNS}
        row['report_path'] = str(path.relative_to(HW_DIR))
        rows.append(row)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS + COVERAGE_COLUMNS)
    if summary['ticker'].duplicated().any():
        raise ValueError('Duplicate tickers among the collected reports.')
    return summary.sort_values('sharpe_ratio', ascending=False, na_position='last').reset_index(drop=True)


def plot_risk_return(summary, path):
    """Scatter of mean rolling volatility vs total return, colored by Sharpe."""
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    points = ax.scatter(summary['avg_volatility'] * 100, summary['total_return'] * 100,
                        c=summary['sharpe_ratio'], cmap='RdYlGn', s=90, edgecolors='black', linewidths=0.5)
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(6, 4), textcoords='offset points', fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.margins(x=0.12, y=0.12)
    ax.grid(True, alpha=0.4)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Mean 20-day rolling volatility, annualized (%)',
           ylabel='Total return over the window (%)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(title, frame):
    """Print a ranking with percentages formatted for display."""
    shown = frame.copy()
    for col in ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'total_return']:
        shown[col] = shown[col].map('{:+.2%}'.format if col != 'avg_volatility' else '{:.2%}'.format)
    shown['sharpe_ratio'] = shown['sharpe_ratio'].map('{:.2f}'.format)
    print(f'\n{title}')
    print(shown[['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown',
                 'avg_daily_return', 'last_date', 'trading_days']].to_string(index=False))


def main():
    tickers, latest = load_tickers_and_latest(DB_PATH)
    start, end = six_complete_months(latest)
    run_dir = REPORTS_ROOT / date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Latest observation in ohlc_hw_cleaned: {latest}')
    print(f'Analysis window (six complete months): {start} to {end}')
    print(f'Reports folder: {run_dir}')

    successful, failures = [], []
    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            successful.append(run_report(ticker, start, end, run_dir))
        except Exception as exc:
            message = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', exc)}"
            failures.append(ticker)
            tqdm.write(f'FAILED {ticker}: {message}')

    print(f'\nSuccessfully generated {len(successful)}/{len(tickers)} reports')
    (run_dir / 'failed_stocks.txt').write_text(''.join(t + '\n' for t in failures))
    if failures:
        print('Failed tickers saved to', run_dir / 'failed_stocks.txt')

    if not successful:
        print('No report succeeded, so there is nothing to summarize.')
        return

    summary = collect_reports(successful)
    summary.to_csv(run_dir / 'portfolio_summary.csv', index=False)
    print('Portfolio summary saved to', run_dir / 'portfolio_summary.csv')

    short = summary.loc[pd.to_datetime(summary['last_date']) < pd.Timestamp(end) - pd.Timedelta(days=7)]
    if not short.empty:
        print(f'\nNote: {len(short)} tickers have data ending well before {end} '
              f'({", ".join(sorted(short["ticker"]))}); their metrics cover fewer days, so compare with care.')

    ranked = summary.dropna(subset=['sharpe_ratio'])
    print_table('Top 10 by Sharpe ratio', ranked.head(10))
    print_table('Bottom 10 by Sharpe ratio', ranked.tail(10))
    excluded = summary.loc[summary['sharpe_ratio'].isna(), 'ticker'].tolist()
    if excluded:
        print('\nExcluded from rankings (undefined Sharpe):', ', '.join(excluded))

    plot_risk_return(summary, run_dir / 'risk_return.png')
    print('\nRisk-return plot saved to', run_dir / 'risk_return.png')


if __name__ == '__main__':
    main()
