"""Run 02_stock_report_template.ipynb for every ticker in ohlc_hw_cleaned and summarize the results.

Run from this homework folder after 01_data_cleaning.ipynb has written ohlc_hw_cleaned:
    python 03_run_all_stocks.py
Outputs go to reports/<run date>/: one executed {ticker}_report.ipynb per ticker, failed_stocks.txt,
portfolio_summary.csv and risk_return_profile.png.
"""
# Set up: 
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid')

# File paths:
hw_dir = Path(__file__).resolve().parent
template = hw_dir / '02_stock_report_template.ipynb'
db_path = hw_dir / 'data' / 'data.db'

# Glued names and CSV columns:
metric_names = ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']
coverage_names = ['ticker', 'start_date', 'end_date', 'first_date', 'last_date', 'trading_days']
summary_cols = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio',
                'total_return', 'report_path']


def load_tickers_and_latest(database):
    """Return the tickers and the latest observation date, both from ohlc_hw_cleaned."""
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker']
        latest = pd.read_sql_query('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    return tickers.tolist(), latest


def six_complete_months(latest_date):
    """Six calendar months before the latest observation's month (HW2 convention)."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose a window from an empty date range.')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def run_report(ticker, start, end, folder):
    """Run one independent report; propagate errors so the caller can record failures."""
    output = folder / f'{ticker}_report.ipynb'

    pm.execute_notebook(
        str(template), str(output),
        parameters={'ticker': ticker, 'start_date': start, 'end_date': end,
                    'db_path': str(db_path.relative_to(hw_dir))},
        cwd=str(hw_dir), progress_bar=False,
    )
    return output


def run_all_reports(tickers, start, end, folder):
    """Run every ticker with the same window; log failures and return the successful report paths."""
    successful_paths, failed = [], []

    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            successful_paths.append(run_report(ticker, start, end, folder))
        except Exception as exc:
            failed.append(ticker)
            tqdm.write(f'FAILED: {ticker}: {str(exc).strip().splitlines()[-1]}') # last line of the error is the reason

    (folder / 'failed_stocks.txt').write_text(''.join(ticker + '\n' for ticker in failed))
    print(f'Successfully generated {len(successful_paths)}/{len(tickers)} reports')
    return successful_paths


def collect_reports(paths):
    """Build a summary from the scraps of successful reports only; reject empty or duplicated input."""
    rows = []

    for path in paths:
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in coverage_names + metric_names}
        row['report_path'] = path.relative_to(hw_dir).as_posix()
        rows.append(row)
    if not rows:
        raise ValueError('No successful reports to summarize; see failed_stocks.txt.')

    summary = pd.DataFrame(rows)
    if summary['ticker'].duplicated().any():
        raise ValueError('Duplicate tickers in this run.')
    summary[metric_names] = summary[metric_names].astype(float)  # undefined Sharpe is glued as 'nan'
    return summary.sort_values('sharpe_ratio', ascending=False, na_position='last').reset_index(drop=True)


def print_coverage(summary):
    """Disclose the observed dates of each report and flag histories that end early."""
    latest = summary['last_date'].max()
    print(f"\nRequested window for every ticker: {summary['start_date'].iloc[0]} to {summary['end_date'].iloc[0]}")
    coverage = summary.sort_values('ticker').groupby(['first_date', 'last_date', 'trading_days'])['ticker']
    for (first, last, days), tickers in coverage:
        print(f'Observed {first} to {last} ({days} prices): {", ".join(tickers)}')
    short = summary['last_date'].lt(latest).sum()
    if short:
        print(f'{short} of {len(summary)} tickers end before {latest}; their metrics cover fewer days, '
              'so compare them with the others with care.')


def print_rankings(summary):
    """Print the top and bottom 10 by Sharpe ratio, excluding undefined values."""
    ranked = summary.loc[summary['sharpe_ratio'].notna(), ['ticker'] + metric_names]
    excluded = summary.loc[summary['sharpe_ratio'].isna(), 'ticker'].tolist()
    print('\nTop 10 by Sharpe ratio:')
    print(ranked.head(10).to_string(index=False, float_format='{:.4f}'.format))
    print('\nBottom 10 by Sharpe ratio:')
    print(ranked.tail(10).to_string(index=False, float_format='{:.4f}'.format))
    if excluded:
        print('Excluded from the rankings (undefined Sharpe ratio):', excluded)


def plot_risk_return(summary, path):
    """Scatter mean rolling volatility vs total return, colored by Sharpe; label the top 5 by return."""
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    latest = summary['last_date'].max()
    full = summary['last_date'].eq(latest)
    norm = plt.Normalize(summary['sharpe_ratio'].min(), summary['sharpe_ratio'].max())
    groups = [(full, 'o', f'data through {latest}'), (~full, '^', f'shorter history (ends before {latest})')]
    for mask, marker, label in groups:
        if mask.any():
            points = ax.scatter(summary.loc[mask, 'avg_volatility'] * 100, summary.loc[mask, 'total_return'] * 100,
                                c=summary.loc[mask, 'sharpe_ratio'], cmap='RdYlGn', norm=norm, marker=marker,
                                s=90, edgecolor='black', label=label)
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    legend = ax.legend(title=f"Window {summary['start_date'].iloc[0]} to {summary['end_date'].iloc[0]}")
    for handle in legend.legend_handles:
        handle.set_facecolor('white')  # marker shape shows coverage; color is the Sharpe ratio
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.margins(x=0.1, y=0.1)
    ax.grid(True)
    ax.set(xlabel='Mean 20-day rolling volatility, annualized (%)', ylabel='Total return (%)',
           title='Risk-Return Profile of Portfolio')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run all reports for the last six complete months and write the portfolio summary."""
    if not db_path.is_file():
        raise FileNotFoundError(f'{db_path} not found; run 01_data_cleaning.ipynb first.')
    tickers, latest = load_tickers_and_latest(db_path)
    start_date, end_date = six_complete_months(latest)
    run_dir = hw_dir / 'reports' / datetime.now().strftime('%Y-%m-%d')  # run date, not the data window
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ['failed_stocks.txt', 'portfolio_summary.csv', 'risk_return_profile.png']:
        (run_dir / name).unlink(missing_ok=True)  # a same-day rerun must not leave an earlier summary behind
    print(f'Latest cleaned observation: {latest} | Window: {start_date} to {end_date} | Output: {run_dir}')

    successful_paths = run_all_reports(tickers, start_date, end_date, run_dir)
    if not successful_paths:
        print('Every report failed, so there is no portfolio summary; see failed_stocks.txt.')
        return

    summary = collect_reports(successful_paths)
    print_coverage(summary)
    summary[summary_cols].to_csv(run_dir / 'portfolio_summary.csv', index=False)  # fractions, not percents
    print_rankings(summary)
    plot_risk_return(summary, run_dir / 'risk_return_profile.png')
    print('\nSaved', run_dir / 'portfolio_summary.csv', 'and', run_dir / 'risk_return_profile.png')


if __name__ == '__main__':
    main()
