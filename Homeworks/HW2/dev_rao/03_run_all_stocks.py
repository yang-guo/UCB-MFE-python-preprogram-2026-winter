"""HW2 part 3: run the stock report for every ticker in ohlc_hw_cleaned.

usage (from the repo's .venv, after running 01_data_cleaning.ipynb):
    python 03_run_all_stocks.py
    python 03_run_all_stocks.py --tickers AAPL MSFT    # quick test on a few tickers
    python 03_run_all_stocks.py --db path/to/data.db

What it does:
1. reads tickers + the latest date from ohlc_hw_cleaned
2. picks the last 6 complete months before the latest date's month
3. runs 02_stock_report_template.ipynb with papermill for each ticker -> reports/{run_date}/{ticker}_report.ipynb
4. logs failures to failed_stocks.txt and keeps going
5. reads the glued metrics from the successful reports only and saves portfolio_summary.csv + a risk/return plot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
import traceback
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # no window, just save the figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / '02_stock_report_template.ipynb'
DEFAULT_DB = HERE / 'data' / 'data.db'

SUMMARY_COLUMNS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
                   'sharpe_ratio', 'total_return', 'report_path']
# extra columns so the shorter histories are visible in the csv too
COVERAGE_COLUMNS = ['first_date', 'last_date', 'trading_days', 'short_coverage']


def six_complete_months(latest_date) -> tuple[str, str]:
    """Six calendar months before the month of the latest observation.

    e.g. latest 2024-12-06 -> ('2024-06-01', '2024-11-30'). Same rule even if the
    latest date is a month end, since one trading date doesn't prove the month is complete.
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('no latest date, is ohlc_hw_cleaned empty?')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def load_tickers_and_latest(db_path: Path) -> tuple[list[str], str]:
    """Distinct tickers and MAX(ts), both from the same cleaned table."""
    uri = db_path.resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
        latest = pd.read_sql_query('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    return tickers, latest


def make_kernel() -> str:
    """Register a throwaway kernel that uses *this* python, so the reports run in the
    same environment as the script (same idea as the lecture's report_runtime helper)."""
    kernel_root = Path(tempfile.mkdtemp(prefix='hw2_kernel_'))
    kernel_name = 'hw2-report'
    kernel_dir = kernel_root / 'kernels' / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'HW2 report', 'language': 'python',
    }), encoding='utf-8')
    prior = os.environ.get('JUPYTER_PATH')
    os.environ['JUPYTER_PATH'] = str(kernel_root) + (os.pathsep + prior if prior else '')
    return kernel_name


def run_report(ticker: str, start: str, end: str, out_dir: Path, db_path: Path, kernel_name: str) -> Path:
    """Run the template for one ticker. Errors are raised so the caller can log them."""
    output = out_dir / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={'ticker': ticker, 'start_date': start, 'end_date': end, 'db_path': str(db_path.resolve())},
        kernel_name=kernel_name, cwd=str(HERE), progress_bar=False,
    )
    return output


def read_report(path: Path, ticker: str, start: str, end: str) -> dict:
    """Pull the glued metrics out of one finished report and check it's the right one."""
    scraps = sb.read_notebook(str(path)).scraps
    if 'report_complete' not in scraps or scraps['report_complete'].data is not True:
        raise ValueError(f'{path.name} did not finish (no report_complete scrap)')
    row = {name: scraps[name].data for name in ['ticker', 'start_date', 'end_date'] + SUMMARY_COLUMNS[1:-1] + COVERAGE_COLUMNS}
    if (row['ticker'], row['start_date'], row['end_date']) != (ticker, start, end):
        raise ValueError(f'{path.name} has the wrong ticker/window, maybe a stale file')
    # NaN was glued as None
    for key in ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']:
        row[key] = math.nan if row[key] is None else float(row[key])
    row['report_path'] = path.relative_to(HERE).as_posix()
    return row


def plot_risk_return(summary: pd.DataFrame, start: str, end: str, out_path: Path) -> None:
    """Scatter: x = mean rolling vol, y = total return, colour = Sharpe, top 5 by return labelled."""
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    full = summary[~summary['short_coverage']]
    short = summary[summary['short_coverage']]
    vmin, vmax = summary['sharpe_ratio'].min(), summary['sharpe_ratio'].max()
    common = dict(c=None, cmap='RdYlGn', vmin=vmin, vmax=vmax, s=110, edgecolors='black', linewidths=0.8)
    points = None
    if not full.empty:
        points = ax.scatter(full['avg_volatility'] * 100, full['total_return'] * 100,
                            **{**common, 'c': full['sharpe_ratio']}, marker='o', label='full window')
    if not short.empty:
        points = ax.scatter(short['avg_volatility'] * 100, short['total_return'] * 100,
                            **{**common, 'c': short['sharpe_ratio']}, marker='^',
                            label='data ends early (shorter window)')
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(6, 6), textcoords='offset points', fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.grid(True, alpha=0.4)
    ax.margins(0.12)
    # gray legend markers, the colour only means Sharpe
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='lightgray', mec='black', ms=10, label='full window'),
                       Line2D([], [], marker='^', ls='', color='lightgray', mec='black', ms=10,
                              label='data ends early (shorter window)')], loc='best')
    ax.set(xlabel='Average 20-day rolling volatility, annualized (%)', ylabel='Total return (%)')
    ax.set_title('Risk-Return Profile of Portfolio', fontsize=16, fontweight='bold')
    ax.text(0.01, 0.01, f'requested window {start} to {end}. triangles = history ends before the window does, '
            'so their returns cover fewer days', transform=ax.transAxes, fontsize=9, color='dimgray')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate HW2 stock reports for every ticker.')
    parser.add_argument('--db', type=Path, default=DEFAULT_DB, help='working database with ohlc_hw_cleaned')
    parser.add_argument('--tickers', nargs='*', help='only run these tickers (for testing)')
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.is_file():
        print(f'no database at {db_path}. run 01_data_cleaning.ipynb first.')
        return 1

    tickers, latest = load_tickers_and_latest(db_path)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers]
    start, end = six_complete_months(latest)
    run_dir = HERE / 'reports' / date.today().isoformat()   # run date, not the analysis dates
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f'latest date in ohlc_hw_cleaned: {latest}')
    print(f'analysis window (last 6 complete months): {start} to {end}')
    print(f'{len(tickers)} tickers -> {run_dir}')

    kernel_name = make_kernel()
    rows, failures = [], []
    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            path = run_report(ticker, start, end, run_dir, db_path, kernel_name)
            rows.append(read_report(path, ticker, start, end))
        except Exception as exc:  # one bad ticker shouldn't stop the batch
            message = str(exc).strip().splitlines()
            short = next((line for line in reversed(message) if line.strip()), type(exc).__name__)
            failures.append({'ticker': ticker, 'error': f'{type(exc).__name__}: {short}'[:300],
                             'traceback': traceback.format_exc()})
            tqdm.write(f'FAILED {ticker}: {failures[-1]["error"]}')

    print(f'Successfully generated {len(rows)}/{len(tickers)} reports')
    (run_dir / 'failed_stocks.txt').write_text(
        ''.join(f"{f['ticker']}\t{f['error']}\n" for f in failures), encoding='utf-8')
    if failures:
        (run_dir / 'failed_stocks_traceback.log').write_text(
            '\n\n'.join(f"=== {f['ticker']} ===\n{f['traceback']}" for f in failures), encoding='utf-8')
        print(f'failed tickers saved to {run_dir / "failed_stocks.txt"}')

    if not rows:
        print('every report failed, so there is no summary to build. check failed_stocks.txt')
        return 1

    summary = pd.DataFrame(rows)
    if summary.duplicated('ticker').any():
        raise ValueError('duplicate tickers in this run')
    summary_df = summary[SUMMARY_COLUMNS + COVERAGE_COLUMNS].sort_values(
        'sharpe_ratio', ascending=False, na_position='last').reset_index(drop=True)
    summary_df.to_csv(run_dir / 'portfolio_summary.csv', index=False)

    ranked = summary_df.dropna(subset=['sharpe_ratio'])   # undefined Sharpe can't be ranked
    show = ['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown', 'last_date']
    pd.set_option('display.width', 140)
    print('\ntop 10 by Sharpe:')
    print(ranked.head(10)[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nbottom 10 by Sharpe:')
    print(ranked.tail(10)[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    if len(ranked) < 20:
        print('(fewer than 20 ranked stocks, so top and bottom overlap)')
    n_short = int(summary_df['short_coverage'].sum())
    if n_short:
        print(f'\nnote: {n_short} tickers have data ending before {end}, their metrics cover a shorter period')
    if len(ranked) < len(summary_df):
        print('undefined Sharpe (left out of the ranking):', summary_df.loc[summary_df['sharpe_ratio'].isna(), 'ticker'].tolist())

    plot_risk_return(summary_df, start, end, run_dir / 'risk_return_profile.png')
    print(f'\nsaved {run_dir / "portfolio_summary.csv"} and {run_dir / "risk_return_profile.png"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
