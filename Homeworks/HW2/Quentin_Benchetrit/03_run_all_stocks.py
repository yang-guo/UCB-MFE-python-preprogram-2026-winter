"""HW2 Part 3: run the stock report for every ticker of ohlc_hw_cleaned and summarize the results.

Run with `python 03_run_all_stocks.py` after 01_data_cleaning.ipynb. Output goes to reports/<run date>/.
"""
import json
import logging
import os
import sqlite3
import sys
import tempfile
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
DB_PATH = Path('data/data.db')
TEMPLATE = Path('02_stock_report_template.ipynb')
KERNEL = 'hw2-reports'
COLUMNS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']

log = logging.getLogger(__name__)


def six_complete_months(latest_date):
    """The six calendar months before the month of the latest date, e.g. 2024-12-06 -> 2024-06-01, 2024-11-30."""
    month_start = pd.Timestamp(latest_date).to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return f'{start:%Y-%m-%d}', f'{end:%Y-%m-%d}'


def use_this_python(folder):
    """Kernel spec pointing to the interpreter running this script, visible to this process only."""
    kernel_dir = Path(folder) / 'kernels' / KERNEL
    kernel_dir.mkdir(parents=True)
    spec = {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': KERNEL, 'language': 'python'}
    (kernel_dir / 'kernel.json').write_text(json.dumps(spec))
    os.environ['JUPYTER_PATH'] = os.pathsep.join(filter(None, [str(folder), os.environ.get('JUPYTER_PATH')]))


def run_reports(tickers, start, end, run_dir):
    """Run the template for each ticker; a failure is logged and the loop continues."""
    successes, failures = [], []
    for ticker in tqdm(tickers, desc='Generating reports'):
        output = run_dir / f'{ticker}_report.ipynb'
        try:
            pm.execute_notebook(str(TEMPLATE), str(output), kernel_name=KERNEL, cwd='.', progress_bar=False,
                                parameters={'ticker': ticker, 'start_date': start, 'end_date': end,
                                            'db_path': str(DB_PATH)})
            successes.append(output)
        except Exception as exc:
            failures.append(ticker)
            log.error('%s failed: %s', ticker, getattr(exc, 'evalue', exc))
    return successes, failures


def collect(report_paths):
    """Summary table from the saved results of the successful reports, sorted by Sharpe ratio."""
    rows = []
    for path in report_paths:
        scraps = sb.read_notebook(str(path)).scraps
        rows.append({name: scraps[name].data for name in COLUMNS + ['last_date']} | {'report_path': path.as_posix()})
    summary = pd.DataFrame(rows)
    summary[COLUMNS[1:]] = summary[COLUMNS[1:]].astype(float)
    return summary.sort_values('sharpe_ratio', ascending=False, ignore_index=True)


def plot_risk_return(summary, path):
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    points = ax.scatter(summary['avg_volatility'] * 100, summary['total_return'] * 100, c=summary['sharpe_ratio'],
                        cmap='RdYlGn', s=80, edgecolors='black', linewidths=0.5)
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100), xytext=(5, 5),
                    textcoords='offset points', fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Average 20-day volatility, annualized (%)',
           ylabel='Total return (%)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.chdir(HW_DIR)
    with closing(sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)) as conn:
        tickers = [row[0] for row in conn.execute('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker')]
        latest = conn.execute('SELECT MAX(ts) FROM ohlc_hw_cleaned').fetchone()[0]
    start, end = six_complete_months(latest)
    run_dir = Path('reports') / f'{date.today():%Y-%m-%d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    for handler in (logging.FileHandler(run_dir / 'run.log', mode='w'), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        log.addHandler(handler)
    log.info('%d tickers, window %s to %s (latest date %s)', len(tickers), start, end, latest)

    with tempfile.TemporaryDirectory() as kernel_home:
        use_this_python(kernel_home)
        successes, failures = run_reports(tickers, start, end, run_dir)
    (run_dir / 'failed_stocks.txt').write_text(''.join(f'{ticker}\n' for ticker in failures))
    print(f'Successfully generated {len(successes)}/{len(tickers)} reports')
    if not successes:
        print('No successful report, nothing to summarize.')
        return 1

    summary = collect(successes)
    summary[COLUMNS + ['report_path']].to_csv(run_dir / 'portfolio_summary.csv', index=False)
    plot_risk_return(summary, run_dir / 'risk_return_profile.png')

    ranked = summary.dropna(subset=['sharpe_ratio'])[COLUMNS]
    print(f'\nTop 10 by Sharpe ratio\n{ranked.head(10).round(4).to_string(index=False)}')
    print(f'\nBottom 10 by Sharpe ratio\n{ranked.tail(10).round(4).to_string(index=False)}')
    short = summary.loc[summary['last_date'] < summary['last_date'].max(), 'ticker']
    if len(short):
        print(f'\nShorter history, no data after {summary["last_date"].min()}: {", ".join(short)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
