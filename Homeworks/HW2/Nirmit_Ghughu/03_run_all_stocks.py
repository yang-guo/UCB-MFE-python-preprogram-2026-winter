"""HW2 Part 3: run the single-stock report for every ticker and summarize the portfolio.

Usage (from the repository root, after running 01_data_cleaning.ipynb):

    .venv/bin/python Homeworks/HW2/Nirmit_Ghughu/03_run_all_stocks.py
    .venv/bin/python Homeworks/HW2/Nirmit_Ghughu/03_run_all_stocks.py --tickers AAPL IBM MRK

Steps:
1. Read the ticker list and the latest observation date from ohlc_hw_cleaned.
2. Choose the analysis window: the six complete calendar months before the latest date's month.
3. Execute 02_stock_report_template.ipynb with papermill for each ticker into reports/{run date}/.
4. Log failures (failed_stocks.txt) and continue with the remaining tickers.
5. Collect scrapbook results from this run's successful reports only, then save
   portfolio_summary.csv and a risk/return scatter plot.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use('Agg')                    # the script only saves figures, it never opens a window
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HW_DIR = Path(__file__).resolve().parent
TEMPLATE = HW_DIR / '02_stock_report_template.ipynb'
DEFAULT_DB = HW_DIR / 'data' / 'data.db'
SUMMARY_COLS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
                'sharpe_ratio', 'total_return', 'report_path']


def six_complete_months(latest_date: str) -> tuple[str, str]:
    """Return the six calendar months before the month containing `latest_date` as ISO strings.

    The month of the latest observation is excluded even when that date is a month-end,
    because a final trading date alone does not prove the month was delivered completely.
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose a window: the cleaned table has no dates.')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def load_universe(db_path: Path) -> tuple[list[str], str]:
    """Read all tickers and the latest observation date from ohlc_hw_cleaned (read-only)."""
    if not db_path.is_file():
        raise FileNotFoundError(f'{db_path} not found. Run 01_data_cleaning.ipynb first.')
    with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
        latest = pd.read_sql_query('SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', conn)['latest'].iloc[0]
    return tickers, latest


def register_kernel() -> str:
    """Register a temporary Jupyter kernel that runs this same Python interpreter.

    The reports then use the packages of whatever environment launched the script,
    instead of whichever `python` happens to be first on PATH.
    """
    kernel_root = Path(tempfile.mkdtemp(prefix='hw2_kernel_'))
    kernel_name = 'hw2-report'
    kernel_dir = kernel_root / 'kernels' / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'HW2 report', 'language': 'python',
    }))
    prior = os.environ.get('JUPYTER_PATH')
    os.environ['JUPYTER_PATH'] = str(kernel_root) + (os.pathsep + prior if prior else '')
    return kernel_name


def run_report(ticker: str, start: str, end: str, db_path: Path, run_dir: Path, kernel_name: str) -> Path:
    """Execute the template for one ticker; papermill raises if any cell fails."""
    output = run_dir / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={'ticker': ticker, 'start_date': start, 'end_date': end, 'db_path': str(db_path)},
        kernel_name=kernel_name, cwd=str(HW_DIR), progress_bar=False,
    )
    return output


def collect_summary(report_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the portfolio summary from the scrapbook values of successful reports only.

    Returns the seven required summary columns and a coverage table (actual first/last dates).
    """
    rows = []
    for path in report_paths:
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in
               ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio',
                'total_return', 'first_date', 'last_date', 'trading_days']}
        row['report_path'] = str(path.relative_to(HW_DIR))
        rows.append(row)
    collected = pd.DataFrame(rows)
    if collected['ticker'].duplicated().any():
        raise ValueError('Duplicate tickers among collected reports.')
    numeric = ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']
    collected[numeric] = collected[numeric].astype(float)       # None (undefined) becomes NaN
    summary = (collected[SUMMARY_COLS]
               .sort_values('sharpe_ratio', ascending=False, na_position='last')
               .reset_index(drop=True))
    coverage = collected[['ticker', 'first_date', 'last_date', 'trading_days']].sort_values('ticker')
    return summary, coverage


def plot_risk_return(summary: pd.DataFrame, short: set[str], start: str, end: str, output: Path) -> None:
    """Scatter of mean rolling volatility vs. total return, colored by Sharpe; top 5 by return labeled."""
    fig, ax = plt.subplots(figsize=(11, 7))
    full = ~summary['ticker'].isin(short)
    points = None
    for mask, marker, label in [(full, 'o', 'Full window'), (~full, 's', 'Data ends before window end')]:
        part = summary[mask]
        if part.empty:
            continue
        points = ax.scatter(part['avg_volatility'], part['total_return'], c=part['sharpe_ratio'],
                            cmap='RdYlGn', vmin=summary['sharpe_ratio'].min(), vmax=summary['sharpe_ratio'].max(),
                            s=90, marker=marker, edgecolors='black', linewidths=0.6, label=label)
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in summary.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 6),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel='Mean 20-day rolling volatility (annualized)', ylabel='Total return',
           title=f'Risk-Return Profile of Portfolio\n{start} to {end} (top 5 by return labeled)')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', type=Path, default=DEFAULT_DB, help='database containing ohlc_hw_cleaned')
    parser.add_argument('--tickers', nargs='+', help='run only these tickers (default: all in the table)')
    args = parser.parse_args()
    db_path = args.db.expanduser().resolve()

    all_tickers, latest = load_universe(db_path)
    tickers = [t.strip().upper() for t in args.tickers] if args.tickers else all_tickers
    start, end = six_complete_months(latest)
    run_dir = HW_DIR / 'reports' / date.today().isoformat()      # run date, not the analysis window
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Latest cleaned observation: {latest} -> analysis window {start} to {end}')
    print(f'Writing {len(tickers)} reports to {run_dir.relative_to(HW_DIR)}/')

    kernel_name = register_kernel()
    successes, failures = [], []
    for ticker in tqdm(tickers, desc='Generating reports'):
        try:
            successes.append(run_report(ticker, start, end, db_path, run_dir, kernel_name))
        except Exception as exc:                                  # one bad ticker must not stop the batch
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            failures.append((ticker, message))
            tqdm.write(f'FAILED {ticker}: {message}')

    (run_dir / 'failed_stocks.txt').write_text(''.join(f'{t}\t{msg}\n' for t, msg in failures))
    print(f'\nSuccessfully generated {len(successes)}/{len(tickers)} reports')
    if failures:
        print(f'Failed tickers saved to {(run_dir / "failed_stocks.txt").relative_to(HW_DIR)}')
    if not successes:
        print('No successful reports, so there is no portfolio summary to build.')
        return 1

    summary, coverage = collect_summary(successes)
    csv_path = run_dir / 'portfolio_summary.csv'
    summary.to_csv(csv_path, index=False)                        # fractions, not formatted percentages

    end_ts = pd.Timestamp(end)
    short = coverage[pd.to_datetime(coverage['last_date']) < end_ts - pd.Timedelta(days=7)]
    if not short.empty:
        print(f'\nCoverage note: {len(short)} of {len(coverage)} stocks have data ending before {end}:')
        for last, group in short.groupby('last_date'):
            print(f'  ends {last}: {", ".join(group["ticker"])}')
        print('Their metrics describe a shorter period, so compare them with the others with care.')

    ranked = summary.dropna(subset=['sharpe_ratio'])
    excluded = summary.loc[summary['sharpe_ratio'].isna(), 'ticker'].tolist()
    if excluded:
        print(f'\nExcluded from ranking (undefined Sharpe): {", ".join(excluded)}')
    display_cols = ['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown', 'avg_daily_return']
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 120):
        print('\nTop 10 by Sharpe ratio:')
        print(ranked.head(10)[display_cols].to_string(index=False))
        print('\nBottom 10 by Sharpe ratio:')
        print(ranked.tail(10)[display_cols].to_string(index=False))

    plot_path = run_dir / 'portfolio_risk_return.png'
    plot_risk_return(summary.dropna(subset=['avg_volatility', 'total_return']), set(short['ticker']), start, end, plot_path)
    print(f'\nSaved {csv_path.relative_to(HW_DIR)} and {plot_path.relative_to(HW_DIR)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
