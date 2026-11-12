"""Execute every cleaned ticker with Papermill and collect only current successes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import logging
import matplotlib
matplotlib.use("Agg")  # Batch output does not require a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import nbformat
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
FIELDS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
          'sharpe_ratio', 'total_return', 'report_path', 'actual_start', 'actual_end', 'observations']


def analysis_window(latest):
    """Six calendar months before the month containing the latest observation."""
    month = pd.Timestamp(latest).normalize().replace(day=1)
    return (month - pd.DateOffset(months=6)).strftime('%Y-%m-%d'), (month - pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def collect_output(path):
    """Read the one tagged JSON output from a successful executed notebook."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if 'summary-output' in cell.metadata.get('tags', []):
            for output in cell.get('outputs', []):
                if output.output_type == 'stream':
                    for line in output.text.splitlines():
                        if line.startswith('HW2_SUMMARY='):
                            return json.loads(line.removeprefix('HW2_SUMMARY='))
    raise ValueError('Executed report has no named HW2_SUMMARY output')


def run(db_path, output_root, kernel_name='python3', selected_tickers=None):
    """Run all or selected cleaned tickers, isolate failures and save summaries."""
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f'{db_path}: run 01_data_cleaning.ipynb first')
    with sqlite3.connect(db_path) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn).ticker.tolist()
        latest = conn.execute('SELECT MAX(ts) FROM ohlc_hw_cleaned').fetchone()[0]
    if latest is None:
        raise ValueError('Cleaned table is empty')
    if selected_tickers is not None:
        requested = list(dict.fromkeys(t.strip().upper() for t in selected_tickers))
        unknown = sorted(set(requested) - set(tickers))
        if unknown:
            raise ValueError(f'Tickers not found in cleaned data: {", ".join(unknown)}')
        if not requested:
            raise ValueError('Select at least one ticker')
        tickers = requested
    start, end = analysis_window(latest)
    out = Path(output_root).resolve() / datetime.now().strftime('%Y-%m-%d')
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out / 'batch.log', level=logging.INFO, force=True)
    successes, failures = [], []
    print(f'Common requested window: {start} to {end}. Actual coverage may differ by ticker.')
    for number, ticker in enumerate(tqdm(tickers, desc='Generating reports')):
        # Encode unsafe ticker characters to prevent paths escaping the output directory.
        from urllib.parse import quote
        filename = quote(ticker, safe='')
        if filename in {'.', '..'}:
            filename = f'ticker_{number}'
        path = out / f'{filename}_report.ipynb'
        try:
            pm.execute_notebook(str(BASE / '02_stock_report_template.ipynb'), str(path),
                                parameters={'ticker': ticker, 'start_date': start, 'end_date': end, 'db_path': str(db_path)},
                                cwd=str(BASE), kernel_name=kernel_name, progress_bar=False)
            result = collect_output(path)
            result['report_path'] = str(path.relative_to(out.parent.parent)) if path.is_relative_to(out.parent.parent) else str(path)
            successes.append(result)
        except Exception as error:
            failures.append(ticker)
            logging.exception('Failed %s', ticker)
            print(f'{ticker}: {type(error).__name__}: {error}')
    (out / 'failed_stocks.txt').write_text(''.join(f'{ticker}\n' for ticker in failures))
    summary = pd.DataFrame(successes, columns=FIELDS)
    for column in ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']:
        summary[column] = pd.to_numeric(summary[column], errors='coerce')
    summary = summary.sort_values('sharpe_ratio', ascending=False, na_position='last')
    summary.to_csv(out / 'portfolio_summary.csv', index=False)
    (out / 'run_metadata.json').write_text(json.dumps({'requested_start': start, 'requested_end': end,
        'database_latest': latest, 'attempted': len(tickers), 'successful': len(successes), 'failed': failures}, indent=2))
    print(f'Successfully generated {len(successes)}/{len(tickers)} reports')
    if summary.empty:
        print('All reports failed: no portfolio comparison is available. CSV contains headers only.')
        return summary
    ranked = summary.loc[np.isfinite(summary.sharpe_ratio)]
    print('Top 10 (finite Sharpe only):\n', ranked.head(10).to_string(index=False))
    print('Bottom 10 (groups may overlap):\n', ranked.tail(10).to_string(index=False))
    fig, ax = plt.subplots(figsize=(11, 7))
    finite_xy = np.isfinite(summary.avg_volatility) & np.isfinite(summary.total_return)
    colored = finite_xy & np.isfinite(summary.sharpe_ratio)
    if colored.any():
        points = ax.scatter(summary.loc[colored, 'avg_volatility'], summary.loc[colored, 'total_return'],
                            c=summary.loc[colored, 'sharpe_ratio'], cmap='viridis', s=65)
        fig.colorbar(points, ax=ax, label='Annualized Sharpe ratio')
    unknown = finite_xy & ~colored
    if unknown.any():
        ax.scatter(summary.loc[unknown, 'avg_volatility'], summary.loc[unknown, 'total_return'], color='gray', label='Undefined Sharpe')
        ax.legend()
    for row in summary.loc[finite_xy].nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(5, 5), textcoords='offset points')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Mean 20-day annualized rolling volatility', ylabel='Total adjusted-price return')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'portfolio_risk_return.png', dpi=150)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-path', type=Path, default=BASE / 'data/data.db')
    parser.add_argument('--output-root', type=Path, default=BASE / 'reports')
    parser.add_argument('--kernel-name', default='python3')
    parser.add_argument('--tickers', nargs='+', help='Space-separated tickers to test; defaults to all cleaned tickers')
    args = parser.parse_args()
    run(args.db_path, args.output_root, args.kernel_name, args.tickers)
