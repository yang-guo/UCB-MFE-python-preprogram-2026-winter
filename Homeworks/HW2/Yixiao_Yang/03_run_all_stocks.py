"""Execute the HW2 template for every cleaned ticker and collect successful outputs."""

import argparse
from contextlib import closing
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import shutil
from uuid import uuid4
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm


# Standalone utilities: no adjacent Python helper modules are needed.
from contextlib import contextmanager
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
import sys
import sqlite3
from matplotlib.ticker import PercentFormatter
GRAY = '#9aa3ad'


def open_database(path, readonly=True):
    """Open an existing DB without accidentally creating an empty SQLite file."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Supply the bCourses data.db; no synthetic fallback.')
    return sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True)


def save_notebook_atomic(notebook, path):
    """Publish a complete executed notebook by atomic rename, avoiding partial reads."""
    import nbformat
    target = Path(path)
    with NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                            dir=target.parent, delete=False) as temp:
        temp.write(nbformat.writes(notebook))
        temp.flush()
        os.fsync(temp.fileno())
        temp_path = Path(temp.name)
    temp_path.replace(target)


@contextmanager
def notebook_kernel():
    """Temporarily expose an ipykernel spec for this interpreter, then restore lookup."""
    with TemporaryDirectory(prefix='hw2-kernel-') as tmp:
        name = 'hw2-python'
        folder = Path(tmp) / 'kernels' / name
        folder.mkdir(parents=True)
        (folder / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'HW2 Python', 'language': 'python'}), encoding='utf-8')
        previous = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = tmp + (os.pathsep + previous if previous else '')
        try:
            yield name
        finally:
            if previous is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = previous


def plot_portfolio(summary, dataset_label='COURSE DATABASE'):
    """Risk/return scatter; undefined Sharpe shown gray and excluded from rankings."""
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    finite_xy = np.isfinite(summary[['avg_volatility', 'total_return']]).all(axis=1)
    shown = summary.loc[finite_xy]
    defined = np.isfinite(shown['sharpe_ratio'])
    if defined.any():
        points = ax.scatter(shown.loc[defined, 'avg_volatility'], shown.loc[defined, 'total_return'],
                            c=shown.loc[defined, 'sharpe_ratio'], cmap='viridis', s=85,
                            edgecolors='white', linewidths=.6)
        fig.colorbar(points, ax=ax, label='Annualized Sharpe ratio')
    if (~defined).any():
        ax.scatter(shown.loc[~defined, 'avg_volatility'], shown.loc[~defined, 'total_return'],
                   color=GRAY, s=75, label='Sharpe undefined')
        ax.legend(frameon=False)
    for i, row in enumerate(shown.nlargest(5, 'total_return').itertuples()):
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return),
                    xytext=(7, 8 + (i % 2) * 9), textcoords='offset points', fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0, color=GRAY, linewidth=.7)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Mean rolling annualized volatility',
           ylabel='Total adjusted-price return')
    ax.grid(alpha=.2)
    ax.margins(.2)
    fig.suptitle(f'{dataset_label} — individual stocks; not a weighted portfolio\n'
                 'Common requested window; actual coverage may differ (see CSV)', fontsize=10, color='#526273')
    return fig


BASE = Path(__file__).resolve().parent
SUMMARY_COLUMNS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
                   'sharpe_ratio', 'total_return', 'report_path', 'start_date', 'end_date',
                   'actual_start', 'actual_end', 'trading_days', 'return_count',
                   'annualized_return', 'annualized_volatility', 'dataset_label']


def six_complete_months(latest_date):
    """Six calendar months before the month containing the latest cleaned date."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('No latest date: cleaned table is empty.')
    month = latest.to_period('M').start_time
    return ((month - pd.DateOffset(months=6)).strftime('%Y-%m-%d'),
            (month - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))


def collect_report(path, ticker, start, end, run_id):
    """Read only a report that just executed successfully; verify identity and run."""
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if cell.execution_count is None or any(o.output_type == 'error' for o in cell.outputs):
                raise ValueError('Report contains an unexecuted or failed code cell.')
    scraps = sb.read_notebook(str(path)).scraps
    values = scraps['report_metrics'].data
    if scraps['run_id'].data != run_id:
        raise ValueError('Stale report run ID.')
    if (values['ticker'], values['start_date'], values['end_date']) != (ticker, start, end):
        raise ValueError('Report parameters do not match the current request.')
    if not values['data_quality_ok']:
        raise ValueError('Report failed quality validation.')
    values['report_path'] = path.name
    return {c: values[c] for c in SUMMARY_COLUMNS}


def run_batch(db_path, output_root, template=None, dataset_label='COURSE DATABASE', tickers=None):
    """Run all tickers independently; return (summary, failures, dated output directory).

    `tickers` is a test hook; CLI always queries all cleaned tickers. Existing same-day
    output directories are archived so failed/stale files cannot masquerade as this run.
    """
    db_path = Path(db_path).resolve()
    template = Path(template or BASE / '02_stock_report_template.ipynb').resolve()
    try:
        report_db_path = str(db_path.relative_to(BASE))
    except ValueError:
        report_db_path = str(db_path)
    with closing(open_database(db_path)) as conn:
        available = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
        latest = conn.execute('SELECT MAX(ts) FROM ohlc_hw_cleaned').fetchone()[0]
    start, end = six_complete_months(latest)
    symbols = available if tickers is None else list(tickers)
    root = Path(output_root).resolve()
    run_date = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    run_id = uuid4().hex
    output = root / run_date
    if output.exists():
        archive = root / '_archive'
        archive.mkdir(parents=True, exist_ok=True)
        output.rename(archive / f'{run_date}_{run_id[:10]}')
    output.mkdir(parents=True)
    logger = logging.getLogger('hw2.' + run_id)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / 'batch.log', encoding='utf-8')
    logger.addHandler(handler)
    rows, failures, successful = [], [], []
    print(f'Database latest: {latest}; requested {start} to {end}; output run date: {run_date}')
    try:
        with notebook_kernel() as kernel:
            for ticker in tqdm(symbols, desc='Generating reports'):
                try:
                    if not isinstance(ticker, str) or not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=_-]*', ticker):
                        raise ValueError('Ticker cannot safely be used as a report filename.')
                    path = output / f'{ticker}_report.ipynb'
                    working = output / f'.{ticker}_working.ipynb'
                    executed = pm.execute_notebook(str(template), str(working), parameters={
                        'ticker': ticker, 'start_date': start, 'end_date': end,
                        'db_path': report_db_path, 'dataset_label': dataset_label, 'run_id': run_id},
                        cwd=str(BASE), kernel_name=kernel, progress_bar=False)
                    save_notebook_atomic(executed, path)
                    row = collect_report(path, ticker, start, end, run_id)
                    working.unlink(missing_ok=True)
                    rows.append(row)
                    successful.append(path.name)
                    logger.info('SUCCESS %s', ticker)
                except Exception as exc:
                    failures.append({'ticker': ticker, 'error': str(exc)})
                    logger.exception('FAILED %s', ticker)
                    print(f'FAILED {ticker}: {type(exc).__name__}; see batch.log')
        summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        summary = summary.sort_values('sharpe_ratio', ascending=False, na_position='last').reset_index(drop=True)
        if not summary.empty:
            summary.to_csv(output / 'portfolio_summary.csv', index=False)
            ranked = summary.loc[np.isfinite(summary['sharpe_ratio'])]
            print('\nTop 10 (defined Sharpe only):\n', ranked.head(10).to_string(index=False))
            print('\nBottom 10 (defined Sharpe only):\n', ranked.tail(10).to_string(index=False))
            print('Top/bottom groups may overlap with fewer than 20 defined Sharpe values.')
            fig = plot_portfolio(summary, dataset_label)
            fig.savefig(output / 'portfolio_dashboard.png', dpi=140)
            plt.close(fig)
        else:
            print('No successful reports; no summary CSV or dashboard was produced.')
        (output / 'failed_stocks.txt').write_text(''.join(str(x['ticker']) + '\n' for x in failures), encoding='utf-8')
        (output / 'failed_stocks.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
        (output / 'manifest.json').write_text(json.dumps({
            'run_id': run_id, 'run_date': run_date, 'dataset_label': dataset_label,
            'latest_cleaned_date': latest, 'start_date': start, 'end_date': end,
            'attempted': len(symbols), 'successful_reports': successful, 'failures': failures,
            'summary_created': bool(len(summary)),
            'coverage_note': 'Actual first/last dates differ by ticker; no trailing extrapolation.'
        }, indent=2), encoding='utf-8')
        print(f'Successfully generated {len(summary)}/{len(symbols)} reports')
        return summary, failures, output
    finally:
        handler.close()
        logger.removeHandler(handler)


def execute_cleaning(db_path, source_db=None):
    """Run the self-contained cleaning notebook, then save its complete outputs."""
    target = Path(db_path).expanduser().resolve()
    if not target.is_relative_to(BASE):
        raise ValueError('With --clean, --db-path must be inside this homework directory.')
    params = {'db_path': str(target.relative_to(BASE)), 'audit_dir': 'outputs/cleaning'}
    if source_db is not None:
        source = Path(source_db).expanduser().resolve()
        params['source_db'] = str(Path(os.path.relpath(source, BASE)))
    working = BASE / 'outputs/01_data_cleaning_working.ipynb'
    working.parent.mkdir(parents=True, exist_ok=True)
    with notebook_kernel() as kernel:
        executed = pm.execute_notebook(str(BASE / '01_data_cleaning.ipynb'), str(working),
            parameters=params, cwd=str(BASE), kernel_name=kernel, progress_bar=False)
    for cell in executed.cells:
        if cell.cell_type == 'code' and (cell.execution_count is None or
                any(o.output_type == 'error' for o in cell.outputs)):
            raise ValueError('Cleaning notebook did not execute completely.')
    save_notebook_atomic(executed, BASE / '01_data_cleaning.ipynb')
    working.unlink(missing_ok=True)


def main():
    """Optionally clean first, then report all tickers from the local working DB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-path', type=Path, default=BASE / 'data/data.db')
    parser.add_argument('--output-root', type=Path, default=BASE / 'reports')
    parser.add_argument('--dataset-label', default='COURSE DATABASE')
    parser.add_argument('--clean', action='store_true', help='Execute the cleaning notebook before reports')
    parser.add_argument('--source-db', type=Path, help='Read-only original database; requires --clean')
    args = parser.parse_args()
    if args.source_db is not None and not args.clean:
        parser.error('--source-db requires --clean')
    if not args.db_path.is_absolute():
        args.db_path = BASE / args.db_path
    if not args.output_root.is_absolute():
        args.output_root = BASE / args.output_root
    if args.clean:
        execute_cleaning(args.db_path, args.source_db)
    summary, failures, _ = run_batch(args.db_path, args.output_root, dataset_label=args.dataset_label)
    return 2 if summary.empty else 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
