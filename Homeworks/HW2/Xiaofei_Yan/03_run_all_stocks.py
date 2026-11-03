"""Generate one Papermill risk report per stock in the cleaned homework database.

Run this script with the HW2 Python environment active, after executing Part 1 to
create `ohlc_hw_cleaned`. Report metrics are collected from a named JSON line
emitted by the corresponding successful report notebook; no existing report files
are scanned or reused.
"""
from contextlib import closing, contextmanager
from datetime import date
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import papermill as pm
from tqdm.auto import tqdm


# Resolve every input and output relative to this script, not the shell's current folder.
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / 'data' / 'data.db'
TEMPLATE_PATH = SCRIPT_DIR / '02_stock_report_template.ipynb'
REPORTS_ROOT = SCRIPT_DIR / 'reports'
# The report template prints one JSON record with this marker at the end of Part 2.
# This is a named, machine-readable alternative to adding a Scrapbook dependency.
METRICS_MARKER = 'HW2_METRICS_JSON='


def six_complete_months(latest_date):
    """Return the six complete calendar months before latest_date's month."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose an analysis window: the cleaned table has no dates.')
    # Exclude the month containing the latest observation: its data delivery may be partial.
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


@contextmanager
def current_python_kernel():
    """Expose this script's interpreter as a temporary Jupyter kernel for Papermill."""
    prior_jupyter_path = os.environ.get('JUPYTER_PATH')
    # Papermill starts a separate kernel; point it at the same Python used to run this script.
    # Keep the kernel spec temporary so this script does not change user-level Jupyter settings.
    with tempfile.TemporaryDirectory(prefix='hw2_report_kernel_') as temporary_root:
        kernel_root = Path(temporary_root)
        kernel_name = 'hw2-current-python'
        kernel_dir = kernel_root / 'kernels' / kernel_name
        kernel_dir.mkdir(parents=True)
        kernel_spec = {
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'HW2 current Python',
            'language': 'python',
        }
        (kernel_dir / 'kernel.json').write_text(json.dumps(kernel_spec), encoding='utf-8')
        search_paths = [str(kernel_root)]
        if prior_jupyter_path:
            search_paths.append(prior_jupyter_path)
        os.environ['JUPYTER_PATH'] = os.pathsep.join(search_paths)
        try:
            yield kernel_name
        finally:
            if prior_jupyter_path is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = prior_jupyter_path


def load_tickers_and_latest_date(database_path):
    """Read all cleaned tickers and the latest cleaned observation from one database."""
    if not database_path.is_file():
        raise FileNotFoundError(f'Local homework database not found: {database_path}')
    with closing(sqlite3.connect(database_path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", connection
        )['name'].tolist()
        if 'ohlc_hw_cleaned' not in tables:
            raise ValueError('Table ohlc_hw_cleaned is missing; run Part 1 first.')
        # Read symbols and the date anchor from the same cleaned table and database.
        tickers = pd.read_sql_query(
            'SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', connection
        )['ticker'].dropna().astype(str).tolist()
        latest_date = pd.read_sql_query(
            'SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned', connection
        )['latest'].iloc[0]
    tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not tickers or pd.isna(latest_date):
        raise ValueError('The cleaned table contains no tickers or observation dates.')
    return tickers, latest_date


def run_report(ticker, start_date, end_date, output_path, database_path, kernel_name):
    """Execute the single-stock template and return metrics from that exact output."""
    # Each ticker gets a fresh execution of the parameterized single-stock template.
    pm.execute_notebook(
        str(TEMPLATE_PATH),
        str(output_path),
        parameters={
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'db_path': str(database_path.resolve()),
        },
        kernel_name=kernel_name,
        cwd=str(SCRIPT_DIR),
        progress_bar=False,
    )
    return read_report_metrics(output_path)


def read_report_metrics(report_path):
    """Read the single named JSON metrics record emitted by a successful notebook."""
    # Only inspect the notebook just executed for this ticker; do not scan reports/ for files.
    notebook = nbformat.read(report_path, as_version=4)
    payloads = []
    # The marker makes the metric record identifiable even when a failed or stale notebook exists.
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.get('output_type') != 'stream':
                continue
            text = output.get('text', '')
            if isinstance(text, list):
                text = ''.join(text)
            payloads.extend(
                line[len(METRICS_MARKER):]
                for line in text.splitlines()
                if line.startswith(METRICS_MARKER)
            )
    if len(payloads) != 1:
        raise ValueError(f'Expected one named metrics record in {report_path.name}; found {len(payloads)}.')
    metrics = json.loads(payloads[0])
    required = {
        'ticker', 'start_date', 'end_date', 'actual_start_date', 'actual_end_date',
        'trading_days', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
        'sharpe_ratio', 'total_return',
    }
    if not required.issubset(metrics):
        missing = sorted(required - set(metrics))
        raise ValueError(f'Metrics record is missing fields: {missing}')
    return metrics


def generate_portfolio_plot(summary, output_path):
    """Save the portfolio risk-return scatter plot colored by Sharpe ratio."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    # A missing Sharpe ratio cannot be used as a color value; show those stocks in gray.
    defined_sharpe = summary['sharpe_ratio'].notna()
    points = None
    if defined_sharpe.any():
        points = ax.scatter(
            summary.loc[defined_sharpe, 'avg_volatility'] * 100,
            summary.loc[defined_sharpe, 'total_return'] * 100,
            c=summary.loc[defined_sharpe, 'sharpe_ratio'],
            cmap='viridis', s=55, alpha=0.85, edgecolor='white', linewidth=0.5,
        )
        fig.colorbar(points, ax=ax, label='Sharpe ratio')
    if (~defined_sharpe).any():
        ax.scatter(
            summary.loc[~defined_sharpe, 'avg_volatility'] * 100,
            summary.loc[~defined_sharpe, 'total_return'] * 100,
            color='gray', marker='x', s=55, label='Sharpe ratio N/A',
        )
        ax.legend()
    # Label the five highest total returns, as requested for the dashboard.
    for row in summary.nlargest(5, 'total_return').itertuples():
        if pd.notna(row.avg_volatility) and pd.notna(row.total_return):
            ax.annotate(
                row.ticker,
                (row.avg_volatility * 100, row.total_return * 100),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
            )
    ax.set(
        title='Risk-Return Profile of Portfolio',
        xlabel='Mean rolling volatility, annualized (%)',
        ylabel='Total return (%)',
    )
    ax.margins(x=0.12, y=0.12)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def print_sharpe_rankings(summary):
    """Print the top and bottom ten among reports with defined Sharpe ratios."""
    ranked = summary.dropna(subset=['sharpe_ratio']).sort_values(
        'sharpe_ratio', ascending=False
    )
    if ranked.empty:
        print('No reports have a defined Sharpe ratio; rankings are unavailable.')
        return
    columns = ['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown']
    print('\nTop 10 stocks by Sharpe ratio:')
    print(ranked.head(10)[columns].to_string(index=False))
    print('\nBottom 10 stocks by Sharpe ratio:')
    print(ranked.tail(10)[columns].sort_values('sharpe_ratio').to_string(index=False))


def main():
    """Generate reports for all stocks and summarize this run's successful reports."""
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f'Stock report template not found: {TEMPLATE_PATH}')
    tickers, latest_date = load_tickers_and_latest_date(DB_PATH)
    start_date, end_date = six_complete_months(latest_date)
    # Folder date records when this batch ran; analysis dates remain database-anchored.
    run_date = date.today().isoformat()
    run_dir = REPORTS_ROOT / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = run_dir / 'portfolio_summary.csv'
    plot_path = run_dir / 'portfolio_summary.png'
    # A same-day all-failed rerun must not leave a stale summary looking current.
    summary_csv.unlink(missing_ok=True)
    plot_path.unlink(missing_ok=True)

    print(f'Latest cleaned observation: {latest_date}')
    print(f'Common analysis window: {start_date} to {end_date}')
    print(f'Report run date: {run_date}')
    print(
        'Coverage note: each report shows its actual date range. Metrics for shorter histories '
        'use only available observations and are not full-window comparisons; no trailing prices are extrapolated.'
    )

    # Append a row only after Papermill finishes and the named metrics record is read.
    successful_rows = []
    failures = []
    with current_python_kernel() as kernel_name:
        # Put every stock, including the first, inside the same error-handling path.
        for ticker in tqdm(tickers, desc='Generating reports'):
            safe_ticker = re.sub(r'[^A-Za-z0-9._-]+', '_', ticker)
            report_path = run_dir / f'{safe_ticker}_report.ipynb'
            try:
                metrics = run_report(
                    ticker, start_date, end_date, report_path, DB_PATH, kernel_name
                )
                if metrics['ticker'] != ticker:
                    raise ValueError(f'Report ticker mismatch: expected {ticker}, got {metrics["ticker"]}.')
                if metrics['start_date'] != start_date or metrics['end_date'] != end_date:
                    raise ValueError(f'{ticker} report did not use the common requested window.')
                row = {
                    'ticker': ticker,
                    'avg_daily_return': metrics['avg_daily_return'],
                    'avg_volatility': metrics['avg_volatility'],
                    'max_drawdown': metrics['max_drawdown'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_return': metrics['total_return'],
                    'report_path': str(report_path.relative_to(SCRIPT_DIR)),
                }
                successful_rows.append(row)
                print(
                    f'{ticker}: actual coverage {metrics["actual_start_date"]} '
                    f'to {metrics["actual_end_date"]} ({metrics["trading_days"]} observations).'
                )
            except Exception as error:
                message = ' '.join(str(error).split())
                failures.append({'ticker': ticker, 'error': message})
                print(f'FAILED: {ticker}: {message}')

    failure_file = run_dir / 'failed_stocks.txt'
    failure_file.write_text(
        ''.join(f"{failure['ticker']}: {failure['error']}\n" for failure in failures),
        encoding='utf-8',
    )
    print(f'Successfully generated {len(successful_rows)}/{len(tickers)} reports')

    # Do not create an empty or stale dashboard when every report failed.
    if not successful_rows:
        print('No reports succeeded; no portfolio summary was created.')
        return

    # Metrics stay as fractions in the CSV; percentages are used only in the plot labels.
    summary_df = pd.DataFrame(successful_rows)
    summary_df['sharpe_ratio'] = pd.to_numeric(summary_df['sharpe_ratio'], errors='coerce')
    summary_df = summary_df.sort_values('sharpe_ratio', ascending=False, na_position='last')
    summary_df.to_csv(summary_csv, index=False)
    print_sharpe_rankings(summary_df)
    generate_portfolio_plot(summary_df, plot_path)
    print(f'Portfolio summary CSV: {summary_csv}')
    print(f'Risk-return plot: {plot_path}')
    print(f'Failed ticker log: {failure_file}')


if __name__ == '__main__':
    main()
