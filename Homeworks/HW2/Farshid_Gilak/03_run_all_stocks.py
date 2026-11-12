"""HW2 Part 3: generate a risk report for every stock and summarize the portfolio.

For each ticker in ``ohlc_hw_cleaned`` this script executes ``02_stock_report_template.ipynb``
with papermill over one common window: the six complete calendar months before the month that
contains the latest cleaned observation. It then collects the named results (Scrapbook scraps)
from the reports that succeeded in *this* run, and writes a portfolio summary CSV and a
risk-return chart.

Run from anywhere, after ``01_data_cleaning.ipynb`` has created the cleaned table::

    python 03_run_all_stocks.py                      # all tickers, run date = today
    python 03_run_all_stocks.py --tickers AAPL MSFT  # a subset, e.g. for a quick test

Outputs, in ``reports/{run date}/``:

* ``{TICKER}_report.ipynb``   one executed report per ticker
* ``portfolio_summary.csv``   one row per successful report; metrics stored as fractions
* ``risk_return_profile.png`` average volatility vs total return, coloured by Sharpe ratio
* ``failed_stocks.txt``       one failed ticker per line, with its error (empty if none failed)
* ``run.log``                 the full log of the run

The process exits with status 1 when no report succeeds, in which case no summary is written.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
from contextlib import chdir, closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # write image files; never open a window

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HW_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HW_DIR / '02_stock_report_template.ipynb'
DEFAULT_DB = HW_DIR / 'data' / 'data.db'
DEFAULT_OUTPUT_ROOT = HW_DIR / 'reports'
CLEAN_TABLE = 'ohlc_hw_cleaned'

# Columns required by the assignment, followed by coverage columns that disclose each report's actual period.
SUMMARY_COLUMNS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio',
                   'total_return', 'report_path']
COVERAGE_COLUMNS = ['first_date', 'last_date', 'trading_days', 'covers_requested_end', 'repaired_rows']
REQUIRED_SCRAPS = ['ticker', 'start_date', 'end_date', 'data_quality_ok', 'avg_daily_return', 'avg_volatility',
                   'max_drawdown', 'sharpe_ratio', 'total_return', *COVERAGE_COLUMNS]

log = logging.getLogger('hw2.batch')


def portable_path(path: Path) -> str:
    """Path relative to the homework folder when possible, so the CSV works on any machine."""
    path = path.resolve()
    return path.relative_to(HW_DIR).as_posix() if path.is_relative_to(HW_DIR) else str(path)


# ---------------------------------------------------------------------------
# Inputs: database, tickers, analysis window
# ---------------------------------------------------------------------------

def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite database read-only, so the batch can never modify it."""
    if not db_path.is_file():
        raise FileNotFoundError(f'Database not found: {db_path}. Run 01_data_cleaning.ipynb first.')
    return sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)


def load_tickers(db_path: Path) -> tuple[list[str], str]:
    """Return the sorted tickers in the cleaned table and its latest observation date (YYYY-MM-DD)."""
    with closing(connect_read_only(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if CLEAN_TABLE not in tables:
            raise RuntimeError(f'{db_path.name} has no {CLEAN_TABLE} table; run 01_data_cleaning.ipynb first.')
        tickers = [row[0] for row in conn.execute(f'SELECT DISTINCT ticker FROM {CLEAN_TABLE} ORDER BY ticker')]
        latest = conn.execute(f'SELECT MAX(ts) FROM {CLEAN_TABLE}').fetchone()[0]
    return tickers, latest


def six_complete_months(latest_observation: str | None) -> tuple[str, str]:
    """Six calendar months before the month containing the latest observation.

    The month of the latest observation is always excluded, even when that observation falls on
    a month-end: a final trading date alone does not prove the month's delivery is complete.

    >>> six_complete_months('2024-12-06')
    ('2024-06-01', '2024-11-30')
    >>> six_complete_months('2025-01-01')
    ('2024-07-01', '2024-12-31')
    >>> six_complete_months('2024-03-31')
    ('2023-09-01', '2024-02-29')
    >>> six_complete_months(None)
    Traceback (most recent call last):
    ...
    ValueError: The cleaned table has no observations to anchor the window.
    """
    if latest_observation is None or pd.isna(latest_observation):
        raise ValueError('The cleaned table has no observations to anchor the window.')
    month_start = pd.Timestamp(latest_observation).to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return f'{start:%Y-%m-%d}', f'{end:%Y-%m-%d}'


def check_template(template: Path) -> None:
    """Fail early unless the template exists and has a cell tagged 'parameters' for papermill."""
    if not template.is_file():
        raise FileNotFoundError(f'Report template not found: {template}')
    notebook = nbformat.read(template, as_version=4)
    if not any('parameters' in cell.metadata.get('tags', []) for cell in notebook.cells):
        raise ValueError(f'{template.name} has no cell tagged "parameters"; papermill cannot inject values.')


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def register_kernel(kernel_dir: Path) -> str:
    """Register a temporary Jupyter kernel that runs *this* Python interpreter.

    Reports then use the same environment as the script, independent of whatever kernels the
    user has installed. The kernel spec lives in ``kernel_dir`` and is found via JUPYTER_PATH.
    """
    name = 'hw2-report'
    spec_dir = kernel_dir / 'kernels' / name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'HW2 report (batch)', 'language': 'python',
    }), encoding='utf-8')
    prior = os.environ.get('JUPYTER_PATH')
    os.environ['JUPYTER_PATH'] = str(kernel_dir) + (os.pathsep + prior if prior else '')
    return name


def run_report(ticker: str, start: str, end: str, run_dir: Path, *, template: Path, db_path: Path,
               kernel_name: str, timeout: int) -> Path:
    """Execute the template for one ticker and return the output path; any failure propagates.

    Papermill records the input, output and parameter paths inside each report. They are passed
    relative to the homework folder (and papermill runs from there), so the committed reports do
    not contain this machine's absolute paths.
    """
    output = run_dir / f'{ticker}_report.ipynb'
    output.unlink(missing_ok=True)  # never leave an older report in place of this run's attempt
    # Resolve against the caller's working directory before switching to the homework folder.
    template_rel, output_rel, db_rel = portable_path(template), portable_path(output), portable_path(db_path)
    with chdir(HW_DIR):
        pm.execute_notebook(
            template_rel, output_rel,
            parameters={'ticker': ticker, 'start_date': start, 'end_date': end, 'db_path': db_rel},
            kernel_name=kernel_name, cwd=str(HW_DIR), progress_bar=False, execution_timeout=timeout,
        )
    return output


def describe_error(error: Exception) -> str:
    """One-line error description; for a failed notebook, the exception raised inside it."""
    if isinstance(error, pm.PapermillExecutionError):
        return f'{error.ename}: {error.evalue}'
    return f'{type(error).__name__}: {error}'


def run_batch(tickers: list[str], start: str, end: str, run_dir: Path, **report_options) \
        -> tuple[dict[str, Path], dict[str, str]]:
    """Run every ticker, continuing past failures. Returns ({ticker: report path}, {ticker: error})."""
    succeeded, failed = {}, {}
    for ticker in tqdm(tickers, desc='Generating reports', unit='report'):
        try:
            succeeded[ticker] = run_report(ticker, start, end, run_dir, **report_options)
            log.debug('%s: report written to %s', ticker, succeeded[ticker].name)
        except Exception as error:  # noqa: BLE001 - one bad ticker must not stop the batch
            failed[ticker] = describe_error(error)
            log.warning('%s: report failed (%s)', ticker, failed[ticker])
    return succeeded, failed


# ---------------------------------------------------------------------------
# Collection and summary
# ---------------------------------------------------------------------------

def collect_summary(succeeded: dict[str, Path], start: str, end: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build the portfolio summary from the scraps of this run's successful reports only.

    A report is rejected (and reported as failed) if a scrap is missing, if it describes a
    different ticker or window, or if its input data failed the quality checks.
    """
    rows, rejected = [], {}
    for ticker, path in succeeded.items():
        try:
            scraps = sb.read_notebook(str(path)).scraps
            missing = [name for name in REQUIRED_SCRAPS if name not in scraps]
            if missing:
                raise ValueError(f'missing scraps {missing}')
            row = {name: scraps[name].data for name in REQUIRED_SCRAPS}
            if (row['ticker'], row['start_date'], row['end_date']) != (ticker, start, end):
                raise ValueError(f'report describes {row["ticker"]} {row["start_date"]}..{row["end_date"]}')
            if not row['data_quality_ok']:
                raise ValueError('input data failed the report quality checks')
        except Exception as error:  # noqa: BLE001
            rejected[ticker] = f'collection failed: {describe_error(error)}'
            log.warning('%s: %s', ticker, rejected[ticker])
            continue
        row['report_path'] = portable_path(path)
        rows.append(row)

    summary = pd.DataFrame(rows, columns=[*REQUIRED_SCRAPS, 'report_path'])
    metric_cols = ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']
    summary[metric_cols] = summary[metric_cols].astype(float)  # undefined metrics (None) become NaN
    summary = summary[SUMMARY_COLUMNS + COVERAGE_COLUMNS]
    summary = summary.sort_values('sharpe_ratio', ascending=False, na_position='last').reset_index(drop=True)
    return summary, rejected


def rank_by_sharpe(summary: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top and bottom ``n`` by Sharpe ratio, excluding undefined Sharpe ratios.

    With fewer than 2n ranked stocks the two groups overlap; that is expected, not an error.
    """
    ranked = summary.dropna(subset=['sharpe_ratio']).sort_values('sharpe_ratio', ascending=False)
    return ranked.head(n), ranked.tail(n).iloc[::-1]


def format_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Percent strings for console output only; the CSV keeps fractions."""
    shown = frame[['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown',
                   'avg_daily_return', 'last_date']].copy()
    for col in ['total_return', 'avg_volatility', 'max_drawdown']:
        shown[col] = shown[col].map(lambda v: 'N/A' if pd.isna(v) else f'{v:+.2%}' if col != 'avg_volatility'
                                    else f'{v:.2%}')
    shown['avg_daily_return'] = shown['avg_daily_return'].map(lambda v: 'N/A' if pd.isna(v) else f'{v:+.3%}')
    shown['sharpe_ratio'] = shown['sharpe_ratio'].map(lambda v: 'N/A' if pd.isna(v) else f'{v:.2f}')
    return shown


def plot_risk_return(summary: pd.DataFrame, start: str, end: str, path: Path) -> None:
    """Scatter of mean rolling volatility vs total return, coloured by Sharpe ratio; top 5 returns labelled."""
    defined = summary.dropna(subset=['sharpe_ratio'])
    undefined = summary.loc[summary['sharpe_ratio'].isna()]
    fig, ax = plt.subplots(figsize=(11, 7.5), layout='constrained')

    # Diverging colours centred on a Sharpe of 0: blue = positive risk-adjusted return, red = negative.
    limit = max(float(defined['sharpe_ratio'].abs().max()), 0.5) if not defined.empty else 1.0
    points = ax.scatter(defined['avg_volatility'], defined['total_return'], c=defined['sharpe_ratio'],
                        cmap='RdBu', norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                        s=110, edgecolor='#2b2b2b', linewidth=0.7, zorder=3)
    fig.colorbar(points, ax=ax, label='Sharpe ratio (annualized, rf = 2%)')
    if not undefined.empty:
        ax.scatter(undefined['avg_volatility'], undefined['total_return'], s=110, facecolor='none',
                   edgecolor='#7a7a7a', linewidth=1.2, label='Sharpe undefined', zorder=3)
    # Mark shorter histories so they are not compared like-for-like with complete windows.
    short = summary.loc[~summary['covers_requested_end'].astype(bool)]
    if not short.empty:
        ax.scatter(short['avg_volatility'], short['total_return'], s=18, marker='x', color='#2b2b2b',
                   zorder=4, label='History ends before window end')
    if not (undefined.empty and short.empty):
        ax.legend(loc='lower right', frameon=True, facecolor='white')

    ax.axhline(0, color='#5f5f5f', linewidth=0.9)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.margins(x=0.12, y=0.12)
    ax.grid(True, alpha=0.6)
    ax.set_title('Risk-Return Profile of Portfolio', fontsize=16, fontweight='bold')
    ax.set_xlabel('Average 20-day rolling volatility (annualized)')
    ax.set_ylabel('Total return (adjusted close)')
    fig.supxlabel(f'Requested window {start} to {end}. {len(short)} of {len(summary)} stocks have histories '
                  'ending before the window end; their returns cover a shorter period (marked x).',
                  fontsize=9, color='#5f5f5f')
    label_top_returns(fig, ax, summary.nlargest(5, 'total_return'))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def label_top_returns(fig: plt.Figure, ax: plt.Axes, rows: pd.DataFrame) -> None:
    """Label each row's point with ticker and return, just to the right of the point.

    Labels are placed from the highest return down. A label that would overlap one already placed
    is pushed straight down below it and joined to its point by a thin leader line.
    """
    style = dict(fontsize=10, fontweight='bold', color='#2b2b2b')
    fig.canvas.draw()                      # fix the layout so pixel positions are final
    renderer = fig.canvas.get_renderer()
    to_points = 72 / fig.dpi               # pixels -> typographic points (annotation offsets)
    pad, gap = 10 / to_points, 3 / to_points
    placed = []                            # (left, right, centre_y) of placed labels, in pixels
    for row in rows.sort_values('total_return', ascending=False).itertuples():
        text = f'{row.ticker} {row.total_return:+.1%}'
        probe = ax.text(0, 0, text, **style)  # measure the label's size, then discard the probe
        size = probe.get_window_extent(renderer)
        probe.remove()
        px, py = ax.transData.transform((row.avg_volatility, row.total_return))
        left, right, y = px + pad, px + pad + size.width, py
        for other_left, other_right, other_y in sorted(placed, key=lambda item: -item[2]):
            if left < other_right and other_left < right and abs(y - other_y) < size.height + gap:
                y = other_y - size.height - gap
        placed.append((left, right, y))
        shift = (y - py) * to_points
        ax.annotate(text, (row.avg_volatility, row.total_return), xytext=(10, shift), textcoords='offset points',
                    va='center', **style,
                    arrowprops=dict(arrowstyle='-', color='#7a7a7a', lw=0.8, shrinkB=6) if abs(shift) > 3 else None)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class TqdmHandler(logging.Handler):
    """Console handler that prints above an active tqdm progress bar instead of through it."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


def configure_logging(run_dir: Path) -> None:
    """Log everything to run.log; show warnings and progress messages on the console."""
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    file_handler = logging.FileHandler(run_dir / 'run.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
    console = TqdmHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(file_handler)
    log.addHandler(console)
    logging.getLogger('papermill').setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options; every default points inside this homework folder."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--db-path', type=Path, default=DEFAULT_DB, help='database with ohlc_hw_cleaned')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE, help='parameterized report notebook')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT, help='folder for dated runs')
    parser.add_argument('--run-date', default=date.today().isoformat(), help='run folder name (YYYY-MM-DD)')
    parser.add_argument('--tickers', nargs='+', help='run only these tickers (default: all)')
    parser.add_argument('--kernel', help='existing Jupyter kernel to use (default: this interpreter)')
    parser.add_argument('--timeout', type=int, default=600, help='per-cell timeout in seconds')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the batch end to end; returns the process exit status."""
    args = parse_args(argv)
    date.fromisoformat(args.run_date)  # reject a malformed folder name early
    run_dir = args.output_root / args.run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)

    check_template(args.template)
    all_tickers, latest = load_tickers(args.db_path)
    start, end = six_complete_months(latest)
    tickers = all_tickers if not args.tickers else [t.strip().upper() for t in args.tickers]
    log.info('Latest cleaned observation: %s -> analysis window %s to %s (same for every ticker)', latest, start, end)
    log.info('Running %d reports into %s', len(tickers), portable_path(run_dir))

    # Outputs from an earlier run on the same date would be stale; remove them before starting.
    for name in ['portfolio_summary.csv', 'risk_return_profile.png', 'failed_stocks.txt']:
        (run_dir / name).unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix='hw2_kernel_') as kernel_dir:
        kernel_name = args.kernel or register_kernel(Path(kernel_dir))
        succeeded, failed = run_batch(tickers, start, end, run_dir, template=args.template, db_path=args.db_path,
                                      kernel_name=kernel_name, timeout=args.timeout)

    summary, rejected = collect_summary(succeeded, start, end)
    failed.update(rejected)
    (run_dir / 'failed_stocks.txt').write_text(
        ''.join(f'{ticker}\t{error}\n' for ticker, error in sorted(failed.items())), encoding='utf-8')
    log.info('Successfully generated %d/%d reports', len(summary), len(tickers))
    if failed:
        log.info('Failed tickers (see failed_stocks.txt): %s', ', '.join(sorted(failed)))

    if summary.empty:
        log.error('No successful reports: no portfolio summary was produced.')
        return 1

    summary.to_csv(run_dir / 'portfolio_summary.csv', index=False)
    plot_risk_return(summary, start, end, run_dir / 'risk_return_profile.png')

    top, bottom = rank_by_sharpe(summary)
    with pd.option_context('display.width', 140, 'display.max_columns', 20):
        log.info('\nTop %d by Sharpe ratio:\n%s', len(top), format_for_display(top).to_string(index=False))
        log.info('\nBottom %d by Sharpe ratio:\n%s', len(bottom), format_for_display(bottom).to_string(index=False))
    unranked = summary.loc[summary['sharpe_ratio'].isna(), 'ticker'].tolist()
    if unranked:
        log.info('Excluded from ranking (Sharpe undefined): %s', ', '.join(unranked))
    short = summary.loc[~summary['covers_requested_end'].astype(bool)]
    if not short.empty:
        log.info('\n%d of %d histories end before %s, so their metrics cover a shorter period '
                 '(last observation in parentheses): %s', len(short), len(summary), end,
                 ', '.join(f'{r.ticker} ({r.last_date})' for r in short.itertuples()))
    log.info('\nSaved %s and %s', 'portfolio_summary.csv', 'risk_return_profile.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
