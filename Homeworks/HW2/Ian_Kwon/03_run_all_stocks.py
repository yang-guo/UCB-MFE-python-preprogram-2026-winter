"""Generate a risk report for every ticker in ohlc_hw_cleaned and summarize the portfolio.

Usage (from any folder, with the course environment active):

    python 03_run_all_stocks.py                      # all tickers
    python 03_run_all_stocks.py --tickers AAPL MSFT  # a subset, e.g. for testing

Steps:
1. Read the ticker list and the latest observation date from ohlc_hw_cleaned.
2. Choose the analysis window: the six complete calendar months before the month that
   contains the latest cleaned observation (anchored to the data, not the computer clock).
3. Execute 02_stock_report_template.ipynb once per ticker with papermill, saving
   reports/{run_date}/{ticker}_report.ipynb. A failure is logged and the loop continues.
4. Collect the metrics each *successful* report saved with scrapbook, then write
   portfolio_summary.csv and a risk-return scatter plot.

Run 01_data_cleaning.ipynb first: it creates data/data.db with the ohlc_hw_cleaned table.
"""
from __future__ import annotations

import argparse
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

matplotlib.use('Agg')  # write figures to files; no display needed
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HW_DIR = Path(__file__).resolve().parent
TEMPLATE = HW_DIR / '02_stock_report_template.ipynb'
DEFAULT_DB = HW_DIR / 'data' / 'data.db'
REPORTS_ROOT = HW_DIR / 'reports'

# Required columns first (the assignment's summary_df), then coverage columns for disclosure.
SUMMARY_COLUMNS = ['ticker', 'avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio',
                   'total_return', 'report_path']
COVERAGE_COLUMNS = ['first_date', 'last_date', 'trading_days']

log = logging.getLogger('run_all_stocks')


def six_complete_months(latest_date) -> tuple[str, str]:
    """Six calendar months before the month containing `latest_date`, as ISO strings.

    Example: latest 2024-12-06 -> ('2024-06-01', '2024-11-30'). The rule is the same when the
    latest date is a month-end, because a final trading date alone does not prove that the
    month's delivery is complete.
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose a window from an empty date range.')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def load_universe(db_path: Path) -> tuple[list[str], str]:
    """Return (sorted tickers, latest ts) from ohlc_hw_cleaned in one read-only connection."""
    if not db_path.is_file():
        raise FileNotFoundError(f'{db_path} not found: run 01_data_cleaning.ipynb first.')
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        tickers = [row[0] for row in conn.execute('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker')]
        latest = conn.execute('SELECT MAX(ts) FROM ohlc_hw_cleaned').fetchone()[0]
    if not tickers or latest is None:
        raise ValueError('ohlc_hw_cleaned is empty.')
    return tickers, latest


def last_session_in_window(db_path: Path, end: str) -> str:
    """Latest trading date on or before `end` for any ticker: the last session a full-window report can have."""
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        return conn.execute('SELECT MAX(ts) FROM ohlc_hw_cleaned WHERE ts <= ?', (end,)).fetchone()[0]


def register_kernel(kernel_root: Path) -> str:
    """Register a temporary Jupyter kernel that runs *this* Python interpreter.

    Papermill's default 'python3' kernel may point at a different Python installation that
    lacks pandas or scrapbook. The kernel spec lives in a temporary folder and only affects
    this process (via JUPYTER_PATH); nothing is installed for the user.
    """
    kernel_name = 'hw2-report'
    kernel_dir = kernel_root / 'kernels' / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'HW2 report (current Python)',
        'language': 'python',
    }), encoding='utf-8')
    prior = os.environ.get('JUPYTER_PATH')
    os.environ['JUPYTER_PATH'] = str(kernel_root) + (os.pathsep + prior if prior else '')
    return kernel_name


def run_report(ticker: str, start: str, end: str, out_dir: Path, db_path: Path, kernel_name: str) -> Path:
    """Execute the template for one ticker and return the output path; errors propagate."""
    output = out_dir / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={'ticker': ticker, 'start_date': start, 'end_date': end, 'db_path': str(db_path)},
        kernel_name=kernel_name, cwd=str(HW_DIR), progress_bar=False,
    )
    return output


def short_error(exc: Exception) -> str:
    """One-line description of a report failure (papermill errors carry the notebook exception)."""
    if isinstance(exc, pm.PapermillExecutionError):
        return f'{exc.ename}: {exc.evalue}'
    return f'{type(exc).__name__}: {exc}'


def collect_summary(report_paths: dict[str, Path]) -> pd.DataFrame:
    """Build the portfolio summary from the scraps of successful reports only.

    Never globs the folder: failed or stale notebooks may be sitting next to the good ones.
    """
    rows = []
    for ticker, path in report_paths.items():
        scraps = sb.read_notebook(str(path)).scraps
        summary = scraps['report_summary'].data
        if summary['ticker'] != ticker:
            raise ValueError(f'{path} contains results for {summary["ticker"]}, expected {ticker}.')
        rows.append({**summary, 'report_path': str(path.relative_to(HW_DIR).as_posix())})
    summary = pd.DataFrame(rows)
    if summary['ticker'].duplicated().any():
        raise ValueError('Duplicate tickers among collected reports.')
    numeric = ['avg_daily_return', 'avg_volatility', 'max_drawdown', 'sharpe_ratio', 'total_return']
    summary[numeric] = summary[numeric].astype(float)  # JSON null -> NaN
    summary = summary[SUMMARY_COLUMNS + COVERAGE_COLUMNS]
    return summary.sort_values('sharpe_ratio', ascending=False, na_position='last').reset_index(drop=True)


def plot_risk_return(summary: pd.DataFrame, path: Path, start: str, end: str, last_session: str) -> None:
    """Scatter of mean rolling volatility vs total return, colored by Sharpe; top 5 returns labeled.

    Circles cover the whole window; triangles mark tickers whose data ends earlier.
    """
    data = summary.dropna(subset=['avg_volatility', 'total_return'])
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    sharpe = data['sharpe_ratio'].dropna()
    # Diverging colors centred on a Sharpe ratio of 0 (red = negative, green = positive).
    norm = (TwoSlopeNorm(vcenter=0, vmin=sharpe.min(), vmax=sharpe.max())
            if len(sharpe) and sharpe.min() < 0 < sharpe.max() else None)
    full_window = data['last_date'] >= last_session
    points = None
    for mask, marker, label in [(full_window, 'o', f'Data through {last_session}'),
                                (~full_window, '^', 'Data ends earlier (shorter period)')]:
        group = data.loc[mask]
        if group.empty:
            continue
        defined = group['sharpe_ratio'].notna()
        points = ax.scatter(group.loc[defined, 'avg_volatility'], group.loc[defined, 'total_return'],
                            c=group.loc[defined, 'sharpe_ratio'], cmap='RdYlGn', norm=norm, marker=marker,
                            s=100, edgecolors='#333333', linewidths=0.6, zorder=3)
        ax.scatter(group.loc[~defined, 'avg_volatility'], group.loc[~defined, 'total_return'],
                   color='lightgray', marker=marker, edgecolors='#333333', s=100, zorder=3)
        ax.scatter([], [], marker=marker, color='white', edgecolors='#333333', s=100, label=label)
    if data['sharpe_ratio'].isna().any():
        ax.scatter([], [], color='lightgray', edgecolors='#333333', s=100, label='Sharpe undefined')
    ax.legend(loc='lower right')
    if points is not None:
        fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in data.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 6),
                    textcoords='offset points', fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.margins(0.12)
    ax.grid(True, alpha=0.5)
    ax.set(title='Risk-Return Profile of Portfolio',
           xlabel='Average 20-day rolling volatility (annualized)',
           ylabel=f'Total return, {start} to {end} (or last available date)')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def format_table(frame: pd.DataFrame) -> str:
    """Readable text table with percentages for display (the CSV keeps fractions)."""
    shown = frame[['ticker', 'total_return', 'avg_daily_return', 'avg_volatility', 'max_drawdown',
                   'sharpe_ratio', 'last_date']].copy()
    for col in ['total_return', 'avg_volatility', 'max_drawdown']:
        shown[col] = shown[col].map(lambda v: f'{v:+.2%}' if col != 'avg_volatility' else f'{v:.2%}')
    shown['avg_daily_return'] = shown['avg_daily_return'].map(lambda v: f'{v:+.3%}')
    shown['sharpe_ratio'] = shown['sharpe_ratio'].map(lambda v: 'N/A' if pd.isna(v) else f'{v:.2f}')
    return shown.to_string(index=False)


def configure_logging(log_file: Path) -> None:
    """Log to the console (through tqdm, so the progress bar stays intact) and to a file."""

    class TqdmHandler(logging.Handler):
        def emit(self, record):
            tqdm.write(self.format(record))

    log.setLevel(logging.INFO)
    log.handlers.clear()
    console = TqdmHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(console)
    log.addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tickers', nargs='+', help='Run only these tickers (default: all in ohlc_hw_cleaned).')
    parser.add_argument('--db-path', type=Path, default=DEFAULT_DB, help='Working database (default: data/data.db).')
    args = parser.parse_args(argv)
    db_path = args.db_path.resolve()

    all_tickers, latest = load_universe(db_path)
    tickers = [t.strip().upper() for t in args.tickers] if args.tickers else all_tickers
    start, end = six_complete_months(latest)

    run_date = date.today().isoformat()  # names the output folder only; not the analysis window
    out_dir = REPORTS_ROOT / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir / 'run_log.txt')
    log.info(f'Latest cleaned observation: {latest} -> analysis window {start} to {end}')
    log.info(f'Running {len(tickers)} reports into {out_dir.relative_to(HW_DIR).as_posix()}/')

    successes: dict[str, Path] = {}
    failures: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix='hw2_kernel_') as kernel_root:
        kernel_name = register_kernel(Path(kernel_root))
        for ticker in tqdm(tickers, desc='Generating reports'):
            try:
                successes[ticker] = run_report(ticker, start, end, out_dir, db_path, kernel_name)
            except Exception as exc:  # any failure is recorded; the loop continues
                failures[ticker] = short_error(exc)
                log.error(f'FAILED {ticker}: {failures[ticker]}')
                partial = out_dir / f'{ticker}_report.ipynb'
                if partial.exists():  # papermill saves partial output; mark it so it is never mistaken
                    partial.replace(out_dir / f'{ticker}_report_FAILED.ipynb')

    (out_dir / 'failed_stocks.txt').write_text(''.join(f'{t}\n' for t in failures), encoding='utf-8')
    log.info(f'Successfully generated {len(successes)}/{len(tickers)} reports')
    if failures:
        log.info(f'Failed tickers saved to failed_stocks.txt: {", ".join(failures)}')
    if not successes:
        log.error('No report succeeded, so there is no portfolio summary. See run_log.txt for the errors.')
        return 1

    summary = collect_summary(successes)
    summary_path = out_dir / 'portfolio_summary.csv'
    summary.to_csv(summary_path, index=False)
    last_session = last_session_in_window(db_path, end)
    plot_risk_return(summary, out_dir / 'risk_return_profile.png', start, end, last_session)

    ranked = summary.dropna(subset=['sharpe_ratio'])  # undefined Sharpe ratios are not ranked
    log.info(f'\nTop {min(10, len(ranked))} by Sharpe ratio:\n{format_table(ranked.head(10))}')
    log.info(f'\nBottom {min(10, len(ranked))} by Sharpe ratio:\n{format_table(ranked.tail(10))}')
    if len(ranked) < 20:
        log.info('(Fewer than 20 ranked reports, so the top and bottom lists overlap.)')
    unranked = summary.loc[summary['sharpe_ratio'].isna(), 'ticker'].tolist()
    if unranked:
        log.info(f'Excluded from ranking (Sharpe undefined): {", ".join(unranked)}')

    short = summary.loc[summary['last_date'] < last_session]
    log.info(f'\nCoverage: {len(summary) - len(short)} of {len(summary)} tickers have data through '
             f'{last_session}, the last trading session in the window.')
    if not short.empty:
        log.info(f'{len(short)} tickers end earlier, so their metrics cover a shorter period than {start} to '
                 f'{end} and are not fully comparable with the full-window tickers (nothing is extrapolated):')
        for (last_date, days), group in short.groupby(['last_date', 'trading_days']):
            log.info(f'  last date {last_date} ({days} trading days): {", ".join(group["ticker"])}')
    log.info(f'\nSaved {summary_path.relative_to(HW_DIR).as_posix()} and risk_return_profile.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
