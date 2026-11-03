"""Execute all stock reports locally and summarize only current-run successes.

Author: Jiajing Wang
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import logging
import sqlite3
from contextlib import closing
import sys

ROOT = Path(__file__).resolve().parent
# Optional task-local installation; normal installations use requirements.txt.
if (ROOT / '.local_packages').is_dir():
    sys.path.insert(0, str(ROOT / '.local_packages'))
import papermill as pm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
def six_complete_months(latest_date):
    """Return the six calendar months before the latest observation's month."""
    # Exclude the latest month, even if the latest date is at month-end.
    first_day = pd.Timestamp(latest_date).replace(day=1)
    start_date = first_day - pd.DateOffset(months=6)
    end_date = first_day - pd.Timedelta(days=1)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def plot_portfolio(summary, output):
    """Plot return versus mean rolling volatility, colored by defined Sharpe."""
    fig, ax = plt.subplots(figsize=(12,8))
    valid = summary[['avg_volatility','total_return','sharpe_ratio']].notna().all(axis=1)
    ranked = summary.loc[valid]
    if not ranked.empty:
        points = ax.scatter(ranked.avg_volatility, ranked.total_return, c=ranked.sharpe_ratio,
                            cmap='RdYlGn', s=80, edgecolors='#444444', linewidths=.5)
        fig.colorbar(points, ax=ax, label='Annualized Sharpe ratio')
    unranked = summary.loc[~valid & summary.avg_volatility.notna() & summary.total_return.notna()]
    if not unranked.empty:
        ax.scatter(unranked.avg_volatility, unranked.total_return, color='gray', label='Undefined Sharpe')
        ax.legend()
    for row in summary.nlargest(5,'total_return').itertuples():
        if pd.notna(row.avg_volatility) and pd.notna(row.total_return):
            ax.annotate(row.ticker,(row.avg_volatility,row.total_return),xytext=(5,5),textcoords='offset points')
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Mean annualized 20-day rolling volatility',
           ylabel='Total adjusted-price return')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    fig.text(.5,.015,'Observed coverage differs by stock; compare actual_start, actual_end and n_prices. Author: Jiajing Wang',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,1]); fig.savefig(output,dpi=150); plt.close(fig)


# 3.1 Script Structure
# Run Steps 1-4 in order; Sections 3.2 and 3.3 follow below.
def run_batch(db_path=ROOT / "data/data.db", tickers=None, run_date=None):
    """Generate stock reports, record failures, and save the portfolio summary."""
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        # Step 1: Load all unique tickers from the cleaned table.
        all_tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn
        )["ticker"].tolist()

        # Step 2: Anchor the analysis window to the latest cleaned date.
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]

    if not latest:
        raise ValueError("The cleaned table is empty; run the cleaning notebook first")
    # Example: 2024-12-06 gives 2024-06-01 through 2024-11-30.
    start_date, end_date = six_complete_months(latest)
    print(f"Latest cleaned date: {latest}; analysis: {start_date} to {end_date}")

    # Step 3: Create reports/{run date}/, independently of the analysis dates.
    run_date = run_date or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    output = ROOT / "reports" / run_date
    output.mkdir(parents=True, exist_ok=True)

    # Record errors in the same folder as this run's reports.
    logger = logging.getLogger("hw2_batch")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "execution.log", mode="w")
    logger.addHandler(handler)
    tickers = all_tickers if tickers is None else tickers
    successes, failed = [], []

    # Step 4: Execute the template once per ticker, with a progress indicator.
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            if ticker not in all_tickers:
                raise ValueError(f"Ticker not in cleaned data: {ticker}")
            report_path = output / f"{ticker}_report.ipynb"
            metrics_path = output / f"{ticker}_metrics.csv"
            # Do not reuse an old result if this execution fails.
            metrics_path.unlink(missing_ok=True)

            # Pass the stock and historical dates to Papermill.
            pm.execute_notebook(
                input_path=str(ROOT / "02_stock_report_template.ipynb"),
                output_path=str(report_path),
                parameters={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "db_path": str(db_path),
                    "output_dir": str(output),
                },
                cwd=str(ROOT),
                kernel_name="python3",
                progress_bar=False,
                log_output=False,
                start_timeout=120,
                execution_timeout=300,
            )

            # Keep each successful report's CSV results for Section 3.3.
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
            if metrics["ticker"] != ticker:
                raise ValueError("Metrics CSV ticker does not match the request")
            metrics["report_path"] = str(report_path.relative_to(ROOT))
            successes.append(metrics)
            logger.info("SUCCESS %s", ticker)
        except Exception as error:
            # Log the failure and continue with the next stock.
            failed.append(ticker)
            logger.exception("FAILED %s: %s", ticker, error)

    # 3.2 Progress Tracking: save failed tickers and report the success count.
    (output/'failed_stocks.txt').write_text(''.join(t+'\n' for t in failed))

    # 3.3 Summary Dashboard: combine successful results, rank, and plot.
    summary_path=output/'portfolio_summary.csv'
    figure_path=output/'portfolio_risk_return.png'
    if successes:
        summary=pd.DataFrame(successes).sort_values('sharpe_ratio',ascending=False,na_position='last')
        summary.to_csv(summary_path,index=False)
        defined=summary.dropna(subset=['sharpe_ratio'])
        columns=['ticker','sharpe_ratio','total_return','actual_start','actual_end','n_prices']
        print('\nTop 10 by defined Sharpe:\n',defined.head(10)[columns].to_string(index=False))
        print('\nBottom 10 by defined Sharpe:\n',defined.tail(10)[columns].to_string(index=False))
        print('Top/bottom groups may overlap when fewer than 20 defined Sharpe values exist.')
        print('Coverage differs by stock; no prices were extrapolated.')
        plot_portfolio(summary,figure_path)
    else:
        # Remove only stale summaries from a prior invocation in this run folder.
        summary_path.unlink(missing_ok=True); figure_path.unlink(missing_ok=True)
        summary=pd.DataFrame()
        print('All reports failed; no portfolio summary was generated.')
    print(f'Successfully generated {len(successes)}/{len(tickers)} reports')
    handler.close()
    logger.removeHandler(handler)
    return summary,failed


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-path',type=Path,default=ROOT/'data/data.db')
    parser.add_argument('--tickers',nargs='+',help='Optional smoke-test subset; omit for the full assignment batch')
    parser.add_argument('--run-date',help='Optional reproducible output-folder date (YYYY-MM-DD)')
    args=parser.parse_args()
    _,failed=run_batch(args.db_path,args.tickers,args.run_date)
    if failed:
        print('See failed_stocks.txt and execution.log for details.')
