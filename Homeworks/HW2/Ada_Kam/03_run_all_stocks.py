import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
import papermill as pm
import scrapbook as sb
from contextlib import closing
from tqdm.auto import tqdm
from datetime import date
from matplotlib.ticker import PercentFormatter

hw_dir = Path(__file__).resolve().parent
db_path = hw_dir / 'data' / 'data.db'
template = hw_dir / '02_stock_report_template.ipynb'

def load_tickers():
    """return sorted ticker list"""

    if not db_path.is_file():
        raise FileNotFoundError(f"{db_path} not found; run 01_data_cleaning.ipynb first")
    
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)['ticker'].tolist()
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]

    return tickers, latest
def find_date_range(latest):
    """find t-6 months before 'latest;, returns start and end date"""
    if latest is None:
        raise ValueError("No observations in ohlc_hw_cleaned: cannot determine a date range")
    
    month_start = pd.Timestamp(latest).normalize().replace(day=1)
    end_dt = month_start - pd.Timedelta(days=1)
    start_dt = (month_start - pd.DateOffset(months=6))

    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

def create_output_dir(base=hw_dir):
    """Create reports under output"""
    run_date = date.today().strftime('%Y-%m-%d')
    output_dir = base / 'reports' / run_date
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def run_report(ticker, start, end, folder):
    """Run report template for individual ticker"""
    output = folder / f'{ticker}_report.ipynb'
    pm.execute_notebook(
        str(template), 
        str(output),
        parameters={'ticker':ticker, 'start_date': start, 'end_date': end, 'db_path': str(db_path), 'volatility_window': 20},
        cwd=str(hw_dir), kernel_name = 'python3', progress_bar=False,
    )

    return output

def run_all_reports(tickers, start_date, end_date, output_dir):
    """Run every ticker, log failures, return path for successful runs"""
    succeeded, failed = {}, {}

    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            succeeded[ticker] = run_report(ticker, start_date, end_date, output_dir)
        except Exception as e:
            reason = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
            failed[ticker] = reason
            tqdm.write(f"FAILED {ticker}: {reason}")

    print(f"Successfully generated {len(succeeded)}/{len(tickers)} reports")

    failed_file = output_dir / 'failed_stocks.txt'
    failed_file.write_text(''.join(f"{t}\t{r}\n" for t, r in failed.items()), encoding='utf-8')

    if failed:
        print(f"{len(failed)} failed tickers written to {failed_file}")
    return succeeded, failed

COLUMNS = ['ticker', 'first_date', 'last_date', 'n_days', 'avg_daily_return', 'avg_volatility',
           'max_drawdown', 'sharpe_ratio', 'total_return', 'report_path']

def summary_dashboard(succeeded):
    """read glued summary from successful report into one DF"""
    rows = []

    for ticker, path in succeeded.items():
        row = dict(sb.read_notebook(str(path)).scraps['summary'].data)
        row['report_path'] = str(path.relative_to(hw_dir))
        rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS)

def plot_risk_return(summary_df, output_dir):
    """Scatter plot avg volatility vs total return, coloured by Sharpe, top 5 returns labelled. Saves a PNG."""
    defined = summary_df['sharpe_ratio'].notna()
    fig, ax = plt.subplots(figsize=(12, 8))

    points = ax.scatter(summary_df.loc[defined, 'avg_volatility'], summary_df.loc[defined, 'total_return'],
                        c=summary_df.loc[defined, 'sharpe_ratio'], cmap='RdYlGn', s=90, edgecolors='black', linewidths=0.5)
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    if (~defined).any():
        ax.scatter(summary_df.loc[~defined, 'avg_volatility'], summary_df.loc[~defined, 'total_return'],
                   color='lightgray', s=90, edgecolors='black', linewidths=0.5, label='Sharpe undefined')
        ax.legend(loc='lower right')

    for _, row in summary_df.nlargest(5, 'total_return').iterrows():
        ax.annotate(row['ticker'], (row['avg_volatility'], row['total_return']),
                    xytext=(6, 6), textcoords='offset points', fontsize=10, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel('Average rolling volatility (annualized)')
    ax.set_ylabel('Total return')
    ax.set_title('Risk-Return Profile of Portfolio', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)

    fig_path = output_dir / 'risk_return_profile.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig_path

if __name__ == "__main__":
    tickers, latest = load_tickers()
    start_date, end_date = find_date_range(latest)
    print(f"{len(tickers)} tickers; latest observation {latest}; window {start_date} to {end_date}")
    output_dir = create_output_dir()
    succeeded, failed = run_all_reports(tickers, start_date, end_date, output_dir)

    if not succeeded:
        print("All reports failed, no summary created, see failed_stocks.txt")
        raise SystemExit(1)
    
    summary_df = summary_dashboard(succeeded)
    csv_path = output_dir / 'portfolio_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")

    ranked = summary_df.dropna(subset=['sharpe_ratio']).sort_values('sharpe_ratio', ascending=False)
    cols = ['ticker', 'sharpe_ratio', 'total_return', 'avg_volatility', 'max_drawdown', 'last_date']
    print("\nTop 10 by Sharpe:\n", ranked.head(10)[cols].to_string(index=False))
    print("\nBottom 10 by Sharpe:\n", ranked.tail(10)[cols].to_string(index=False))

    fig_path = plot_risk_return(summary_df, output_dir)
    print(f"Saved risk-return plot to {fig_path}")




