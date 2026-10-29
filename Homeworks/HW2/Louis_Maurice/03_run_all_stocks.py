"""Run 02_stock_report_template.ipynb for every ticker over the last 6 complete months and summarize."""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / 'data' / 'data.db'
TEMPLATE = HERE / '02_stock_report_template.ipynb'


def load_tickers_and_last_date():
    with sqlite3.connect(DB_PATH) as conn:
        tickers = pd.read_sql_query('SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker', conn)['ticker'].tolist()
        last_date = pd.read_sql_query('SELECT MAX(ts) AS ts FROM ohlc_hw_cleaned', conn, parse_dates=['ts'])['ts'].iloc[0]
    return tickers, last_date


def last_six_complete_months(last_date):
    end = last_date + pd.offsets.MonthEnd(0)
    if end != last_date:
        end = last_date - pd.offsets.MonthEnd(1)
    start = end - pd.offsets.MonthBegin(6)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def run_reports(tickers, start_date, end_date, out_dir):
    paths, failed = {}, []
    for ticker in tqdm(tickers, desc='Generating reports'):
        output = out_dir / f'{ticker}_report.ipynb'
        try:
            pm.execute_notebook(
                str(TEMPLATE), str(output),
                parameters={'ticker': ticker, 'start_date': start_date, 'end_date': end_date, 'db_path': str(DB_PATH)},
                cwd=str(HERE), progress_bar=False,
            )
            paths[ticker] = output
        except Exception as exc:
            print(f'FAILED {ticker}: {exc}')
            failed.append(ticker)
    return paths, failed


def collect_summary(paths):
    rows = []
    for ticker, path in paths.items():
        scraps = sb.read_notebook(str(path)).scraps
        rows.append({
            'ticker': ticker,
            'avg_daily_return': scraps['avg_daily_return'].data,
            'avg_volatility': scraps['avg_volatility'].data,
            'max_drawdown': scraps['max_drawdown'].data,
            'sharpe_ratio': scraps['sharpe_ratio'].data,
            'total_return': scraps['total_return'].data,
            'report_path': str(path.relative_to(HERE)),
        })
    return pd.DataFrame(rows).sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)


def plot_risk_return(summary_df, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.scatter(summary_df['avg_volatility'] * 100, summary_df['total_return'] * 100,
                        c=summary_df['sharpe_ratio'], cmap='RdYlGn', s=80, edgecolor='black')
    fig.colorbar(points, ax=ax, label='Sharpe ratio')
    for row in summary_df.nlargest(5, 'total_return').itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100), xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set(title='Risk-Return Profile of Portfolio', xlabel='Average rolling volatility, annualized (%)', ylabel='Total return (%)')
    ax.grid(True)
    fig.savefig(path, dpi=150, bbox_inches='tight')


def main():
    tickers, last_date = load_tickers_and_last_date()
    start_date, end_date = last_six_complete_months(last_date)
    out_dir = HERE / 'reports' / end_date
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Last observation {last_date.date()} -> reports for {start_date} to {end_date} in {out_dir}')

    paths, failed = run_reports(tickers, start_date, end_date, out_dir)
    print(f'Successfully generated {len(paths)}/{len(tickers)} reports')
    (out_dir / 'failed_stocks.txt').write_text('\n'.join(failed))

    summary_df = collect_summary(paths)
    summary_df.to_csv(out_dir / 'portfolio_summary.csv', index=False)
    print('\nTop 10 by Sharpe ratio:\n', summary_df.head(10).drop(columns='report_path').to_string(index=False))
    print('\nBottom 10 by Sharpe ratio:\n', summary_df.tail(10).drop(columns='report_path').to_string(index=False))
    plot_risk_return(summary_df, out_dir / 'risk_return.png')


if __name__ == '__main__':
    main()
