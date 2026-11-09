"""Generate stock risk reports from the cleaned HW2 database."""

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import argparse
import json
import papermill as pm
from tqdm import tqdm
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parent
DB_PATH = HW2_DIR / "data" / "data.db"


def load_batch_inputs():
    """Read the ticker list and latest trading date from SQLite."""
    db_uri = DB_PATH.as_uri() + "?mode=ro"

    with closing(sqlite3.connect(db_uri, uri=True)) as connection:
        ticker_rows = connection.execute(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker"
        ).fetchall()
        latest_date = connection.execute(
            "SELECT MAX(ts) FROM ohlc_hw_cleaned"
        ).fetchone()[0]

    if latest_date is None:
        raise ValueError("The cleaned table is empty. Run the cleaning notebook first.")

    return [row[0] for row in ticker_rows], date.fromisoformat(latest_date)


def report_window(latest_date):
    """Return the six calendar months before the latest observation's month."""
    current_month_start = latest_date.replace(day=1)
    end_date = current_month_start - timedelta(days=1)

    if latest_date.month > 6:
        start_date = date(latest_date.year, latest_date.month - 6, 1)
    else:
        start_date = date(latest_date.year - 1, latest_date.month + 6, 1)

    return start_date, end_date


def read_report_metrics(report_path):
    """Read the named JSON output from an executed report."""
    notebook = json.loads(report_path.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            payload = output.get("data", {}).get("application/json", {})
            if "portfolio_metrics" in payload:
                return payload["portfolio_metrics"]

    raise ValueError("The report did not produce portfolio_metrics.")


def generate_reports(tickers, start_date, end_date, output_dir):
    """Run one notebook per ticker and collect metrics from successful runs."""
    template = HW2_DIR / "02_stock_report_template.ipynb"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failed = []

    for ticker in tqdm(tickers, desc="Generating reports"):
        report_path = output_dir / f"{ticker}_report.ipynb"

        try:
            pm.execute_notebook(
                str(template),
                str(report_path),
                parameters={
                    "ticker": ticker,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                kernel_name="python3",
                cwd=str(HW2_DIR),
                progress_bar=False,
            )
            metrics = read_report_metrics(report_path)
            if metrics["ticker"] != ticker:
                raise ValueError(f"Expected {ticker}, but the report returned {metrics['ticker']}.")
        except Exception as error:
            failed.append(ticker)
            tqdm.write(f"{ticker} failed: {error}")
            continue

        metrics["report_path"] = report_path.relative_to(HW2_DIR).as_posix()
        results.append(metrics)

    failure_log = output_dir / "failed_stocks.txt"
    failure_log.write_text("".join(f"{ticker}\n" for ticker in failed), encoding="utf-8")
    print(f"Successfully generated {len(results)}/{len(tickers)} reports")
    return results

def write_portfolio_summary(results, output_dir):
    """Save the portfolio CSV and chart, and print the Sharpe rankings."""
    summary = pd.DataFrame(results).sort_values(
        "sharpe_ratio", ascending=False, na_position="last"
    )
    csv_path = output_dir / "portfolio_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path.name} with {len(summary)} stocks.")

    ranking_columns = [
        "ticker", "sharpe_ratio", "total_return", "actual_start", "actual_end",
    ]
    ranked = summary.dropna(subset=["sharpe_ratio"])[ranking_columns]

    if ranked.empty:
        print("No stocks have a defined Sharpe ratio.")
    else:
        print("\nTop 10 by Sharpe ratio")
        print(ranked.head(10).to_string(index=False))
        print("\nBottom 10 by Sharpe ratio")
        print(ranked.tail(10).to_string(index=False))

    plotted = summary.dropna(
        subset=["avg_volatility", "total_return", "sharpe_ratio"]
    )
    if plotted.empty:
        print("No stocks have all three metrics needed for the scatter plot.")
        return

    fig = Figure(figsize=(10, 7), layout="constrained")
    ax = fig.subplots()
    points = ax.scatter(
        plotted["avg_volatility"],
        plotted["total_return"],
        c=plotted["sharpe_ratio"],
        cmap="viridis",
        s=65,
    )
    fig.colorbar(points, ax=ax, label="Sharpe ratio")

    for row in plotted.nlargest(5, "total_return").itertuples():
        ax.annotate(
            row.ticker,
            (row.avg_volatility, row.total_return),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set(
        title="Risk-Return Profile of Portfolio",
        xlabel="Mean annualized rolling volatility",
        ylabel="Total return over actual coverage",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "portfolio_risk_return.png", dpi=150)

    
def main():
    """Run the batch for all tickers or test a single ticker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", type=str.upper, help="Run one ticker for testing.")
    args = parser.parse_args()

    tickers, latest_date = load_batch_inputs()
    start_date, end_date = report_window(latest_date)
    if args.ticker:
        if args.ticker not in tickers:
            parser.error(f"Ticker {args.ticker} is absent from the cleaned database.")
        tickers = [args.ticker]

    output_dir = HW2_DIR / "reports" / date.today().isoformat()
    print(f"Report window: {start_date} to {end_date}")
    results = generate_reports(tickers, start_date, end_date, output_dir)
    if not results:
        raise SystemExit("No reports succeeded. See failed_stocks.txt.")

    if args.ticker is None:
        write_portfolio_summary(results, output_dir)


if __name__ == "__main__":
    main()

    