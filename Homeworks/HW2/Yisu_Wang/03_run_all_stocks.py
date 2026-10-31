# Generate one notebook report per stock and summarize successful runs.
from contextlib import closing
from datetime import date
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

homework_dir = Path(__file__).resolve().parent
repo_root = homework_dir.parents[2]
db_path = repo_root / "Lectures" / "Lecture 4" / "data" / "data.db"
template = homework_dir / "02_stock_report_template.ipynb"

def six_complete_months(latest_date):
    """Return the six calendar months before the latest observation's month."""
    month_start = pd.Timestamp(latest_date).to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def main():
    """Run stock reports and build a portfolio summary from successful reports."""
    if not db_path.is_file() or not template.is_file():
        raise FileNotFoundError(f"Check database and template paths:\n{db_path}\n{template}")

    # Read the stock list and latest date from the cleaned table.
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
        tickers = [row[0] for row in conn.execute(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned WHERE ticker IS NOT NULL ORDER BY ticker"
        )]
    if latest is None or not tickers:
        raise ValueError("ohlc_hw_cleaned contains no stock observations.")

    start_date, end_date = six_complete_months(latest)
    run_dir = homework_dir / "reports" / date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Latest database date: {latest}")
    print(f"Common requested window: {start_date} to {end_date}")
    print(f"Stocks: {len(tickers)} | Output: {run_dir}")

    # Execute each report and collect only results from successful executions.
    rows, failures = [], []
    metrics = ["avg_daily_return", "avg_volatility", "max_drawdown",
               "sharpe_ratio", "total_return"]
    coverage = ["ticker", "trading_days", "actual_start_date", "actual_end_date"]

    for ticker in tqdm(tickers, desc="Generating reports"):
        report_path = run_dir / f"{ticker}_report.ipynb"
        try:
            pm.execute_notebook(
                str(template), str(report_path),
                parameters={"ticker": ticker, "start_date": start_date,
                            "end_date": end_date, "db_path": str(db_path)},
                cwd=str(repo_root), progress_bar=False
            )
            scraps = sb.read_notebook(str(report_path)).scraps
            row = {name: scraps[name].data for name in coverage + metrics}
            row["report_path"] = str(report_path)
            rows.append(row)
        except Exception as exc:
            failures.append(f"{ticker}\t{type(exc).__name__}: {exc}")
            tqdm.write(f"Failed: {ticker}")

    (run_dir / "failed_stocks.txt").write_text(
        "\n".join(failures) + ("\n" if failures else ""), encoding="utf-8"
    )
    print(f"Successfully generated {len(rows)}/{len(tickers)} reports")
    if not rows:
        print(f"No reports succeeded. Inspect {run_dir / 'failed_stocks.txt'}")
        return

    # Keep numeric values as fractions; undefined Sharpe values remain in the CSV.
    summary = pd.DataFrame(rows)
    for name in metrics:
        summary[name] = pd.to_numeric(summary[name], errors="coerce")
    summary = summary[coverage + metrics + ["report_path"]]
    summary = summary.sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary.to_csv(run_dir / "portfolio_summary.csv", index=False)

    ranked = summary.dropna(subset=["sharpe_ratio"])
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10).to_string(index=False))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10).to_string(index=False))
    print("\nActual start/end dates and trading days are included in portfolio_summary.csv.")
    print("Stocks with shorter histories may cover different actual periods.")

    # Plot stocks with defined Sharpe, volatility, and total return.
    plotted = ranked.dropna(subset=["avg_volatility", "total_return"])
    if plotted.empty:
        print("No stocks have all three metrics needed for the comparison chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    points = ax.scatter(plotted["avg_volatility"] * 100, plotted["total_return"] * 100,
                        c=plotted["sharpe_ratio"], cmap="viridis", s=65)
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in plotted.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(5, 5), textcoords="offset points")
    ax.set(title="Risk-Return Profile of Portfolio",
           xlabel="Mean rolling volatility, annualized (%)",
           ylabel="Total return (%)")
    ax.grid(alpha=0.3)
    fig.savefig(run_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary and chart to {run_dir}")


if __name__ == "__main__":
    main()