"""HW2 Part 3: generate a risk report for every cleaned ticker with papermill.

Run with the course environment active (any working folder):

    python 03_run_all_stocks.py

Requires the ohlc_hw_cleaned table in data/data.db, created by 01_data_cleaning.ipynb.
Outputs go to reports/{run date}/: one {ticker}_report.ipynb per success,
failed_stocks.txt, portfolio_summary.csv and risk_return_profile.png.
"""
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "data.db"
TEMPLATE = HERE / "02_stock_report_template.ipynb"
METRICS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]


def six_complete_months(latest_date):
    """Six calendar months before the latest observation's month (HW2 convention)."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError('Cannot choose a window from an empty date range.')
    month_start = latest.to_period('M').start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def load_tickers_and_latest_date(db_path):
    """Return the cleaned tickers and the latest cleaned observation date."""
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
        latest = pd.read_sql_query("SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn)["latest"].iloc[0]
    return tickers, latest


def register_report_kernel(kernel_root):
    """Make reports run with this Python interpreter; return the kernel name.

    Same idea as the lecture's report_runtime helper: a kernel spec pointing at
    sys.executable lives in a temporary folder that is added to JUPYTER_PATH
    for this process only, so nothing is installed for the user.
    """
    kernel_name = "hw2-report"
    kernel_dir = Path(kernel_root) / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "HW2 report",
        "language": "python",
    }), encoding="utf-8")
    prior_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + prior_path if prior_path else "")
    return kernel_name


def run_report(ticker, start_date, end_date, run_dir, kernel_name):
    """Execute the template for one ticker; papermill raises if any cell fails."""
    output = run_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                    "db_path": str(DB_PATH)},
        kernel_name=kernel_name, cwd=str(HERE), progress_bar=False,
    )
    return output


def collect_reports(paths):
    """Build the portfolio summary from the glued metrics of successful reports only."""
    rows = []
    for path in paths:
        scraps = sb.read_notebook(str(path)).scraps
        row = {"ticker": scraps["ticker"].data}
        row.update({name: scraps[name].data for name in METRICS})
        row["report_path"] = str(path.relative_to(HERE))
        # Actual coverage, since some histories end before the requested window
        row.update({name: scraps[name].data for name in ["first_date", "last_date", "trading_days"]})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_risk_return(summary, path):
    """Scatter mean rolling volatility vs total return, colored by Sharpe; label the top 5 by return."""
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")  # undefined Sharpe stays visible
    points = ax.scatter(summary["avg_volatility"] * 100, summary["total_return"] * 100,
                        c=summary["sharpe_ratio"], cmap=cmap, s=70, edgecolor="black")
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    for row in summary.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100),
                    xytext=(5, 5), textcoords="offset points", fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(title="Risk-Return Profile of Portfolio",
           xlabel="Mean 20-day rolling volatility, annualized (%)", ylabel="Total return (%)")
    ax.grid(True, alpha=0.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    tickers, latest_date = load_tickers_and_latest_date(DB_PATH)
    start_date, end_date = six_complete_months(latest_date)
    run_dir = HERE / "reports" / date.today().isoformat()  # run date, not the analysis window
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Latest cleaned observation: {latest_date}")
    print(f"Analysis window (six complete months): {start_date} to {end_date}")
    print(f"Writing reports to {run_dir}")

    successful_paths, failed_tickers = [], []
    with tempfile.TemporaryDirectory(prefix="hw2_kernel_") as kernel_root:
        kernel_name = register_report_kernel(kernel_root)
        for ticker in tqdm(tickers, desc="Generating reports"):
            try:
                successful_paths.append(run_report(ticker, start_date, end_date, run_dir, kernel_name))
            except Exception as exc:
                # Log and keep going; papermill errors carry the failing cell's error name/value
                error = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', exc)}"
                failed_tickers.append(ticker)
                tqdm.write(f"FAILED {ticker}: {error}")

    print(f"Successfully generated {len(successful_paths)}/{len(tickers)} reports")
    (run_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failed_tickers))
    if not successful_paths:
        print("All reports failed, so there is no portfolio summary. See failed_stocks.txt.")
        return 1

    summary_df = collect_reports(successful_paths)
    summary_df = summary_df.sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary_df.to_csv(run_dir / "portfolio_summary.csv", index=False)

    # Rankings use defined Sharpe ratios only; groups overlap with fewer than 20 successes
    ranked = summary_df.dropna(subset=["sharpe_ratio"])
    columns = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "last_date"]
    print("\nTop 10 by Sharpe ratio:")
    print(ranked.head(10)[columns].to_string(index=False, float_format="{:.4f}".format))
    print("\nBottom 10 by Sharpe ratio:")
    print(ranked.tail(10)[columns].to_string(index=False, float_format="{:.4f}".format))

    # Disclose uneven coverage (Nov 30 itself may not be a trading day, so compare with the latest report)
    latest_report_date = summary_df["last_date"].max()
    short = summary_df[summary_df["last_date"] < latest_report_date]
    if not short.empty:
        print(f"\nNote: {len(short)} of {len(summary_df)} reports end before {latest_report_date} "
              f"(last dates: {', '.join(sorted(short['last_date'].unique()))}); "
              "their metrics cover a shorter period, so compare them with care.")

    plot_risk_return(summary_df, run_dir / "risk_return_profile.png")
    print(f"\nSaved portfolio_summary.csv and risk_return_profile.png to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
