"""Generate single-stock reports for every ticker in ohlc_hw_cleaned.

Run from this directory after 01_data_cleaning.ipynb has written ohlc_hw_cleaned:

    python 03_run_all_stocks.py

The analysis window is the six complete calendar months before the month of
the latest cleaned observation (anchored to the database, not the wall clock).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HW_DIR = Path(__file__).resolve().parent
TEMPLATE = HW_DIR / "02_stock_report_template.ipynb"
DB_PATH = HW_DIR / "data" / "data.db"
REPORTS_ROOT = HW_DIR / "reports"


def six_complete_months(latest_date: str) -> tuple[str, str]:
    """Six calendar months before the latest observation's month (HW2 convention)."""
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError("Cannot choose a window from an empty date range.")
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def prepare_kernel() -> str:
    """Register a one-off Jupyter kernel in a system temp dir (not in reports/)."""
    kernel_name = "hw2-report"
    kernel_root = Path(tempfile.mkdtemp(prefix="hw2-kernel-"))
    kernel_dir = kernel_root / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "HW2 report",
        "language": "python",
    }), encoding="utf-8")
    prior = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + prior if prior else "")
    return kernel_name


def load_universe(db_path: Path) -> tuple[list[str], str]:
    if not db_path.is_file():
        raise FileNotFoundError(
            f"{db_path} is missing. Copy data.db into data/ and run 01_data_cleaning.ipynb."
        )
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if "ohlc_hw_cleaned" not in tables:
            raise FileNotFoundError("ohlc_hw_cleaned not found. Run 01_data_cleaning.ipynb first.")
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn
        )["ticker"].tolist()
        latest = pd.read_sql_query(
            "SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn
        )["latest"].iloc[0]
    if not tickers or pd.isna(latest):
        raise ValueError("ohlc_hw_cleaned has no tickers or dates.")
    return tickers, str(latest)


def run_report(ticker: str, start: str, end: str, folder: Path, kernel_name: str) -> Path:
    """Execute the template independently; let the caller record failures."""
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE),
        str(output),
        parameters={
            "ticker": ticker,
            "start_date": start,
            "end_date": end,
            "db_path": str(DB_PATH.resolve()),
            "volatility_window": 20,
        },
        kernel_name=kernel_name,
        cwd=str(HW_DIR),
        progress_bar=False,
    )
    return output


def collect_reports(paths: list[Path]) -> pd.DataFrame:
    """Build a summary from successful paths only; reject empty or duplicated input."""
    rows = []
    names = [
        "ticker", "start_date", "end_date", "trading_days",
        "avg_daily_return", "avg_volatility", "max_drawdown",
        "sharpe_ratio", "total_return", "actual_start", "actual_end",
    ]
    for path in paths:
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in names}
        try:
            row["report_path"] = str(path.resolve().relative_to(HW_DIR))
        except ValueError:
            row["report_path"] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError("No successful reports to summarize; inspect failed_stocks.txt.")
    summary = pd.DataFrame(rows)
    if summary.duplicated(["ticker", "start_date", "end_date"]).any():
        raise ValueError("Duplicate report keys in this run.")
    return summary.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)


def save_risk_return_plot(summary: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    points = ax.scatter(
        summary["avg_volatility"],
        summary["total_return"],
        c=summary["sharpe_ratio"],
        cmap="viridis",
    )
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    labeled = summary.nlargest(5, "total_return")
    for row in labeled.itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return),
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(
        xlabel="Mean rolling volatility (annualized, fraction)",
        ylabel="Total return (fraction)",
        title="Risk-Return Profile of Portfolio",
    )
    ax.grid(True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tickers, latest = load_universe(DB_PATH)
    start_date, end_date = six_complete_months(latest)
    assert six_complete_months("2025-01-15") == ("2024-07-01", "2024-12-31")
    assert six_complete_months("2024-12-06") == ("2024-06-01", "2024-11-30")

    run_date = datetime.now().strftime("%Y-%m-%d")
    run_dir = REPORTS_ROOT / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    kernel_name = prepare_kernel()

    print(f"Database latest observation: {latest}")
    print(f"Common requested window: {start_date} → {end_date}")
    print(f"Reports directory: {run_dir}")
    print(f"Tickers: {len(tickers)}")

    successful_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    for symbol in tqdm(tickers, desc="Generating reports"):
        try:
            successful_paths.append(run_report(symbol, start_date, end_date, run_dir, kernel_name))
        except Exception as exc:
            failures.append({"ticker": symbol, "error": str(exc)})
            print(f"FAILED: {symbol}")

    (run_dir / "failed_stocks.txt").write_text(
        "".join(item["ticker"] + "\n" for item in failures), encoding="utf-8"
    )
    

    n_ok, n_all = len(successful_paths), len(tickers)
    print(f"Successfully generated {n_ok}/{n_all} reports")
    if n_ok == 0:
        print("All reports failed; no portfolio summary was written. See failed_stocks.txt.")
        return

    summary_df = collect_reports(successful_paths)
    csv_cols = [
        "ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
        "sharpe_ratio", "total_return", "report_path",
        "trading_days", "actual_start", "actual_end",
    ]
    summary_df[csv_cols].to_csv(run_dir / "portfolio_summary.csv", index=False)

    ranked = summary_df.dropna(subset=["sharpe_ratio"])
    print("\nHistories differ by ticker. Compare names with actual_start/actual_end; "
          "prices are not extrapolated through trailing gaps.")
    print("\nTop 10 by Sharpe (undefined Sharpe excluded):")
    print(ranked.head(10)[["ticker", "sharpe_ratio", "total_return", "avg_volatility",
                           "trading_days", "actual_end"]].to_string(index=False))
    print("\nBottom 10 by Sharpe:")
    print(ranked.tail(10)[["ticker", "sharpe_ratio", "total_return", "avg_volatility",
                           "trading_days", "actual_end"]].to_string(index=False))
    if len(ranked) < 20:
        print("Fewer than 20 ranked names: top and bottom groups may overlap.")

    save_risk_return_plot(summary_df, run_dir / "risk_return_profile.png")
    print(f"\nWrote {run_dir / 'portfolio_summary.csv'}")


if __name__ == "__main__":
    main()
