"""HW2 · Part 3 - run the single-stock report for every ticker and summarise the portfolio.

Usage (after 01_data_cleaning.ipynb has created ohlc_hw_cleaned in data/data.db):

    python 03_run_all_stocks.py

Pipeline: tickers + latest date come from ``ohlc_hw_cleaned`` -> one common six-complete-month
window -> ``02_stock_report_template.ipynb`` executed once per ticker with papermill -> named
results (scrapbook) collected from the *successful* reports only -> ranked summary, CSVs and a
risk/return chart in ``reports/<run date>/``.

Useful flags:  --tickers IBM NVDA   run a subset
               --demo-failure       also request a bogus ticker to show that one failure does not stop the batch
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only saved, never shown
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HW_DIR = Path(__file__).resolve().parent
DEFAULT_DB = HW_DIR / "data" / "data.db"
TEMPLATE = HW_DIR / "02_stock_report_template.ipynb"

# Summary columns requested by the assignment, in order.
SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return", "report_path"]
# Named results that the template exports with sb.glue.
SCRAP_NAMES = ["ticker", "start_date", "end_date", "actual_start", "actual_end", "observations",
               "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]


# --------------------------------------------------------------------------------------- window and universe
def six_complete_months(latest_date: str) -> tuple[str, str]:
    """The six calendar months *before* the month that contains `latest_date`.

    A last trading date alone does not prove the month was fully delivered, so even a month-end
    observation does not make its own month "complete": 2024-12-06 -> 2024-06-01 .. 2024-11-30.
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError("Cannot choose a window from an empty date range.")
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def load_universe(db_path: Path) -> tuple[list[str], str]:
    """All tickers in ohlc_hw_cleaned and the latest cleaned observation date (read-only)."""
    if not db_path.is_file():
        raise FileNotFoundError(f"{db_path} not found. Run 01_data_cleaning.ipynb to create it.")
    with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "ohlc_hw_cleaned" not in tables:
            raise LookupError("Table ohlc_hw_cleaned is missing. Run 01_data_cleaning.ipynb first.")
        tickers = [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker")]
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    if not tickers or latest is None:
        raise ValueError("ohlc_hw_cleaned is empty.")
    return tickers, latest


# ------------------------------------------------------------------------------------------------- execution
def register_kernel() -> str:
    """Create a throw-away kernelspec that launches *this* interpreter and return its name.

    This guarantees the reports run in the same environment as the script, without installing
    anything into the user's Jupyter configuration.
    """
    root = Path(tempfile.mkdtemp(prefix="hw2-kernel-"))
    spec_dir = root / "kernels" / "hw2-report"
    spec_dir.mkdir(parents=True)
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "HW2 report", "language": "python"}), encoding="utf-8")
    previous = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(root) + (os.pathsep + previous if previous else "")
    return "hw2-report"


def scrub_local_paths(notebook: Path) -> None:
    """Papermill records absolute input/output paths in the notebook metadata; keep only file names."""
    nb = nbformat.read(notebook, as_version=4)
    info = nb.metadata.get("papermill")
    if info:
        info["input_path"], info["output_path"] = Path(info["input_path"]).name, Path(info["output_path"]).name
        nbformat.write(nb, notebook)


def run_report(ticker: str, start: str, end: str, folder: Path, db_path: Path, kernel: str) -> Path:
    """Execute the template for one ticker. Errors propagate so the caller can record them."""
    output = folder / f"{ticker}_report.ipynb"
    output.unlink(missing_ok=True)  # never let a stale file from an earlier run look like a result
    db_param = os.path.relpath(db_path, HW_DIR) if db_path.is_relative_to(HW_DIR) else str(db_path)
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": Path(db_param).as_posix()},
        kernel_name=kernel, cwd=str(HW_DIR), progress_bar=False)
    scrub_local_paths(output)
    return output


def run_batch(tickers: list[str], start: str, end: str, folder: Path, db_path: Path) -> tuple[list[Path], list[dict]]:
    """Run every ticker; one failure is logged and the loop continues (including for the first ticker)."""
    kernel = register_kernel()
    failed_dir = folder / "_failed"
    successes, failures = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successes.append(run_report(ticker, start, end, folder, db_path, kernel))
        except Exception as exc:  # noqa: BLE001 - any failure for one stock must not stop the others
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            failures.append({"ticker": ticker, "error_type": type(exc).__name__, "error": message})
            partial = folder / f"{ticker}_report.ipynb"
            if partial.exists():  # papermill saves a notebook even when a cell failed: keep it out of the results
                failed_dir.mkdir(exist_ok=True)
                shutil.move(str(partial), failed_dir / partial.name)
    return successes, failures


# ------------------------------------------------------------------------------------------------- results
def collect_reports(paths: list[Path]) -> pd.DataFrame:
    """Read the named scraps from successful reports only; reject an empty or duplicated set."""
    rows = []
    for path in paths:
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in SCRAP_NAMES}
        resolved = path.resolve()
        row["report_path"] = (resolved.relative_to(HW_DIR) if resolved.is_relative_to(HW_DIR) else resolved).as_posix()
        rows.append(row)
    if not rows:
        raise ValueError("No successful reports to summarise.")
    frame = pd.DataFrame(rows)
    if frame["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers among the collected reports.")
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")  # None (undefined) -> NaN
    return frame


def risk_return_chart(summary: pd.DataFrame, coverage: pd.DataFrame, window: tuple[str, str], path: Path) -> None:
    """Scatter of mean rolling volatility vs total return, coloured by Sharpe; top 5 returns labelled."""
    data = summary.merge(coverage[["ticker", "truncated"]], on="ticker")
    fig, ax = plt.subplots(figsize=(11, 7.5), layout="constrained")
    defined = data.dropna(subset=["sharpe_ratio"])
    undefined = data[data["sharpe_ratio"].isna()]
    norm = plt.Normalize(defined["sharpe_ratio"].min(), defined["sharpe_ratio"].max()) if len(defined) else None
    sc = None
    for truncated, marker, label in [(False, "o", "full requested window"), (True, "D", "history ends early")]:
        part = defined[defined["truncated"] == truncated]
        if len(part):
            sc = ax.scatter(part["avg_volatility"] * 100, part["total_return"] * 100, c=part["sharpe_ratio"], cmap="viridis",
                            norm=norm, marker=marker, s=110, edgecolor="black", linewidth=0.6, label=label)
    if len(undefined):
        ax.scatter(undefined["avg_volatility"] * 100, undefined["total_return"] * 100, color="lightgrey", marker="x", s=80,
                   label="Sharpe undefined")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    for row in data.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(title="Risk-Return Profile of Portfolio", xlabel="Mean rolling 20-day volatility, annualised (%)",
           ylabel="Total return over the observed period (%)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.text(0.5, -0.01, f"Requested window {window[0]} to {window[1]}. Diamonds: history ends before the window does, so total return "
             "covers a shorter span than the circles - compare with care.", ha="center", fontsize=9, color="#555555")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="database holding ohlc_hw_cleaned (default: data/data.db)")
    parser.add_argument("--run-date", default=date.today().isoformat(), help="name of the output folder (default: today)")
    parser.add_argument("--out-root", type=Path, default=HW_DIR / "reports", help="parent of the dated output folder")
    parser.add_argument("--tickers", nargs="+", help="run only these tickers")
    parser.add_argument("--demo-failure", action="store_true", help="also request NOT_A_TICKER to demonstrate error handling")
    args = parser.parse_args()

    db_path = args.db.resolve()
    tickers, latest = load_universe(db_path)
    start, end = six_complete_months(latest)
    if args.tickers:
        wanted = [t.upper() for t in args.tickers]
        tickers = [t for t in tickers if t in wanted] + [t for t in wanted if t not in tickers]  # unknown names fail loudly in the loop
    if args.demo_failure:
        tickers = tickers + ["NOT_A_TICKER"]
    folder = args.out_root / args.run_date
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Database latest observation {latest} -> analysis window {start} to {end}")
    print(f"{len(tickers)} tickers | output folder: {folder}")

    successes, failures = run_batch(tickers, start, end, folder, db_path)
    (folder / "failed_stocks.txt").write_text("".join(f"{item['ticker']}\n" for item in failures))
    (folder / "run_log.json").write_text(json.dumps(
        {"window": [start, end], "database_latest": latest, "succeeded": [p.name for p in successes], "failed": failures}, indent=2))
    print(f"\nSuccessfully generated {len(successes)}/{len(tickers)} reports")
    for item in failures:
        print(f"  FAILED {item['ticker']}: {item['error_type']}: {item['error']}")
    if not successes:
        print("Every report failed, so there is nothing to summarise. See failed_stocks.txt and run_log.json.")
        return 1

    collected = collect_reports(successes)
    summary_df = (collected[SUMMARY_COLUMNS].sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True))
    summary_df.to_csv(folder / "portfolio_summary.csv", index=False)

    coverage = collected[["ticker", "start_date", "end_date", "actual_start", "actual_end", "observations"]].copy()
    coverage["days_short_of_window_end"] = (pd.to_datetime(coverage["end_date"]) - pd.to_datetime(coverage["actual_end"])).dt.days
    coverage["truncated"] = coverage["days_short_of_window_end"] > 7
    coverage.sort_values("ticker").to_csv(folder / "coverage_summary.csv", index=False)

    ranked = summary_df.dropna(subset=["sharpe_ratio"])
    show = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown"]
    fmt = {"sharpe_ratio": "{:.2f}".format, "total_return": "{:+.2%}".format, "avg_volatility": "{:.2%}".format, "max_drawdown": "{:.2%}".format}
    print(f"\nTop 10 by Sharpe ({len(ranked)} stocks with a defined Sharpe):")
    print(ranked.head(10)[show].to_string(index=False, formatters=fmt))
    print("\nBottom 10 by Sharpe:")
    print(ranked.tail(10)[show].to_string(index=False, formatters=fmt))
    truncated = coverage[coverage["truncated"]]
    print(f"\n{len(truncated)} of {len(coverage)} reports have a history that ends before {end} "
          f"(earliest end {coverage['actual_end'].min()}); their returns cover a shorter span - see coverage_summary.csv.")

    risk_return_chart(collected, coverage, (start, end), folder / "risk_return_profile.png")
    print(f"\nSaved: portfolio_summary.csv, coverage_summary.csv, risk_return_profile.png, failed_stocks.txt in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
