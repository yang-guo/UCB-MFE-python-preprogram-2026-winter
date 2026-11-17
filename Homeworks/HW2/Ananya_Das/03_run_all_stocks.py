"""Generate the HW2 risk report for every ticker and summarize the portfolio (Part 3).

Usage (from any folder; paths resolve relative to this file)::

    python 03_run_all_stocks.py                      # all tickers, run date = today
    python 03_run_all_stocks.py --run-date 2026-11-18 --tickers AAPL KO

Steps:

1. read the tickers and the latest observation date from ``ohlc_hw_cleaned``;
2. choose the six complete calendar months before the month of that latest
   observation (anchored to the data, never to the computer clock);
3. run ``02_stock_report_template.ipynb`` with Papermill for each ticker into
   ``reports/{run_date}/{ticker}_report.ipynb``, logging and continuing on failure;
4. write ``failed_stocks.txt``, then collect the ``report_metrics`` scraps of the
   reports that succeeded in *this* run only and write ``portfolio_summary.csv``
   (fractions) and ``risk_return_profile.png``.

The script must run in the same environment as the notebook kernel; the child
reports are executed with this interpreter (see ``interpreter_kernel``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
from contextlib import closing, contextmanager
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import papermill as pm  # noqa: E402
import scrapbook as sb  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "02_stock_report_template.ipynb"
DEFAULT_DB = HERE / "data" / "data.db"
DEFAULT_OUTPUT = HERE / "reports"
TABLE = "ohlc_hw_cleaned"
KERNEL_NAME = "hw2-report"

SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
                   "total_return", "report_path"]
COVERAGE_COLUMNS = ["first_date", "last_date", "trading_days", "full_window"]
SUMMARY_FILE, FAILED_FILE, PLOT_FILE, LOG_FILE = ("portfolio_summary.csv", "failed_stocks.txt",
                                                  "risk_return_profile.png", "run.log")

log = logging.getLogger("hw2.run_all_stocks")


# ------------------------------------------------------------------ inputs

def six_complete_months(latest) -> tuple[str, str]:
    """Six calendar months before the month containing ``latest``.

    ``2024-12-06 -> ('2024-06-01', '2024-11-30')``. The rule is the same when
    ``latest`` is itself a month-end: a final trading date alone does not
    certify that the delivery for that month is complete.
    """
    latest = pd.Timestamp(latest)
    if pd.isna(latest):
        raise ValueError("cannot choose a window without a latest observation date")
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def read_universe(db_path: Path) -> tuple[list[str], str]:
    """All tickers and the latest observation date in the cleaned table (read-only)."""
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as con:
        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone() is None:
            raise LookupError(f"{TABLE} not found in {db_path}; run 01_data_cleaning.ipynb first")
        tickers = [row[0] for row in con.execute(f"SELECT DISTINCT ticker FROM {TABLE} ORDER BY ticker")]
        latest = con.execute(f"SELECT MAX(ts) FROM {TABLE}").fetchone()[0]
    if not tickers or latest is None:
        raise LookupError(f"{TABLE} is empty")
    return tickers, latest


# ------------------------------------------------------------- execution

@contextmanager
def interpreter_kernel():
    """Temporarily register a Jupyter kernel that runs *this* Python interpreter.

    The kernel spec lives in a temporary folder that is put on ``JUPYTER_PATH``
    for this process only, so reports always use the same packages as the
    script and nothing is installed into the user's Jupyter configuration.
    """
    with tempfile.TemporaryDirectory(prefix="hw2_kernel_") as root:
        spec_dir = Path(root) / "kernels" / KERNEL_NAME
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "HW2 report (script interpreter)", "language": "python"}))
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = root + (os.pathsep + previous if previous else "")
        try:
            yield KERNEL_NAME
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous


def run_report(ticker: str, start: str, end: str, output: Path, db_path: Path, kernel_name: str) -> Path:
    """Execute the template for one ticker; any failure propagates to the caller."""
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": str(db_path)},
        kernel_name=kernel_name, cwd=str(HERE), progress_bar=False, log_output=False)
    return output


def describe_error(exc: Exception) -> str:
    """One-line reason for the failure log."""
    if isinstance(exc, pm.PapermillExecutionError):
        return f"{exc.ename}: {str(exc.evalue).splitlines()[0] if exc.evalue else ''}"
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def collect_report(path: Path, ticker: str, start: str, end: str) -> dict:
    """Read one successful report's scraps into a summary row.

    The report must have finished (``report_complete``) and must belong to
    the requested ticker and window; otherwise ``ValueError`` is raised and
    the ticker is treated as failed. Undefined metrics (``None``) become ``NaN``.
    """
    scraps = sb.read_notebook(str(path)).scraps
    missing = {"report_complete", "report_metrics", "ticker", "start_date", "end_date"} - set(scraps.keys())
    if missing or scraps["report_complete"].data is not True:
        raise ValueError(f"report did not complete; missing scraps: {sorted(missing)}")
    identity = (scraps["ticker"].data, scraps["start_date"].data, scraps["end_date"].data)
    if identity != (ticker, start, end):
        raise ValueError(f"report is for {identity}, expected {(ticker, start, end)}")
    metrics = {k: (np.nan if v is None else v) for k, v in scraps["report_metrics"].data.items()}
    return {
        "ticker": ticker,
        "avg_daily_return": metrics["avg_daily_return"],
        "avg_volatility": metrics["avg_volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "total_return": metrics["total_return"],
        "report_path": path.resolve().relative_to(HERE).as_posix() if path.resolve().is_relative_to(HERE)
        else str(path.resolve()),
        "first_date": scraps["first_date"].data,
        "last_date": scraps["last_date"].data,
        "trading_days": scraps["trading_days"].data,
    }


# --------------------------------------------------------------- summary

def summarize(rows: list[dict]) -> pd.DataFrame:
    """Portfolio summary sorted by Sharpe ratio (descending; undefined Sharpe last).

    ``full_window`` is True when a report covers the same first and last
    trading dates as the longest report in the run, i.e. the whole common window.
    """
    if not rows:
        raise ValueError("no successful reports to summarize")
    summary = pd.DataFrame(rows)
    if summary["ticker"].duplicated().any():
        raise ValueError("duplicate tickers among collected reports")
    summary["full_window"] = (summary["first_date"].eq(summary["first_date"].min())
                              & summary["last_date"].eq(summary["last_date"].max()))
    summary = summary.sort_values(["sharpe_ratio", "ticker"], ascending=[False, True], na_position="last")
    return summary[SUMMARY_COLUMNS + COVERAGE_COLUMNS].reset_index(drop=True)


def rankings(summary: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top and bottom ``n`` by Sharpe, excluding undefined Sharpe ratios.

    The two lists overlap when fewer than ``2 * n`` Sharpe ratios are defined.
    """
    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False)
    return ranked.head(n), ranked.tail(n).sort_values("sharpe_ratio")


def format_for_display(frame: pd.DataFrame) -> str:
    """Percentages for display only; the CSV keeps fractions."""
    shown = frame[["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown",
                   "avg_daily_return", "last_date"]].copy()
    for col in ["total_return", "avg_volatility", "max_drawdown"]:
        shown[col] = shown[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:+.2%}" if col != "avg_volatility" else f"{v:.2%}")
    shown["avg_daily_return"] = shown["avg_daily_return"].map(lambda v: "N/A" if pd.isna(v) else f"{v:+.4%}")
    shown["sharpe_ratio"] = shown["sharpe_ratio"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.2f}")
    return shown.to_string(index=False)


def plot_risk_return(summary: pd.DataFrame, path: Path, start: str, end: str) -> plt.Figure:
    """Scatter of mean rolling volatility vs total return, coloured by Sharpe ratio.

    The five highest total returns are annotated. Reports that end before the
    common window ends are drawn as triangles; undefined Sharpe ratios are gray.
    """
    fig, ax = plt.subplots(figsize=(11, 7.5))
    defined = summary["sharpe_ratio"].notna()
    limit = np.nanmax(np.abs(summary["sharpe_ratio"])) if defined.any() else 1.0
    scatter = None
    for full, marker, label in [(True, "o", "full window"), (False, "^", "shorter history")]:
        part = summary[(summary["full_window"] == full) & defined]
        if len(part):
            scatter = ax.scatter(part["avg_volatility"] * 100, part["total_return"] * 100, c=part["sharpe_ratio"],
                                 cmap="RdYlGn", vmin=-limit, vmax=limit, marker=marker, s=90,
                                 edgecolors="black", linewidths=0.6, label=label)
    undefined = summary[~defined]
    if len(undefined):
        ax.scatter(undefined["avg_volatility"] * 100, undefined["total_return"] * 100, color="#9e9e9e",
                   marker="s", s=80, edgecolors="black", linewidths=0.6, label="Sharpe undefined")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    for row in summary.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility * 100, row.total_return * 100), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.margins(0.12)
    ax.grid(True, alpha=0.3)
    legend = ax.legend(loc="best", fontsize=9)
    for handle in legend.legend_handles:      # marker shape carries the meaning; colour belongs to the colourbar
        if handle.get_label() != "Sharpe undefined":
            handle.set_array(None)            # drop the colour mapping, otherwise it repaints the marker
            handle.set_facecolor("white")
    ax.set(xlabel="Mean 20-day rolling volatility, annualized (%)", ylabel="Total return (%)")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005, f"Requested window {start} to {end}; triangles: history ends before the window does. "
             "Top 5 total returns labelled.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    return fig


# ------------------------------------------------------------------ main

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB, help="database with ohlc_hw_cleaned")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="parent folder for run folders")
    parser.add_argument("--run-date", default=date.today().isoformat(),
                        help="names the output folder (default: today); independent of the analysis window")
    parser.add_argument("--tickers", nargs="+", help="run only these tickers (default: every ticker in the table)")
    args = parser.parse_args(argv)
    args.run_date = pd.Timestamp(args.run_date).strftime("%Y-%m-%d")
    return args


def _configure_logging(run_dir: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    handler = logging.FileHandler(run_dir / LOG_FILE, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)


def main(argv=None) -> int:
    args = parse_args(argv)
    db_path = args.db_path.resolve()
    all_tickers, latest = read_universe(db_path)
    start, end = six_complete_months(latest)
    tickers = [t.strip().upper() for t in args.tickers] if args.tickers else all_tickers

    run_dir = args.output_root.resolve() / args.run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir)
    for stale in [SUMMARY_FILE, PLOT_FILE]:
        (run_dir / stale).unlink(missing_ok=True)   # never leave an earlier run's summary behind
    print(f"Latest observation in {TABLE}: {latest} -> window {start} to {end} (six complete months)")
    print(f"Run date {args.run_date}: writing {len(tickers)} reports to {run_dir}")
    log.info("window %s to %s from latest %s; %d tickers", start, end, latest, len(tickers))

    rows, failures = [], []
    with interpreter_kernel() as kernel_name:
        for ticker in tqdm(tickers, desc="Generating reports"):
            output = run_dir / f"{ticker}_report.ipynb"
            try:
                run_report(ticker, start, end, output, db_path, kernel_name)
                rows.append(collect_report(output, ticker, start, end))
                (run_dir / f"{ticker}_report_FAILED.ipynb").unlink(missing_ok=True)
                log.info("%s: report written to %s", ticker, output.name)
            except Exception as exc:  # one bad ticker must not stop the batch
                reason = describe_error(exc)
                failures.append((ticker, reason))
                log.error("%s failed: %s", ticker, reason)
                tqdm.write(f"FAILED {ticker}: {reason}")
                if output.exists():
                    output.replace(run_dir / f"{ticker}_report_FAILED.ipynb")

    print(f"Successfully generated {len(rows)}/{len(tickers)} reports")
    (run_dir / FAILED_FILE).write_text("".join(f"{t}\t{reason}\n" for t, reason in failures))
    if failures:
        print(f"{len(failures)} failed; tickers and reasons saved to {run_dir / FAILED_FILE}")
    if not rows:
        print("No report succeeded, so there is no portfolio summary. See failed_stocks.txt and run.log.")
        log.error("all reports failed; no summary written")
        return 1

    summary = summarize(rows)
    summary.to_csv(run_dir / SUMMARY_FILE, index=False)
    top, bottom = rankings(summary)
    undefined = summary.loc[summary["sharpe_ratio"].isna(), "ticker"].tolist()
    print(f"\nTop {len(top)} by Sharpe ratio:\n{format_for_display(top)}")
    print(f"\nBottom {len(bottom)} by Sharpe ratio:\n{format_for_display(bottom)}")
    if len(top) + len(bottom) > summary["sharpe_ratio"].notna().sum():
        print("(fewer than 20 defined Sharpe ratios, so the top and bottom lists overlap)")
    if undefined:
        print(f"Excluded from rankings (undefined Sharpe ratio): {', '.join(undefined)}")
    short = summary[~summary["full_window"]]
    if len(short):
        print(f"\nCoverage: {len(short)} of {len(summary)} reports end before the window does "
              f"(last observations: {', '.join(sorted(short['last_date'].unique()))}). Their metrics describe a "
              "shorter period, so total returns are not directly comparable; nothing was extrapolated.")
    plot_risk_return(summary, run_dir / PLOT_FILE, start, end)
    print(f"\nSaved {run_dir / SUMMARY_FILE} and {run_dir / PLOT_FILE}")
    log.info("summary of %d reports written", len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
