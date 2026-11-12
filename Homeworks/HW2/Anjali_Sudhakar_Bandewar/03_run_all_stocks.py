"""Generate a risk report for every stock in ``ohlc_hw_cleaned`` and summarize the batch.

Run order: execute ``01_data_cleaning.ipynb`` first (it creates ``ohlc_hw_cleaned``), then::

    python 03_run_all_stocks.py                       # every ticker
    python 03_run_all_stocks.py --tickers AAPL KO     # a smoke test on a few tickers

What it does (HW2 Part 3):

1. Reads the tickers and the latest observation date from ``ohlc_hw_cleaned``.
2. Picks the six complete calendar months before the month containing that latest
   date (Dec 6, 2024 -> June 1 to Nov 30, 2024). The computer clock is not used.
3. Runs ``02_stock_report_template.ipynb`` with papermill once per ticker, saving
   ``reports/{run_date}/{ticker}_report.ipynb``. A failure is logged and the batch
   continues; failed tickers go to ``failed_stocks.txt`` (errors in ``failed_stocks.json``).
4. Collects Scrapbook metrics **only from reports that succeeded in this run**
   and writes ``portfolio_summary.csv`` (the seven required columns, sorted by
   Sharpe ratio), ``report_coverage.csv`` (observed dates per report) and
   ``risk_return_scatter.png``.

Histories end on different dates (some on 2024-10-18, others on 2024-12-06), so a
common requested window does not mean common observed coverage. Each report's
observed dates are recorded and partial coverage is reported; prices are never
extrapolated past a ticker's last observation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the script only saves figures; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import papermill as pm  # noqa: E402
import scrapbook as sb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from tqdm import tqdm  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR / "02_stock_report_template.ipynb"
DEFAULT_DB = SCRIPT_DIR / "data" / "data.db"
DEFAULT_REPORTS_ROOT = SCRIPT_DIR / "reports"

SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                   "sharpe_ratio", "total_return", "report_path"]
# Scrap name in the report -> summary column (identical names by design).
METRIC_SCRAPS = SUMMARY_COLUMNS[1:-1]
COVERAGE_SCRAPS = ["first_date", "last_date", "trading_days"]
# A report counts as covering the window if its observations start and end within
# this many calendar days of the requested bounds (allows for weekends/holidays).
COVERAGE_TOLERANCE_DAYS = 7
# Outputs of a previous run on the same day that must not survive into this run.
RUN_OUTPUTS = ["portfolio_summary.csv", "report_coverage.csv", "risk_return_scatter.png",
               "failed_stocks.txt", "failed_stocks.json", "run_manifest.json"]


# ---------------------------------------------------------------------------
# Inputs: tickers and the analysis window
# ---------------------------------------------------------------------------

def six_complete_months(latest_date) -> tuple[str, str]:
    """Six calendar months before the month containing ``latest_date`` (HW2 rule).

    The latest observation's own month is always excluded, even when that
    observation falls on a month-end: a final trading date alone does not show
    the delivery for that month is complete.

    >>> six_complete_months("2024-12-06")
    ('2024-06-01', '2024-11-30')
    """
    latest = pd.Timestamp(latest_date)
    if pd.isna(latest):
        raise ValueError("Cannot choose a window without a latest observation date.")
    month_start = latest.to_period("M").start_time
    start = month_start - pd.DateOffset(months=6)
    end = month_start - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def load_universe(db_path: Path) -> tuple[list[str], str]:
    """Distinct tickers and the latest ``ts`` in ``ohlc_hw_cleaned`` (read-only)."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}. Run 01_data_cleaning.ipynb first.")
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        tickers = [row[0] for row in conn.execute(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker")]
        (latest,) = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()
    if not tickers or latest is None:
        raise ValueError("ohlc_hw_cleaned is empty; run 01_data_cleaning.ipynb first.")
    return tickers, latest


# ---------------------------------------------------------------------------
# Running the reports
# ---------------------------------------------------------------------------

def run_report(ticker: str, start_date: str, end_date: str, out_dir: Path, db_path: Path,
               kernel_name: str | None = None) -> Path:
    """Execute the template for one ticker; raise on any failure so the caller can log it.

    ``kernel_name=None`` uses the notebook's own kernel ("python3"), which
    resolves to the Python environment running this script.
    """
    output = Path(out_dir) / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(TEMPLATE), str(output),
        parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                    "db_path": str(Path(db_path).resolve())},
        kernel_name=kernel_name, cwd=str(SCRIPT_DIR), progress_bar=False,
    )
    return output


def run_batch(tickers, start_date, end_date, out_dir, runner):
    """Run ``runner(ticker, start, end, out_dir)`` for every ticker, never stopping on a failure.

    Returns ``(successes, failures)``: ticker -> report path, and ticker -> error
    message. Only paths returned by a completed call are successes; a notebook
    papermill saved for a failed run is never treated as a result.
    """
    successes, failures = {}, {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            successes[ticker] = Path(runner(ticker, start_date, end_date, out_dir))
        except Exception as exc:  # any failure for one ticker must not stop the batch
            message = f"{type(exc).__name__}: {exc}".strip()
            if isinstance(exc, pm.PapermillExecutionError):
                message = f"{exc.ename}: {exc.evalue}"
            failures[ticker] = message
            tqdm.write(f"FAILED {ticker}: {message}")
    return successes, failures


def write_failures(failures: dict, out_dir: Path) -> None:
    """``failed_stocks.txt`` (one ticker per line, empty if none) and ``failed_stocks.json`` (errors)."""
    out_dir = Path(out_dir)
    (out_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failures))
    (out_dir / "failed_stocks.json").write_text(json.dumps(
        [{"ticker": t, "error": e} for t, e in failures.items()], indent=2))


# ---------------------------------------------------------------------------
# Collecting results
# ---------------------------------------------------------------------------

def read_scraps(path: Path) -> dict:
    """Named Scrapbook outputs of one executed report as a plain dict."""
    return {name: scrap.data for name, scrap in sb.read_notebook(str(path)).scraps.items()}


def relative_path(path: Path, base_dir: Path = SCRIPT_DIR) -> str:
    """Path relative to the homework folder when possible, so outputs stay portable."""
    path = Path(path)
    return str(path.relative_to(base_dir) if path.is_relative_to(base_dir) else path)


def _as_float(value) -> float:
    """Scrapbook stores undefined metrics as None; turn them back into NaN."""
    return np.nan if value is None else float(value)


def collect_results(successes: dict, start_date: str, end_date: str, reader=read_scraps,
                    base_dir: Path = SCRIPT_DIR) -> pd.DataFrame:
    """One row per successful report: summary metrics plus observed coverage.

    Every report must belong to this run's request (same ticker, start and end)
    and pass its own data-quality check; anything else is an error rather than
    silently mixed into the summary.
    """
    if not successes:
        raise ValueError("No successful reports to summarize.")
    rows = []
    for ticker, path in successes.items():
        scraps = reader(path)
        missing = [k for k in METRIC_SCRAPS + COVERAGE_SCRAPS + ["ticker", "start_date", "end_date"]
                   if k not in scraps]
        if missing:
            raise ValueError(f"{path}: missing scraps {missing}")
        if (scraps["ticker"], scraps["start_date"], scraps["end_date"]) != (ticker, start_date, end_date):
            raise ValueError(f"{path}: report is for {scraps['ticker']} {scraps['start_date']}.."
                             f"{scraps['end_date']}, not this run's request")
        if scraps.get("data_quality_ok") is not True:
            raise ValueError(f"{path}: report did not pass its data-quality check")
        rows.append({
            "ticker": ticker,
            **{name: _as_float(scraps[name]) for name in METRIC_SCRAPS},
            "report_path": relative_path(path, base_dir),
            "first_date": scraps["first_date"], "last_date": scraps["last_date"],
            "trading_days": int(scraps["trading_days"]),
        })
    results = pd.DataFrame(rows)
    if results["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers among this run's reports.")
    return results


def portfolio_summary(results: pd.DataFrame) -> pd.DataFrame:
    """The seven required columns, sorted by Sharpe ratio (descending, undefined last)."""
    return (results[SUMMARY_COLUMNS]
            .sort_values("sharpe_ratio", ascending=False, na_position="last", kind="stable")
            .reset_index(drop=True))


def coverage_table(results: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Requested vs observed dates per report, with a ``full_coverage`` flag."""
    tolerance = pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)
    first = pd.to_datetime(results["first_date"])
    last = pd.to_datetime(results["last_date"])
    table = results[["ticker", "first_date", "last_date", "trading_days"]].copy()
    table.insert(1, "requested_start", start_date)
    table.insert(2, "requested_end", end_date)
    table["full_coverage"] = ((first <= pd.Timestamp(start_date) + tolerance)
                              & (last >= pd.Timestamp(end_date) - tolerance))
    return table.sort_values("ticker").reset_index(drop=True)


def top_and_bottom(summary: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top and bottom ``n`` by Sharpe, excluding undefined Sharpe values.

    With fewer than ``2 * n`` ranked stocks the two groups overlap.
    """
    ranked = summary.dropna(subset=["sharpe_ratio"]).sort_values(
        "sharpe_ratio", ascending=False, kind="stable")
    return ranked.head(n), ranked.tail(n)


def sharpe_norm(values) -> plt.Normalize:
    """Color scale for Sharpe ratios that is symmetric around zero.

    Zero always maps to the middle of the diverging colormap, negative values to
    the red half and positive values to the green half, with equal intensity for
    equal magnitude. The limit is the largest absolute defined value, so the
    scale still works when every value is positive, negative or identical; it
    falls back to +/-1 when all values are zero or undefined.
    """
    finite = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    limit = float(finite.abs().max()) if not finite.empty else 0.0
    if limit == 0.0:
        limit = 1.0
    return plt.Normalize(vmin=-limit, vmax=limit)


def plot_risk_return(summary: pd.DataFrame, coverage: pd.DataFrame, path: Path,
                     start_date: str, end_date: str) -> None:
    """Scatter of mean rolling volatility vs total return, colored by Sharpe; top 5 returns labeled.

    Round markers cover the full window; triangles mark reports whose
    observations end (or start) early, so their total return covers a shorter
    period and is not directly comparable.
    """
    data = summary.merge(coverage[["ticker", "full_coverage", "last_date"]], on="ticker")
    fig, ax = plt.subplots(figsize=(12, 8))
    defined = data["sharpe_ratio"].notna()
    norm = sharpe_norm(data["sharpe_ratio"])
    points = None
    for full, marker in [(True, "o"), (False, "^")]:
        part = data[defined & data["full_coverage"].eq(full)]
        if not part.empty:
            points = ax.scatter(part["avg_volatility"], part["total_return"], c=part["sharpe_ratio"],
                                cmap="RdYlGn", norm=norm, marker=marker, s=90, edgecolors="black",
                                linewidths=0.6)
    undefined = data[~defined]
    if not undefined.empty:
        ax.scatter(undefined["avg_volatility"], undefined["total_return"], color="0.6", marker="x", s=70)
    if points is not None:
        colorbar = fig.colorbar(points, ax=ax, label="Sharpe ratio (annualized; red < 0 < green)")
        colorbar.ax.axhline(0, color="black", lw=1.2)  # mark the zero midpoint
    for row in data.nlargest(5, "total_return").itertuples():
        ax.annotate(row.ticker, (row.avg_volatility, row.total_return), xytext=(6, 6),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel="Average 20-day rolling volatility (annualized)",
           ylabel="Total return over observed period", title="Risk-Return Profile of Portfolio")
    ax.grid(alpha=0.3)
    partial = data[~data["full_coverage"]]
    handles = [Line2D([], [], marker="o", ls="", color="0.4", label=f"Full window {start_date} to {end_date}")]
    if not partial.empty:
        handles.append(Line2D([], [], marker="^", ls="", color="0.4",
                              label=f"Partial: data end {', '.join(sorted(partial['last_date'].unique()))}"))
    if not undefined.empty:
        handles.append(Line2D([], [], marker="x", ls="", color="0.6", label="Sharpe undefined"))
    ax.legend(handles=handles, loc="best")
    ax.margins(0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coverage_message(coverage: pd.DataFrame) -> str:
    """Plain-language disclosure of partial coverage for the console."""
    partial = coverage[~coverage["full_coverage"]]
    if partial.empty:
        return "All successful reports cover the full requested window."
    ends = ", ".join(f"{d} ({n})" for d, n in partial["last_date"].value_counts().sort_index().items())
    return (f"{len(partial)} of {len(coverage)} reports cover only part of the requested window "
            f"(last observations: {ends}). Their metrics describe a shorter period, so compare their "
            f"total returns with full-window stocks cautiously; no prices were extrapolated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers", nargs="+", help="Run only these tickers (default: all).")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT,
                        help="Reports go to <reports-root>/<run-date>/ (default: reports/).")
    parser.add_argument("--run-date", default=date.today().isoformat(),
                        help="Output folder name, YYYY-MM-DD (default: today).")
    parser.add_argument("--kernel", default=None, help="Jupyter kernel name (default: notebook's).")
    return parser.parse_args(argv)


def main(argv=None, runner=None, reader=read_scraps) -> int:
    """Run the batch; returns a process exit code (1 when no report succeeded)."""
    args = parse_args(argv)
    run_date = date.fromisoformat(args.run_date).isoformat()
    all_tickers, latest = load_universe(args.db_path)
    tickers = all_tickers
    if args.tickers:
        requested = [t.strip().upper() for t in args.tickers]
        unknown = sorted(set(requested) - set(all_tickers))
        if unknown:
            print(f"Note: not in ohlc_hw_cleaned (their reports will fail): {', '.join(unknown)}")
        tickers = requested
    start_date, end_date = six_complete_months(latest)

    out_dir = Path(args.reports_root).resolve() / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_OUTPUTS:  # stale results from an earlier run today must not be mistaken for this run's
        (out_dir / name).unlink(missing_ok=True)

    print(f"Latest observation in ohlc_hw_cleaned: {latest}")
    print(f"Analysis window (six complete months): {start_date} to {end_date}")
    print(f"Run date / output folder: {out_dir}")

    if runner is None:
        def runner(ticker, start, end, folder):
            return run_report(ticker, start, end, folder, args.db_path, args.kernel)

    successes, failures = run_batch(tickers, start_date, end_date, out_dir, runner)
    write_failures(failures, out_dir)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "run_date": run_date, "latest_observation": latest, "start_date": start_date,
        "end_date": end_date, "tickers_requested": tickers,
        "successful_reports": {t: relative_path(p) for t, p in successes.items()},
        "failed_tickers": list(failures)}, indent=2))
    print(f"Successfully generated {len(successes)}/{len(tickers)} reports")

    if not successes:
        print("No report succeeded, so there is no portfolio summary. "
              f"See {out_dir / 'failed_stocks.json'} for the errors.")
        return 1
    if failures:
        print(f"Failed tickers saved to {out_dir / 'failed_stocks.txt'}: {', '.join(failures)}")

    results = collect_results(successes, start_date, end_date, reader=reader)
    summary = portfolio_summary(results)
    coverage = coverage_table(results, start_date, end_date)
    summary.to_csv(out_dir / "portfolio_summary.csv", index=False)
    coverage.to_csv(out_dir / "report_coverage.csv", index=False)
    plot_risk_return(summary, coverage, out_dir / "risk_return_scatter.png", start_date, end_date)

    top, bottom = top_and_bottom(summary)
    shown = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown"]
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(f"\nTop {len(top)} by Sharpe ratio:\n{top[shown].to_string(index=False)}")
        print(f"\nBottom {len(bottom)} by Sharpe ratio:\n{bottom[shown].to_string(index=False)}")
    n_undefined = int(summary["sharpe_ratio"].isna().sum())
    if n_undefined:
        print(f"\n{n_undefined} stock(s) with undefined Sharpe ratio excluded from the rankings.")
    if set(top["ticker"]) & set(bottom["ticker"]):
        print("Note: fewer than 20 ranked stocks, so the top and bottom groups overlap.")
    print("\n" + coverage_message(coverage))
    print(f"\nSaved: {out_dir / 'portfolio_summary.csv'}, {out_dir / 'report_coverage.csv'}, "
          f"{out_dir / 'risk_return_scatter.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
