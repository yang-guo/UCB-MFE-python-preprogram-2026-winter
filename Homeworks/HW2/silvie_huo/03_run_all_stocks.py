#!/usr/bin/env python3
"""Automated multi-stock risk reports.

Runs ``02_stock_report_template.ipynb`` once per ticker with papermill, collects
each report's headline numbers through scrapbook, and builds a portfolio summary
(CSV, rankings and a risk-return scatter plot).

Pipeline
--------
1. Read every ticker from ``ohlc_hw_cleaned``.
2. Work out ONE common analysis window for all tickers: the six calendar months
   *before* the month that contains the latest observation in the database
   (the database's latest date, never the computer clock).
       latest = 2024-12-06  ->  2024-06-01 .. 2024-11-30
       latest = 2024-12-31  ->  2024-06-01 .. 2024-11-30   (month-end changes nothing:
                                a final trading date alone does not certify a complete month)
3. Create ``reports/<run date>/`` (the run date is today's date and is independent
   of the historical window), then execute the template for each ticker, logging and
   skipping any ticker that fails.
4. Collect the named scrap ``report_metrics`` from each *successfully executed*
   report (never by scanning the folder: failed or stale notebooks may live there),
   and write ``portfolio_summary.csv``, ``risk_return_profile.png`` and
   ``failed_stocks.txt`` next to the reports.

Usage
-----
    pip install papermill scrapbook tqdm ipykernel pandas matplotlib jinja2
    python 03_run_all_stocks.py
    python 03_run_all_stocks.py --tickers AAPL MSFT --run-date 2025-01-02

Comparability note
------------------
All tickers share the same requested window, but histories can start later or end
earlier. A report only ever uses the prices that exist: nothing is extrapolated
through a trailing gap. Each report's real coverage (first/last observation, number
of days) is recorded in the summary, and reports with shorter history are flagged
because their total return, volatility and drawdown cover fewer days and are not
strictly comparable with full-window tickers.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the script only saves figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

try:
    import papermill as pm
    import scrapbook as sb
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - friendly message instead of a traceback
    sys.exit(
        f"Missing dependency '{exc.name}'. Install the requirements with:\n"
        "    pip install papermill scrapbook tqdm ipykernel pandas matplotlib jinja2"
    )

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
TABLE_NAME = "ohlc_hw_cleaned"
WINDOW_MONTHS = 6
SCRAP_NAME = "report_metrics"  # name used by sb.glue(...) in the template notebook

# Relative to the project folder (main() switches to it), so no absolute paths are stored in outputs.
DEFAULT_DB = Path("data") / "data.db"
DEFAULT_TEMPLATE = Path("02_stock_report_template.ipynb")
DEFAULT_OUTPUT_ROOT = Path("reports")

# Spec columns first, then coverage columns that disclose each report's real history.
SUMMARY_COLUMNS = [
    "ticker", "avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio",
    "total_return", "report_path",
]
COVERAGE_COLUMNS = ["first_obs", "last_obs", "n_days", "coverage_pct", "full_window_coverage"]
METRIC_FIELDS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]

log = logging.getLogger("stock_reports")


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite database read-only (and fail instead of silently creating a new file)."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def load_tickers(db_path: Path) -> list[str]:
    """Return the sorted, de-duplicated tickers (upper-case, stripped) in ``ohlc_hw_cleaned``."""
    with closing(connect_readonly(db_path)) as conn:
        rows = conn.execute(
            f'SELECT DISTINCT ticker FROM "{TABLE_NAME}" WHERE ticker IS NOT NULL'
        ).fetchall()
    return sorted({str(r[0]).strip().upper() for r in rows if str(r[0]).strip()})


def load_trading_dates(db_path: Path) -> pd.DatetimeIndex:
    """Return every distinct observation date in the table as a sorted DatetimeIndex.

    Parsed in pandas so the result does not depend on how ``ts`` text is formatted.
    """
    with closing(connect_readonly(db_path)) as conn:
        raw = pd.read_sql_query(f'SELECT DISTINCT ts FROM "{TABLE_NAME}"', conn)["ts"]
    dates = pd.to_datetime(raw, errors="coerce").dropna().dt.normalize()
    return pd.DatetimeIndex(dates.drop_duplicates().sort_values())


def compute_analysis_window(latest_obs, months: int = WINDOW_MONTHS) -> tuple[str, str]:
    """Return ``(start, end)`` as ``YYYY-MM-DD`` for the last ``months`` complete calendar months.

    The window is the ``months`` calendar months *before the month containing*
    ``latest_obs``. That month is treated as incomplete even if ``latest_obs`` falls
    on its last day, so the rule is reproducible and independent of the computer clock.

    >>> compute_analysis_window("2024-12-06")
    ('2024-06-01', '2024-11-30')
    >>> compute_analysis_window("2024-12-31")
    ('2024-06-01', '2024-11-30')
    >>> compute_analysis_window("2025-01-15")
    ('2024-07-01', '2024-12-31')
    """
    month = pd.Timestamp(latest_obs).to_period("M")
    start = (month - months).start_time.normalize()
    end = (month - 1).end_time.normalize()
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Running reports
# --------------------------------------------------------------------------- #
def safe_filename(ticker: str) -> str:
    """Make a ticker safe to use in a file name (e.g. ``BRK/B`` -> ``BRK_B``)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker) or "_"


def _num(value) -> float:
    """Scrap value -> float (``None``/missing -> NaN)."""
    return np.nan if value is None else float(value)


def read_report_metrics(report_path: Path, ticker: str, start: str, end: str) -> dict:
    """Read and validate the ``report_metrics`` scrap from one executed report.

    Raises ``ValueError`` if the scrap is missing, or if the report was produced for a
    different ticker or window than requested (guards against stale/mismatched files).
    """
    scraps = sb.read_notebook(str(report_path)).scraps
    if SCRAP_NAME not in scraps:
        raise ValueError(f"scrap '{SCRAP_NAME}' not found (is scrapbook installed in the notebook kernel?)")
    data = scraps[SCRAP_NAME].data

    got = (str(data.get("ticker", "")).upper(), data.get("start_date"), data.get("end_date"))
    if got != (ticker.upper(), start, end):
        raise ValueError(f"report was generated for {got}, expected {(ticker.upper(), start, end)}")

    metrics = {field: _num(data.get(field)) for field in METRIC_FIELDS}
    metrics.update(
        first_obs=pd.Timestamp(data["first_obs"]),
        last_obs=pd.Timestamp(data["last_obs"]),
        n_days=int(data["n_days"]),
    )
    return metrics


def run_ticker(ticker: str, report_path: Path, args, start: str, end: str) -> dict:
    """Execute the template for one ticker and return its validated metrics.

    Any existing file at ``report_path`` is deleted first so a failed run can never leave
    an old report behind that looks current. Exceptions propagate to the caller.
    """
    report_path.unlink(missing_ok=True)
    pm.execute_notebook(
        str(args.template),
        str(report_path),
        parameters={
            "ticker": ticker,
            "start_date": start,
            "end_date": end,
            "db_path": args.db_path.as_posix(),
        },
        kernel_name=args.kernel,
        progress_bar=False,  # we draw our own tqdm bar
        log_output=False,
        execution_timeout=args.timeout,
    )
    return read_report_metrics(report_path, ticker, start, end)


def describe_error(exc: Exception) -> str:
    """One-line description of an exception (papermill errors carry ``ename``/``evalue``)."""
    name = getattr(exc, "ename", None) or type(exc).__name__
    detail = getattr(exc, "evalue", None) or str(exc)
    return f"{name}: {' '.join(str(detail).split())}"[:400]


class TqdmLoggingHandler(logging.Handler):
    """Print log records through ``tqdm.write`` so they do not break the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never crash the run
            self.handleError(record)


def setup_logging(log_file: Path) -> None:
    """Log everything to ``run.log``; show only warnings/errors on the console."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    console = TqdmLoggingHandler(level=logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(file_handler)
    log.addHandler(console)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def display_path(path: Path) -> str:
    """Path relative to the project folder when possible, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(Path(path).resolve())


def build_summary(results: dict[str, dict], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the portfolio summary DataFrame, sorted by Sharpe ratio (descending).

    ``results`` maps ticker -> metrics (+ ``report_path``) for *successful* reports only.
    ``calendar`` holds the database's trading dates inside the common window; it is used to
    flag reports whose actual history is shorter than the window. NaN Sharpe values are kept
    in the table (sorted last) but excluded from rankings by the caller.

    Columns: the spec columns, then first_obs, last_obs, n_days, coverage_pct and
    full_window_coverage. All returns/volatilities are raw fractions.
    """
    rows = []
    n_cal = len(calendar)
    for ticker, m in results.items():
        full = bool(
            n_cal and m["first_obs"] <= calendar[0] and m["last_obs"] >= calendar[-1] and m["n_days"] >= n_cal
        )
        rows.append({
            "ticker": ticker,
            **{f: m[f] for f in METRIC_FIELDS},
            "report_path": display_path(m["report_path"]),
            "first_obs": m["first_obs"].strftime("%Y-%m-%d"),
            "last_obs": m["last_obs"].strftime("%Y-%m-%d"),
            "n_days": m["n_days"],
            "coverage_pct": m["n_days"] / n_cal if n_cal else np.nan,
            "full_window_coverage": full,
        })
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS + COVERAGE_COLUMNS)
    return df.sort_values("sharpe_ratio", ascending=False, na_position="last", kind="mergesort").reset_index(drop=True)


def _pct(x: float, signed: bool = False) -> str:
    """Display-only percent format ('N/A' for NaN); calculations and CSVs keep fractions."""
    if pd.isna(x):
        return "N/A"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def format_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Format a slice of the summary for printing (percentages formatted here only)."""
    return pd.DataFrame({
        "ticker": df["ticker"],
        "sharpe": df["sharpe_ratio"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.2f}"),
        "total_return": df["total_return"].map(lambda v: _pct(v, signed=True)),
        "avg_volatility": df["avg_volatility"].map(_pct),
        "max_drawdown": df["max_drawdown"].map(_pct),
        "days": df["n_days"],
        "full_window": df["full_window_coverage"].map({True: "yes", False: "NO"}),
    })


def print_rankings(summary_df: pd.DataFrame) -> None:
    """Print the top 10 and bottom 10 tickers by Sharpe ratio.

    Undefined (NaN) Sharpe ratios are excluded. With fewer than 20 usable tickers the two
    groups overlap, which is stated explicitly.
    """
    ranked = summary_df.dropna(subset=["sharpe_ratio"])
    excluded = len(summary_df) - len(ranked)
    if excluded:
        print(f"\n({excluded} ticker(s) with undefined Sharpe ratio are excluded from the rankings.)")
    if ranked.empty:
        print("\nNo defined Sharpe ratios - rankings skipped.")
        return

    print("\nTop 10 by Sharpe ratio:")
    print(format_ranking(ranked.head(10)).to_string(index=False))
    print("\nBottom 10 by Sharpe ratio (lowest last):")
    print(format_ranking(ranked.tail(10)).to_string(index=False))
    if len(ranked) < 20:
        print(f"\nNote: only {len(ranked)} ranked ticker(s), so the top and bottom groups may overlap.")


def print_coverage_report(summary_df: pd.DataFrame, calendar: pd.DatetimeIndex, start: str, end: str) -> None:
    """Disclose which reports cover less than the common window and how to read them."""
    partial = summary_df[~summary_df["full_window_coverage"]]
    full_n = len(summary_df) - len(partial)
    print(f"\nCoverage check - common window {start} to {end} "
          f"({len(calendar)} trading dates in the database)")
    print(f"  {full_n}/{len(summary_df)} successful report(s) cover the full window.")
    if partial.empty:
        return
    print(f"  {len(partial)} report(s) have shorter history inside the window:")
    for _, row in partial.head(15).iterrows():
        print(f"    {row['ticker']:<8} {row['first_obs']} -> {row['last_obs']}  "
              f"({row['n_days']}/{len(calendar)} days, {row['coverage_pct']:.0%})")
    if len(partial) > 15:
        print(f"    ... and {len(partial) - 15} more (see first_obs/last_obs/n_days in portfolio_summary.csv)")
    print("  Reading these: each statistic uses only the days that exist (prices are never extrapolated\n"
          "  through a trailing gap), so total return, volatility and drawdown cover fewer days and are\n"
          "  not like-for-like with full-window tickers. They are drawn as triangles in the scatter plot.")


def plot_risk_return(summary_df: pd.DataFrame, out_path: Path, start: str, end: str) -> Path | None:
    """Save the risk-return scatter plot and return its path (``None`` if nothing is plottable).

    X = average rolling volatility, Y = total return, colour = Sharpe ratio (colourbar).
    Triangles mark reports with shorter-than-window history; gray crosses mark undefined
    Sharpe ratios. The five highest-return stocks are annotated.
    """
    df = summary_df.dropna(subset=["avg_volatility", "total_return"])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="#FBFCFE")
    has_sharpe = df["sharpe_ratio"].notna()

    # Colour scale centred on 0 when Sharpe values span it (green = good, red = bad).
    vmin, vmax = (df.loc[has_sharpe, "sharpe_ratio"].min(), df.loc[has_sharpe, "sharpe_ratio"].max()) \
        if has_sharpe.any() else (-1.0, 1.0)
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else Normalize(vmin, vmax)

    mappable = None
    for full, marker in ((True, "o"), (False, "^")):
        sub = df[has_sharpe & (df["full_window_coverage"] == full)]
        if not sub.empty:
            mappable = ax.scatter(sub["avg_volatility"], sub["total_return"], c=sub["sharpe_ratio"],
                                  cmap="RdYlGn", norm=norm, marker=marker, s=95, edgecolor="#2B2F36",
                                  linewidth=0.7, alpha=0.92, zorder=3)
    undefined = df[~has_sharpe]
    if not undefined.empty:
        ax.scatter(undefined["avg_volatility"], undefined["total_return"], c="#9AA3AE", marker="x",
                   s=70, linewidth=1.8, zorder=3)
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, pad=0.02).set_label("Sharpe ratio (annualized)")

    # Annotate the five highest-return stocks, staggering offsets to limit overlaps.
    offsets = [(10, 10), (10, -16), (-48, 10), (-48, -16), (12, 24)]
    for (_, row), (dx, dy) in zip(df.nlargest(5, "total_return").iterrows(), offsets):
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10, fontweight="bold", color="#2B2F36",
                    arrowprops=dict(arrowstyle="-", color="#6B7280", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D5DAE1"), zorder=5)

    ax.axhline(0, color="#2B2F36", lw=1, zorder=2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
    ax.set_xlabel("Average volatility (mean 20-day rolling, annualized)")
    ax.set_ylabel("Total return over the window")
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=16, fontweight="bold", loc="left", pad=14)
    ax.grid(True, color="#D5DAE1", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", mfc="#BBBBBB", mec="#2B2F36", label="Full-window history"),
               Line2D([], [], marker="^", ls="", mfc="#BBBBBB", mec="#2B2F36", label="Shorter history (partial window)")]
    if not undefined.empty:
        handles.append(Line2D([], [], marker="x", ls="", color="#9AA3AE", label="Sharpe undefined"))
    ax.legend(handles=handles, loc="best", fontsize=9.5)

    fig.text(0.5, 0.01, f"Window {start} to {end} - {len(df)} ticker(s) - labelled: five highest total returns",
             ha="center", fontsize=9.5, color="#6B7280")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_date_arg(value: str) -> str:
    """argparse type: accept only a real ``YYYY-MM-DD`` date (it becomes a folder name)."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid run date {value!r}; expected YYYY-MM-DD") from None


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line options (all have sensible defaults next to this script)."""
    p = argparse.ArgumentParser(description="Generate a risk report notebook for every ticker with papermill.")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB,
                   help="SQLite database (default: data/data.db; relative paths resolve from the project folder)")
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                   help="template notebook")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="parent of the dated output folder ")
    p.add_argument("--run-date", type=run_date_arg, default=date.today().isoformat(),
                   help="folder name reports/<run-date>/ (default: today; unrelated to the analysis window)")
    p.add_argument("--tickers", nargs="+", help="only run these tickers (default: all tickers in the database)")
    p.add_argument("--kernel", default=None, help="Jupyter kernel name (default: the one stored in the template)")
    p.add_argument("--timeout", type=int, default=600, help="per-notebook execution timeout in seconds")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the full pipeline. Returns a process exit code (1 if nothing could be produced)."""
    args = parse_args(argv)
    os.chdir(BASE_DIR)  # relative paths resolve from the project folder; the notebook kernel inherits it
    if not args.template.is_file():
        print(f"Template notebook not found: {args.template}")
        return 1

    # 1) tickers  2) common window ------------------------------------------------
    try:
        all_tickers = load_tickers(args.db_path)
        trading_dates = load_trading_dates(args.db_path)
    except (FileNotFoundError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        print(f"Could not read '{TABLE_NAME}' from {args.db_path}: {exc}")
        return 1
    if trading_dates.empty or not all_tickers:
        print(f"Table '{TABLE_NAME}' has no usable tickers/dates.")
        return 1

    latest = trading_dates.max()
    start, end = compute_analysis_window(latest)
    calendar = trading_dates[(trading_dates >= start) & (trading_dates <= end)]

    tickers = all_tickers
    if args.tickers:
        wanted = [t.strip().upper() for t in args.tickers]
        unknown = [t for t in wanted if t not in all_tickers]
        if unknown:
            print(f"Ignoring tickers not in the database: {unknown}")
        tickers = [t for t in dict.fromkeys(wanted) if t in all_tickers]
    if not tickers:
        print("No tickers to process.")
        return 1

    # 3) output directory -----------------------------------------------------------
    out_dir = args.output_root / args.run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    # A re-run on the same day must never leave the previous run's summary behind, otherwise an
    # all-failed run would sit next to an old summary that looks current.
    for stale in ("portfolio_summary.csv", "risk_return_profile.png", "failed_stocks.txt"):
        (out_dir / stale).unlink(missing_ok=True)
    setup_logging(out_dir / "run.log")

    print(f"Database           : {args.db_path}")
    print(f"Tickers            : {len(tickers)}")
    print(f"Latest observation : {latest:%Y-%m-%d}")
    print(f"Analysis window    : {start} to {end}  ({len(calendar)} trading dates in the database)")
    print(f"Output directory   : {out_dir}\n")
    log.info("window=%s..%s latest=%s tickers=%d", start, end, f"{latest:%Y-%m-%d}", len(tickers))
    if calendar.empty:
        print("The database has no observations inside the analysis window - nothing to report.")
        return 1

    # 4) run papermill per ticker ---------------------------------------------------
    results: dict[str, dict] = {}
    failed: list[str] = []
    used_names: set[str] = set()

    for ticker in tqdm(tickers, desc="Generating reports"):
        name = safe_filename(ticker)
        while name in used_names:  # two tickers sanitising to the same file name
            name += "_"
        used_names.add(name)
        report_path = out_dir / f"{name}_report.ipynb"
        try:
            metrics = run_ticker(ticker, report_path, args, start, end)
            metrics["report_path"] = report_path
            results[ticker] = metrics
            log.info("%s OK (%d days, %s -> %s)", ticker, metrics["n_days"],
                     f"{metrics['first_obs']:%Y-%m-%d}", f"{metrics['last_obs']:%Y-%m-%d}")
        except Exception as exc:  # noqa: BLE001 - one bad ticker must never stop the run
            failed.append(ticker)
            log.error("%s FAILED - %s", ticker, describe_error(exc))

    n_ok, n_total = len(results), len(tickers)
    failed_file = out_dir / "failed_stocks.txt"
    failed_file.write_text("".join(f"{t}\n" for t in failed), encoding="utf-8")  # always rewritten: never stale

    print(f"\nSuccessfully generated {n_ok}/{n_total} reports")
    if failed:
        print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")
        print(f"  list saved to {failed_file}; reasons are in {out_dir / 'run.log'}")

    # 5) summary --------------------------------------------------------------------
    if not results:
        print("\nAll reports failed - no portfolio summary, rankings or plot were created.")
        return 1

    summary_df = build_summary(results, calendar)
    csv_path = out_dir / "portfolio_summary.csv"
    summary_df.to_csv(csv_path, index=False)  # raw fractions; percentages are formatted only for display

    print_coverage_report(summary_df, calendar, start, end)
    print_rankings(summary_df)

    plot_path = plot_risk_return(summary_df, out_dir / "risk_return_profile.png", start, end)
    print(f"\nSaved summary : {csv_path}")
    print(f"Saved plot    : {plot_path if plot_path else 'skipped (no plottable points)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())