"""
Automated multi-stock analysis with papermill.

Requires:
    2_stock_report_template.ipynb with 'parameters' tagged cell
      and a final scrapbook glue cell named "metrics".
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DB_PATH = Path(os.environ.get("DB_PATH", "data/data.db"))
TEMPLATE_NB = Path("02_stock_report_template.ipynb")
REPORTS_ROOT = Path("reports")
RUN_DATE = date.today().isoformat()          # run-day folder name
OUTPUT_DIR = REPORTS_ROOT / RUN_DATE
FAILED_FILE = OUTPUT_DIR / "failed_stocks.txt"
LOG_FILE = OUTPUT_DIR / "run_log.txt"
SUMMARY_CSV = OUTPUT_DIR / "portfolio_summary.csv"
SUMMARY_PNG = OUTPUT_DIR / "portfolio_summary.png"

# load tickers from the cleaned table
def load_tickers(db_path: Path) -> list[str]:
    """Return sorted, unique tickers from ohlc_hw_cleaned."""
    with sqlite3.connect(db_path) as con:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker",
            con,
        )["ticker"].tolist()
    return tickers


# last 6 complete months anchored to the latest observation
def latest_observation_date(db_path: Path) -> pd.Timestamp:
    with sqlite3.connect(db_path) as con:
        row = pd.read_sql_query(
            "SELECT MAX(ts) AS max_ts FROM ohlc_hw_cleaned", con
        )
    if row["max_ts"].isna().iloc[0]:
        raise RuntimeError("ohlc_hw_cleaned is empty.")
    return pd.Timestamp(row["max_ts"].iloc[0])


def compute_window(latest_ts: pd.Timestamp) -> tuple[str, str]:
    """
    Six complete calendar months BEFORE the month containing latest_ts.
    Example: latest = 2024-12-06  ->  start = 2024-06-01, end = 2024-11-30
    """
    last_month = latest_ts.to_period("M") - 1          # last complete month
    start_month = last_month - 5                       # six months inclusive
    start_date = start_month.start_time.date().isoformat()
    end_date = last_month.end_time.date().isoformat()
    return start_date, end_date


# run one notebook via papermill
def run_one(ticker: str, start_date: str, end_date: str) -> Path:
    """Execute the template for one ticker; return output notebook path."""
    out_nb = OUTPUT_DIR / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        input_path=str(TEMPLATE_NB),
        output_path=str(out_nb),
        parameters={
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "db_path": str(DB_PATH),
        },
        kernel_name="python3",
        progress_bar=False,
        log_output=False,
    )
    return out_nb


# collect metrics via scrapbook (only from successes)
def collect_metrics(output_nb: Path, ticker: str) -> dict | None:
    """Read the glued 'metrics' dict from a successfully executed notebook."""
    try:
        nb = sb.read_notebook(str(output_nb))
    except Exception as e:  # noqa: BLE001
        print(f"[collect] cannot read {output_nb}: {e}")
        return None
    if "metrics" not in nb.scraps.data_dict:
        print(f"[collect] no 'metrics' glue in {output_nb.name}")
        return None
    metrics = nb.scraps.data_dict["metrics"]
    # Validate required keys; guard against malformed glues.
    required = {"avg_daily_return", "avg_volatility", "max_drawdown",
                "sharpe_ratio", "total_return"}
    if not required.issubset(metrics.keys()):
        print(f"[collect] missing keys in {output_nb.name}")
        return None
    return {
        "ticker": ticker,
        "avg_daily_return": metrics["avg_daily_return"],
        "avg_volatility": metrics["avg_volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "total_return": metrics["total_return"],
        "report_path": str(output_nb),
    }


# summary plot
def plot_summary(summary_df: pd.DataFrame, out_png: Path) -> None:

    df = summary_df.dropna(subset=["sharpe_ratio"]).copy()
    if df.empty:
        print("[plot] no valid Sharpe values; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df["avg_volatility"], df["total_return"],
        c=df["sharpe_ratio"], cmap="viridis",
        s=80, edgecolor="k", linewidth=0.4, alpha=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Ratio")

    # annotate top-5 by total_return
    top5 = df.nlargest(5, "total_return")
    for _, r in top5.iterrows():
        ax.annotate(r["ticker"],
                    (r["avg_volatility"], r["total_return"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("Average Rolling Volatility (annualized)")
    ax.set_ylabel("Total Return (fraction)")
    ax.set_title("Risk-Return Profile of Portfolio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] saved -> {out_png}")


# Main
def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[main] DB           : {DB_PATH.resolve()}")
    print(f"[main] Template     : {TEMPLATE_NB}")
    print(f"[main] Output dir   : {OUTPUT_DIR.resolve()}")

    # window
    try:
        tickers = load_tickers(DB_PATH)
        latest_ts = latest_observation_date(DB_PATH)
        start_date, end_date = compute_window(latest_ts)
    except Exception as e:  # noqa: BLE001
        print(f"[fatal] cannot initialize: {e}")
        return 1

    print(f"[main] Tickers      : {len(tickers)}")
    print(f"[main] Latest obs   : {latest_ts.date()}")
    print(f"[main] Window       : {start_date} -> {end_date}")

    if not tickers:
        print("[fatal] no tickers in ohlc_hw_cleaned.")
        return 1

    # per-ticker execution
    successes: list[Path] = []
    failures: list[tuple[str, str]] = []
    log_lines: list[str] = []

    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            out_nb = run_one(ticker, start_date, end_date)
            successes.append(out_nb)
            log_lines.append(f"OK   {ticker} -> {out_nb.name}")
        except Exception as e:  # noqa: BLE001
            # Log the full traceback for debugging.
            tb = traceback.format_exc(limit=5)
            failures.append((ticker, f"{type(e).__name__}: {e}"))
            log_lines.append(f"FAIL {ticker}: {type(e).__name__}: {e}")
            log_lines.append(tb)

    # persist logs
    (OUTPUT_DIR / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    if failures:
        FAILED_FILE.write_text(
            "\n".join(f"{t}\t{msg}" for t, msg in failures),
            encoding="utf-8",
        )
    print(f"\n[main] Successfully generated {len(successes)}/{len(tickers)} reports")
    if failures:
        print(f"[main] {len(failures)} failures logged to {FAILED_FILE.name}")

    # collect metrics (only from successes)
    rows: list[dict] = []
    for nb in successes:
        ticker = nb.name.split("_report")[0]
        metrics = collect_metrics(nb, ticker)
        if metrics is not None:
            rows.append(metrics)

    if not rows:
        print("[summary] no usable metrics were collected; "
              "not writing portfolio_summary.csv.")
        return 2 if failures else 0

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        "sharpe_ratio", ascending=False, na_position="last",
    ).reset_index(drop=True)

    #display top/bottom (overlap allowed if < 20 successes)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n TOP 10 by Sharpe")
    print(summary_df.head(10).to_string(index=False))
    if len(summary_df) > 1:
        print("\n BOTTOM 10 by Sharpe")
        print(summary_df.tail(10).to_string(index=False))

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[summary] wrote {SUMMARY_CSV}")

    plot_summary(summary_df, SUMMARY_PNG)
    return 0


if __name__ == "__main__":
    sys.exit(main())