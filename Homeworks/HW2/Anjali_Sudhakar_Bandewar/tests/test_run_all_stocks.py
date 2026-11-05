"""Tests for 03_run_all_stocks.py using a synthetic database and mocked report execution.

No notebook is executed: a fake ``runner`` stands in for papermill and a fake
``reader`` stands in for Scrapbook, so each scenario is fast and deterministic.
"""

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "03_run_all_stocks.py"
spec = importlib.util.spec_from_file_location("run_all_stocks", SCRIPT)
ras = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ras)

START, END = "2024-06-01", "2024-11-30"


def make_db(path, last_dates):
    """Tiny ohlc_hw_cleaned with one row per business day up to each ticker's last date."""
    rows = []
    for ticker, last in last_dates.items():
        for ts in pd.bdate_range("2024-05-01", last):
            rows.append((ticker, ts.strftime("%Y-%m-%d"), 1.0, 1.0, 1.0, 1.0, 1.0, 100))
    with sqlite3.connect(path) as conn:
        pd.DataFrame(rows, columns=["ticker", "ts", "open", "high", "low", "close", "adj_close", "volume"]) \
            .to_sql("ohlc_hw_cleaned", conn, index=False)
    return path


def scraps_for(ticker, sharpe, last_date="2024-11-29", start=START, end=END):
    return {"ticker": ticker, "start_date": start, "end_date": end, "data_quality_ok": True,
            "first_date": "2024-06-03", "last_date": last_date, "trading_days": 100,
            "avg_daily_return": 0.001, "avg_volatility": 0.2 + 0.01 * len(ticker),
            "max_drawdown": -0.1, "sharpe_ratio": sharpe, "total_return": 0.05 * (sharpe or 0)}


# --- window selection --------------------------------------------------------

@pytest.mark.parametrize("latest, expected", [
    ("2024-12-06", ("2024-06-01", "2024-11-30")),     # this database
    ("2024-11-30", ("2024-05-01", "2024-10-31")),     # month-end: its month still excluded
    ("2025-01-15", ("2024-07-01", "2024-12-31")),     # crosses a year boundary
    ("2024-08-31", ("2024-02-01", "2024-07-31")),     # leap-year February
])
def test_six_complete_months(latest, expected):
    assert ras.six_complete_months(latest) == expected


def test_six_complete_months_rejects_missing_date():
    with pytest.raises(ValueError):
        ras.six_complete_months(None)


def test_load_universe_reads_cleaned_table(tmp_path):
    db = make_db(tmp_path / "t.db", {"BBB": "2024-10-18", "AAA": "2024-12-06"})
    tickers, latest = ras.load_universe(db)
    assert tickers == ["AAA", "BBB"] and latest == "2024-12-06"
    assert ras.six_complete_months(latest) == (START, END)


# --- batch execution and failures --------------------------------------------

def test_run_batch_continues_after_failures(tmp_path):
    def runner(ticker, start, end, folder):
        if ticker in {"AAA", "CCC"}:
            raise ValueError(f"Insufficient history for {ticker}")
        return Path(folder) / f"{ticker}_report.ipynb"

    successes, failures = ras.run_batch(["AAA", "BBB", "CCC", "DDD"], START, END, tmp_path, runner)
    assert list(successes) == ["BBB", "DDD"]          # the first ticker failing does not stop the loop
    assert list(failures) == ["AAA", "CCC"]
    assert "Insufficient history for AAA" in failures["AAA"]
    ras.write_failures(failures, tmp_path)
    assert (tmp_path / "failed_stocks.txt").read_text() == "AAA\nCCC\n"


def test_write_failures_empty_file_when_none(tmp_path):
    ras.write_failures({}, tmp_path)
    assert (tmp_path / "failed_stocks.txt").read_text() == ""


# --- summary -----------------------------------------------------------------

def test_summary_columns_sorting_and_undefined_sharpe(tmp_path):
    fake = {"AAA": scraps_for("AAA", 0.5), "BBB": scraps_for("BBB", None),
            "CCC": scraps_for("CCC", 2.0), "DDD": scraps_for("DDD", -1.0)}
    successes = {t: tmp_path / f"{t}_report.ipynb" for t in fake}
    results = ras.collect_results(successes, START, END, reader=lambda p: fake[p.name.split("_")[0]],
                                  base_dir=tmp_path)
    summary = ras.portfolio_summary(results)
    assert list(summary.columns) == ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                                     "sharpe_ratio", "total_return", "report_path"]
    assert summary["ticker"].tolist() == ["CCC", "AAA", "DDD", "BBB"]   # NaN Sharpe last
    assert np.isnan(summary.loc[3, "sharpe_ratio"])
    assert summary.loc[0, "report_path"] == "CCC_report.ipynb"
    top, bottom = ras.top_and_bottom(summary)
    assert "BBB" not in set(top["ticker"]) | set(bottom["ticker"])
    assert top["ticker"].tolist() == ["CCC", "AAA", "DDD"]               # fewer than 20: groups overlap
    assert set(top["ticker"]) == set(bottom["ticker"])


def test_collect_rejects_reports_from_another_request(tmp_path):
    fake = scraps_for("AAA", 1.0, start="2024-01-01", end="2024-06-30")
    with pytest.raises(ValueError, match="not this run's request"):
        ras.collect_results({"AAA": tmp_path / "AAA_report.ipynb"}, START, END, reader=lambda p: fake)


def test_collect_rejects_empty_success_list():
    with pytest.raises(ValueError):
        ras.collect_results({}, START, END)


def test_coverage_flags_tickers_ending_early(tmp_path):
    results = pd.DataFrame([
        {"ticker": "FULL", "first_date": "2024-06-03", "last_date": "2024-11-29", "trading_days": 125},
        {"ticker": "PART", "first_date": "2024-06-03", "last_date": "2024-10-18", "trading_days": 97},
    ])
    cov = ras.coverage_table(results, START, END)
    assert cov.set_index("ticker")["full_coverage"].to_dict() == {"FULL": True, "PART": False}
    msg = ras.coverage_message(cov)
    assert "1 of 2 reports cover only part" in msg and "2024-10-18" in msg and "no prices were extrapolated" in msg


# --- Sharpe color scale ------------------------------------------------------

@pytest.mark.parametrize("values, limit", [
    ([-0.5, 0.2, 1.5], 1.5),            # mixed signs
    ([0.4, 0.9, 2.0], 2.0),             # all positive
    ([-2.5, -0.3], 2.5),                # all negative
    ([0.7, 0.7, 0.7], 0.7),             # identical
    ([0.0, 0.0], 1.0),                  # all zero -> fallback
    ([np.nan, np.nan], 1.0),            # none defined -> fallback
    ([np.nan, -1.2, 0.5], 1.2),         # NaN ignored
])
def test_sharpe_norm_is_symmetric_around_zero(values, limit):
    norm = ras.sharpe_norm(values)
    assert (norm.vmin, norm.vmax) == (-limit, limit)
    assert norm(0.0) == pytest.approx(0.5)
    for v in values:
        if np.isfinite(v) and v != 0:
            assert (norm(v) > 0.5) == (v > 0)       # sign decides the side of the colormap


@pytest.mark.parametrize("sharpes", [[1.0, 0.5, 2.0], [-1.0, -0.5, -2.0], [0.3, 0.3, 0.3], [None, None, None],
                                     [1.0, None, -0.4]])
def test_scatter_renders_for_edge_cases(tmp_path, sharpes):
    fake = {t: scraps_for(t, s, last_date="2024-10-18" if t == "BB" else "2024-11-29")
            for t, s in zip(["A", "BB", "CCC"], sharpes)}
    results = ras.collect_results({t: tmp_path / f"{t}_report.ipynb" for t in fake}, START, END,
                                  reader=lambda p: fake[p.name.split("_")[0]], base_dir=tmp_path)
    summary = ras.portfolio_summary(results)
    out = tmp_path / "scatter.png"
    ras.plot_risk_return(summary, ras.coverage_table(results, START, END), out, START, END)
    assert out.stat().st_size > 0


# --- end to end with mocks ---------------------------------------------------

def test_main_end_to_end_with_mocks(tmp_path):
    db = make_db(tmp_path / "t.db", {"AAA": "2024-12-06", "BBB": "2024-10-18", "CCC": "2024-12-06"})
    reports_root = tmp_path / "reports"
    stale = reports_root / "2026-10-29"
    stale.mkdir(parents=True)
    (stale / "ZZZ_report.ipynb").write_text("stale report from an earlier run")
    fake = {"AAA": scraps_for("AAA", 1.0), "BBB": scraps_for("BBB", 0.3, last_date="2024-10-18")}

    def runner(ticker, start, end, folder):
        assert (start, end) == (START, END)
        if ticker == "CCC":
            raise RuntimeError("kernel died")
        return Path(folder) / f"{ticker}_report.ipynb"

    code = ras.main(["--db-path", str(db), "--reports-root", str(reports_root), "--run-date", "2026-10-29"],
                    runner=runner, reader=lambda p: fake[p.name.split("_")[0]])
    assert code == 0
    summary = pd.read_csv(stale / "portfolio_summary.csv")
    assert summary["ticker"].tolist() == ["AAA", "BBB"]            # stale ZZZ and failed CCC excluded
    assert len(summary.columns) == 7
    cov = pd.read_csv(stale / "report_coverage.csv")
    assert cov.set_index("ticker")["full_coverage"].to_dict() == {"AAA": True, "BBB": False}
    assert (stale / "failed_stocks.txt").read_text() == "CCC\n"
    assert (stale / "risk_return_scatter.png").stat().st_size > 0


def test_main_all_fail_writes_no_summary(tmp_path, capsys):
    db = make_db(tmp_path / "t.db", {"AAA": "2024-12-06", "BBB": "2024-12-06"})
    out = tmp_path / "reports" / "2026-10-29"
    out.mkdir(parents=True)
    (out / "portfolio_summary.csv").write_text("stale summary from an earlier run\n")

    def runner(ticker, start, end, folder):
        raise ValueError("boom")

    code = ras.main(["--db-path", str(db), "--reports-root", str(tmp_path / "reports"),
                     "--run-date", "2026-10-29"], runner=runner)
    printed = capsys.readouterr().out
    assert code == 1
    assert "Successfully generated 0/2 reports" in printed
    assert "no portfolio summary" in printed
    assert not (out / "portfolio_summary.csv").exists()            # stale summary removed, none written
    assert not (out / "risk_return_scatter.png").exists()
    assert (out / "failed_stocks.txt").read_text() == "AAA\nBBB\n"
