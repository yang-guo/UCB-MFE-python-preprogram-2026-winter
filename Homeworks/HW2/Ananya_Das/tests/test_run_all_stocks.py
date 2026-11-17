import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# papermill 2.6 calls datetime.utcnow(), deprecated in Python 3.12; that is third-party noise, not ours
pytestmark = pytest.mark.filterwarnings("ignore:datetime.datetime.utcnow:DeprecationWarning")

SCRIPT = Path(__file__).resolve().parents[1] / "03_run_all_stocks.py"
spec = importlib.util.spec_from_file_location("run_all_stocks", SCRIPT)
ras = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ras)


# ------------------------------------------------------------- the window

@pytest.mark.parametrize("latest, expected", [
    ("2024-12-06", ("2024-06-01", "2024-11-30")),      # the assignment's example
    ("2024-11-30", ("2024-05-01", "2024-10-31")),      # latest on a month-end: still excludes its month
    ("2024-12-01", ("2024-06-01", "2024-11-30")),      # first day of a month
    ("2025-01-15", ("2024-07-01", "2024-12-31")),      # crosses a year boundary
    ("2024-03-31", ("2023-09-01", "2024-02-29")),      # leap-year February end
    ("2023-08-31", ("2023-02-01", "2023-07-31")),
])
def test_six_complete_months(latest, expected):
    assert ras.six_complete_months(latest) == expected


def test_window_is_exactly_six_whole_months():
    for latest in pd.date_range("2020-01-01", "2026-12-31", freq="7D"):
        start, end = map(pd.Timestamp, ras.six_complete_months(latest))
        assert start.day == 1 and (end + pd.Timedelta(days=1)).day == 1
        assert (end.to_period("M") - start.to_period("M")).n == 5
        assert end < latest.to_period("M").start_time


def test_six_complete_months_rejects_missing_date():
    with pytest.raises(ValueError):
        ras.six_complete_months(None)


# ------------------------------------------------------ summary / ranking

def rows(n, sharpe=None):
    out = []
    for i in range(n):
        out.append({"ticker": f"T{i:02d}", "avg_daily_return": 0.001 * i, "avg_volatility": 0.2 + 0.01 * i,
                    "max_drawdown": -0.05 * (i + 1), "sharpe_ratio": float(i) if sharpe is None else sharpe[i],
                    "total_return": 0.01 * i, "report_path": f"reports/x/T{i:02d}_report.ipynb",
                    "first_date": "2024-06-03", "last_date": "2024-11-29" if i % 2 else "2024-10-18",
                    "trading_days": 120})
    return out


def test_summary_columns_order_and_sharpe_sort():
    summary = ras.summarize(rows(4, sharpe=[0.5, np.nan, 2.0, -1.0]))
    assert list(summary.columns)[:7] == ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                                         "sharpe_ratio", "total_return", "report_path"]
    assert summary["ticker"].tolist() == ["T02", "T00", "T03", "T01"]          # NaN Sharpe last
    assert summary["full_window"].tolist() == [False, False, True, True]


def test_rankings_exclude_undefined_and_may_overlap():
    summary = ras.summarize(rows(5, sharpe=[0.5, np.nan, 2.0, -1.0, 1.0]))
    top, bottom = ras.rankings(summary)
    assert top["ticker"].tolist() == ["T02", "T04", "T00", "T03"]
    assert bottom["ticker"].tolist() == ["T03", "T00", "T04", "T02"]
    assert "T01" not in set(top["ticker"]) | set(bottom["ticker"])


def test_rankings_with_many_reports_take_ten_each():
    top, bottom = ras.rankings(ras.summarize(rows(30)))
    assert len(top) == len(bottom) == 10 and set(top["ticker"]).isdisjoint(bottom["ticker"])
    assert top["sharpe_ratio"].iloc[0] == 29 and bottom["sharpe_ratio"].iloc[0] == 0


def test_summarize_rejects_empty_and_duplicates():
    with pytest.raises(ValueError):
        ras.summarize([])
    with pytest.raises(ValueError):
        ras.summarize(rows(2) + rows(1))


# ------------------------------------------------------------- end to end

def synthetic_db(path):
    """GOOD1/GOOD2 trade all of 2024 to 2024-12-06; FLAT never moves; SHORT has only 10 November days."""
    full = pd.bdate_range("2024-01-02", "2024-12-06").strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    frames = []
    for ticker, closes in [("GOOD1", 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(full))))),
                           ("GOOD2", 50 * np.exp(np.cumsum(rng.normal(-0.0005, 0.02, len(full))))),
                           ("FLAT", np.full(len(full), 20.0))]:
        frames.append(pd.DataFrame({"ts": full, "ticker": ticker, "close": closes}))
    frames.append(pd.DataFrame({"ts": pd.bdate_range("2024-11-01", periods=10).strftime("%Y-%m-%d"),
                                "ticker": "SHORT", "close": np.linspace(10, 11, 10)}))
    data = pd.concat(frames, ignore_index=True)
    data = data.assign(open=data["close"], high=data["close"] * 1.01, low=data["close"] * 0.99,
                       adj_close=data["close"], volume=1000)
    with sqlite3.connect(path) as con:
        data.to_sql("ohlc_hw_cleaned", con, index=False)
    return path


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    root = tmp_path_factory.mktemp("batch")
    db = synthetic_db(root / "synthetic.db")
    run_dir = root / "reports" / "2030-01-15"
    run_dir.mkdir(parents=True)
    (run_dir / "STALE_report.ipynb").write_text("{}")          # leftover from another run: must be ignored
    code = ras.main(["--db-path", str(db), "--output-root", str(root / "reports"), "--run-date", "2030-01-15"])
    return code, run_dir, db, root


def test_batch_runs_every_ticker_and_continues_after_failure(batch, capsys):
    code, run_dir, _, _ = batch
    assert code == 0
    assert sorted(p.name for p in run_dir.glob("*_report.ipynb")) == [
        "FLAT_report.ipynb", "GOOD1_report.ipynb", "GOOD2_report.ipynb", "STALE_report.ipynb"]
    assert (run_dir / "SHORT_report_FAILED.ipynb").exists()
    failed = (run_dir / "failed_stocks.txt").read_text().splitlines()
    assert len(failed) == 1 and failed[0].startswith("SHORT\tValueError: SHORT: insufficient history")
    assert (run_dir / "run.log").read_text().count("SHORT failed") == 1


def test_batch_summary_uses_this_runs_successes_only(batch):
    _, run_dir, _, _ = batch
    summary = pd.read_csv(run_dir / "portfolio_summary.csv")
    assert set(summary["ticker"]) == {"GOOD1", "GOOD2", "FLAT"}                 # not SHORT, not STALE
    assert list(summary.columns[:7]) == ras.SUMMARY_COLUMNS
    assert summary["ticker"].iloc[-1] == "FLAT" and np.isnan(summary["sharpe_ratio"].iloc[-1])
    assert (summary["first_date"] == "2024-06-03").all() and (summary["last_date"] == "2024-11-29").all()
    assert summary["report_path"].str.endswith("_report.ipynb").all()
    assert (run_dir / "risk_return_profile.png").stat().st_size > 10_000


def test_batch_metrics_match_a_direct_calculation(batch):
    _, run_dir, db, _ = batch
    summary = pd.read_csv(run_dir / "portfolio_summary.csv").set_index("ticker")
    with sqlite3.connect(db) as con:
        p = pd.read_sql("SELECT adj_close FROM ohlc_hw_cleaned WHERE ticker='GOOD1' "
                        "AND ts BETWEEN '2024-06-01' AND '2024-11-30' ORDER BY ts", con)["adj_close"]
    r = np.log(p / p.shift(1)).dropna()
    assert summary.loc["GOOD1", "total_return"] == pytest.approx(p.iloc[-1] / p.iloc[0] - 1)
    assert summary.loc["GOOD1", "avg_daily_return"] == pytest.approx(r.mean())
    assert summary.loc["GOOD1", "avg_volatility"] == pytest.approx((r.rolling(20).std() * np.sqrt(252)).mean())
    assert summary.loc["GOOD1", "max_drawdown"] == pytest.approx((p / p.cummax() - 1).min())
    assert summary.loc["GOOD1", "sharpe_ratio"] == pytest.approx((r.mean() * 252 - 0.02) / (r.std() * np.sqrt(252)))


def test_collect_report_rejects_a_report_for_another_window(batch):
    _, run_dir, _, _ = batch
    with pytest.raises(ValueError, match="expected"):
        ras.collect_report(run_dir / "GOOD1_report.ipynb", "GOOD1", "2024-01-01", "2024-06-30")


def test_all_failed_run_writes_no_summary(batch, capsys):
    _, _, db, root = batch
    out = root / "reports_fail"
    stale = out / "2030-01-16"
    stale.mkdir(parents=True)
    (stale / "portfolio_summary.csv").write_text("old")
    code = ras.main(["--db-path", str(db), "--output-root", str(out), "--run-date", "2030-01-16",
                     "--tickers", "SHORT", "NOPE"])
    printed = capsys.readouterr().out
    assert code == 1
    assert "Successfully generated 0/2 reports" in printed and "no portfolio summary" in printed
    assert not (stale / "portfolio_summary.csv").exists() and not (stale / "risk_return_profile.png").exists()
    reasons = dict(line.split("\t", 1) for line in (stale / "failed_stocks.txt").read_text().splitlines())
    assert reasons["NOPE"].startswith("LookupError") and reasons["SHORT"].startswith("ValueError")
