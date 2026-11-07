"""Independent reconciliation of the recorded experiment and cash ledger."""

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from research import DATABASE, HERE, OUTPUT, save_json


def audit():
    candidates = sorted(
        OUTPUT.glob("final_test_corporate_actions_v*"),
        key=lambda path: int(path.name.rsplit("v", 1)[1]),
    )
    directory = candidates[-1]
    status = json.loads((directory / "evaluation_status.json").read_text())
    assert status["status"] == "completed"
    frozen = json.loads((OUTPUT / "frozen_experiment.json").read_text())
    config = frozen["config"]["strategy"]
    history = pd.read_csv(directory / "history.csv", parse_dates=["date"])
    fills = pd.read_csv(directory / "fills.csv", parse_dates=["date", "entry_date", "scheduled_exit"])
    orders = pd.read_csv(directory / "orders.csv", parse_dates=["signal_date", "entry_date"])
    lots = pd.read_csv(directory / "open_lots.csv")
    initial = config["initial_capital"]
    movements = np.where(
        fills.side.eq("buy"), -fills.gross - fills.fee,
        fills.gross - fills.fee,
    )
    daily_movements = pd.Series(movements, index=fills.date).groupby(level=0).sum()
    rebuilt_cash = initial + daily_movements.reindex(history.date, fill_value=0).cumsum()
    np.testing.assert_allclose(rebuilt_cash, history.cash, atol=1e-6, rtol=0)
    assert history.cash.min() >= -1e-6
    np.testing.assert_allclose(history.nav, history.cash + history.positions_value)
    sales = fills.loc[~fills.side.eq("buy")]
    assert sales.date.ge(sales.entry_date + pd.Timedelta(days=90)).all()
    assert sales.scheduled_exit.eq(sales.entry_date + pd.Timedelta(days=90)).all()
    ordinary = fills.loc[fills.side.isin(["buy", "sell"])]
    np.testing.assert_allclose(ordinary.fee, ordinary.gross * .001)
    assert fills.loc[fills.side.eq("cash_claim_release"), "fee"].eq(0).all()
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        calendar = pd.DatetimeIndex(connection.execute(
            "SELECT date FROM calendar ORDER BY date"
        ).fetchdf().date.astype("datetime64[ns]"))
        member = connection.execute(
            "SELECT symbol,quarter_start FROM universe_membership"
        ).fetchdf()
        final_quotes = connection.execute(
            "SELECT symbol,adj_close FROM prices WHERE date='2026-09-30'"
        ).fetchdf().set_index("symbol").adj_close
    signal_positions = calendar.get_indexer(orders.signal_date)
    entry_positions = calendar.get_indexer(orders.entry_date)
    assert np.array_equal(entry_positions, signal_positions + 1)
    expected_sessions = calendar[(calendar >= "2016-04-01") & (calendar <= "2026-09-30")]
    assert history.date.tolist() == expected_sessions.tolist()
    maturities = calendar.searchsorted(sales.scheduled_exit)
    actual_sales = calendar.get_indexer(sales.date)
    assert ((actual_sales - maturities >= 0) & (actual_sales - maturities <= 5)).all()
    buys = fills.loc[fills.side.eq("buy"), ["symbol", "date"]].copy()
    buys["quarter_start"] = buys.date.dt.to_period("Q").dt.start_time
    member["quarter_start"] = member.quarter_start.astype("datetime64[ns]")
    checked = buys.merge(member, on=["symbol", "quarter_start"], how="left", indicator=True)
    assert checked._merge.eq("both").all()
    final_marks = lots.symbol.map(final_quotes)
    assert final_marks.notna().all()
    np.testing.assert_allclose(lots.last_price, final_marks)
    final_positions = float((lots.units * final_marks).sum())
    assert abs(final_positions - history.iloc[-1].positions_value) < 1e-6
    assert pd.Timestamp(frozen["latest_training_outcome"]) < pd.Timestamp("2016-01-01")
    assert hashlib.sha256((OUTPUT / "frozen_model.joblib").read_bytes()).hexdigest() == frozen["model_sha256"]
    forecast_hash = hashlib.sha256((OUTPUT / "final_test/predictions.csv").read_bytes()).hexdigest()
    for revision in candidates:
        start_record = json.loads((revision / "test_started.json").read_text())
        assert start_record["predictions_sha256"] == forecast_hash
        assert start_record["frozen_model_sha256"] == frozen["model_sha256"]
    report = {
        "status": "passed", "completed_run": directory.name,
        "market_sessions": len(history), "elapsed_calendar_days": 3834,
        "orders": len(orders), "fills": len(fills),
        "minimum_cash": float(history.cash.min()),
        "maximum_cash_reconciliation_error": float(abs(rebuilt_cash.to_numpy() - history.cash).max()),
        "final_positions_marked_at_actual_test_end_quotes": len(lots),
        "final_cash": float(history.iloc[-1].cash),
        "final_positions_value": final_positions,
        "same_frozen_model_and_forecasts_across_corrections": True,
        "latest_training_outcome": frozen["latest_training_outcome"],
        "all_new_buys_in_historical_quarterly_membership": True,
        "all_orders_planned_one_session_before_entry": True,
        "all_realized_lots_preserve_90_day_maturity": True,
    }
    (HERE / "results").mkdir(exist_ok=True)
    save_json(HERE / "results/accounting_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    audit()
