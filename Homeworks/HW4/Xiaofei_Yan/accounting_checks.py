"""Run small hand-verifiable accounting/timing examples required by HW4."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from portfolio_backtest import simulate, UnreliableHeldPrice


def run_checks():
    """Check fees, holiday maturity, repeated lots, gaps and causal sizing."""
    # This is an explicit toy session calendar, not an exchange calendar.
    calendar = pd.bdate_range("2014-12-31", "2015-05-05").difference(pd.DatetimeIndex(["2015-01-01"]))
    quotes = pd.DataFrame({"date": calendar, "symbol": "DEMO", "adj_close": 100.0})
    benchmark = quotes[["date", "adj_close"]].copy()
    scores = pd.DataFrame({"date": [pd.Timestamp("2014-12-31")], "symbol": ["DEMO"], "score": [0.1]})
    start, end = "2015-01-02", "2015-05-05"
    result = simulate(quotes, benchmark, scores, start, end)
    expected_units = (1_000_000 / 3) / (100 * 1.0005)
    expected_fee = expected_units * 100 * .0005
    assert len(result.trades) == 2
    assert np.isclose(result.trades.iloc[0].units, expected_units)
    assert np.isclose(result.history.iloc[-1].nav, 1_000_000 - 2 * expected_fee)
    assert result.trades.iloc[1].date == pd.Timestamp("2015-04-02")
    assert result.positions.empty
    changes = result.trades.groupby("date").cash_change.sum().reindex(result.history.index, fill_value=0)
    assert np.allclose(result.history.cash, 1_000_000 + changes.cumsum())
    assert np.allclose(result.history.nav, result.history.cash + result.history.holdings_value)
    assert np.isclose(result.trades.realized_pnl.sum(), result.history.iloc[-1].nav - 1_000_000)
    assert (result.trades.loc[result.trades.side.eq("BUY"), "signal_date"] < result.trades.loc[result.trades.side.eq("BUY"), "date"]).all()

    weekend = simulate(quotes, benchmark, scores, start, end, holding_days=2)
    assert weekend.trades.iloc[1].date == pd.Timestamp("2015-01-05")
    merger = pd.DataFrame({"date":[pd.Timestamp("2015-01-05")],"symbol":["DEMO"],
                           "cash_per_actual_share":[105.0],"adjusted_to_actual_factor":[1.0],"source":["synthetic verified-event fixture"]})
    redeemed = simulate(quotes, benchmark, scores, start, end, corporate_actions=merger)
    assert redeemed.trades.side.tolist() == ["BUY", "CASH_MERGER"]
    assert redeemed.positions.empty
    assert np.isclose(redeemed.history.iloc[-1].nav, 1_000_000-1_000_000/3+expected_units*105*(1-.0005))
    warrant = merger.assign(cash_per_actual_share=.1, event_type='CASH_REDEMPTION', timing='after_close')
    expired = simulate(quotes, benchmark, scores, start, end, corporate_actions=warrant)
    assert expired.trades.side.tolist() == ['BUY', 'CASH_REDEMPTION']
    assert np.isclose(expired.history.iloc[-1].nav, 1_000_000-1_000_000/3+expected_units*.1*(1-.0005))
    sold_before_redemption = simulate(quotes, benchmark, scores, start, end, corporate_actions=warrant, holding_days=2)
    assert sold_before_redemption.trades.side.tolist() == ['BUY', 'SELL']
    successor = quotes.loc[quotes.date.ge(pd.Timestamp('2015-01-05'))].assign(symbol='NEW',adj_close=200.0)
    converted_quotes = pd.concat([quotes.loc[quotes.date.lt(pd.Timestamp('2015-01-05'))],successor])
    stock_event = pd.DataFrame({'date':[pd.Timestamp('2015-01-05')],'symbol':['DEMO'],
                                'successor_symbol':['NEW'],'adjusted_unit_ratio':[.5],
                                'source':['synthetic stock conversion']})
    converted = simulate(converted_quotes,benchmark,scores,start,end,stock_mergers=stock_event)
    assert converted.trades.side.tolist() == ['BUY','STOCK_MERGER','SELL']
    assert converted.trades.iloc[-1].date == pd.Timestamp('2015-04-02')
    assert np.isclose(converted.trades.iloc[-1].units,expected_units*.5)
    assert np.isclose(converted.history.iloc[-1].nav,1_000_000-2*expected_fee)
    repeated_scores = pd.DataFrame({"date": pd.to_datetime(["2014-12-31", "2015-01-30", "2015-02-27"]), "symbol": "DEMO", "score": .1})
    overlapping = simulate(quotes, benchmark, repeated_scores, start, "2015-03-03")
    assert len(overlapping.positions) == 3 and overlapping.positions.lot_id.nunique() == 3
    assert overlapping.history.cash.ge(0).all()

    missing_entry = simulate(quotes.loc[quotes.date.ne(pd.Timestamp(start))], benchmark, scores, start, end)
    assert missing_entry.trades.empty and missing_entry.history.nav.eq(1_000_000).all()
    assert "rejected_missing_quote" in missing_entry.diagnostics.event.tolist()
    no_orders = simulate(quotes, benchmark, scores.iloc[:0], start, end)
    assert no_orders.trades.empty and no_orders.history.nav.eq(1_000_000).all()
    missing_exit = simulate(quotes.loc[quotes.date.ne(pd.Timestamp("2015-04-02"))], benchmark, scores, start, end)
    assert missing_exit.trades.iloc[1].date == pd.Timestamp("2015-04-03")
    short_gap = calendar[(calendar > pd.Timestamp(start))][:5]
    stale = simulate(quotes.loc[~quotes.date.isin(short_gap)], benchmark, scores, start, end)
    assert len(stale.diagnostics.loc[stale.diagnostics.event.eq("stale_mark")]) == 5
    assert stale.trades.iloc[1].date == pd.Timestamp("2015-04-02")
    long_gap = calendar[(calendar > pd.Timestamp(start))][:6]
    try:
        simulate(quotes.loc[~quotes.date.isin(long_gap)], benchmark, scores, start, end)
    except UnreliableHeldPrice as error:
        assert error.partial_result is not None
        assert error.partial_result.history.index[-1] < error.failed_date
        assert error.partial_result.history.stale_marks.max() <= 5
        assert 'DEMO' in error.partial_result.positions.symbol.tolist()
        assert (error.partial_result.trades.date <= error.partial_result.history.index[-1]).all()
    else:
        raise AssertionError("A six-session held-price gap should stop the backtest")

    # Scenario exits use only information available before maturity. A much
    # higher later quote must not affect the assumed exit or release cash early.
    maturity = pd.Timestamp("2015-04-02")
    gap_quotes = quotes.loc[quotes.date.le(pd.Timestamp(start)) | quotes.date.gt(maturity)].copy()
    gap_quotes.loc[gap_quotes.date.gt(maturity), "adj_close"] = 500.0
    hypothetical = simulate(gap_quotes, benchmark, scores, start, end,
                            missing_price_policy="last_before_maturity")
    assumed = hypothetical.trades.loc[hypothetical.trades.side.eq("ASSUMED_STALE_EXIT")]
    assert len(assumed) == 1
    assert assumed.iloc[0].date == maturity
    assert assumed.iloc[0].price_date == pd.Timestamp(start)
    assert assumed.iloc[0].price == 100.0
    assert assumed.iloc[0].stale_sessions > 5
    assert hypothetical.positions.empty
    assert np.isclose(hypothetical.history.iloc[-1].nav, 1_000_000 - 2 * expected_fee)
    before_maturity = hypothetical.history.loc[pd.Timestamp(start):maturity-pd.Timedelta(days=1)]
    assert np.allclose(before_maturity.cash, 1_000_000 - 1_000_000/3)
    assert (hypothetical.trades.loc[hypothetical.trades.side.eq("SELL")].empty)

    altered = quotes.copy()
    altered.loc[altered.date > pd.Timestamp(start), "adj_close"] = 200.0
    changed = simulate(altered, benchmark, scores, start, end)
    assert np.isclose(changed.trades.iloc[0].units, expected_units)
    flagged = pd.DataFrame({"symbol": ["DEMO"], "date": [pd.Timestamp("2015-01-05")], "reason": ["unresolved_price_scale_transition"]})
    try:
        simulate(quotes, benchmark, scores, start, end, quality_audit=flagged)
    except UnreliableHeldPrice:
        pass
    else:
        raise AssertionError("An unresolved held-price transition should stop valuation")

    evidence = {
        "status": "passed",
        "cases": ["flat-price round-trip fees", "daily cash reconciliation", "NAV identity", "realized PnL", "signal before entry", "weekend maturity", "overlapping lots and open-position end NAV", "missing entry", "no orders", "missing maturity quote waits for observed fill", "five-session stale marks", "six-session gap error", "future-price change does not change first purchase", "unresolved held-price transition"],
        "flat_price_expected_fee_per_side": expected_fee,
        "flat_price_expected_final_value": 1_000_000 - 2 * expected_fee,
        "flat_price_observed_final_value": float(result.history.iloc[-1].nav),
    }
    evidence['cases'].append('verified cash merger closes early at sourced payout with fees')
    evidence['cases'].append('after-close warrant redemption follows scheduled market sales; no exercise')
    evidence['cases'].append('automatic stock merger preserves cost, maturity and cash; successor sold on original due date')
    evidence['cases'].append('long gap preserves unresolved position and completed prefix; no retrospective deletion or failed-session NAV')
    evidence['cases'].append('optional stale-exit scenario waits for maturity, uses pre-maturity quote, records assumed exit and fees; later quotes do not change it')
    folder = Path(__file__).parent / "outputs"
    folder.mkdir(exist_ok=True)
    (folder / "accounting_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    run_checks()
