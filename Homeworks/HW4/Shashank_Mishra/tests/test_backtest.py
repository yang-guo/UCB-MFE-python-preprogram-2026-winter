"""Hand-checkable accounting and timing tests.  Run:  python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw4.backtest import Backtester, performance  # noqa: E402
from hw4.data import MarketData  # noqa: E402
from hw4.features import make_labels  # noqa: E402

CAL = pd.bdate_range("2020-01-01", periods=120)          # toy market calendar
SYMS = ["A", "B"]


class FixedPlan:
    """Strategy stub: on given signal dates return fixed dollar budgets."""

    def __init__(self, plans):
        self.plans = {pd.Timestamp(k): v for k, v in plans.items()}

    def is_signal_date(self, d):
        return d in self.plans

    def plan(self, snap):
        return dict(self.plans[snap.date])


def prices(a=100.0, b=50.0):
    px = np.empty((len(CAL), 2))
    px[:, 0], px[:, 1] = a, b
    return px


def test_round_trip_matches_hand_calculation():
    px = prices()
    px[CAL >= pd.Timestamp("2020-04-01"), 0] = 110.0       # A rises 10% before maturity
    bt = Backtester(CAL, SYMS, px, fee_bps=10)
    res = bt.run(FixedPlan({CAL[0]: {"A": 500_000}}), CAL[1], CAL[-1])
    buy, sell = res.fills.iloc[0], res.fills.iloc[1]

    units = 500_000 / (100 * 1.001)                        # fee-inclusive budget
    assert buy.date == CAL[1] and buy.price == 100 and buy.units == pytest.approx(units)
    assert buy.fee == pytest.approx(units * 100 * 0.001)
    # Maturity = entry + 90 calendar days = 2020-04-01 (a Wednesday session).
    assert sell.date == pd.Timestamp("2020-04-01") == CAL[1] + pd.Timedelta(days=90)
    proceeds = units * 110 * 0.999
    assert sell.cash_change == pytest.approx(proceeds)
    assert sell.realized_pnl == pytest.approx(proceeds - 500_000)
    assert res.daily.nav.iloc[-1] == pytest.approx(500_000 + proceeds)
    # NAV on the entry day = cash + units * close; only the entry fee is lost.
    assert res.daily.nav.iloc[0] == pytest.approx(1_000_000 - units * 100 * 0.001)


def test_missing_quote_at_maturity_sells_next_quoted_close():
    px = prices()
    due = CAL[1] + pd.Timedelta(days=90)
    gap = (CAL >= due) & (CAL < due + pd.Timedelta(days=3))  # no quotes for A on maturity +2 days
    px[gap, 0] = np.nan
    res = Backtester(CAL, SYMS, px, fee_bps=0).run(FixedPlan({CAL[0]: {"A": 100_000}}), CAL[1], CAL[-1])
    sell = res.fills.query("side == 'SELL'").iloc[0]
    assert sell.date == CAL[~np.isnan(px[:, 0]) & (CAL >= due)][0]
    assert sell.sessions_late >= 1
    # Stale marks were carried during the gap and counted.
    assert res.daily.loc[CAL[gap], "stale_lots"].eq(1).all()


def test_no_quote_at_entry_is_skipped_not_filled():
    px = prices()
    px[1, 1] = np.nan
    res = Backtester(CAL, SYMS, px).run(FixedPlan({CAL[0]: {"A": 10_000, "B": 10_000}}), CAL[1], CAL[-1])
    assert set(res.fills.query("side == 'BUY'").symbol) == {"A"}
    assert res.skipped_orders.symbol.tolist() == ["B"]


def test_budgets_above_cash_are_rejected():
    plans = {CAL[0]: {"A": 600_000}, CAL[1]: {"B": 600_000}}  # second plan exceeds remaining cash
    with pytest.raises(ValueError, match="exceed cash"):
        Backtester(CAL, SYMS, prices()).run(FixedPlan(plans), CAL[1], CAL[-1])


def test_two_lots_same_symbol_keep_their_own_maturities():
    plans = {CAL[0]: {"A": 100_000}, CAL[10]: {"A": 100_000}}
    res = Backtester(CAL, SYMS, prices()).run(FixedPlan(plans), CAL[1], CAL[-1])
    sells = res.fills.query("side == 'SELL'").sort_values("date")
    assert sells.date.tolist() == [CAL[1] + pd.Timedelta(days=90),
                                   CAL[CAL >= CAL[11] + pd.Timedelta(days=90)][0]]


def test_labels_use_next_session_entry_and_first_quote_after_90_days():
    px = prices()
    px[:, 0] = np.linspace(100, 219, len(CAL))
    md = MarketData(CAL, pd.Index(SYMS), px, px, np.ones_like(px), pd.Series(1.0, index=CAL),
                    pd.Series("X", index=SYMS), {})
    panel = pd.DataFrame({"date": [CAL[0]], "symbol": ["A"], "entry_date": [CAL[1]]})
    lab = make_labels(panel, md).iloc[0]
    exit_day = CAL[CAL >= CAL[1] + pd.Timedelta(days=90)][0]
    assert lab.entry_price == px[1, 0]
    assert lab.label_end == exit_day
    assert lab.target == pytest.approx(px[CAL.get_loc(exit_day), 0] / px[1, 0] - 1)


def test_performance_metrics_on_known_series():
    idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
    m = performance(pd.Series([1_000_000.0, 2_000_000.0], index=idx))
    assert m.final_value == 2_000_000
    assert m.CAGR == pytest.approx(2 ** (365.25 / 366) - 1)
    assert m.max_drawdown == 0


def test_ticker_reuse_after_long_gap_is_frozen_not_sold():
    """A stock stops quoting (acquired) and the ticker returns much later as a
    different company: the old lot must stay at its last mark, never sell."""
    cal = pd.bdate_range("2019-01-01", periods=400)
    px = np.full((len(cal), 2), 50.0)
    px[30:300, 0] = np.nan          # quotes stop before maturity...
    px[300:, 0] = 5.0               # ...and an unrelated issuer reuses the ticker
    plan = FixedPlan({cal[0]: {"A": 100_000}})
    res = Backtester(cal, SYMS, px, fee_bps=0, freeze_after_sessions=63).run(plan, cal[1], cal[-1])
    assert res.fills.query("side == 'SELL'").empty
    assert res.open_lots.frozen.all() and res.open_lots.mark.eq(50.0).all()
    assert res.daily.nav.iloc[-1] == pytest.approx(1_000_000)
