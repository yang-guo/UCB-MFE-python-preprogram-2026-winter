"""A small lot-based, long-only daily backtester (my own implementation).

Design follows the HW4 rules and the Lecture 8 separation of concerns
(strategy / order book / daily loop / analytics), but the code is new.

Timeline of one market session ``d`` (all fills at the close of ``d``):

1. **Buys** planned at the previous session's close are filled at ``d``'s
   close. The budget is fee-inclusive: units = budget / (price * (1 + fee)).
   A buy with no quote at ``d`` is skipped (never filled at a stale price).
2. **Sells**: every lot whose 90-calendar-day maturity is on/before ``d`` is
   sold at ``d``'s close if the stock quotes; otherwise it stays open and
   is sold at the next quoted close (flagged if more than 5 sessions late).
3. **Mark to market** at the last observed adjusted close (stale marks are
   counted and reported).
4. **Plan** (signal dates only): the strategy sees the portfolio snapshot and
   predictions known at ``d``'s close and returns dollar budgets for buys at
   the next session. Total budgets may not exceed current cash, so the same
   cash can never be spent twice.

Positions are held in adjusted-price units, so dividends are already in the
return and are not added again. Cash earns 0%.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Lot:
    lot_id: int
    symbol: str
    col: int
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp
    frozen_mark: float = np.nan     # set when the stock stops quoting for too long


@dataclass
class Snapshot:
    date: pd.Timestamp
    cash: float
    nav: float
    holdings_value: pd.Series      # market value by symbol (aggregated across lots)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    incidents: pd.DataFrame
    skipped_orders: pd.DataFrame
    initial_cash: float
    fee_rate: float
    info: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, calendar, symbols, adj_close, initial_cash=1_000_000.0, fee_bps=10.0,
                 holding_days=90, late_exit_flag_sessions=5, freeze_after_sessions=63):
        self.calendar = pd.DatetimeIndex(calendar)
        self.symbols = pd.Index(symbols)
        self.px = np.asarray(adj_close, dtype=float)
        assert self.px.shape == (len(self.calendar), len(self.symbols))
        self.initial_cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000.0
        self.holding_days = holding_days
        self.late_flag = late_exit_flag_sessions
        # After ~3 months without quotes a later quote may belong to a different
        # issuer reusing the ticker (e.g. ITG: acquired 2019, new company 2026).
        # Such lots are frozen at their last pre-gap mark and never sold.
        self.freeze_after = freeze_after_sessions

    def run(self, strategy, start, end):
        cal = self.calendar
        i0, i1 = cal.searchsorted(pd.Timestamp(start)), cal.searchsorted(pd.Timestamp(end), side="right") - 1
        cash = self.initial_cash
        lots: dict[int, Lot] = {}
        next_id = 0
        last_px = np.full(len(self.symbols), np.nan)
        last_q = np.full(len(self.symbols), -1)
        pending: dict[str, float] = {}
        pending_signal = None
        fills, daily, incidents, skipped = [], [], [], []

        # The strategy's first plan happens at the close *before* the first
        # tradable session; that close is the signal date for the first buy.
        if i0 > 0 and strategy.is_signal_date(cal[i0 - 1]):
            pending = strategy.plan(Snapshot(cal[i0 - 1], cash, cash, pd.Series(dtype=float)))
            pending_signal = cal[i0 - 1]
            assert sum(pending.values()) <= cash + 1e-6

        for i in range(i0, i1 + 1):
            d = cal[i]
            row = self.px[i]
            quoted = np.isfinite(row)
            last_px[quoted] = row[quoted]
            last_q[quoted] = i

            # 1) buys planned at the previous close
            for sym, budget in pending.items():
                col = self.symbols.get_loc(sym)
                p = row[col]
                if not np.isfinite(p) or budget <= 0:
                    skipped.append({"signal_date": pending_signal, "date": d, "symbol": sym, "budget": budget,
                                    "reason": "no quote at entry"})
                    continue
                budget = min(budget, cash)             # guard against float dust
                units = budget / (p * (1 + self.fee_rate))
                notional = units * p
                fee = notional * self.fee_rate
                assert notional + fee <= cash + 1e-6, "would spend cash twice"
                cash -= notional + fee
                lot = Lot(next_id, sym, col, units, pending_signal, d, p, fee,
                          d + pd.Timedelta(days=self.holding_days))
                lots[next_id] = lot
                next_id += 1
                fills.append({"date": d, "side": "BUY", "lot_id": lot.lot_id, "symbol": sym, "units": units,
                              "price": p, "notional": notional, "fee": fee, "cash_change": -(notional + fee),
                              "signal_date": pending_signal, "entry_date": d, "due_date": lot.due_date,
                              "realized_pnl": 0.0})
            pending, pending_signal = {}, None
            cash = max(cash, 0.0)

            # 2) maturity sells
            for lot_id in [k for k, l in lots.items() if l.due_date <= d and np.isnan(l.frozen_mark)]:
                lot = lots[lot_id]
                p = row[lot.col]
                if not np.isfinite(p):
                    continue                            # wait for the next quoted close
                notional = lot.units * p
                fee = notional * self.fee_rate
                pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
                cash += notional - fee
                late = cal.searchsorted(lot.due_date)
                fills.append({"date": d, "side": "SELL", "lot_id": lot_id, "symbol": lot.symbol, "units": lot.units,
                              "price": p, "notional": notional, "fee": fee, "cash_change": notional - fee,
                              "signal_date": lot.signal_date, "entry_date": lot.entry_date,
                              "due_date": lot.due_date, "realized_pnl": pnl, "sessions_late": i - late})
                if i - late > self.late_flag:
                    incidents.append({"date": d, "symbol": lot.symbol, "lot_id": lot_id,
                                      "issue": f"sold {i - late} sessions after maturity (no quotes)"})
                del lots[lot_id]

            # 3) mark to market at last observed price
            value_by_col = {}
            stale_value, stale_n = 0.0, 0
            for lot in lots.values():
                gap = i - last_q[lot.col]
                if np.isnan(lot.frozen_mark) and gap > self.freeze_after:
                    lot.frozen_mark = last_px[lot.col]
                    incidents.append({"date": d, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                      "issue": f"frozen: no quote for >{self.freeze_after} sessions; held at last mark, never sold"})
                frozen = not np.isnan(lot.frozen_mark)
                v = lot.units * (lot.frozen_mark if frozen else last_px[lot.col])
                value_by_col[lot.col] = value_by_col.get(lot.col, 0.0) + v
                if frozen or gap > 0:
                    stale_value += v
                    stale_n += 1
                    if gap == self.late_flag + 1:
                        incidents.append({"date": d, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                          "issue": f"no quote for >{self.late_flag} sessions; carried at last mark"})
            holdings = sum(value_by_col.values())
            nav = cash + holdings
            daily.append({"date": d, "cash": cash, "holdings": holdings, "nav": nav, "n_lots": len(lots),
                          "n_names": len(value_by_col), "stale_lots": stale_n, "stale_value": stale_value})

            # 4) plan tomorrow's buys
            if i < i1 and strategy.is_signal_date(d):
                hv = pd.Series({self.symbols[c]: v for c, v in value_by_col.items()}, dtype=float)
                pending = strategy.plan(Snapshot(d, cash, nav, hv))
                pending_signal = d
                total = sum(pending.values())
                if total > cash + 1e-6:
                    raise ValueError(f"{d.date()}: strategy budgets {total:,.2f} exceed cash {cash:,.2f}")

        def mark(l):
            return last_px[l.col] if np.isnan(l.frozen_mark) else l.frozen_mark

        open_lots = pd.DataFrame([{**l.__dict__, "mark": mark(l), "frozen": not np.isnan(l.frozen_mark),
                                   "last_quote": cal[last_q[l.col]] if last_q[l.col] >= 0 else pd.NaT,
                                   "market_value": l.units * mark(l)} for l in lots.values()])
        return BacktestResult(pd.DataFrame(daily).set_index("date"), pd.DataFrame(fills), open_lots,
                              pd.DataFrame(incidents), pd.DataFrame(skipped), self.initial_cash, self.fee_rate)


class TopNStrategy:
    """Every ``every`` sessions buy the top-N predicted stocks, equal weight.

    * Cohort budget = min(cash, NAV / cohorts); with ~6.4 overlapping cohorts
      per 90 days and cohorts=6 the book stays close to fully invested.
    * Screens use only signal-date information: dated universe rank, price,
      and a data-quality screen for recent >=100% one-day moves.
    * At most ``sector_cap`` of a cohort's names from one (snapshot) sector,
      and a name's total exposure is capped at ``name_cap`` of NAV.
    """

    def __init__(self, predictions, signal_dates, top_n=25, cohorts=6, max_rank=2000,
                 min_price=5.0, max_abs_ret=1.0, sector_cap=0.3, name_cap=0.05, min_mom_rank=None):
        self.pred = {d: g for d, g in predictions.groupby("date")}
        self.signal_dates = set(pd.DatetimeIndex(signal_dates))
        self.top_n, self.cohorts, self.max_rank = top_n, cohorts, max_rank
        self.min_price, self.max_abs_ret = min_price, max_abs_ret
        self.sector_cap, self.name_cap = sector_cap, name_cap
        self.min_mom_rank = min_mom_rank      # optional trend screen on the 12-1 momentum percentile
        self.log = []

    def is_signal_date(self, d):
        return d in self.signal_dates and d in self.pred

    def plan(self, snap):
        g = self.pred[snap.date]
        g = g.loc[g.universe_rank.le(self.max_rank) & g.close.ge(self.min_price)
                  & g.max_abs_ret_252.lt(self.max_abs_ret) & g.score.notna()]
        if self.min_mom_rank is not None:
            g = g.loc[g.r_mom_12_1.ge(self.min_mom_rank)]
        g = g.sort_values(["score", "symbol"], ascending=[False, True])
        per_sector = max(1, int(np.floor(self.top_n * self.sector_cap)))
        picks, counts = [], {}
        for sym, sec in zip(g.symbol, g.sector):
            if counts.get(sec, 0) >= per_sector:
                continue
            picks.append(sym)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) == self.top_n:
                break
        budget = min(snap.cash, snap.nav / self.cohorts)
        if not picks or budget <= 1.0:
            return {}
        each = budget / len(picks)
        orders = {}
        for sym in picks:
            room = self.name_cap * snap.nav - snap.holdings_value.get(sym, 0.0)
            amt = min(each, max(room, 0.0))
            if amt > 1.0:
                orders[sym] = amt
        self.log.append({"date": snap.date, "candidates": len(g), "picks": len(picks),
                         "budget": budget, "ordered": sum(orders.values())})
        return orders


class BuyAndHold:
    """Benchmark helper: all cash into one series at ``start``'s close."""

    @staticmethod
    def run(series, start, end, initial_cash=1_000_000.0):
        s = series.loc[start:end]
        return pd.DataFrame({"nav": initial_cash * s / s.iloc[0]})


def performance(nav, initial=1_000_000.0, rf_annual=0.02, periods=252):
    """Metrics from a daily NAV whose first row is the first test close.

    ``initial`` is the capital just before the first close, so the first daily
    return includes entry fees (and drawdown measures from the starting $1M).
    CAGR uses elapsed calendar days / 365.25; Sharpe uses 252 sessions and a
    2% annual risk-free rate converted to its daily equivalent.
    """
    nav = nav.dropna()
    r = pd.concat([pd.Series([nav.iloc[0] / initial - 1], index=[nav.index[0]]), nav.pct_change().dropna()])
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    nav = pd.concat([pd.Series([initial], index=[nav.index[0] - pd.Timedelta(days=1)]), nav])
    rf_d = (1 + rf_annual) ** (1 / periods) - 1
    ex = r - rf_d
    dd = nav / nav.cummax() - 1
    return pd.Series({
        "start": nav.index[1].date(), "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "ann_vol": r.std() * np.sqrt(periods),
        "sharpe_rf2": ex.mean() / ex.std() * np.sqrt(periods),
        "max_drawdown": dd.min(),
    })
