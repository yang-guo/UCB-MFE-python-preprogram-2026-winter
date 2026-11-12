"""My backtester for HW4: long only, tracked lot by lot, money split into tranches.

Built from scratch for HW4. Lectures/modeling/backtest.py was handy for
checking the timing conventions, but it isn't imported anywhere.

Order of events on each trading day:
  a) A stock we own that has printed no price for `silent_limit` days in a row
     is assumed bought out. Its lots are closed at the last price it printed.
  b) Lots that are 90+ days old get sold at today's close, if there is one.
     No price today means we try again tomorrow.
  c) On the day after a signal, every tranche that has nothing left waiting
     to mature spends its cash on that signal's top picks.
  d) Mark the book at the most recent price of each holding and save the day.

Every tranche keeps its own cash pile and can only spend what is in it, so
the cash balance can't dip below zero.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Holding:
    ticket: int
    tranche: int
    symbol: str
    picked_on: pd.Timestamp
    bought_on: pd.Timestamp
    buy_px: float
    shares: float
    outlay: float
    sell_after: pd.Timestamp
    sold_on: pd.Timestamp | None = None
    sell_px: float | None = None
    cash_back: float | None = None


@dataclass
class Tranche:
    slot: int
    cash: float
    opens_on: pd.Timestamp
    book: list = field(default_factory=list)

    def can_rebuy(self, day):
        # waits until its opening day, then until every unsold holding is old enough
        if day < self.opens_on:
            return False
        return not any(h.sold_on is None and h.sell_after > day for h in self.book)


class StaggeredBook:
    def __init__(self, closes, first_day, last_day, start_cash=1_000_000.0, tranches=3, basket=25,
                 fee_bps=5.0, horizon_days=90, gap_days=30, silent_limit=10):
        in_window = (closes.index >= pd.Timestamp(first_day)) & (closes.index <= pd.Timestamp(last_day))
        self.calendar = closes.index[in_window]
        self.closes = closes.loc[: self.calendar[-1]]
        self.carried_px = self.closes.ffill()
        # how many rows since each stock last had a real price (0 on a day it trades)
        row_no = pd.Series(np.arange(len(self.closes)), index=self.closes.index)
        last_seen_row = self.closes.notna().mul(row_no, axis=0).where(self.closes.notna()).ffill()
        self.days_silent = last_seen_row.rsub(row_no, axis=0)
        self.silent_limit = silent_limit
        self.start_cash = start_cash
        self.basket = basket
        self.fee = fee_bps / 10_000
        self.horizon = pd.Timedelta(days=horizon_days)
        each = start_cash / tranches
        self.tranches = [Tranche(t, each, self.calendar[0] + pd.Timedelta(days=gap_days * t))
                         for t in range(tranches)]
        self.holdings, self.ledger, self.snapshots = [], [], []

    def run(self, picks_by_date):
        """picks_by_date maps signal date -> Series of scores per symbol (bigger = better)."""
        trading_days = list(self.closes.index)
        row_of = {d: r for r, d in enumerate(trading_days)}
        buy_schedule = {}
        for signal_day, ranking in picks_by_date.items():
            r = row_of.get(signal_day)
            if r is not None and r + 1 < len(trading_days):
                buy_schedule[trading_days[r + 1]] = (signal_day, ranking)

        for day in self.calendar:
            px_row = self.closes.loc[day]
            self._close_vanished(day)
            self._close_matured(day, px_row)
            if day in buy_schedule:
                signal_day, ranking = buy_schedule[day]
                for tr in self.tranches:
                    if tr.can_rebuy(day):
                        self._open_basket(day, signal_day, tr, ranking, px_row)
            self._snapshot(day)
        return self.collect()

    def _exit(self, h, day, px, kind):
        notional = h.shares * px
        h.sold_on, h.sell_px = day, px
        h.cash_back = notional * (1 - self.fee)
        self.tranches[h.tranche].cash += h.cash_back
        self.ledger.append(dict(date=day, side=kind, ticket=h.ticket, tranche=h.tranche, symbol=h.symbol,
                                price=px, shares=h.shares, cash_delta=h.cash_back,
                                commission=notional * self.fee))

    def _close_matured(self, day, px_row):
        for h in self.holdings:
            if h.sold_on is not None or h.sell_after > day:
                continue
            px = px_row.get(h.symbol)
            if px is not None and not pd.isna(px):
                self._exit(h, day, float(px), "sell")

    def _close_vanished(self, day):
        # backward looking only: the stock has already gone quiet for silent_limit rows
        quiet = self.days_silent.loc[day]
        for h in self.holdings:
            if h.sold_on is None and h.bought_on < day and quiet.get(h.symbol, 0) >= self.silent_limit:
                self._exit(h, day, float(self.carried_px.at[day, h.symbol]), "buyout_close")

    def _open_basket(self, day, signal_day, tr, ranking, px_row):
        """Each pick gets cash / basket, fixed before any buying. If a pick has no
        price today it is skipped and its share just stays in the tranche."""
        winners = ranking.dropna().sort_values(ascending=False)[: self.basket].index
        slice_cash = tr.cash / self.basket
        for sym in winners:
            px = px_row.get(sym)
            if px is None or pd.isna(px) or px <= 0:
                self.ledger.append(dict(date=day, side="no_price", ticket=None, tranche=tr.slot, symbol=sym,
                                        price=np.nan, shares=0.0, cash_delta=0.0, commission=0.0))
                continue
            net_spend = slice_cash / (1 + self.fee)
            qty = net_spend / px
            tr.cash -= slice_cash
            h = Holding(len(self.holdings), tr.slot, sym, signal_day, day, float(px), qty,
                        slice_cash, day + self.horizon)
            tr.book.append(h)
            self.holdings.append(h)
            self.ledger.append(dict(date=day, side="buy", ticket=h.ticket, tranche=tr.slot, symbol=sym,
                                    price=float(px), shares=qty, cash_delta=-slice_cash,
                                    commission=slice_cash - net_spend))

    def _snapshot(self, day):
        px_marks = self.carried_px.loc[day]
        live = [h for h in self.holdings if h.sold_on is None]
        mkt = sum(h.shares * px_marks[h.symbol] for h in live)
        cash = sum(tr.cash for tr in self.tranches)
        self.snapshots.append(dict(date=day, cash=cash, holdings=mkt, nav=cash + mkt, open_lots=len(live)))

    def collect(self):
        daily = pd.DataFrame(self.snapshots).set_index("date")
        ledger = pd.DataFrame(self.ledger)
        lots = pd.DataFrame([vars(h) for h in self.holdings])
        final_day = self.calendar[-1]
        still_held = pd.DataFrame()
        if len(lots) and lots.sold_on.isna().any():
            still_held = lots[lots.sold_on.isna()].copy()
            still_held["mark"] = [self.carried_px.at[final_day, s] for s in still_held.symbol]
            still_held["value"] = still_held.mark * still_held.shares
            last_print = self.closes.loc[:final_day].apply(lambda col: col.last_valid_index())
            still_held["last_quote_date"] = still_held.symbol.map(last_print)
            still_held["stale"] = (final_day - still_held.last_quote_date).dt.days > 7
        return dict(daily=daily, fills=ledger, lots=lots, open_lots=still_held)


def summarize_nav(nav, rf_annual=0.02):
    """The four required numbers (plus volatility) for one daily NAV path."""
    nav = nav.dropna()
    span_years = (nav.index[-1] - nav.index[0]) / pd.Timedelta(days=365.25)
    growth = nav.iloc[-1] / nav.iloc[0]
    day_ret = nav.pct_change().dropna()
    rf_daily = np.expm1(np.log1p(rf_annual) / 252)   # 2% a year compounded over 252 sessions
    vol = day_ret.std()
    return pd.Series({
        "Final value": nav.iloc[-1],
        "CAGR": growth ** (1 / span_years) - 1,
        "Max drawdown": (nav / nav.cummax()).min() - 1,
        "Sharpe": (day_ret - rf_daily).mean() / vol * np.sqrt(252),
        "Volatility": vol * np.sqrt(252),
    })


def index_as_nav(levels, first_day, last_day, start_cash=1_000_000.0):
    window = levels.loc[(levels.index >= pd.Timestamp(first_day)) & (levels.index <= pd.Timestamp(last_day))]
    return window / window.iloc[0] * start_cash
