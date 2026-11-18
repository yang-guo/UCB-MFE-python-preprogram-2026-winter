"""Portfolio, OrderBook and Trader classes for a constrained daily-rebalance backtest.

Rules implemented (Trader.rebalance):
  1. Wanted trade = target shares - current shares, kept only if |value| > 5 bps of NAV AND > 1 share.
  2. 90-day holding period: only shares from lots held >= 90 days can be sold (FIFO). The rest is blocked.
  3. TC = 2 bps: a sell returns (1 - tc) of its value; a buy costs (1 + tc) of its value.
  4. Cash rules, applied after TC:
       - post-trade cash > 20 bps of NAV  -> scale SELLS down
       - post-trade cash < 0              -> scale BUYS down (this also covers blocked sells)
     Shares are floored after scaling, so the cash limits are never broken.
  5. Every day's TC, cash, NAV, scale factors and unfilled orders go to the OrderBook.
"""
import numpy as np
import pandas as pd

##### CLAUDE CODE #####
## Ideas are mine, implementation was Claude's, I was running short on time to get all the requirements done #


class Portfolio:
    """Cash plus share lots. Each lot keeps its buy date so the holding period can be enforced."""

    def __init__(self, cash):
        self.start_cash = float(cash)   # starting NAV, never changes
        self.cash = float(cash)
        self.lots = pd.DataFrame({'symbol': pd.Series(dtype=object),
                                  'buy_date': pd.Series(dtype='datetime64[ns]'),
                                  'shares': pd.Series(dtype=float)})
        self.last_price = pd.Series(dtype=float)   # carries prices forward for symbols that miss a day

    def update_prices(self, prices):
        self.last_price = prices.dropna().combine_first(self.last_price)

    def shares(self):
        return self.lots.groupby('symbol')['shares'].sum()

    def nav(self):
        held = self.shares()
        return self.cash + (held * self.last_price.reindex(held.index)).sum()

    def sellable(self, date, hold_days):
        aged = self.lots[self.lots['buy_date'] <= date - pd.Timedelta(days=hold_days)]
        return aged.groupby('symbol')['shares'].sum()

    def buy(self, shares, date):
        if shares.empty:
            return
        new_lots = pd.DataFrame({'symbol': shares.index, 'buy_date': date, 'shares': shares.values})
        self.lots = new_lots if self.lots.empty else pd.concat([self.lots, new_lots], ignore_index=True)

    def sell(self, shares):
        """FIFO: oldest lots are consumed first (these are the aged, sellable lots)."""
        if shares.empty:
            return
        self.lots = self.lots.sort_values(['symbol', 'buy_date'], ignore_index=True)
        cum = self.lots.groupby('symbol')['shares'].cumsum()
        qty = self.lots['symbol'].map(shares).fillna(0)
        self.lots['shares'] = (cum - qty).clip(lower=0).clip(upper=self.lots['shares'])
        self.lots = self.lots[self.lots['shares'] > 0].reset_index(drop=True)


class OrderBook:
    """History of executed trades, unfilled orders and daily summaries."""

    def __init__(self):
        self._trades, self._unfilled, self._daily = [], [], []

    def log_trades(self, date, executed, prices, tc):
        if executed.empty:
            return
        px = prices.reindex(executed.index)
        value = executed * px
        self._trades.append(pd.DataFrame({
            'date': date, 'symbol': executed.index,
            'side': np.where(executed > 0, 'BUY', 'SELL'),
            'shares': executed.values, 'price': px.values,
            'value': value.values, 'tc': value.abs().values * tc}))

    def log_unfilled(self, date, wanted, executed, blocked_by_hold):
        missed = wanted - executed.reindex(wanted.index, fill_value=0)
        missed = missed[missed != 0]
        if missed.empty:
            return
        reason = np.where(blocked_by_hold.reindex(missed.index, fill_value=0) > 0, 'hold_90d',
                          np.where(wanted[missed.index] > 0, 'buy_cash_limit', 'sell_cash_cap'))
        self._unfilled.append(pd.DataFrame({
            'date': date, 'symbol': missed.index,
            'wanted': wanted[missed.index].values,
            'executed': executed.reindex(missed.index, fill_value=0).values,
            'reason': reason}))

    def log_day(self, **row):
        self._daily.append(row)

    @property
    def trades(self):
        return pd.concat(self._trades, ignore_index=True) if self._trades else pd.DataFrame()

    @property
    def unfilled(self):
        return pd.concat(self._unfilled, ignore_index=True) if self._unfilled else pd.DataFrame()

    @property
    def daily(self):
        return pd.DataFrame(self._daily).set_index('date')


class Trader:
    def __init__(self, portfolio, orderbook, tc_bps=2, hold_days=90, min_trade_bps=5, max_cash_bps=20):
        self.portfolio, self.orderbook = portfolio, orderbook
        self.tc = tc_bps / 1e4
        self.hold_days = hold_days
        self.min_trade = min_trade_bps / 1e4
        self.max_cash = max_cash_bps / 1e4

    def _check_day(self, date, prices, pre_cash, pre_shares, pre_nav, executed):
        """Daily invariants. Raises AssertionError (with the date) the moment any is broken."""
        p, tc = self.portfolio, self.tc
        tol = 1e-8 * pre_nav

        # 4) No trade in a security whose price is missing today (checked first: the others need prices)
        missing = executed.index[prices.reindex(executed.index).isna()]
        assert missing.empty, f'{date:%Y-%m-%d}: traded with missing price: {list(missing)}'

        # 1) Position weights + cash weight = 1
        held = p.shares()
        held_px = p.last_price.reindex(held.index)
        assert held_px.notna().all(), f'{date:%Y-%m-%d}: holding with no price: {list(held_px[held_px.isna()].index)}'
        nav = p.nav()
        total_w = (held * held_px).sum() / nav + p.cash / nav
        assert abs(total_w - 1) < 1e-9, f'{date:%Y-%m-%d}: weights + cash = {total_w:.12f}'

        # 2) Change in cash = sells - buys - TC   (buys +value, sells -value)
        traded_val = executed * prices.reindex(executed.index)
        exp_dcash = -traded_val.sum() - traded_val.abs().sum() * tc
        assert abs((p.cash - pre_cash) - exp_dcash) < tol, \
            f'{date:%Y-%m-%d}: cash change {p.cash - pre_cash:.4f} != sells - buys - TC {exp_dcash:.4f}'

        # 3) New position value = old position value + traded value; NAV falls by exactly TC
        syms = pre_shares.index.union(held.index)
        d_val = (held.reindex(syms, fill_value=0) - pre_shares.reindex(syms, fill_value=0)) * p.last_price.reindex(syms)
        bad = (d_val - traded_val.reindex(syms, fill_value=0)).abs() > tol
        assert not bad.any(), f'{date:%Y-%m-%d}: position change != trade for {list(syms[bad])}'
        assert abs(nav - (pre_nav - traded_val.abs().sum() * tc)) < tol, \
            f'{date:%Y-%m-%d}: NAV {nav:.4f} != pre-trade NAV - TC {pre_nav - traded_val.abs().sum() * tc:.4f}'

    def rebalance(self, date, target_weights, prices):
        p, tc = self.portfolio, self.tc
        p.update_prices(prices)
        nav = p.nav()
        pre_cash, pre_shares = p.cash, p.shares()
        px = prices[prices > 0].dropna()

        # 1) Wanted trades: delta from target, > 5 bps of NAV and > 1 share of value
        current = p.shares().reindex(px.index, fill_value=0)
        target = target_weights.reindex(px.index, fill_value=0) * nav / px
        delta_val = (target - current) * px
        keep = (delta_val.abs() > self.min_trade * nav) & (delta_val.abs() > px)
        wanted = np.trunc(target - current)[keep]
        wanted = wanted[wanted != 0]

        # 2) Sells limited to shares held >= hold_days
        want_sell = -wanted[wanted < 0]
        sell = np.minimum(want_sell, p.sellable(date, self.hold_days).reindex(want_sell.index, fill_value=0))
        blocked = want_sell - sell
        buy = wanted[wanted > 0]

        S = (sell * px[sell.index]).sum()
        B = (buy * px[buy.index]).sum()

        # 3) Cash cap: if cash would end above max_cash -> scale sells down
        sell_scale = 1.0
        cash_cap = self.max_cash * nav
        if S > 0 and p.cash + S * (1 - tc) - B * (1 + tc) > cash_cap:
            sell_scale = float(np.clip((cash_cap - p.cash + B * (1 + tc)) / (S * (1 - tc)), 0, 1))
            sell = np.floor(sell * sell_scale)
            S = (sell * px[sell.index]).sum()

        # 4) Cash floor: if buys cost more than cash + sell proceeds -> scale buys down
        buy_scale = 1.0
        budget = p.cash + S * (1 - tc)
        if B > 0 and B * (1 + tc) > budget:
            buy_scale = max(budget, 0) / (B * (1 + tc))
            buy = np.floor(buy * buy_scale)
            B = (buy * px[buy.index]).sum()

        # 5) Execute
        sell, buy = sell[sell > 0], buy[buy > 0]
        p.sell(sell)
        p.buy(buy, date)
        p.cash += S * (1 - tc) - B * (1 + tc)
        day_tc = (S + B) * tc

        executed = pd.concat([-sell, buy])
        self._check_day(date, prices, pre_cash, pre_shares, nav, executed)
        self.orderbook.log_trades(date, executed, px, tc)
        self.orderbook.log_unfilled(date, wanted, executed, blocked)
        self.orderbook.log_day(date=date, nav=p.nav(), cash=p.cash, cash_bps=p.cash / p.nav() * 1e4,
                               tc=day_tc, buy_value=B, sell_value=S,
                               buy_scale=buy_scale, sell_scale=sell_scale,
                               n_blocked_hold=int((blocked > 0).sum()))

    def run(self, target_weights, prices):
        """target_weights / prices: wide DataFrames (date index x symbol columns).
        Rebalances on every date that has targets; other dates are marked to market only."""
        for date, day_prices in prices.iterrows():
            if date in target_weights.index:
                self.rebalance(date, target_weights.loc[date], day_prices)
            else:
                self.portfolio.update_prices(day_prices)
                self.orderbook.log_day(date=date, nav=self.portfolio.nav(), cash=self.portfolio.cash,
                                       cash_bps=self.portfolio.cash / self.portfolio.nav() * 1e4, tc=0.0,
                                       buy_value=0.0, sell_value=0.0, buy_scale=1.0, sell_scale=1.0,
                                       n_blocked_hold=0)
        return self.orderbook.daily



class Backtest:
    def __init__(self, portfolio, trader, prices_wide, total_return, bad_days):
        self.portfolio = portfolio
        self.trader = trader
        self.prices_wide = prices_wide
        self.total_return = total_return      # wide: date x [portfolio, sp500], both start at 1
        self.bad_days = bad_days              # output of bad_days_screener

    @property
    def orderbook(self):  return self.trader.orderbook
    @property
    def daily(self):      return self.trader.orderbook.daily
    @property
    def nav(self):        return self.trader.orderbook.daily['nav']
    @property
    def trades(self):     return self.trader.orderbook.trades
    @property
    def unfilled(self):   return self.trader.orderbook.unfilled
    @property
    def holdings(self):   return self.trader.portfolio.shares().sort_values(ascending=False)
    @property
    def last_day(self):   return self.daily.index.max()
    @property
    def total_return_long(self):
        return self.total_return.reset_index(names='date').melt(
            id_vars='date', var_name='portfolio', value_name='total_return_index')

    def day(self, date=None):
        """Fills, unfilled orders and daily summary for one date (default: last day)."""
        date = pd.Timestamp(date) if date is not None else self.last_day
        return {'trades': self.trades[self.trades['date'] == date],
                'unfilled': self.unfilled[self.unfilled['date'] == date],
                'summary': self.daily.loc[date]}