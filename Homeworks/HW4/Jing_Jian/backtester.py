"""Independent HW4 cash-and-lot backtester; no lecture backtester imports.

Scores must be supplied as-of the previous close and already restricted to the
universe eligible for the entry quarter. Only scores (never future labels) enter
the trading engine. Each purchase is a separate 90-calendar-day lot.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataGapError(ValueError):
    """A held position cannot be priced or exited under the stated policy."""


@dataclass(frozen=True)
class StrategyConfig:
    initial_cash: float = 1_000_000.0
    top_n: int = 10
    monthly_fraction: float = 1 / 3
    cost_bps: float = 10.0  # Charged on purchase and sale notional.
    max_stale_sessions: int = 5

    def __post_init__(self):
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("Initial cash must be positive and finite")
        if self.top_n < 1 or not 0 < self.monthly_fraction <= 1:
            raise ValueError("Require top_n >= 1 and 0 < monthly_fraction <= 1")
        if not 0 <= self.cost_bps < 10_000 or self.max_stale_sessions < 0:
            raise ValueError("Invalid transaction cost or stale-mark allowance")


@dataclass
class Lot:
    lot_id: int
    symbol: str
    entry_date: pd.Timestamp
    maturity: pd.Timestamp
    units: float
    mark: float
    last_quote_index: int
    cash_settlement: bool = False


@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: pd.DataFrame
    action_ledger: pd.DataFrame


def entry_schedule(calendar, start, end):
    """First evaluation close, then the first available session of each month."""
    calendar = pd.DatetimeIndex(calendar)
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return dates[~dates.to_period("M").duplicated()]


def run_backtest(prices, calendar, scores, start, end, config=None, *, price_matrix=None,
                 corporate_actions=()):
    """Trade prior-close scores at monthly entry closes with fixed dollar budgets.

    Adjusted-price fractional units represent total-return exposure; do not add
    dividends. Missing entry quotes cancel that order without redistribution.
    Missing maturity quotes stop the run. Other missing held-stock quotes may be
    marked at the last observed price for at most max_stale_sessions sessions.
    Cash earns zero; there is no borrowing. Sales do not enlarge prior budgets.
    """
    config = config or StrategyConfig()
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise ValueError("Calendar must be sorted and unique")
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if dates.empty or calendar.get_loc(dates[0]) == 0:
        raise ValueError("Need evaluation sessions and a preceding signal session")
    if price_matrix is None:
        if prices.duplicated(["date", "symbol"]).any():
            raise ValueError("Duplicate price key")
        if not (np.isfinite(prices.adj_close) & prices.adj_close.gt(0)).all():
            raise ValueError("Prices must be positive and finite; quarantine bad rows first")
        quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    else:
        # A bounded, prebuilt matrix avoids retaining millions of long rows as
        # well as a second pivoted copy during the full-universe experiment.
        if price_matrix.index.has_duplicates or price_matrix.columns.has_duplicates:
            raise ValueError("Duplicate matrix dates or symbols")
        quotes = price_matrix.reindex(calendar)
        values = quotes.to_numpy()
        if (np.isinf(values) | (values <= 0)).any():
            raise ValueError("Observed matrix prices must be positive and finite")
    if scores.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate score key")
    if not np.isfinite(scores.prediction).all():
        raise ValueError("Predictions must be finite")
    score_groups = {date: group for date, group in scores.groupby("date")}
    entry_dates = set(entry_schedule(calendar, start, end))
    cash = config.initial_cash
    previous_nav = config.initial_cash
    fee_rate = config.cost_bps / 10_000
    lots = []
    fills = []
    history = []
    diagnostics = []
    action_ledger = []
    next_lot_id = 0

    for day in dates:
        day_index = calendar.get_loc(day)
        signal_date = calendar[day_index - 1]

        # Plan using only yesterday's portfolio and scores. Today's prices and
        # today's sale proceeds cannot change the chosen names or dollar budgets.
        orders = []
        if day in entry_dates:
            candidates = score_groups.get(signal_date)
            if candidates is not None:
                selected = candidates.loc[candidates.prediction.gt(0)].sort_values(
                    ["prediction", "symbol"], ascending=[False, True]
                ).head(config.top_n)
                total_budget = min(cash, previous_nav * config.monthly_fraction)
                if len(selected) and total_budget > 1e-8:
                    orders = [(symbol, total_budget / len(selected)) for symbol in selected.symbol]

        today = quotes.loc[day]
        # A mandatory merger conversion is not a discretionary sale or a new
        # model decision. Retain the lot id, entry date, and original maturity.
        for action in corporate_actions:
            if day != pd.Timestamp(action["effective_date"]):
                continue
            for lot in lots:
                if lot.symbol == action["old_symbol"]:
                    old_units = lot.units
                    is_cash = action.get("kind", "stock") == "cash"
                    lot.units *= (action["cash_per_adjusted_unit"] if is_cash
                                  else action["adjusted_unit_ratio"])
                    lot.symbol = action["new_symbol"]
                    if is_cash:
                        # Reserve a confirmed cash entitlement until the original
                        # horizon; do not release merger cash for early purchases.
                        lot.cash_settlement = True
                        lot.mark = 1.0
                    action_ledger.append(dict(
                        date=day, lot_id=lot.lot_id,
                        event=("mandatory_cash_settlement" if is_cash else
                               "mandatory_symbol_change" if action.get("kind") == "rename"
                               else "mandatory_stock_conversion"),
                        old_symbol=action["old_symbol"], new_symbol=action["new_symbol"],
                        old_units=old_units, new_units=lot.units,
                        share_exchange_ratio=action.get("share_exchange_ratio", 0),
                        cash_amount=lot.units if is_cash else 0.0,
                        original_maturity=lot.maturity,
                    ))
        # Exit each matured lot independently, even when its ticker has left
        # the entry universe. Never interpret disappearance as a cash payout.
        remaining = []
        for lot in lots:
            if day >= lot.maturity:
                if lot.cash_settlement:
                    cash += lot.units
                    fills.append(dict(date=day, signal_date=pd.NaT, side="SETTLEMENT",
                                      lot_id=lot.lot_id, symbol=lot.symbol, units=lot.units,
                                      price=1.0, notional=lot.units, fee=0.0))
                    continue
                price = today.get(lot.symbol, np.nan)
                if not np.isfinite(price):
                    raise DataGapError(f"Missing maturity exit: {lot.symbol} on {day.date()}")
                notional = lot.units * price
                fee = notional * fee_rate
                cash += notional - fee
                fills.append(dict(date=day, signal_date=pd.NaT, side="SELL",
                                  lot_id=lot.lot_id, symbol=lot.symbol, units=lot.units,
                                  price=price, notional=notional, fee=fee))
            else:
                remaining.append(lot)
        lots = remaining

        for symbol, budget in orders:
            score_symbol = symbol
            for action in corporate_actions:
                if (action.get("kind") == "rename" and symbol == action["old_symbol"]
                        and day >= pd.Timestamp(action["effective_date"])):
                    symbol = action["new_symbol"]
            price = today.get(symbol, np.nan)
            if not np.isfinite(price):
                diagnostics.append(dict(date=day, symbol=symbol, event="missing_entry_quote"))
                continue
            # Budget includes fees, so a purchase never spends more than planned.
            notional = budget / (1 + fee_rate)
            fee = notional * fee_rate
            units = notional / price
            cash -= notional + fee
            lots.append(Lot(next_lot_id, symbol, day, day + pd.Timedelta(days=90),
                            units, price, day_index))
            if score_symbol != symbol:
                action_ledger.append(dict(
                    date=day, lot_id=next_lot_id, event="entry_symbol_alias",
                    old_symbol=score_symbol, new_symbol=symbol, old_units=units,
                    new_units=units, share_exchange_ratio=1.0, cash_amount=0.0,
                    original_maturity=day + pd.Timedelta(days=90),
                ))
            fills.append(dict(date=day, signal_date=signal_date, side="BUY",
                              lot_id=next_lot_id, symbol=symbol, units=units,
                              price=price, notional=notional, fee=fee))
            next_lot_id += 1

        stale_count = 0
        for lot in lots:
            if lot.cash_settlement:
                continue
            price = today.get(lot.symbol, np.nan)
            if np.isfinite(price):
                lot.mark = price
                lot.last_quote_index = day_index
            else:
                age = day_index - lot.last_quote_index
                if age > config.max_stale_sessions:
                    raise DataGapError(f"Held quote gap: {lot.symbol} on {day.date()} ({age} sessions)")
                stale_count += 1
                diagnostics.append(dict(date=day, symbol=lot.symbol, event="stale_mark"))

        market_value = sum(lot.units * lot.mark for lot in lots)
        if cash < -1e-6 or not np.isfinite(cash + market_value):
            raise ValueError(f"Invalid portfolio accounting on {day.date()}")
        cash = max(0.0, cash)  # Only remove floating-point roundoff below zero.
        previous_nav = cash + market_value
        history.append(dict(date=day, cash=cash, market_value=market_value,
                            nav=previous_nav, open_lots=len(lots), stale_lots=stale_count))

    open_lots = pd.DataFrame([
        dict(lot_id=lot.lot_id, symbol=lot.symbol, entry_date=lot.entry_date,
             maturity=lot.maturity, units=lot.units, last_price=lot.mark,
             market_value=lot.units * lot.mark,
             reserved_settlement=lot.cash_settlement,
             last_quote_date=calendar[lot.last_quote_index])
        for lot in lots
    ])
    return BacktestResult(
        pd.DataFrame(history).set_index("date"),
        pd.DataFrame(fills, columns=["date", "signal_date", "side", "lot_id", "symbol",
                                   "units", "price", "notional", "fee"]),
        open_lots,
        pd.DataFrame(diagnostics, columns=["date", "symbol", "event"]),
        pd.DataFrame(action_ledger, columns=["date", "lot_id", "event", "old_symbol",
                                           "new_symbol", "old_units", "new_units",
                                           "share_exchange_ratio", "cash_amount", "original_maturity"]),
    )


def performance(nav, initial_cash=1_000_000.0):
    """Include first-close fees in returns and drawdown, before any purchases."""
    nav = nav.astype(float)
    if len(nav) < 2 or not (np.isfinite(nav) & nav.gt(0)).all():
        raise ValueError("Need at least two positive, finite portfolio values")
    returns = nav.pct_change()
    returns.iloc[0] = nav.iloc[0] / initial_cash - 1
    daily_rf = 1.02 ** (1 / 252) - 1
    volatility = returns.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    peaks = nav.cummax().clip(lower=initial_cash)
    return {
        "final_value": float(nav.iloc[-1]),
        "cagr": float((nav.iloc[-1] / initial_cash) ** (1 / years) - 1),
        "max_drawdown": float((nav / peaks - 1).min()),
        "sharpe": float((returns.mean() - daily_rf) / volatility * np.sqrt(252))
        if volatility > 0 else None,
    }
