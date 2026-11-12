"""HW4 lot-based cash accounting, independently implemented for this submission.

Scheduling and missing-price assumptions follow Lecture 8/1_backtesting.ipynb;
this module does not import the course OMS or Backtester implementation.
Input scores must come from features eligible at their signal date, including
membership for the following session's entry quarter. Future label availability
must never be used to filter scores.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


class UnreliableHeldPrice(RuntimeError):
    """Stop when an existing position cannot be valued defensibly."""

    def __init__(self, message, partial_result=None, failed_date=None):
        super().__init__(message)
        self.partial_result = partial_result
        self.failed_date = failed_date


@dataclass
class Position:
    """One purchase, with its own maturity even for repeated stock purchases."""

    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass
class SimulationResult:
    """Auditable account history, trades, remaining positions and diagnostics."""

    history: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    diagnostics: pd.DataFrame


def simulate(prices, benchmark, scores, start, end, quality_audit=None,
             initial_cash=1_000_000.0, top_n=10, cohort_fraction=1/3,
             fee_bps=5.0, holding_days=90, max_stale_sessions=5, corporate_actions=None,
             stock_mergers=None):
    """Plan from yesterday's balance, sell mature lots, then fill planned buys.

    Prices have symbol/date/adj_close; benchmark has date/adj_close; scores
    have date/symbol/score, where date is the preceding close's signal date.
    The benchmark is invested at the first test close. Fractional adjusted-price
    units include total-return exposure; dividends are not added separately.
    """
    if initial_cash <= 0 or not np.isfinite(initial_cash):
        raise ValueError("initial_cash must be finite and positive")
    if top_n < 1 or holding_days < 1 or max_stale_sessions < 0:
        raise ValueError("Invalid count, holding period, or stale-price limit")
    if not 0 < cohort_fraction <= 1 or not np.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError("Invalid allocation fraction or fee")
    p, b, s = prices.copy(), benchmark.copy(), scores.copy()
    for frame in (p, b, s):
        frame["date"] = pd.to_datetime(frame.date)
    if p.duplicated(["date", "symbol"]).any() or s.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate price or score keys")
    if b.date.duplicated().any():
        raise ValueError("Duplicate benchmark dates")
    b = b.set_index("date").adj_close.sort_index()
    if not (np.isfinite(b) & b.gt(0)).all():
        raise ValueError("Invalid benchmark prices")
    calendar = pd.DatetimeIndex(b.index)
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if days.empty or calendar.get_loc(days[0]) == 0:
        raise ValueError("Need test sessions and one preceding signal session")
    if not p.date.isin(calendar).all():
        raise ValueError("Price dates must be on the benchmark calendar")
    valid_prices = np.isfinite(p.adj_close) & p.adj_close.gt(0)
    p = p.loc[valid_prices]

    unresolved, blocked = {}, set()
    if quality_audit is not None and not quality_audit.empty:
        for row in quality_audit.itertuples():
            key = (pd.Timestamp(row.date), row.symbol)
            blocked.add(key)
            if row.reason == "unresolved_price_scale_transition":
                unresolved.setdefault(key[0], set()).add(row.symbol)
    quotes = {date: group.set_index("symbol").adj_close for date, group in p.groupby("date")}
    predictions = {date: group.set_index("symbol").score for date, group in s.groupby("date")}
    conversions = {}
    retired_symbols = set()
    if stock_mergers is not None and not stock_mergers.empty:
        for row in stock_mergers.itertuples():
            if not np.isfinite(row.adjusted_unit_ratio) or row.adjusted_unit_ratio <= 0 or not row.source:
                raise ValueError("Stock mergers require a sourced positive conversion ratio")
            conversions.setdefault(pd.Timestamp(row.date), []).append(row)
    events = {}
    if corporate_actions is not None and not corporate_actions.empty:
        actions = corporate_actions.copy()
        actions["date"] = pd.to_datetime(actions.date)
        if actions.duplicated(["date", "symbol"]).any():
            raise ValueError("Duplicate corporate actions")
        for row in actions.itertuples():
            payout = row.cash_per_actual_share * row.adjusted_to_actual_factor
            if not np.isfinite(payout) or payout <= 0 or not row.source:
                raise ValueError("Cash mergers need a positive payout and a verified source")
            event_type = getattr(row, "event_type", "CASH_MERGER")
            timing = getattr(row, "timing", "before_open")
            if event_type not in {"CASH_MERGER", "CASH_REDEMPTION"} or timing not in {"before_open", "after_close"}:
                raise ValueError("Unsupported corporate-action type or timing")
            events.setdefault(row.date, {})[row.symbol] = (payout, row.source, event_type, timing)

    cash, nav, rate = float(initial_cash), float(initial_cash), fee_bps / 10_000
    positions, marks, last_seen = {}, {}, {}
    trades, diagnostics, history = [], [], []
    next_id = 1
    prior = calendar[calendar.get_loc(days[0]) - 1]
    history.append(dict(date=prior, cash=cash, holdings_value=0.0, nav=nav,
                        benchmark_nav=initial_cash, fees=0.0, open_lots=0, stale_marks=0))

    def settle_cash_events(day, selected_events):
        """Book a sourced compulsory payout; this is not an exchange fill."""
        nonlocal cash
        fees = 0.0
        for lot_id, lot in list(positions.items()):
            if lot.symbol not in selected_events:
                continue
            payout, source, event_type, _ = selected_events[lot.symbol]
            notional = lot.units * payout
            fee = notional * rate
            cash += notional - fee
            fees += fee
            trades.append(dict(date=day, signal_date=pd.NaT, lot_id=lot_id, symbol=lot.symbol,
                               side=event_type, units=lot.units, price=payout, fee=fee,
                               cash_change=notional-fee,
                               realized_pnl=lot.units*(payout-lot.entry_price)-lot.entry_fee-fee,
                               source=source))
            diagnostics.append(dict(date=day, symbol=lot.symbol, event=event_type.lower()))
            del positions[lot_id]
        return fees

    for day in days:
        prior_positions = [asdict(lot) for lot in positions.values()]
        prior_trade_count, prior_diagnostic_count = len(trades), len(diagnostics)

        def stop(message):
            """Return only completed sessions; never value the failed session."""
            partial = SimulationResult(
                pd.DataFrame(history).set_index("date"),
                pd.DataFrame(trades[:prior_trade_count], columns=["date", "signal_date", "lot_id", "symbol", "side", "units", "price", "fee", "cash_change", "realized_pnl", "source", "related_symbol", "received_units"]),
                pd.DataFrame(prior_positions, columns=Position.__dataclass_fields__),
                pd.DataFrame(diagnostics[:prior_diagnostic_count], columns=["date", "symbol", "event"]),
            )
            raise UnreliableHeldPrice(message, partial_result=partial, failed_date=day)

        session = calendar.get_loc(day)
        signal_date = calendar[session - 1]
        # Freeze dollar budgets before seeing this session's prices or sales.
        budgets = {}
        if day.to_period("M") != signal_date.to_period("M"):
            ranking = predictions.get(signal_date, pd.Series(dtype=float))
            ranking = ranking.loc[np.isfinite(ranking)]
            # Stable ticker tie-break, independent of SQL/input row order.
            ranking = ranking.sort_index().sort_values(ascending=False, kind="stable").head(top_n)
            if len(ranking):
                budget = min(cash, nav * cohort_fraction) / len(ranking)
                budgets = dict.fromkeys(ranking.index, budget)

        cash_events = events.get(day, {})
        before_open = {symbol:event for symbol,event in cash_events.items() if event[3] == "before_open"}
        after_close = {symbol:event for symbol,event in cash_events.items() if event[3] == "after_close"}
        day_fees = settle_cash_events(day, before_open)
        for event in conversions.get(day, []):
            retired_symbols.add(event.symbol)
            for lot in positions.values():
                if lot.symbol != event.symbol:
                    continue
                successor_quotes = quotes.get(day, pd.Series(dtype=float))
                if event.successor_symbol not in successor_quotes or (day, event.successor_symbol) in blocked:
                    stop(f"Missing verified successor quote: {event.successor_symbol} on {day.date()}")
                old_units = lot.units
                new_units = old_units * event.adjusted_unit_ratio
                trades.append(dict(date=day, signal_date=pd.NaT, lot_id=lot.lot_id,
                                   symbol=lot.symbol, side="STOCK_MERGER", units=old_units,
                                   price=np.nan, fee=0.0, cash_change=0.0, realized_pnl=0.0,
                                   source=event.source, related_symbol=event.successor_symbol,
                                   received_units=new_units))
                # Preserve total entry cost, entry fee, entry date and maturity.
                lot.entry_price = lot.entry_price * old_units / new_units
                lot.units = new_units
                lot.symbol = event.successor_symbol
                diagnostics.append(dict(date=day, symbol=event.symbol, event="verified_stock_merger"))
        held_symbols = {lot.symbol for lot in positions.values()}
        if held_symbols & unresolved.get(day, set()):
            stop(f"Unresolved held-price transition on {day.date()}: {held_symbols & unresolved[day]}")
        observed = quotes.get(day, pd.Series(dtype=float))
        observed = observed.loc[[symbol for symbol in observed.index if (day, symbol) not in blocked and symbol not in before_open and symbol not in retired_symbols]]
        for symbol, price in observed.items():
            marks[symbol], last_seen[symbol] = float(price), session
        stale = 0
        for symbol in held_symbols:
            age = session - last_seen[symbol]
            if age > max_stale_sessions:
                stop(f"Held quote gap exceeds limit: {symbol} on {day.date()}")
            if age:
                stale += 1
                diagnostics.append(dict(date=day, symbol=symbol, event="stale_mark"))

        for lot_id, lot in list(positions.items()):
            if day >= lot.due_date and lot.symbol in observed:
                price = float(observed[lot.symbol])
                notional = lot.units * price
                fee = notional * rate
                change = notional - fee
                cash += change
                day_fees += fee
                trades.append(dict(date=day, signal_date=pd.NaT, lot_id=lot_id, symbol=lot.symbol,
                                   side="SELL", units=lot.units, price=price, fee=fee,
                                   cash_change=change, realized_pnl=lot.units * (price-lot.entry_price)-lot.entry_fee-fee))
                del positions[lot_id]

        for symbol, budget in budgets.items():
            if budget <= 1e-8:
                continue
            if symbol not in observed:
                diagnostics.append(dict(date=day, symbol=symbol, event="rejected_missing_quote"))
                continue
            price = float(observed[symbol])
            units = budget / (price * (1 + rate))
            notional, fee = units * price, units * price * rate
            if notional + fee > cash + 1e-8:
                raise ValueError("Planned purchase exceeds cash")
            cash = max(0.0, cash - notional - fee)
            positions[next_id] = Position(next_id, symbol, units, day, price, fee,
                                          day + pd.Timedelta(days=holding_days))
            trades.append(dict(date=day, signal_date=signal_date, lot_id=next_id, symbol=symbol,
                               side="BUY", units=units, price=price, fee=fee,
                               cash_change=-(notional+fee), realized_pnl=0.0))
            day_fees += fee
            next_id += 1
        day_fees += settle_cash_events(day, after_close)
        holdings = sum(lot.units * marks[lot.symbol] for lot in positions.values())
        nav = cash + holdings
        history.append(dict(date=day, cash=cash, holdings_value=holdings, nav=nav,
                            benchmark_nav=initial_cash * b.loc[day] / b.loc[days[0]],
                            fees=day_fees, open_lots=len(positions), stale_marks=stale))

    return SimulationResult(
        pd.DataFrame(history).set_index("date"),
        pd.DataFrame(trades, columns=["date", "signal_date", "lot_id", "symbol", "side", "units", "price", "fee", "cash_change", "realized_pnl", "source", "related_symbol", "received_units"]),
        pd.DataFrame([asdict(lot) for lot in positions.values()], columns=Position.__dataclass_fields__),
        pd.DataFrame(diagnostics, columns=["date", "symbol", "event"]),
    )
