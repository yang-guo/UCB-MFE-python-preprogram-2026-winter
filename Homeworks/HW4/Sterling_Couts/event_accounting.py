"""Separate accounting revision; the frozen predictor and original engine remain intact.

Documented corporate actions are not market sales. Cash consideration becomes a
receivable, then spendable cash after the declared settlement lag. Replacement
exposures inherit the original maturity. Fractional adjusted units and zero
corporate-action commissions follow the existing stylized exposure convention.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from hw4.engine import plan_budgets
from hw4.settings import TradingRules


@dataclass
class EventLot:
    lot_id: int
    origin_symbol: str
    symbol: str
    units: float
    entry_date: pd.Timestamp
    maturity: pd.Timestamp
    basis: float  # Remaining fee-inclusive cost, allocated at corporate events.


def run_event_account(
    quotes,
    benchmark,
    scores,
    raw_quotes,
    events,
    start,
    end,
    rules=None,
    settlement_sessions=2,
):
    """Run unchanged forecasts with explicit event records and claim accounting.

    Event raw-price conversion bases must be supplied and audited in advance.
    Source dates describe history; they never become a predictor or future-coverage
    eligibility rule. Retired tickers cannot receive new fills after extinction.
    The two-session cash lag is a simulation assumption, not a broker settlement fact.
    """
    rules = TradingRules() if rules is None else rules
    if settlement_sessions < 0 or int(settlement_sessions) != settlement_sessions:
        raise ValueError("Settlement lag must be a nonnegative session count")
    if (
        not quotes.index.equals(benchmark.index)
        or not quotes.index.is_monotonic_increasing
    ):
        raise ValueError("Quote and benchmark calendars must match")
    if quotes.index.has_duplicates or quotes.columns.has_duplicates:
        raise ValueError("Duplicate price axes")
    values = quotes.to_numpy()
    if np.isinf(values).any() or (values <= 0).any():
        raise ValueError("Prices must be positive finite or missing")
    if (
        scores.duplicated(["date", "symbol"]).any()
        or not np.isfinite(scores.score).all()
    ):
        raise ValueError("Invalid forecast keys or values")
    dates = quotes.index
    chosen = np.flatnonzero(
        (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    )
    if not len(chosen) or chosen[0] == 0:
        raise ValueError("Need evaluation dates and preceding signal session")
    by_date = {d: g.set_index("symbol").score for d, g in scores.groupby("date")}
    event_days = {}
    for event in events:
        event = dict(event)
        day = pd.Timestamp(event["date"])
        if day not in dates:
            raise ValueError(
                f"Event needs an explicit market-session booking date: {day}"
            )
        event_days.setdefault(day, []).append(event)
    if len({e["id"] for e in events}) != len(events):
        raise ValueError("Duplicate event IDs")

    cash, nav, realized, next_id = rules.initial_cash, rules.initial_cash, 0.0, 0
    lots, claims, marks, last_seen, retired = {}, [], {}, {}, set()
    fills, event_ledger, cash_ledger, incidents, applied_events = [], [], [], [], []
    holdings = {}
    history = [
        dict(
            date=dates[chosen[0] - 1],
            cash=cash,
            holdings=0.0,
            receivables=0.0,
            nav=nav,
            benchmark_nav=nav,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees=0.0,
            open_lots=0,
            stale_holdings=0,
        )
    ]
    initial_benchmark = benchmark.iloc[chosen[0]]
    status = "complete"
    cost = rules.fee_bps / 10000
    for ix in chosen:
        day, signal = dates[ix], dates[ix - 1]
        # Same prior-close planner as the frozen original. Claims count in NAV,
        # but never in the cash available to place orders.
        exposures = {**holdings, "__receivables__": sum(c["amount"] for c in claims)}
        budgets = plan_budgets(
            by_date.get(signal, pd.Series(dtype=float)), cash, nav, exposures, rules
        )
        observed = quotes.iloc[ix].dropna()
        held_symbols = {lot.symbol for lot in lots.values()}
        for symbol, price in observed.items():
            if symbol in marks and symbol in held_symbols:
                ratio = float(price) / marks[symbol]
                if ratio > 2 or ratio < 0.5:
                    incidents.append(
                        dict(
                            date=day,
                            symbol=symbol,
                            event="large_held_price_move",
                            price_ratio=ratio,
                        )
                    )
            marks[symbol], last_seen[symbol] = float(price), ix

        # Mandatory events are booked at the event day's close, before NAV is
        # recorded. Orders in a retiring ticker are rejected on that date.
        for event in event_days.get(day, []):
            old, new = event["old"], event.get("new")
            retired.add(old)
            affected = [lot for lot in lots.values() if lot.symbol == old]
            applied_events.append(
                dict(
                    event_id=event["id"],
                    date=day,
                    symbol=old,
                    affected_lots=len(affected),
                )
            )
            if not affected:
                continue
            basis_date = pd.Timestamp(event["basis_date"])
            if basis_date > day:
                raise ValueError("Future conversion basis")
            old_adjusted = float(quotes.loc[basis_date, old])
            old_raw = float(raw_quotes.loc[basis_date, old])
            if (
                not np.isfinite([old_adjusted, old_raw]).all()
                or min(old_adjusted, old_raw) <= 0
            ):
                raise ValueError("Missing old-security conversion basis")
            if new:
                new_basis_date = basis_date if event["kind"] == "rename" else day
                new_raw = float(raw_quotes.loc[new_basis_date, new])
                new_adjusted = float(quotes.loc[new_basis_date, new])
                if (
                    not np.isfinite([new_raw, new_adjusted]).all()
                    or min(new_raw, new_adjusted) <= 0
                ):
                    raise ValueError("Missing successor conversion basis")
                if new not in observed:
                    raise ValueError("Missing successor event-day quote")
            for lot in affected:
                old_units, old_basis = lot.units, lot.basis
                shares = old_units * old_adjusted / old_raw
                cash_amount = shares * event.get("cash_per_share", 0.0)
                new_units = (
                    shares * event.get("share_ratio", 0.0) * new_raw / new_adjusted
                    if new
                    else 0.0
                )
                share_value = new_units * float(observed[new]) if new else 0.0
                total_value = cash_amount + share_value
                if total_value <= 0:
                    raise ValueError(
                        "A zero-recovery assumption is not an event record"
                    )
                cash_basis = old_basis * cash_amount / total_value
                event_pnl = cash_amount - cash_basis
                realized += event_pnl
                release_ix = ix + settlement_sessions
                if cash_amount:
                    claims.append(
                        dict(
                            event_id=event["id"],
                            lot_id=lot.lot_id,
                            amount=cash_amount,
                            release_ix=release_ix,
                        )
                    )
                event_ledger.append(
                    dict(
                        date=day,
                        event_id=event["id"],
                        kind=event["kind"],
                        lot_id=lot.lot_id,
                        origin_symbol=lot.origin_symbol,
                        old_symbol=old,
                        new_symbol=new,
                        old_units=old_units,
                        actual_share_equivalent=shares,
                        cash_entitlement=cash_amount,
                        new_units=new_units,
                        stock_value=share_value,
                        old_basis=old_basis,
                        cash_basis=cash_basis,
                        new_basis=old_basis - cash_basis,
                        realized_pnl=event_pnl,
                        original_maturity=lot.maturity,
                        cash_available_date=dates[release_ix]
                        if release_ix < len(dates)
                        else pd.NaT,
                    )
                )
                if new:
                    lot.symbol, lot.units, lot.basis = (
                        new,
                        new_units,
                        old_basis - cash_basis,
                    )
                else:
                    del lots[lot.lot_id]

        for claim in list(claims):
            if ix >= claim["release_ix"]:
                cash += claim["amount"]
                cash_ledger.append(
                    dict(
                        date=day,
                        event_id=claim["event_id"],
                        lot_id=claim["lot_id"],
                        cash_change=claim["amount"],
                    )
                )
                claims.remove(claim)
        stale = 0
        for symbol in sorted({lot.symbol for lot in lots.values()}):
            age = ix - last_seen[symbol]
            if age:
                stale += 1
                incidents.append(
                    dict(date=day, symbol=symbol, event="stale_mark", sessions=age)
                )
            if age > rules.max_stale_sessions:
                status = "blocked_data_gap"
                incidents.append(
                    dict(
                        date=day,
                        symbol=symbol,
                        event=status,
                        sessions=age,
                        last_quote=dates[last_seen[symbol]],
                    )
                )
        if status != "complete":
            break
        daily_fees = 0.0
        for lot_id, lot in list(lots.items()):
            if day < lot.maturity:
                continue
            if lot.symbol not in observed:
                incidents.append(
                    dict(
                        date=day,
                        symbol=lot.symbol,
                        event="deferred_maturity_missing_quote",
                    )
                )
                continue
            price = float(observed[lot.symbol])
            notional = lot.units * price
            fee = notional * cost
            change = notional - fee
            pnl = change - lot.basis
            cash += change
            realized += pnl
            daily_fees += fee
            fills.append(
                dict(
                    date=day,
                    signal_date=pd.NaT,
                    lot_id=lot_id,
                    origin_symbol=lot.origin_symbol,
                    symbol=lot.symbol,
                    side="SELL",
                    price=price,
                    units=lot.units,
                    fee=fee,
                    notional=notional,
                    cash_change=change,
                    maturity=lot.maturity,
                    realized_pnl=pnl,
                )
            )
            del lots[lot_id]
        for symbol, budget in budgets.items():
            if symbol in retired or symbol not in observed:
                incidents.append(
                    dict(
                        date=day,
                        symbol=symbol,
                        event="rejected_retired_security"
                        if symbol in retired
                        else "rejected_entry_missing_quote",
                    )
                )
                continue
            price = float(observed[symbol])
            notional = budget / (1 + cost)
            fee, units = budget - notional, notional / price
            if budget > cash + 1e-7:
                raise ValueError("Insufficient cash; receivables cannot fund purchases")
            lot = EventLot(
                next_id,
                symbol,
                symbol,
                units,
                day,
                day + pd.Timedelta(days=rules.holding_days),
                budget,
            )
            lots[next_id] = lot
            cash -= budget
            if -1e-7 < cash < 0:
                cash = 0.0
            daily_fees += fee
            fills.append(
                dict(
                    date=day,
                    signal_date=signal,
                    lot_id=next_id,
                    origin_symbol=symbol,
                    symbol=symbol,
                    side="BUY",
                    price=price,
                    units=units,
                    fee=fee,
                    notional=notional,
                    cash_change=-budget,
                    maturity=lot.maturity,
                    realized_pnl=0.0,
                )
            )
            next_id += 1
        holdings, unrealized = {}, 0.0
        for lot in lots.values():
            value = lot.units * marks[lot.symbol]
            holdings[lot.symbol] = holdings.get(lot.symbol, 0.0) + value
            unrealized += value - lot.basis
        receivables = sum(claim["amount"] for claim in claims)
        nav = cash + sum(holdings.values()) + receivables
        np.testing.assert_allclose(
            nav - rules.initial_cash, realized + unrealized, rtol=0, atol=1e-6
        )
        if cash < -1e-7:
            raise ValueError("Negative cash")
        history.append(
            dict(
                date=day,
                cash=cash,
                holdings=sum(holdings.values()),
                receivables=receivables,
                nav=nav,
                benchmark_nav=rules.initial_cash
                * benchmark.loc[day]
                / initial_benchmark,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                fees=daily_fees,
                open_lots=len(lots),
                stale_holdings=stale,
            )
        )
    result = dict(
        status=status,
        history=pd.DataFrame(history).set_index("date"),
        fills=pd.DataFrame(fills),
        events=pd.DataFrame(event_ledger),
        settlements=pd.DataFrame(cash_ledger),
        incidents=pd.DataFrame(incidents),
        open_lots=pd.DataFrame([asdict(lot) for lot in lots.values()]),
        open_receivables=pd.DataFrame(claims),
        event_inventory=pd.DataFrame(applied_events),
    )
    return result
