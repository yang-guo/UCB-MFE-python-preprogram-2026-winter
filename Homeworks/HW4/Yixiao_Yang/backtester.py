"""An independent, cash-accounted 90-calendar-day backtester for HW4.

Prices are adjusted-price units, not a claim about actual-share execution.
No dividends are added separately. Decisions and inclusive dollar budgets
are fixed at the preceding close; today's fills cannot change those budgets.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "date", "side", "symbol", "lot_id", "signal_date", "entry_date",
    "maturity_date", "scheduled_exit", "units", "price", "gross_value",
    "fee", "cash_change", "realized_pnl", "position_type", "action_id",
    "action_effective_date", "source_url", "original_symbol", "from_symbol",
    "from_units", "units_per_adjusted_unit",
]
ORDER_COLUMNS = [
    "order_id", "signal_date", "entry_date", "symbol", "score", "budget",
    "signal_cash", "signal_nav", "status",
]
POSITION_COLUMNS = [
    "lot_id", "symbol", "signal_date", "entry_date", "maturity_date",
    "scheduled_exit", "units", "entry_price", "entry_budget", "last_price",
    "last_quote_date", "stale_sessions", "market_value", "unrealized_pnl",
    "position_type", "action_id", "action_effective_date", "source_url",
    "original_symbol", "entry_units",
]
ACTION_COLUMNS = [
    "action_id", "symbol", "effective_date", "cash_per_adjusted_unit", "source_url",
]
SECURITY_ACTION_COLUMNS = [
    "action_id", "symbol", "effective_date", "successor_symbol",
    "units_per_adjusted_unit", "source_url",
]


@dataclass
class BacktestResult:
    history: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    open_positions: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DataQualityError(RuntimeError):
    """A reliable portfolio path is unavailable; retain the valid prefix."""

    def __init__(self, date, symbol, cause, partial_result=None):
        self.date = pd.Timestamp(date)
        self.symbol = str(symbol)
        self.cause = str(cause)
        self.partial_result = partial_result
        self.diagnostic = {
            "date": self.date.isoformat(), "symbol": self.symbol,
            "cause": self.cause,
        }
        super().__init__(f"{self.date.date()} {self.symbol}: {self.cause}")


def _positive_quote(value):
    return pd.notna(value) and np.isfinite(value) and value > 0


def _check_calendar(index):
    index = pd.DatetimeIndex(index)
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Prices require unique, sorted market-session dates")
    if index.tz is not None:
        raise ValueError("Use timezone-naive market dates")
    return index


def _action_schedule(actions, calendar, columns, amount_column, allow_zero=False):
    """Validate dated, sourced actions and expose them only when effective.

    A caller must verify both the corporate action and its conversion from
    actual shares to the database's adjusted-price units. A zero cash amount
    requires independently verified extinguishment, never just missing prices.
    This function cannot verify source documents and does not infer actions.
    """
    if actions is None:
        return {}
    actions = pd.DataFrame(actions).copy()
    if actions.empty:
        return {}
    if not set(columns).issubset(actions.columns):
        raise ValueError(f"Corporate actions must include {columns}")
    actions = actions[columns]
    if actions.isna().any().any():
        raise ValueError("Corporate actions require all identifiers, dates, amounts, and sources")
    string_columns = ["action_id", "symbol", "source_url"]
    if "successor_symbol" in actions:
        string_columns.append("successor_symbol")
    for column in string_columns:
        if not actions[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("Corporate-action identifiers and source URLs must be nonempty strings")
    if not actions.source_url.str.match(r"^https?://[^/\s]+").all():
        raise ValueError("Corporate actions require an HTTP(S) source URL")
    actions["effective_date"] = pd.to_datetime(actions.effective_date)
    dates = pd.DatetimeIndex(actions.effective_date)
    if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
        raise ValueError("Corporate-action effective dates must be timezone-naive calendar dates")
    amounts = pd.to_numeric(actions[amount_column], errors="raise")
    actions[amount_column] = amounts
    sign_ok = amounts.ge(0).all() if allow_zero else amounts.gt(0).all()
    if not np.isfinite(amounts).all() or not sign_ok:
        raise ValueError("Action amounts/ratios must be finite and have the required nonnegative/positive sign")
    if actions.action_id.duplicated().any() or actions.duplicated(["symbol", "effective_date"]).any():
        raise ValueError("Corporate actions require unique IDs and symbol/effective-date pairs")
    schedule = {}
    for action in actions.sort_values(["effective_date", "symbol"]).to_dict("records"):
        position = calendar.searchsorted(action["effective_date"])
        if position == len(calendar):
            continue  # an action after the cutoff cannot affect this run
        observation_date = calendar[position]
        day = schedule.setdefault(observation_date, {})
        if action["symbol"] in day:
            raise ValueError("Two same-symbol actions map to one market session")
        day[action["symbol"]] = action
    return schedule


def run_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    start_date,
    end_date,
    initial_cash: float = 1_000_000.0,
    top_n: int = 20,
    cohort_fraction: float = 1 / 3,
    fee_bps: float = 10.0,
    holding_days: int = 90,
    max_stale_sessions: int = 5,
    cash_actions=None,
    security_actions=None,
) -> BacktestResult:
    """Buy ranked monthly cohorts, track separate lots, and record daily NAV.

    ``scores`` must contain only candidates eligible at the signal close.
    Required columns: signal_date, entry_date, symbol, score. The caller owns
    feature/fit availability and historical membership; this function checks
    that every supplied signal precedes its entry by one market session.

    Each ticket is ``min(signal_cash, fraction * signal_nav) / top_n``.
    Fewer eligible names leave unused slots in cash. This limits each new
    purchase, not the aggregate weight of overlapping lots in one company.
    Missing entry quotes cancel the original ticket without substitutions.
    Missing maturity quotes stop the run instead of inventing an exit.

    Optional sourced cash mergers convert existing stock lots into fixed,
    ring-fenced cash claims on/after their effective dates. Claims remain in
    NAV but cannot fund purchases until the original lot's 90-day maturity.
    One modeled exit fee is charged on release; it is a portfolio convention,
    not a claim about an actual merger fee or payment-settlement timestamp.

    Optional sourced security actions transfer existing lots into a quoted
    successor at a caller-verified adjusted-unit ratio. Original investment
    budgets and maturity dates stay unchanged; conversion itself has no fee.
    """
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if not 0 < cohort_fraction <= 1 or not 0 <= fee_bps < 10_000:
        raise ValueError("Require 0 < cohort_fraction <= 1 and 0 <= fee_bps < 10000")
    if holding_days < 1 or max_stale_sessions < 0:
        raise ValueError("holding_days must be positive and stale limit nonnegative")
    if prices.columns.has_duplicates:
        raise ValueError("Price columns must be unique symbols")
    calendar = _check_calendar(prices.index)
    # Discard any rows after the evaluation cutoff before simulation.
    cutoff = pd.Timestamp(end_date)
    calendar = calendar[calendar <= cutoff]
    run_dates = calendar[calendar >= pd.Timestamp(start_date)]
    if len(run_dates) == 0:
        raise ValueError("No market sessions in the requested interval")
    first_position = calendar.get_indexer([run_dates[0]])[0]
    if first_position == 0:
        raise ValueError("Need the prior market session for initial signal and NAV")
    initial_date = calendar[first_position - 1]
    sim_dates = calendar[first_position - 1:]
    quotes = prices.loc[sim_dates]
    action_schedule = _action_schedule(cash_actions, calendar, ACTION_COLUMNS,
                                       "cash_per_adjusted_unit", allow_zero=True)
    security_schedule = _action_schedule(security_actions, calendar, SECURITY_ACTION_COLUMNS,
                                         "units_per_adjusted_unit")
    cash_ids = {action["action_id"] for day in action_schedule.values() for action in day.values()}
    security_ids = {action["action_id"] for day in security_schedule.values() for action in day.values()}
    if cash_ids & security_ids:
        raise ValueError("Action IDs must also be unique across action types")
    for day, actions in action_schedule.items():
        if set(actions) & set(security_schedule.get(day, {})):
            raise ValueError("Conflicting cash and security conversions for one symbol/session")
    required = ["signal_date", "entry_date", "symbol", "score"]
    if not set(required).issubset(scores.columns):
        raise ValueError(f"scores must include {required}")
    s = scores[required].copy()
    s["signal_date"] = pd.to_datetime(s.signal_date)
    s["entry_date"] = pd.to_datetime(s.entry_date)
    s = s.loc[s.entry_date.between(run_dates[0], run_dates[-1])].copy()
    if s[required].isna().any().any() or not np.isfinite(s.score).all():
        raise ValueError("Scores, symbols, signal dates, and entry dates must be known")
    if s.duplicated(["entry_date", "symbol"]).any():
        raise ValueError("Duplicate stock scores for an entry date")
    unknown = set(s.symbol) - set(quotes.columns)
    if unknown:
        raise ValueError(f"Scored symbols absent from price columns: {sorted(unknown)[:5]}")
    for signal_date, entry_date in s[["signal_date", "entry_date"]].drop_duplicates().itertuples(index=False, name=None):
        entry_position = calendar.get_indexer([entry_date])[0]
        if entry_position <= 0 or calendar[entry_position - 1] != signal_date:
            raise ValueError("A signal must be the immediately preceding market session")
    score_groups = {d: g for d, g in s.groupby("signal_date", sort=True)}
    cash = float(initial_cash)
    fee_rate = fee_bps / 10_000
    lots: dict[int, dict[str, Any]] = {}
    trade_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    history_rows = [{
        "date": initial_date, "cash": cash, "holdings_value": 0.0,
        "nav": cash, "n_open_lots": 0, "stale_lots": 0,
        "stock_value": 0.0, "cash_claim_value": 0.0,
    }]
    pending: dict[pd.Timestamp, list[int]] = {}
    next_lot_id = 1
    stale_marks = 0
    max_cash_error = 0.0

    def snapshot(complete=True):
        positions = []
        for lot in lots.values():
            row = dict(lot)
            row["market_value"] = row["units"] * row["last_price"]
            row["unrealized_pnl"] = row["market_value"] - row["entry_budget"]
            positions.append(row)
        history = pd.DataFrame(history_rows).set_index("date")
        trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
        orders = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
        open_positions = pd.DataFrame(positions, columns=POSITION_COLUMNS)
        realized = float(trades.realized_pnl.sum()) if len(trades) else 0.0
        unrealized = float(open_positions.unrealized_pnl.sum()) if len(open_positions) else 0.0
        diagnostics = {
            "complete": complete,
            "initial_cash": float(initial_cash),
            "first_entry_session": run_dates[0].isoformat(),
            "last_valid_session": history.index[-1].isoformat(),
            "requested_end": cutoff.isoformat(),
            "buy_count": int((trades.side == "BUY").sum()),
            "sell_count": int((trades.side == "SELL").sum()),
            "cash_merger_conversion_count": int((trades.side == "CASH_MERGER_CONVERSION").sum()),
            "cash_claim_release_count": int((trades.side == "CASH_CLAIM_RELEASE").sum()),
            "security_conversion_count": int((trades.side == "SECURITY_CONVERSION").sum()),
            "cancelled_missing_entry_quotes": int((orders.status == "cancelled_missing_quote").sum()),
            "cancelled_effective_action_entries": int((orders.status == "cancelled_cash_action").sum()),
            "cancelled_security_action_entries": int((orders.status == "cancelled_security_action").sum()),
            "open_lots": len(lots),
            "total_fees": float(trades.fee.sum()),
            "gross_trading_volume": float(trades.loc[trades.side.isin(["BUY", "SELL"]), "gross_value"].sum()),
            "gross_cash_claim_releases": float(trades.loc[trades.side.eq("CASH_CLAIM_RELEASE"), "gross_value"].sum()),
            "cash_claim_release_fees": float(trades.loc[trades.side.eq("CASH_CLAIM_RELEASE"), "fee"].sum()),
            "mean_cash_fraction": float((history.cash / history.nav).where(history.nav.gt(0), 0).iloc[1:].mean()),
            "zero_nav_sessions": int(history.nav.eq(0).sum()),
            "stale_lot_session_marks": stale_marks,
            "terminal_stale_lots": int(open_positions.stale_sessions.gt(0).sum()),
            "terminal_cash_claim_lots": int(open_positions.position_type.eq("cash_claim").sum()),
            "terminal_cash_claim_value": float(history.cash_claim_value.iloc[-1]),
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "pnl_reconciliation_error": float(history.nav.iloc[-1] - initial_cash - realized - unrealized),
            "max_cash_reconstruction_error": max_cash_error,
            "cash_interest_rate": 0.0,
            "fee_bps_each_side": float(fee_bps),
            "missing_exit_policy": "stop; do not infer proceeds or delay the exit",
            "position_units": "fractional adjusted-price units",
            "cash_action_policy": "Verified entitlement replaces stock; ring-fenced until original lot maturity; one modeled exit fee on release",
            "security_action_policy": "Verified adjusted-unit transfer to observed successor; original cost and maturity retained; zero conversion fee",
        }
        return BacktestResult(history, trades, orders, open_positions, diagnostics)

    def stop(date, symbol, cause):
        result = snapshot(complete=False)
        result.diagnostics["data_gap"] = {"date": str(date.date()), "symbol": symbol, "cause": cause}
        raise DataQualityError(date, symbol, cause, result)

    def plan(date, nav):
        group = score_groups.get(date)
        if group is None or cash <= 1e-8:
            return
        ranked = group.sort_values(["score", "symbol"], ascending=[False, True], kind="stable").head(top_n)
        ticket = min(cash, nav * cohort_fraction) / top_n
        if ticket <= 1e-8:
            return
        for row in ranked.itertuples(index=False):
            order_id = len(order_rows)
            order_rows.append({
                "order_id": order_id, "signal_date": date,
                "entry_date": row.entry_date, "symbol": row.symbol,
                "score": float(row.score), "budget": ticket,
                "signal_cash": cash, "signal_nav": nav, "status": "planned",
            })
            pending.setdefault(row.entry_date, []).append(order_id)
        assert ticket * len(ranked) <= cash + 1e-7

    # First-test entries use signals already available at the prior close.
    plan(initial_date, initial_cash)
    for date in run_dates:
        row = quotes.loc[date]
        marks = {}
        due_ids = []
        today_actions = action_schedule.get(date, {})
        today_security_actions = security_schedule.get(date, {})
        conversions = {}
        security_conversions = {}
        # Check the whole existing portfolio before booking this day's actions.
        # Failure preserves a coherent history/trade prefix through yesterday.
        for lot_id, lot in lots.items():
            due = pd.notna(lot["scheduled_exit"]) and date >= lot["scheduled_exit"]
            if lot["position_type"] == "stock" and lot["symbol"] in today_actions:
                conversions[lot_id] = today_actions[lot["symbol"]]
            if lot["position_type"] == "cash_claim" or lot_id in conversions:
                if due:
                    due_ids.append(lot_id)
                continue  # a certified cash claim needs no further stock quote
            if lot["symbol"] in today_security_actions:
                action = today_security_actions[lot["symbol"]]
                successor = action["successor_symbol"]
                if successor not in row.index or not _positive_quote(row.get(successor)):
                    stop(date, lot["symbol"], f"Missing valid successor quote for verified security conversion to {successor}")
                security_conversions[lot_id] = action
                price = row[successor]
            else:
                price = row[lot["symbol"]]
            if due:
                if not _positive_quote(price):
                    stop(date, lot["symbol"], "Missing valid quote at scheduled 90-day exit")
                due_ids.append(lot_id)
            elif _positive_quote(price):
                marks[lot_id] = (float(price), date, 0)
            else:
                stale = lot["stale_sessions"] + 1
                if stale > max_stale_sessions:
                    stop(date, lot["symbol"], f"Held position has {stale} missing market-session quotes")
                marks[lot_id] = (lot["last_price"], lot["last_quote_date"], stale)

        # Book conversions only after every existing lot passes the preflight,
        # so a separate unresolved data gap still preserves yesterday's prefix.
        for lot_id, action in security_conversions.items():
            lot = lots[lot_id]
            old_symbol, old_units = lot["symbol"], lot["units"]
            successor = action["successor_symbol"]
            ratio = float(action["units_per_adjusted_unit"])
            lot.update(
                symbol=successor, units=old_units * ratio,
                last_price=float(row[successor]), last_quote_date=date, stale_sessions=0,
                action_id=action["action_id"], action_effective_date=action["effective_date"],
                source_url=action["source_url"],
            )
            trade_rows.append({
                "date": date, "side": "SECURITY_CONVERSION", "symbol": successor,
                "lot_id": lot_id, "signal_date": lot["signal_date"],
                "entry_date": lot["entry_date"], "maturity_date": lot["maturity_date"],
                "scheduled_exit": lot["scheduled_exit"], "units": lot["units"],
                "price": lot["last_price"], "gross_value": lot["units"] * lot["last_price"],
                "fee": 0.0, "cash_change": 0.0, "realized_pnl": 0.0,
                "position_type": "stock", "action_id": lot["action_id"],
                "action_effective_date": lot["action_effective_date"], "source_url": lot["source_url"],
                "original_symbol": lot["original_symbol"], "from_symbol": old_symbol,
                "from_units": old_units, "units_per_adjusted_unit": ratio,
            })
        for lot_id, action in conversions.items():
            lot = lots[lot_id]
            lot.update(
                position_type="cash_claim",
                last_price=float(action["cash_per_adjusted_unit"]),
                stale_sessions=0,
                action_id=action["action_id"],
                action_effective_date=action["effective_date"],
                source_url=action["source_url"],
            )
            trade_rows.append({
                "date": date, "side": "CASH_MERGER_CONVERSION", "symbol": lot["symbol"],
                "lot_id": lot_id, "signal_date": lot["signal_date"],
                "entry_date": lot["entry_date"], "maturity_date": lot["maturity_date"],
                "scheduled_exit": lot["scheduled_exit"], "units": lot["units"],
                "price": lot["last_price"], "gross_value": lot["units"] * lot["last_price"],
                "fee": 0.0, "cash_change": 0.0, "realized_pnl": 0.0,
                "position_type": "cash_claim", "action_id": lot["action_id"],
                "action_effective_date": lot["action_effective_date"], "source_url": lot["source_url"],
                "original_symbol": lot["original_symbol"], "from_symbol": lot["symbol"],
                "from_units": lot["units"], "units_per_adjusted_unit": 1.0,
            })
        for lot_id in due_ids:
            lot = lots.pop(lot_id)
            claim = lot["position_type"] == "cash_claim"
            price = lot["last_price"] if claim else float(row[lot["symbol"]])
            gross = lot["units"] * price
            fee = gross * fee_rate
            proceeds = gross - fee
            cash += proceeds
            trade_rows.append({
                "date": date, "side": "CASH_CLAIM_RELEASE" if claim else "SELL", "symbol": lot["symbol"],
                "lot_id": lot_id, "signal_date": lot["signal_date"],
                "entry_date": lot["entry_date"], "maturity_date": lot["maturity_date"],
                "scheduled_exit": lot["scheduled_exit"], "units": lot["units"],
                "price": price, "gross_value": gross, "fee": fee,
                "cash_change": proceeds, "realized_pnl": proceeds - lot["entry_budget"],
                "position_type": lot["position_type"], "action_id": lot["action_id"],
                "action_effective_date": lot["action_effective_date"], "source_url": lot["source_url"],
                "original_symbol": lot["original_symbol"],
            })
        for lot_id, (price, quote_date, stale) in marks.items():
            lots[lot_id].update(last_price=price, last_quote_date=quote_date, stale_sessions=stale)
            stale_marks += int(stale > 0)

        for order_id in pending.pop(date, []):
            order = order_rows[order_id]
            if order["symbol"] in today_actions:
                order["status"] = "cancelled_cash_action"
                continue  # the old shares are converted before today's close
            if order["symbol"] in today_security_actions:
                order["status"] = "cancelled_security_action"
                continue
            price = row[order["symbol"]]
            if not _positive_quote(price):
                order["status"] = "cancelled_missing_quote"
                continue
            budget = order["budget"]
            if budget > cash + 1e-7:
                raise AssertionError("Fixed orders would overspend cash")
            units = budget / (float(price) * (1 + fee_rate))
            gross = units * float(price)
            fee = gross * fee_rate
            cash -= budget
            if -1e-7 < cash < 0:
                cash = 0.0
            maturity = date + pd.Timedelta(days=holding_days)
            exit_position = calendar.searchsorted(maturity)
            scheduled = calendar[exit_position] if exit_position < len(calendar) else pd.NaT
            lot = {
                "lot_id": next_lot_id, "symbol": order["symbol"],
                "signal_date": order["signal_date"], "entry_date": date,
                "maturity_date": maturity, "scheduled_exit": scheduled,
                "units": units, "entry_price": float(price), "entry_budget": budget,
                "last_price": float(price), "last_quote_date": date, "stale_sessions": 0,
                "position_type": "stock", "action_id": "",
                "action_effective_date": pd.NaT, "source_url": "",
                "original_symbol": order["symbol"], "entry_units": units,
            }
            lots[next_lot_id] = lot
            trade_rows.append({
                "date": date, "side": "BUY", "symbol": order["symbol"],
                "lot_id": next_lot_id, "signal_date": order["signal_date"],
                "entry_date": date, "maturity_date": maturity, "scheduled_exit": scheduled,
                "units": units, "price": float(price), "gross_value": gross, "fee": fee,
                "cash_change": -budget, "realized_pnl": 0.0,
                "position_type": "stock", "action_id": "",
                "action_effective_date": pd.NaT, "source_url": "",
                "original_symbol": order["symbol"],
            })
            order["status"] = "filled"
            next_lot_id += 1

        stock_value = sum(lot["units"] * lot["last_price"] for lot in lots.values()
                          if lot["position_type"] == "stock")
        claim_value = sum(lot["units"] * lot["last_price"] for lot in lots.values()
                          if lot["position_type"] == "cash_claim")
        holdings_value = stock_value + claim_value
        nav = cash + holdings_value
        if cash < -1e-7 or holdings_value < 0 or not np.isfinite(nav) or nav < 0:
            raise AssertionError("Invalid long-only cash accounting")
        reconstructed = initial_cash + sum(trade["cash_change"] for trade in trade_rows)
        max_cash_error = max(max_cash_error, abs(cash - reconstructed))
        if not np.isclose(cash, reconstructed, rtol=1e-12, atol=1e-6):
            raise AssertionError("Cash does not reconcile to recorded trades")
        history_rows.append({
            "date": date, "cash": cash, "holdings_value": holdings_value,
            "nav": nav, "n_open_lots": len(lots),
            "stale_lots": sum(lot["stale_sessions"] > 0 for lot in lots.values()),
            "stock_value": stock_value, "cash_claim_value": claim_value,
        })
        plan(date, nav)

    result = snapshot()
    if not np.isclose(result.diagnostics["pnl_reconciliation_error"], 0, rtol=0, atol=1e-5):
        raise AssertionError("Final realized and unrealized PnL do not reconcile")
    return result


def benchmark_history(series, start, end, initial_cash=1_000_000.0):
    """Invest once at the first study close; add a prior-session cash anchor."""
    benchmark = pd.Series(series, copy=True).astype(float)
    benchmark.index = _check_calendar(benchmark.index)
    window = benchmark.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if window.empty:
        raise ValueError("No benchmark observations in the study interval")
    if not np.isfinite(window).all() or (window <= 0).any():
        raise ValueError("Benchmark has missing or invalid study-window quotes")
    earlier = benchmark.index[benchmark.index < window.index[0]]
    if not len(earlier):
        raise ValueError("Need a prior benchmark market session for the NAV anchor")
    nav = initial_cash * window / window.iloc[0]
    nav = pd.concat([pd.Series([float(initial_cash)], index=[earlier[-1]]), nav])
    nav.index.name = "date"
    return nav.rename("benchmark")


def performance_metrics(nav, initial_cash, start_date, end_date, risk_free=0.02):
    """CAGR, signed maximum drawdown, and daily-return annualized Sharpe.

    The initial capital is the drawdown anchor. The first session's return
    includes entry fees. CAGR measures actual elapsed days between the first
    and last study closes. Sharpe uses 252 sessions and sample volatility.
    """
    nav = pd.Series(nav).astype(float)
    nav.index = _check_calendar(nav.index)
    values = nav.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if values.empty or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Metrics require nonnegative, finite NAV observations")
    if initial_cash <= 0 or risk_free <= -1:
        raise ValueError("Invalid initial capital or annual risk-free rate")
    returns = values.pct_change(fill_method=None)
    returns = returns.mask(values.eq(0) & values.shift(1).eq(0), 0.0)
    returns.iloc[0] = values.iloc[0] / initial_cash - 1
    daily_risk_free = (1 + risk_free) ** (1 / 252) - 1
    excess = returns - daily_risk_free
    volatility = returns.std(ddof=1)
    sharpe = float(np.sqrt(252) * excess.mean() / volatility) if volatility > 0 else float("nan")
    peaks = np.maximum.accumulate(np.r_[initial_cash, values.to_numpy()])
    drawdowns = np.r_[initial_cash, values.to_numpy()] / peaks - 1
    years = (values.index[-1] - values.index[0]).days / 365.25
    total = float(values.iloc[-1] / initial_cash - 1)
    return {
        "final_value": float(values.iloc[-1]),
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 else float("nan"),
        "max_drawdown": float(drawdowns.min()),
        "sharpe": sharpe,
    }
