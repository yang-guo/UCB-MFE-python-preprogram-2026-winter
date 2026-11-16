"""Independent daily, long-only, 90-calendar-day lot backtester.

Design informed by Lectures 2/5/8; no lecture backtester is imported. Signals
contain dollar-budget decisions made at the preceding close. Today's prices
determine fills, not stock selection. All money is denominated in dollars;
units are fractional adjusted-price exposure, not historical broker shares.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .settings import TradingRules


@dataclass(frozen=True)
class Lot:
    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    maturity: pd.Timestamp
    entry_price: float
    entry_fee: float


@dataclass
class Result:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    incidents: pd.DataFrame
    status: str


FILL_COLUMNS = [
    "date",
    "signal_date",
    "lot_id",
    "symbol",
    "side",
    "price",
    "units",
    "notional",
    "fee",
    "cash_change",
    "maturity",
    "realized_pnl",
]


def plan_budgets(scores, cash, nav, holdings, rules):
    """Allocate equal fee-inclusive budgets, subject to prior-close account limits.

    The stock cap aggregates every outstanding lot and this batch's purchases.
    A cap is an entry restriction: price drift never forces premature sales.
    """
    if scores.index.has_duplicates or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Scores require unique symbols and finite values")
    if not np.isfinite([cash, nav]).all() or cash < -1e-8 or nav <= 0:
        raise ValueError("Invalid account value")
    if any(not np.isfinite(x) or x < 0 for x in holdings.values()):
        raise ValueError("Invalid holding exposure")
    if not np.isclose(cash + sum(holdings.values()), nav):
        raise ValueError("Account does not reconcile")
    ranked = scores.rename("score").rename_axis("symbol").reset_index()
    ranked = ranked.sort_values(["score", "symbol"], ascending=[False, True]).head(
        rules.top_n
    )
    total = min(cash, nav * rules.cohort_fraction)
    equal = total / len(ranked) if len(ranked) else 0.0
    cost = rules.fee_bps / 10000
    budgets = {}
    for symbol in ranked.symbol:
        remaining_notional = max(0.0, nav * rules.name_cap - holdings.get(symbol, 0.0))
        budget = min(equal, remaining_notional * (1 + cost))
        if budget > 1e-8:
            budgets[symbol] = budget
    if sum(budgets.values()) > cash + 1e-7:
        raise ValueError("Planned orders exceed cash")
    return budgets


def run_backtest(quotes, benchmark, scores, start, end, rules=None):
    """Run a fresh account. Scores are signal-date/symbol/score observations.

    Monthly signals are precomputed causally and must carry fit audits in the
    research runner. New-entry quotes may be missing: reject, never substitute.
    Existing positions may carry a flagged last mark for max_stale_sessions.
    Longer gaps stop the run with status='blocked_data_gap'; the partial history
    is retained but must not be represented as completed test performance.
    """
    rules = TradingRules() if rules is None else rules
    if rules.initial_cash <= 0 or rules.holding_days < 1 or rules.fee_bps < 0:
        raise ValueError("Invalid capital, holding period or costs")
    if (
        rules.top_n < 1
        or not 0 < rules.cohort_fraction <= 1
        or not 0 < rules.name_cap <= 1
    ):
        raise ValueError("Invalid allocation rules")
    if rules.max_stale_sessions < 0:
        raise ValueError("Invalid stale quote limit")
    if (
        not quotes.index.equals(benchmark.index)
        or not quotes.index.is_monotonic_increasing
    ):
        raise ValueError("Quotes and benchmark must share a sorted market calendar")
    if quotes.index.has_duplicates or quotes.columns.has_duplicates:
        raise ValueError("Duplicate price axes")
    observed_values = quotes.to_numpy()
    if np.isinf(observed_values).any() or (observed_values <= 0).any():
        raise ValueError("Use NaN for missing/invalid quotes; prices must be positive")
    if not (np.isfinite(benchmark) & benchmark.gt(0)).all():
        raise ValueError("Invalid benchmark")
    if (
        scores.duplicated(["date", "symbol"]).any()
        or not np.isfinite(scores.score).all()
    ):
        raise ValueError("Invalid prediction keys or scores")
    calendar = quotes.index
    chosen = np.flatnonzero(
        (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
    )
    if not len(chosen) or chosen[0] == 0:
        raise ValueError("Need evaluation dates and a preceding signal close")
    forecasts = {
        date: group.set_index("symbol").score for date, group in scores.groupby("date")
    }
    cash, next_id, realized = rules.initial_cash, 0, 0.0
    lots, ledger, incidents = {}, [], []
    marks, last_seen = {}, {}
    initial_day = calendar[chosen[0] - 1]
    history = [
        {
            "date": initial_day,
            "cash": cash,
            "holdings": 0.0,
            "nav": cash,
            "benchmark_nav": cash,
            "fees": 0.0,
            "open_lots": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "stale_holdings": 0,
        }
    ]
    benchmark_entry = benchmark.iloc[chosen[0]]
    holdings, prior_nav = {}, cash
    status = "complete"
    fee_rate = rules.fee_bps / 10000
    for day_ix in chosen:
        day, signal = calendar[day_ix], calendar[day_ix - 1]
        day_scores = forecasts.get(signal, pd.Series(dtype=float))
        budgets = plan_budgets(day_scores, cash, prior_nav, holdings, rules)
        observed = quotes.iloc[day_ix].dropna()
        # A positive quote can still contain an adjustment/identifier error.
        # Flag large held moves without clipping PnL or inventing a correction.
        for symbol in sorted({lot.symbol for lot in lots.values()}):
            if symbol in observed and symbol in marks:
                ratio = float(observed[symbol]) / marks[symbol]
                if ratio > 2 or ratio < 0.5:
                    units = sum(
                        lot.units for lot in lots.values() if lot.symbol == symbol
                    )
                    incidents.append(
                        {
                            "date": day,
                            "symbol": symbol,
                            "event": "large_held_price_move",
                            "price_ratio": ratio,
                            "prior_price": marks[symbol],
                            "current_price": float(observed[symbol]),
                            "mark_pnl": units
                            * (float(observed[symbol]) - marks[symbol]),
                        }
                    )
        for symbol, price in observed.items():
            marks[symbol], last_seen[symbol] = float(price), day_ix
        stale_symbols = set()
        for symbol in sorted({lot.symbol for lot in lots.values()}):
            age = day_ix - last_seen[symbol]
            if age:
                stale_symbols.add(symbol)
                incidents.append(
                    {
                        "date": day,
                        "symbol": symbol,
                        "event": "stale_mark",
                        "sessions": age,
                        "last_quote": calendar[last_seen[symbol]],
                    }
                )
            if age > rules.max_stale_sessions:
                status = "blocked_data_gap"
                incidents.append(
                    {
                        "date": day,
                        "symbol": symbol,
                        "event": status,
                        "sessions": age,
                        "last_quote": calendar[last_seen[symbol]],
                    }
                )
        if status != "complete":
            break
        daily_fees = 0.0
        # Maturity is determined when the lot is bought, never from a later ranking.
        for lot_id, lot in list(lots.items()):
            if day < lot.maturity:
                continue
            if lot.symbol not in observed:
                incidents.append(
                    {
                        "date": day,
                        "symbol": lot.symbol,
                        "event": "deferred_maturity_missing_quote",
                    }
                )
                continue
            price = float(observed[lot.symbol])
            notional = lot.units * price
            fee = notional * fee_rate
            change = notional - fee
            pnl = change - lot.units * lot.entry_price - lot.entry_fee
            cash += change
            realized += pnl
            daily_fees += fee
            ledger.append(
                [
                    day,
                    pd.NaT,
                    lot_id,
                    lot.symbol,
                    "SELL",
                    price,
                    lot.units,
                    notional,
                    fee,
                    change,
                    lot.maturity,
                    pnl,
                ]
            )
            del lots[lot_id]
        for symbol, budget in budgets.items():
            if symbol not in observed:
                incidents.append(
                    {
                        "date": day,
                        "symbol": symbol,
                        "event": "rejected_entry_missing_quote",
                    }
                )
                continue
            price = float(observed[symbol])
            notional = budget / (1 + fee_rate)
            fee = budget - notional
            units = notional / price
            if budget > cash + 1e-7:
                raise ValueError("Orders attempted to spend the same cash twice")
            lot = Lot(
                next_id,
                symbol,
                units,
                day,
                day + pd.Timedelta(days=rules.holding_days),
                price,
                fee,
            )
            lots[next_id] = lot
            cash -= budget
            if -1e-7 < cash < 0:
                cash = 0.0
            daily_fees += fee
            ledger.append(
                [
                    day,
                    signal,
                    next_id,
                    symbol,
                    "BUY",
                    price,
                    units,
                    notional,
                    fee,
                    -budget,
                    lot.maturity,
                    0.0,
                ]
            )
            next_id += 1
        holdings = {}
        unrealized = 0.0
        for lot in lots.values():
            value = lot.units * marks[lot.symbol]
            holdings[lot.symbol] = holdings.get(lot.symbol, 0.0) + value
            unrealized += value - lot.units * lot.entry_price - lot.entry_fee
        prior_nav = cash + sum(holdings.values())
        if not np.isclose(
            prior_nav - rules.initial_cash, realized + unrealized, atol=1e-6
        ):
            raise ValueError("PnL does not reconcile")
        history.append(
            {
                "date": day,
                "cash": cash,
                "holdings": sum(holdings.values()),
                "nav": prior_nav,
                "benchmark_nav": rules.initial_cash
                * benchmark.loc[day]
                / benchmark_entry,
                "fees": daily_fees,
                "open_lots": len(lots),
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "stale_holdings": len(stale_symbols),
            }
        )
    result = Result(
        pd.DataFrame(history).set_index("date"),
        pd.DataFrame(ledger, columns=FILL_COLUMNS),
        pd.DataFrame(
            [asdict(lot) for lot in lots.values()], columns=Lot.__dataclass_fields__
        ),
        pd.DataFrame(incidents),
        status,
    )
    validate_accounting(result, rules)
    return result


def validate_accounting(result, rules):
    """Independently reconstruct cash from the transaction ledger and check NAV."""
    h, f = result.history, result.fills
    changes = (
        f.groupby("date")
        .cash_change.sum()
        .reindex(h.index, fill_value=0.0)
        .astype(float)
    )
    np.testing.assert_allclose(h.cash, rules.initial_cash + changes.cumsum(), atol=1e-6)
    np.testing.assert_allclose(h.nav, h.cash + h.holdings, atol=1e-6)
    np.testing.assert_allclose(
        h.nav - rules.initial_cash, h.realized_pnl + h.unrealized_pnl, atol=1e-6
    )
    if h.cash.min() < -1e-7:
        raise ValueError("Negative cash")
    buys, sells = f.loc[f.side.eq("BUY")], f.loc[f.side.eq("SELL")]
    if (
        not buys.signal_date.lt(buys.date).all()
        or not sells.date.ge(sells.maturity).all()
    ):
        raise ValueError("Execution or holding-period timing violation")


def performance(nav, annual_rf=0.02):
    """Metrics from daily simple NAV returns, including initial entry costs.

    The first row is pre-trade capital. CAGR's elapsed period starts at the first
    evaluation close (second row); that first close contributes a cost-only
    strategy return and a zero benchmark return. All fractions remain numeric.
    """
    if (
        len(nav) < 3
        or nav.index.has_duplicates
        or not np.isfinite(nav).all()
        or nav.le(0).any()
    ):
        raise ValueError("Need a valid positive daily NAV history")
    returns = nav.pct_change(fill_method=None).iloc[1:]
    years = (nav.index[-1] - nav.index[1]).days / 365.25
    sd = returns.std(ddof=1)
    rf_daily = (1 + annual_rf) ** (1 / 252) - 1
    return {
        "final_value": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "CAGR": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "annualized_volatility": float(sd * np.sqrt(252)),
        "Sharpe": float(np.sqrt(252) * (returns.mean() - rf_daily) / sd)
        if sd > 1e-12
        else np.nan,
    }


def comparison(result, rules=None):
    """Only completed runs receive headline performance metrics."""
    if result.status != "complete":
        raise ValueError("Cannot publish completed-period metrics for a blocked run")
    rules = TradingRules() if rules is None else rules
    h = result.history
    table = pd.DataFrame(
        {
            "strategy": performance(h.nav, rules.annual_rf),
            "benchmark": performance(h.benchmark_nav, rules.annual_rf),
        }
    ).T
    active = h.nav.pct_change().iloc[1:] - h.benchmark_nav.pct_change().iloc[1:]
    te = active.std(ddof=1) * np.sqrt(252)
    table["tracking_error"] = [te, 0.0]
    table["information_ratio"] = [
        active.mean() * 252 / te if te > 1e-12 else np.nan,
        np.nan,
    ]
    return table
