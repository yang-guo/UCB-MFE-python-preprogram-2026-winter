# Research loop copied from cap_sensitivity_engine.py; prior engines stay frozen.
# Explicit allocator choice isolates candidate replacement and redistribution.
"""Separate allocation experiment adapted from our frozen hw4.engine loop.

The initial change is the budget planner; an explicit optional extension reuses
monthly scores after maturity sales. Orders use previous-close information;
all original maturity, fill, missing-price, fee and accounting rules remain.
A disabled-sector-cap parity test guards divergence from the original engine.
The original implementation is preserved, not edited or dynamically patched.
"""

from dataclasses import asdict, dataclass, field
import numpy as np
import pandas as pd
from hw4.engine import FILL_COLUMNS, Lot, Result, validate_accounting
from hw4.settings import TradingRules
from cap_sensitivity_allocation import plan_sector_budgets as original_plan
from feasible_allocation import plan_feasible_budgets


@dataclass
class ReinvestmentResult(Result):
    """Retain the ordinary ledgers plus the provenance of extra order batches."""

    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_sector_backtest(
    quotes,
    benchmark,
    scores,
    start,
    end,
    rules=None,
    *,
    sectors=None,
    sector_cap=0.20,
    reinvest_maturity_cash=False,
    entry_eligibility=None,
    allocator="original",
):
    """Run a fresh account. Scores are signal-date/symbol/score observations.

    Monthly signals are precomputed causally and must carry fit audits in the
    research runner. Optional extra batches use yesterday's net maturity sales,
    latest monthly scores and rechecked prior-close eligibility. A monthly batch
    takes precedence on overlapping dates. New-entry quotes may be missing:
    reject, never substitute.
    Existing positions may carry a flagged last mark for max_stale_sessions.
    Longer gaps stop the run with status='blocked_data_gap'; the partial history
    is retained but must not be represented as completed test performance.
    """
    if allocator not in {"original", "feasible"}:
        raise ValueError("Unknown allocator")
    plan_sector_budgets = (
        original_plan if allocator == "original" else plan_feasible_budgets
    )
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
    if reinvest_maturity_cash:
        if entry_eligibility is None or not entry_eligibility.index.equals(calendar):
            raise ValueError(
                "Reinvestment requires causal eligibility indexed by entry session."
            )
        if not quotes.columns.equals(entry_eligibility.columns):
            raise ValueError("Eligibility columns must match quote symbols.")
        if entry_eligibility.isna().any().any() or not all(
            dtype == np.dtype("bool") for dtype in entry_eligibility.dtypes
        ):
            raise ValueError("Eligibility must contain only boolean decisions.")
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
    latest_scores, latest_score_date = pd.Series(dtype=float), pd.NaT
    previous_sale_proceeds = 0.0
    decisions = []
    for day_ix in chosen:
        day, signal = calendar[day_ix], calendar[day_ix - 1]
        day_scores = forecasts.get(signal, pd.Series(dtype=float))
        if signal in forecasts:
            latest_scores, latest_score_date = day_scores, signal
        trigger, budget_limit = "monthly" if signal in forecasts else "none", None
        if (
            reinvest_maturity_cash
            and trigger == "none"
            and previous_sale_proceeds > 1e-8
        ):
            trigger, budget_limit = "maturity_reinvestment", previous_sale_proceeds
            # Rankings are carried forward, but present eligibility is not.
            # This boolean row was built exclusively from prior-close information.
            current = entry_eligibility.loc[day].reindex(
                latest_scores.index, fill_value=False
            )
            day_scores = latest_scores.loc[current]
        budgets = plan_sector_budgets(
            day_scores,
            cash,
            prior_nav,
            holdings,
            rules,
            sectors,
            sector_cap,
            budget_limit=budget_limit,
        )
        if reinvest_maturity_cash and trigger != "none":
            decisions.append(
                {
                    "date": day,
                    "signal_date": signal,
                    "model_score_date": latest_score_date,
                    "trigger": trigger,
                    "previous_sale_proceeds": previous_sale_proceeds,
                    "prior_cash": cash,
                    "prior_nav": prior_nav,
                    "planned_spending": sum(budgets.values()),
                    "eligible_ranked_candidates": len(day_scores),
                    "planned_orders": len(budgets),
                }
            )
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
        todays_sale_proceeds = 0.0
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
            todays_sale_proceeds += change
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
        # Only actual completed sales create next-session reinvestment capacity.
        # Unused proceeds are not retried every day; they remain available to
        # the ordinary monthly batch or a later independently triggered batch.
        previous_sale_proceeds = todays_sale_proceeds
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
    if reinvest_maturity_cash:
        result = ReinvestmentResult(
            result.history,
            result.fills,
            result.open_lots,
            result.incidents,
            result.status,
            pd.DataFrame(decisions),
        )
    return result
