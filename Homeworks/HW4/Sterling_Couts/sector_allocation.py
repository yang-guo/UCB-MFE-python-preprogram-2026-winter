"""A sector entry constraint layered over the original stock-budget rule.

Use prior-close NAV and every existing lot's marked exposure. Within a constrained
sector, shrink its proposed orders proportionally; do not choose replacements or
spend the blocked budget elsewhere. This isolates adding one allocation limit.
The limit governs planned entries, not forced liquidation or future price drift.
"""

from dataclasses import replace

import numpy as np

from hw4.engine import plan_budgets


def plan_sector_budgets(
    scores, cash, nav, holdings, rules, sectors, sector_cap, *, budget_limit=None
):
    """Return fee-inclusive orders with unchanged 5% stock-cap semantics."""
    if budget_limit is not None:
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("NAV must be finite and positive.")
        if not np.isfinite(budget_limit) or budget_limit < 0:
            raise ValueError("Reinvestment budget must be finite and nonnegative.")
        if budget_limit == 0:
            return {}
        # Limit total intended spending BEFORE individual and sector caps.
        # Account cash itself is unchanged, so NAV/cash reconciliation still holds.
        rules = replace(
            rules, cohort_fraction=min(rules.cohort_fraction, budget_limit / nav)
        )
    budgets = plan_budgets(scores, cash, nav, holdings, rules)
    if sector_cap is None:
        return budgets
    if not np.isfinite(sector_cap) or not 0 < sector_cap <= 1:
        raise ValueError("Sector cap must be in (0, 1].")
    needed = set(holdings) | set(budgets)
    if any(
        symbol not in sectors
        or not isinstance(sectors[symbol], str)
        or not sectors[symbol].strip()
        for symbol in needed
    ):
        raise ValueError("Every holding and proposed purchase needs a sector label.")
    held, proposed = {}, {}
    fee = rules.fee_bps / 10000
    for symbol, value in holdings.items():
        sector = sectors[symbol]
        held[sector] = held.get(sector, 0.0) + value
    for symbol, budget in budgets.items():
        sector = sectors[symbol]
        proposed[sector] = proposed.get(sector, 0.0) + budget / (1 + fee)
    scales = {
        sector: min(1.0, max(0.0, nav * sector_cap - held.get(sector, 0.0)) / value)
        for sector, value in proposed.items()
    }
    return {
        symbol: budget * scales[sectors[symbol]]
        for symbol, budget in budgets.items()
        if budget * scales[sectors[symbol]] > 1e-8
    }
