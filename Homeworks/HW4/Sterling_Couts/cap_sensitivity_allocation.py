"""Sector-specific caps for an explicitly retrospective 2015 sensitivity study.

The old allocation modules are unchanged. A zero cap prohibits NEW purchases;
it never liquidates an immature lot. Unspent allocations remain in cash, with
no replacement of blocked top-20 names by lower-ranked candidates.
"""

from collections.abc import Mapping

import numpy as np

from sector_allocation import plan_sector_budgets as original_plan


def plan_sector_budgets(
    scores, cash, nav, holdings, rules, sectors, sector_cap, *, budget_limit=None
):
    """Apply individual sector limits after the original stock-budget rule."""
    budgets = original_plan(
        scores, cash, nav, holdings, rules, sectors, None, budget_limit=budget_limit
    )
    if sector_cap is None:
        return budgets
    if any(
        symbol not in sectors
        or not isinstance(sectors[symbol], str)
        or not sectors[symbol].strip()
        for symbol in set(holdings) | set(budgets)
    ):
        raise ValueError("Every exposure needs a sector label.")
    labels = {sectors[s] for s in set(holdings) | set(budgets)}
    if isinstance(sector_cap, Mapping):
        if not labels.issubset(sector_cap):
            raise ValueError("A sector cap is missing.")
        caps = sector_cap
    else:
        caps = {label: sector_cap for label in labels}
    if any(not np.isfinite(cap) or not 0 <= cap <= 1 for cap in caps.values()):
        raise ValueError("Sector caps must be finite and in [0, 1].")
    held, proposed = {}, {}
    fee = rules.fee_bps / 10000
    for symbol, value in holdings.items():
        label = sectors[symbol]
        held[label] = held.get(label, 0.0) + value
    for symbol, budget in budgets.items():
        label = sectors[symbol]
        proposed[label] = proposed.get(label, 0.0) + budget / (1 + fee)
    scales = {
        label: min(1.0, max(0.0, caps[label] * nav - held.get(label, 0.0)) / amount)
        for label, amount in proposed.items()
    }
    return {
        symbol: budget * scales[sectors[symbol]]
        for symbol, budget in budgets.items()
        if budget * scales[sectors[symbol]] > 1e-8
    }
