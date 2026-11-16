"""Rank-aware replacement and redistribution under unchanged entry limits.

Walk down all known rankings, skipping exhausted stock/sector capacity. Admit
at most top_n names with a nominal equal ticket, then redistribute leftover
budget among those names within their remaining caps. This is a deterministic
allocation heuristic, not an optimizer over every possible set of names.
No current fill price, future outcome, borrowing or premature sale is used.
"""

from collections.abc import Mapping

import numpy as np

from sector_allocation import plan_sector_budgets as validate_and_plan


def plan_feasible_budgets(
    scores, cash, nav, holdings, rules, sectors, sector_cap, *, budget_limit=None
):
    """Return fee-inclusive budgets, respecting prior-close account capacity."""
    validate_and_plan(
        scores, cash, nav, holdings, rules, sectors, None, budget_limit=budget_limit
    )
    total = min(cash, nav * rules.cohort_fraction)
    if budget_limit is not None:
        total = min(total, budget_limit)
    if total <= 1e-8 or scores.empty:
        return {}
    required = set(scores.index) | set(holdings)
    if any(
        s not in sectors or not isinstance(sectors[s], str) or not sectors[s].strip()
        for s in required
    ):
        raise ValueError("Every candidate and holding needs a sector label.")
    labels = {sectors[s] for s in required}
    caps = (
        {label: 1.0 if sector_cap is None else sector_cap for label in labels}
        if not isinstance(sector_cap, Mapping)
        else dict(sector_cap)
    )
    if not labels.issubset(caps) or any(
        not np.isfinite(x) or not 0 <= x <= 1 for x in caps.values()
    ):
        raise ValueError("Every sector needs a finite cap in [0, 1].")
    cost = 1 + rules.fee_bps / 10000
    held = {
        label: sum(value for s, value in holdings.items() if sectors[s] == label)
        for label in labels
    }
    sector_room = {
        label: max(0.0, caps[label] * nav - held[label]) * cost for label in labels
    }
    name_room = {
        s: max(0.0, rules.name_cap * nav - holdings.get(s, 0.0)) * cost
        for s in scores.index
    }
    ranking = (
        scores.rename("score")
        .rename_axis("symbol")
        .reset_index()
        .sort_values(["score", "symbol"], ascending=[False, True])
    )
    ticket = total / min(rules.top_n, len(ranking))
    budgets = {}
    remaining = total
    for symbol in ranking.symbol:
        sector = sectors[symbol]
        amount = min(ticket, name_room[symbol], sector_room[sector], remaining)
        if amount <= 1e-8:
            continue
        budgets[symbol] = amount
        name_room[symbol] = max(0.0, name_room[symbol] - amount)
        sector_room[sector] = max(0.0, sector_room[sector] - amount)
        remaining = max(0.0, remaining - amount)
        if len(budgets) == rules.top_n or remaining <= 1e-8:
            break
    # Equal increments among admitted names. Proportional sector scaling keeps
    # their shared residual capacity from being counted once for every stock.
    for _ in range(2 * len(budgets) + len(labels) + 2):
        active = [
            s for s in budgets if name_room[s] > 1e-8 and sector_room[sectors[s]] > 1e-8
        ]
        if not active or remaining <= 1e-8:
            break
        additions = {s: min(remaining / len(active), name_room[s]) for s in active}
        sums = {
            label: sum(x for s, x in additions.items() if sectors[s] == label)
            for label in labels
        }
        for s in active:
            label = sectors[s]
            additions[s] *= min(1.0, sector_room[label] / sums[label])
        spent = sum(additions.values())
        if spent <= 1e-8:
            break
        for s, x in additions.items():
            budgets[s] += x
            name_room[s] = max(0.0, name_room[s] - x)
            sector_room[sectors[s]] = max(0.0, sector_room[sectors[s]] - x)
        remaining = max(0.0, remaining - spent)
    else:
        raise RuntimeError("Budget redistribution failed to converge.")
    if sum(budgets.values()) > total + 1e-7:
        raise ValueError("Allocation exceeds its known cash/cohort/proceeds budget.")
    return budgets
