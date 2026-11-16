"""Selection and calendar-year accounting for the declared 2013–2015 cap study."""

import numpy as np
import pandas as pd


def best_tested(rows, cap_column):
    """Maximize final wealth among complete accounts without unresolved incidents.

    Report every setting within one cent of the maximum. Prefer the largest
    cap in that tie set: an inactive constraint is not a uniquely optimal rule.
    This is a tested-grid maximum conditional on the fixed forecasting strategy.
    """
    frame = pd.DataFrame(rows)
    eligible = frame.loc[
        frame.status.eq("complete")
        & frame.specific_open_data_issue.fillna("").eq("")
        & np.isfinite(frame.final_value)
    ].copy()
    if eligible.empty:
        raise ValueError("No completed, incident-free setting is available")
    highest = float(eligible.final_value.max())
    ties = eligible.loc[(eligible.final_value - highest).abs().le(0.01)]
    chosen = ties.sort_values([cap_column, "variant"], ascending=[False, True]).iloc[0]
    return chosen.to_dict(), sorted(ties[cap_column].unique().tolist()), highest


def refinement_grid(best_cap, coarse_caps):
    """One-point interior checks between the coarse winner's adjacent settings.

    If the unrestricted reference ties or wins, keep it and do not refine its
    already nonrestrictive neighborhood. This is a bounded refinement, not an
    exhaustive search of the continuous cap domain or a guarantee of optimality.
    """
    if best_cap == 1.0:
        return []
    grid = sorted(coarse_caps)
    at = grid.index(best_cap)
    lo, hi = grid[max(at - 1, 0)], grid[min(at + 1, len(grid) - 1)]
    return [
        round(n / 100, 2)
        for n in range(round(lo * 100) + 1, round(hi * 100))
        if round(n / 100, 2) not in grid
    ]


def calendar_year_returns(history):
    """Returns of the SAME continuous account; no resets or retrospective refits."""
    rows = []
    for year in [2013, 2014, 2015]:
        days = history.index[history.index.year == year]
        prior = history.index[history.index.get_loc(days[0]) - 1]
        rows.append(
            {
                "year": year,
                "strategy_return": history.loc[days[-1], "nav"]
                / history.loc[prior, "nav"]
                - 1,
                "benchmark_return": history.loc[days[-1], "benchmark_nav"]
                / history.loc[prior, "benchmark_nav"]
                - 1,
            }
        )
    result = pd.DataFrame(rows)
    for column, nav in [
        ("strategy_return", "nav"),
        ("benchmark_return", "benchmark_nav"),
    ]:
        np.testing.assert_allclose(
            (1 + result[column]).prod(), history[nav].iloc[-1] / history[nav].iloc[0]
        )
    return result
