"""Causal monthly sector stress and reduced new-purchase budgets.

The sector statistic is median stock volatility, not sector-index volatility.
No outcome, current active flag or future coverage enters its construction.
"""

import numpy as np
import pandas as pd


def build_sector_stress(panel, lookback=12, min_stocks=10):
    """Divide current median 63-return volatility by 12 PRIOR monthly readings.

    Reindex the sector/month grid so missing months cannot silently disappear
    from the lookback. A missing/too-small sector stays unknown, never zero.
    Static vendor sector classifications remain a disclosed data limitation.
    """
    if lookback < 1 or min_stocks < 1:
        raise ValueError("Positive lookback and minimum coverage required")
    if panel.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate stock/date observations")
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    months = dates.to_period("M")
    if months.has_duplicates or not months.equals(
        pd.period_range(months[0], months[-1], freq="M")
    ):
        raise ValueError("Require one signal date per consecutive month")
    sectors = sorted(panel.sector.dropna().unique())
    eligible = panel.loc[panel.eligible].copy()
    if eligible.sector.isna().any():
        raise ValueError("Eligible stock lacks a sector")
    valid = np.isfinite(eligible.volatility_63) & eligible.volatility_63.gt(0)
    grouped = eligible.loc[valid].groupby(["date", "sector"])
    frame = grouped.agg(
        sector_volatility=("volatility_63", "median"),
        contributing_stocks=("symbol", "nunique"),
    ).reindex(pd.MultiIndex.from_product([dates, sectors], names=["date", "sector"]))
    frame["contributing_stocks"] = frame.contributing_stocks.fillna(0).astype(int)
    frame.loc[frame.contributing_stocks.lt(min_stocks), "sector_volatility"] = np.nan
    frame = frame.reset_index().sort_values(["sector", "date"])
    frame["historical_median"] = frame.groupby("sector").sector_volatility.transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=lookback).median()
    )
    frame["stress_ratio"] = frame.sector_volatility / frame.historical_median
    frame["reading_known"] = np.isfinite(frame.stress_ratio)
    return frame.sort_values(["date", "sector"]).reset_index(drop=True)


def scale_orders(budgets, sectors, readings, multiplier, threshold=1.5):
    """Scale already capped budgets; no replacement orders or early sales.

    The control (multiplier=1) ignores stress. For treatment rules, an unknown
    reading blocks new purchases and is separately recorded, not called stress.
    Current account state determines baseline budgets anew at each decision;
    retained cash can therefore affect later ordinary monthly purchases.
    """
    if (
        multiplier not in (0.0, 0.5, 1.0)
        or threshold <= 0
        or not np.isfinite(threshold)
    ):
        raise ValueError("Unsupported multiplier or threshold")
    orders, audit = {}, []
    for symbol, budget in budgets.items():
        sector = sectors[symbol]
        ratio = float(readings.get(sector, np.nan))
        known = np.isfinite(ratio) and ratio >= 0
        high = known and ratio >= threshold
        factor = (
            1.0
            if multiplier == 1
            else (0.0 if not known else multiplier if high else 1.0)
        )
        scaled = budget * factor
        if scaled > 1e-8:
            orders[symbol] = scaled
        audit.append(
            {
                "symbol": symbol,
                "sector": sector,
                "stress_ratio": ratio,
                "reading_known": known,
                "high_stress": high,
                "baseline_budget": budget,
                "multiplier": factor,
                "scaled_budget": scaled,
                "withheld_budget": budget - scaled,
            }
        )
    return orders, audit


def forward_risk(panel, quotes, cutoff):
    """Diagnostic-only future risk over each observed entry/exit price path.

    Price gaps leave risk missing. Do not carry marks, clip losses or remove
    the stock from the diagnostic denominator because of future coverage.
    This function is not called by feature construction or trading decisions.
    """
    rows = panel.copy()
    calendar = quotes.index
    entry = calendar.get_indexer(pd.DatetimeIndex(rows.entry_date))
    exit_ = calendar.get_indexer(pd.DatetimeIndex(rows.label_end))
    values = quotes.to_numpy()
    columns = quotes.columns.get_indexer(rows.symbol)
    observed = (
        np.isfinite(rows.target) & rows.label_end.le(pd.Timestamp(cutoff))
    ).to_numpy()
    vol, drawdown = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for i in np.flatnonzero(observed & (entry >= 0) & (exit_ > entry) & (columns >= 0)):
        path = values[entry[i] : exit_[i] + 1, columns[i]]
        if not (np.isfinite(path) & (path > 0)).all():
            continue
        returns = path[1:] / path[:-1] - 1
        vol[i] = np.std(returns, ddof=1) * np.sqrt(252) if len(returns) > 1 else np.nan
        drawdown[i] = np.min(path / np.maximum.accumulate(path) - 1)
    rows["observed"] = observed
    rows["future_volatility"] = vol
    rows["future_drawdown"] = drawdown
    return rows


def diagnose(panel, quotes, benchmark, stress, cutoff="2015-12-31"):
    """Equal-weight within sector/date, then equally weight sector/date means.

    Trading-candidate pool retains its existing individual risk screen. Stress
    is measured separately from the broader eligible pool before that screen.
    Known outcomes after an annual reset's year end are allowed here, provided
    they complete by the overall 2015 cutoff. This differs from account returns.
    """
    candidates = panel.loc[
        panel.eligible
        & panel.risk_screen_pass
        & panel.entry_date.between("2013-01-01", cutoff)
    ].copy()
    rows = forward_risk(candidates, quotes, cutoff).merge(
        stress[["date", "sector", "stress_ratio"]],
        on=["date", "sector"],
        how="left",
        validate="many_to_one",
    )
    rows["regime"] = np.where(
        rows.stress_ratio.isna(),
        "unknown",
        np.where(rows.stress_ratio.ge(1.5), "high", "normal"),
    )
    rows["year"] = rows.entry_date.dt.year
    rows["realized"] = rows.target.where(rows.observed)
    bp = benchmark.reindex(pd.DatetimeIndex(rows.entry_date)).to_numpy()
    be = benchmark.reindex(pd.DatetimeIndex(rows.label_end)).to_numpy()
    rows["excess"] = rows.realized - (be / bp - 1)
    rows["extreme_label"] = rows.observed & rows.target.abs().gt(5)
    rows["pending"] = rows.scheduled_exit.gt(pd.Timestamp(cutoff))
    rows["unavailable_due"] = ~rows.pending & ~rows.observed
    dates = (
        rows.groupby(["date", "sector", "year", "regime"])
        .agg(
            candidates=("symbol", "size"),
            observed=("observed", "sum"),
            pending=("pending", "sum"),
            unavailable_due=("unavailable_due", "sum"),
            mean_return=("realized", "mean"),
            median_return=("realized", "median"),
            mean_excess=("excess", "mean"),
            mean_future_volatility=("future_volatility", "mean"),
            mean_future_drawdown=("future_drawdown", "mean"),
            risk_observed=("future_volatility", "count"),
            stress_ratio=("stress_ratio", "first"),
            extreme_labels=("extreme_label", "sum"),
        )
        .reset_index()
    )
    assert (
        dates.candidates == dates.observed + dates.pending + dates.unavailable_due
    ).all()
    metrics = [
        "mean_return",
        "median_return",
        "mean_excess",
        "mean_future_volatility",
        "mean_future_drawdown",
    ]

    def aggregate(keys):
        return (
            dates.groupby(keys)
            .agg(
                **{m: (m, "mean") for m in metrics},
                sector_dates=("date", "size"),
                dates_with_outcomes=("mean_return", "count"),
                candidates=("candidates", "sum"),
                observed=("observed", "sum"),
                pending=("pending", "sum"),
                unavailable_due=("unavailable_due", "sum"),
                risk_observed=("risk_observed", "sum"),
                extreme_labels=("extreme_labels", "sum"),
            )
            .reset_index()
        )

    # Only dates with observed outcomes in both regimes contribute to this
    # descriptive paired contrast. No iid standard errors are asserted.
    same_date = dates.groupby(["date", "regime"])[metrics].mean().unstack("regime")
    paired = pd.DataFrame(index=same_date.index)
    for metric in metrics:
        values = same_date[metric].reindex(columns=["high", "normal"])
        paired[metric + "_high_minus_normal"] = values.high - values.normal
    return {
        "sector_date_diagnostics": dates,
        "regime_summary": aggregate(["regime"]),
        "annual_regime_summary": aggregate(["year", "regime"]),
        "sector_regime_summary": aggregate(["sector", "regime"]),
        "same_date_contrasts": paired.reset_index(),
        "extreme_labels": rows.loc[
            rows.extreme_label,
            ["date", "symbol", "sector", "regime", "target", "label_end"],
        ],
    }
