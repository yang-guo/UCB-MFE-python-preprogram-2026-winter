"""Outcome-blind score buckets and descriptive mature-label diagnostics."""

import numpy as np
import pandas as pd


def assign_buckets(frame):
    """Bucket all candidates before inspecting targets; equal scores stay together."""
    result = frame.copy()
    groups = result.groupby("date").score
    percentile = (groups.rank(method="average") - 0.5) / groups.transform("size")
    result["decile"] = (np.floor(10 * percentile) + 1).clip(1, 10).astype(int)
    return result


def momentum_scores(reference, panel):
    """Replace only the ranking with observed 252-to-21-session price momentum.

    This ranking is a diagnostic rule, not a calibrated 90-day return forecast.
    Future target values are carried for evaluation but never used to rank.
    """
    joined = reference.merge(
        panel[["date", "symbol", "momentum_252_21"]],
        on=["date", "symbol"],
        validate="one_to_one",
        how="left",
    )
    if not np.isfinite(joined.momentum_252_21).all():
        raise ValueError("A common-pool candidate has no finite momentum feature.")
    joined["score"] = joined.momentum_252_21
    joined = joined.drop(columns="momentum_252_21")
    if "fit_date" in joined:
        joined = joined.rename(columns={"fit_date": "reference_fit_date"})
    return joined


def describe_scores(scores, panel, benchmark, cutoff, period, signal):
    """Keep unknown outcomes in coverage counts and match the benchmark horizon.

    Primary summaries average within each signal date, then equally across dates.
    These overlapping 90-day labels do not constitute independent daily returns
    and cannot be compounded into portfolio wealth or iid t-statistics.
    """
    details = panel[["date", "symbol", "entry_date", "scheduled_exit", "sector"]]
    frame = assign_buckets(
        scores.merge(details, on=["date", "symbol"], how="left", validate="one_to_one")
    )
    assert frame.entry_date.notna().all() and frame.sector.notna().all()
    frame["observed"] = np.isfinite(frame.target) & frame.label_end.le(
        pd.Timestamp(cutoff)
    )
    frame["realized"] = frame.target.where(frame.observed)
    bp = benchmark.reindex(pd.DatetimeIndex(frame.entry_date)).to_numpy()
    be = benchmark.reindex(pd.DatetimeIndex(frame.label_end)).to_numpy()
    frame["benchmark_return"] = np.where(frame.observed, be / bp - 1, np.nan)
    assert np.isfinite(frame.loc[frame.observed, "benchmark_return"]).all()
    frame["excess"] = frame.realized - frame.benchmark_return
    frame["extreme_label"] = frame.observed & frame.target.abs().gt(5.0)
    monthly = (
        frame.groupby(["date", "decile"])
        .agg(
            candidates=("symbol", "size"),
            observed=("observed", "sum"),
            mean_return=("realized", "mean"),
            median_return=("realized", "median"),
            mean_excess=("excess", "mean"),
            extreme_labels=("extreme_label", "sum"),
        )
        .reset_index()
    )
    summary = (
        monthly.groupby("decile")
        .agg(
            signal_dates=("date", "size"),
            dates_with_outcomes=("mean_return", "count"),
            candidates=("candidates", "sum"),
            observed=("observed", "sum"),
            mean_return=("mean_return", "mean"),
            mean_of_monthly_medians=("median_return", "mean"),
            mean_excess=("mean_excess", "mean"),
            extreme_labels=("extreme_labels", "sum"),
        )
        .reset_index()
    )
    summary["coverage"] = summary.observed / summary.candidates
    ics, tails, compositions = [], [], []
    for date, group in frame.groupby("date"):
        observed = group.loc[group.observed]
        ic = (
            observed.score.corr(observed.realized, method="spearman")
            if len(observed) > 4
            and observed.score.nunique() > 1
            and observed.realized.nunique() > 1
            else np.nan
        )
        within = []
        for _, sector in observed.groupby("sector"):
            if (
                len(sector) > 4
                and sector.score.nunique() > 1
                and sector.realized.nunique() > 1
            ):
                within.append(sector.score.corr(sector.realized, method="spearman"))
        ics.append(
            {
                "date": date,
                "rank_ic": ic,
                "mean_within_sector_ic": np.mean(within) if within else np.nan,
                "candidates": len(group),
                "observed": len(observed),
            }
        )
        ranking = group.sort_values(["score", "symbol"], ascending=[False, True])
        for tail, selected in [
            ("top20", ranking.head(20)),
            ("bottom20", ranking.tail(20)),
        ]:
            tails.append(
                {
                    "date": date,
                    "tail": tail,
                    "candidates": len(selected),
                    "observed": int(selected.observed.sum()),
                    "mean_return": selected.realized.mean(),
                    "median_return": selected.realized.median()
                    if selected.observed.any()
                    else np.nan,
                    "mean_excess": selected.excess.mean(),
                }
            )
            for sector, count in selected.sector.value_counts().items():
                compositions.append(
                    {
                        "date": date,
                        "tail": tail,
                        "sector": sector,
                        "count": count,
                        "fraction_of_names": count / len(selected),
                    }
                )
    tables = {
        "monthly_deciles": monthly,
        "decile_summary": summary,
        "monthly_ic": pd.DataFrame(ics),
        "monthly_tails": pd.DataFrame(tails),
        "tail_sector_composition": pd.DataFrame(compositions),
        "extreme_labels": frame.loc[
            frame.extreme_label, ["date", "symbol", "target", "label_end", "decile"]
        ],
    }
    return {
        name: table.assign(period=period, signal=signal)
        for name, table in tables.items()
    }
