"""Explain development gains without deleting winning or losing observations.

Dollar contributions reconcile to the actual saved account. Outcome deciles
are an ex-post diagnostic of forecasts, not investable NAVs or causal proof.
All group membership is assigned before checking future-label availability.
"""

import numpy as np
import pandas as pd

from hw4.data import checksum
from hw4.settings import PROJECT
from research_session import load_context, save_json, verify_saved_sources


OUT = PROJECT / "results/fundamental_changes"
CONTROLS = [
    ("allocation_stress", "T05_reinvestment"),
    ("rank_model_stability", "W10_full_sector20"),
]


def forecast_groups(scores, calendar):
    """Deciles and an equal-weight eligible-label comparison, with censor counts."""
    s = scores.sort_values(
        ["date", "score", "symbol"], ascending=[True, False, True]
    ).copy()
    s["position"] = s.groupby("date").cumcount() + 1
    s["decile"] = np.ceil(s.groupby("date").score.rank(pct=True) * 10).astype(int)
    s["entry_date"] = calendar[calendar.get_indexer(pd.DatetimeIndex(s.date)) + 1]
    known = s.target.notna() & s.label_end.le("2015-12-31")
    s["observed_target"] = s.target.where(known)
    s["known"] = known
    rows, bins = [], []
    for date, group in s.groupby("date"):
        top = group.loc[group.position.le(20)]
        row = dict(
            date=date,
            entry_year=int(group.entry_date.iloc[0].year),
            eligible=len(group),
            observed=int(group.known.sum()),
            missing_or_immature=int((~group.known).sum()),
            top20=len(top),
            top20_observed=int(top.known.sum()),
            eligible_mean=group.observed_target.mean(),
            top20_mean=top.observed_target.mean(),
            rank_IC=group.score.corr(group.observed_target, method="spearman")
            if group.known.sum() >= 5
            else np.nan,
        )
        row["top20_minus_eligible"] = row["top20_mean"] - row["eligible_mean"]
        rows.append(row)
        for decile, bucket in group.groupby("decile"):
            bins.append(
                dict(
                    date=date,
                    entry_year=row["entry_year"],
                    decile=decile,
                    candidates=len(bucket),
                    observed=int(bucket.known.sum()),
                    mean_return=bucket.observed_target.mean(),
                    median_return=bucket.observed_target.median()
                    if bucket.known.any()
                    else np.nan,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(bins)


def run():
    context, protected = load_context()
    monthly, yearly, stock_rows, sector_rows, concentrations = [], [], [], [], []
    inputs = {}
    for family, name in CONTROLS:
        local = PROJECT / "outputs" / family / name
        for file in ["daily_nav.csv", "fills.csv", "forecasts.csv"]:
            inputs[str(local / file)] = checksum(local / file)
        h = pd.read_csv(local / "daily_nav.csv", parse_dates=["date"]).set_index("date")
        fills = pd.read_csv(local / "fills.csv", parse_dates=["date"])
        signed = fills.assign(
            signed=np.where(fills.side.eq("BUY"), fills.units, -fills.units)
        )
        units = (
            signed.groupby(["date", "symbol"])
            .signed.sum()
            .unstack(fill_value=0)
            .reindex(h.index, fill_value=0)
            .cumsum()
        )
        prices = context["quotes"].loc[h.index, units.columns]
        prior_units, moves = units.shift().fillna(0), prices.diff()
        assert not (prior_units.abs().gt(1e-7) & moves.isna()).any().any()
        fees = (
            fills.groupby(["date", "symbol"])
            .fee.sum()
            .unstack(fill_value=0)
            .reindex(index=h.index, columns=units.columns, fill_value=0)
        )
        pnl = prior_units * moves.fillna(0) - fees
        np.testing.assert_allclose(
            pnl.sum(axis=1).iloc[1:], h.nav.diff().iloc[1:], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            pnl.to_numpy().sum(), h.nav.iloc[-1] - 1e6, atol=1e-6, rtol=0
        )
        monthly_nav = h[["nav", "benchmark_nav"]].resample("ME").last()
        returns = monthly_nav.pct_change(fill_method=None).iloc[1:]
        for date, row in returns.iterrows():
            monthly.append(
                dict(
                    variant=name,
                    month=date,
                    strategy_return=row.nav,
                    benchmark_return=row.benchmark_nav,
                    excess=row.nav - row.benchmark_nav,
                )
            )
        for year in [2013, 2014, 2015]:
            dates = h.index[h.index.year == year]
            prior = h.index[h.index.get_loc(dates[0]) - 1]
            stock = pnl.loc[dates].sum().sort_values(ascending=False)
            sector = (
                stock.groupby(context["sectors"]).sum().sort_values(ascending=False)
            )
            net = stock.sum()
            for symbol, value in stock.items():
                stock_rows.append(
                    dict(
                        variant=name,
                        year=year,
                        symbol=symbol,
                        sector=context["sectors"].loc[symbol],
                        net_pnl=value,
                    )
                )
            for label, value in sector.items():
                sector_rows.append(
                    dict(variant=name, year=year, sector=label, net_pnl=value)
                )
            yearly.append(
                dict(
                    variant=name,
                    year=year,
                    net_pnl=net,
                    strategy_return=h.loc[dates[-1], "nav"] / h.loc[prior, "nav"] - 1,
                    benchmark_return=h.loc[dates[-1], "benchmark_nav"]
                    / h.loc[prior, "benchmark_nav"]
                    - 1,
                    largest_contributor=stock.index[0],
                    largest_pnl=stock.iloc[0],
                    top5_share_of_net_gain=stock.head(5).sum() / net,
                    positive_stock_count=int(stock.gt(0).sum()),
                    negative_stock_count=int(stock.lt(0).sum()),
                )
            )
        total = pnl.sum().sort_values(ascending=False)
        concentrations.append(
            dict(
                variant=name,
                net_gain=total.sum(),
                largest_contributor=total.index[0],
                largest_share=total.iloc[0] / total.sum(),
                top5_share=total.head(5).sum() / total.sum(),
                positive_stocks=int(total.gt(0).sum()),
                negative_stocks=int(total.lt(0).sum()),
            )
        )
    pd.DataFrame(monthly).to_csv(OUT / "control_monthly_returns.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "control_year_attribution.csv", index=False)
    pd.DataFrame(stock_rows).to_csv(OUT / "control_stock_year.csv", index=False)
    pd.DataFrame(sector_rows).to_csv(OUT / "control_sector_year.csv", index=False)
    pd.DataFrame(concentrations).to_csv(OUT / "control_concentration.csv", index=False)
    # Both controls use identical monthly predictions. T05's later purchases
    # reuse those ranks; these labels describe next-session entries only.
    a = pd.read_csv(
        PROJECT / "outputs/allocation_stress/T05_reinvestment/forecasts.csv",
        parse_dates=["date", "label_end"],
    )
    b = pd.read_csv(
        PROJECT / "outputs/rank_model_stability/W10_full_sector20/forecasts.csv",
        parse_dates=["date", "label_end"],
    )
    cols = ["date", "symbol", "score"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    monthly_groups, bins = forecast_groups(a, context["quotes"].index)
    monthly_groups.to_csv(OUT / "forecast_monthly_diagnostics.csv", index=False)
    bins.to_csv(OUT / "forecast_monthly_deciles.csv", index=False)
    annual = monthly_groups.groupby("entry_year").agg(
        candidate_months=("date", "size"),
        evaluable_months=("eligible_mean", "count"),
        candidates=("eligible", "sum"),
        observed=("observed", "sum"),
        missing_or_immature=("missing_or_immature", "sum"),
        mean_monthly_IC=("rank_IC", "mean"),
        mean_top20_return=("top20_mean", "mean"),
        mean_eligible_return=("eligible_mean", "mean"),
        mean_top20_minus_eligible=("top20_minus_eligible", "mean"),
    )
    annual.to_csv(OUT / "forecast_annual_diagnostics.csv")
    bins.groupby(["entry_year", "decile"]).agg(
        monthly_mean_return=("mean_return", "mean"),
        candidates=("candidates", "sum"),
        observed=("observed", "sum"),
        evaluable_months=("mean_return", "count"),
    ).to_csv(OUT / "forecast_annual_deciles.csv")
    verify_saved_sources(protected)
    assert all(checksum(path) == digest for path, digest in inputs.items())
    save_json(
        OUT / "diagnostic_verification.json",
        dict(
            controls=2,
            monthly_forecasts_identical=True,
            daily_and_total_dollar_contributions_reconciled=True,
            bin_membership_assigned_before_label_censoring=True,
            saved_account_inputs_unchanged=True,
            code_sha256=checksum(__file__),
            input_sha256=inputs,
            post2015_runs=0,
        ),
    )
    print(pd.DataFrame(concentrations).to_string(index=False))
    print(annual.to_string())


if __name__ == "__main__":
    run()
