"""Time separation and direct arithmetic for the new research helper layer."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_models import (
    ResearchSpec,
    choose_alpha,
    predict_research,
    add_market_features,
)


def toy_panel():
    rng = np.random.default_rng(47)
    calendar = pd.bdate_range("2009-12-01", "2016-01-08")
    days = calendar[:-1][calendar[:-1].to_period("M") != calendar[1:].to_period("M")]
    rows = []
    for date in days:
        if date.year < 2010 or date.year > 2015:
            continue
        entry = calendar[calendar.get_loc(date) + 1]
        for j in range(30):
            x = rng.normal()
            rows.append(
                dict(
                    date=date,
                    entry_date=entry,
                    label_end=entry + pd.Timedelta(days=90),
                    symbol=f"T{j:02}",
                    eligible=True,
                    risk_screen_pass=True,
                    x=x,
                    target=0.04 - 0.1 * x + rng.normal(0, 0.02),
                    sector="Example",
                )
            )
    return pd.DataFrame(rows), calendar


@pytest.mark.parametrize("clip", [True, False])
def test_penalty_choice_cannot_use_unfinished_or_later_outcomes(clip):
    p, calendar = toy_panel()
    spec = ResearchSpec("toy", ("x",), feature_clip=clip)
    a, ta = choose_alpha(p, calendar, spec)
    changed = p.copy()
    changed.loc[changed.label_end.ge("2012-12-31"), "target"] = 1e6
    b, tb = choose_alpha(changed, calendar, spec)
    assert a == b
    pd.testing.assert_frame_equal(ta, tb)
    assert (ta.last_training_label < ta.fit_date).all()
    assert ta.last_validation_label.lt("2012-12-31").all()


def test_quarterly_fits_use_only_mature_outcomes_and_preserve_unknown_candidates():
    p, calendar = toy_panel()
    spec = ResearchSpec("toy", ("x",), schedule="quarterly")
    a, aa, _ = predict_research(p, calendar, spec, 100)
    changed = p.copy()
    future = changed.label_end.ge("2015-12-31")
    changed.loc[future, "target"] = np.nan
    b, ba, _ = predict_research(changed, calendar, spec, 100)
    assert len(a) == len(b) and b.target.isna().any()
    np.testing.assert_allclose(a.score, b.score, rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(aa, ba)
    assert len(aa) == 12 and (aa.last_training_label < aa.fit_date).all()
    fixed, _, _ = predict_research(p, calendar, replace(spec, schedule="fixed"), 100)
    first = a.fit_date.eq(a.fit_date.min())
    np.testing.assert_allclose(a.loc[first, "score"], fixed.loc[first, "score"])


def test_new_trailing_features_are_prefix_invariant_and_match_hand_calculations():
    days = pd.bdate_range("2010-01-01", periods=600)
    quotes = pd.DataFrame(
        {f"T{i}": np.exp(np.arange(600) * (0.0002 + i * 0.00001)) for i in range(12)},
        index=days,
    )
    benchmark = pd.Series(np.exp(np.arange(600) * 0.0001), index=days)
    rows = []
    for day in [days[300], days[400], days[500]]:
        for symbol in quotes:
            rows.append(
                dict(
                    date=day,
                    symbol=symbol,
                    entry_date=day + pd.Timedelta(days=1),
                    label_end=day + pd.Timedelta(days=91),
                    eligible=True,
                    risk_screen_pass=True,
                    momentum_63=0.1,
                    momentum_252_21=0.2,
                    volatility_63=0.25,
                    operating_roa=0.05,
                    fcf_to_assets=0.02,
                )
            )
    p = pd.DataFrame(rows)
    sectors = pd.Series("Example", index=quotes.columns)
    full = add_market_features(p, quotes, benchmark, sectors)
    cutoff = days[400]
    short = add_market_features(
        p.loc[p.date.le(cutoff)], quotes.loc[:cutoff], benchmark.loc[:cutoff], sectors
    )
    features = [c for c in full if c not in p and c != "benchmark_target"]
    pd.testing.assert_frame_equal(
        full.loc[full.date.le(cutoff), features], short[features]
    )
    row = full.loc[full.date.eq(days[300]) & full.symbol.eq("T0")].iloc[0]
    np.testing.assert_allclose(
        row.momentum_126_21, quotes.T0.iloc[279] / quotes.T0.iloc[174] - 1
    )
    np.testing.assert_allclose(row.drawdown_252, 0)
    # An ineligible extreme cannot alter the eligible peer median or ranks.
    p.loc[p.symbol.eq("T11"), "eligible"] = False
    a = add_market_features(p, quotes, benchmark, sectors)
    p.loc[p.symbol.eq("T11"), "momentum_252_21"] = 999
    b = add_market_features(p, quotes, benchmark, sectors)
    np.testing.assert_allclose(
        a.sector_relative_momentum_252_21,
        b.sector_relative_momentum_252_21,
        equal_nan=True,
    )
