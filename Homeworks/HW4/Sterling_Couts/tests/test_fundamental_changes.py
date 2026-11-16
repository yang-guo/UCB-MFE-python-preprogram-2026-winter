"""Hand-calculated examples guard economic definitions and information timing."""

import numpy as np
import pandas as pd
import pytest

from fundamental_changes import FEATURES, build_change_events, join_changes


def record(fiscal="2012-03-31", filing="2012-05-01", **changes):
    return dict(
        symbol="A",
        fiscal_date=fiscal,
        filing_date=filing,
        period="quarter",
        revenue=100.0,
        operating_income=10.0,
        operating_cash_flow=20.0,
        total_assets=200.0,
        total_debt=80.0,
        cash_and_equivalents=20.0,
        source="course",
        **changes,
    )


def example(**changes):
    old = record()
    new = record("2013-03-31", "2013-05-01")
    new.update(operating_income=15, operating_cash_flow=30, total_debt=70, **changes)
    return pd.DataFrame([old, new])


def test_manual_ratios_unknown_metadata_is_not_invented():
    e = build_change_events(example()).iloc[-1]
    np.testing.assert_allclose(
        e[list(FEATURES)].to_numpy(dtype=float), [0.05, 0.05, 0.05]
    )
    assert e.unknown_currency and e.unknown_period_length
    assert e.change_available_date == pd.Timestamp("2013-05-02")


def test_late_prior_is_only_available_prospectively():
    f = example()
    f.loc[0, "filing_date"] = "2013-06-01"
    e = build_change_events(f)
    assert e.iloc[0].pair_status == "no_prior_year"
    assert e.iloc[0][list(FEATURES)].isna().all()
    assert e.iloc[-1].change_fiscal_date == pd.Timestamp("2013-03-31")
    assert e.iloc[-1].change_available_date == pd.Timestamp("2013-06-02")
    assert e.iloc[-1].prior_available_date <= e.iloc[-1].change_available_date
    p = pd.DataFrame(
        dict(
            symbol=["A"] * 3,
            date=pd.to_datetime(["2013-05-01", "2013-06-01", "2013-06-02"]),
            eligible=True,
            risk_screen_pass=True,
        )
    )
    j = join_changes(p, e)
    assert j.operating_margin_change.iloc[:2].isna().all()
    assert j.operating_margin_change.iloc[2] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "column,values",
    [
        ("reported_currency", ["USD", "GBP"]),
        ("source", ["old", "new"]),
        ("accounting_basis", ["going_concern", "liquidation"]),
    ],
)
def test_known_metadata_conflicts_suppress_comparison(column, values):
    f = example()
    f[column] = values
    e = build_change_events(f).iloc[-1]
    assert e.pair_status.startswith("conflict:")
    assert e[list(FEATURES)].isna().all()


def test_invalid_duration_suppresses_flows_but_not_balance_sheet():
    f = example()
    f["period_start"] = ["2012-01-01", "2012-01-01"]
    e = build_change_events(f).iloc[-1]
    assert pd.isna(e.operating_margin_change) and pd.isna(e.cash_profitability_change)
    assert e.net_deleveraging == pytest.approx(0.05)


def test_missing_quarters_do_not_change_year_pair_and_53_week_year_works():
    f = example()
    f.loc[1, "fiscal_date"] = "2013-04-06"
    e = build_change_events(f).iloc[-1]
    assert e.operating_margin_change == pytest.approx(0.05)
    f.loc[1, "fiscal_date"] = "2014-03-31"
    f.loc[1, "filing_date"] = "2014-05-01"
    assert build_change_events(f).iloc[-1].pair_status == "no_prior_year"


def test_zero_denominators_and_negative_profits():
    f = example()
    f.loc[1, ["revenue", "total_assets"]] = 0
    assert build_change_events(f).iloc[-1][list(FEATURES)].isna().all()
    f = example()
    f.loc[1, "operating_income"] = -10
    assert build_change_events(f).iloc[-1].operating_margin_change == pytest.approx(
        -0.2
    )


def test_invalid_filings_and_cutoff():
    f = example()
    f.loc[1, "filing_date"] = f.loc[1, "fiscal_date"]
    assert len(build_change_events(f)) == 1
    f = example()
    assert len(build_change_events(f, "2013-05-01")) == 1


def test_staleness_and_eligible_ranks_preserve_original_order():
    a = example()
    b = example()
    b["symbol"] = "B"
    b.loc[1, "operating_income"] = 30
    c = example()
    c["symbol"] = "C"
    c.loc[1, "operating_income"] = -30
    e = build_change_events(pd.concat([a, b, c]))
    p = pd.DataFrame(
        dict(
            symbol=["B", "C", "A", "A"],
            date=pd.to_datetime(["2013-05-02"] * 3 + ["2015-01-01"]),
            eligible=[True, False, True, True],
            risk_screen_pass=True,
        ),
        index=[8, 2, 10, 6],
    )
    j = join_changes(p, e)
    assert j.index.equals(p.index) and j.symbol.equals(p.symbol)
    np.testing.assert_allclose(j.rank_operating_margin_change.iloc[[0, 2]], [1, 0.5])
    assert pd.isna(j.rank_operating_margin_change.iloc[1])
    assert j.iloc[-1].change_stale and j.iloc[-1][list(FEATURES)].isna().all()


def test_duplicate_and_ambiguous_periods_not_arbitrarily_selected():
    f = example()
    with pytest.raises(ValueError, match="duplicate"):
        build_change_events(pd.concat([f, f.iloc[[0]]]))
    extra = record("2012-04-01", "2012-05-02")
    e = build_change_events(pd.concat([f, pd.DataFrame([extra])])).iloc[-1]
    assert e.pair_status == "ambiguous_prior_year" and e[list(FEATURES)].isna().all()
