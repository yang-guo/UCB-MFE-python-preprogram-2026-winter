"""Unit tests for hw2_cleaning on tiny hand-built tables.

Several cases (conflicting keys, impossible OHLC, missing volume) do not occur in
the supplied database, so these injected examples are the evidence that the
detectors and repairs behave as documented.
"""

import numpy as np
import pandas as pd
import pytest

import hw2_cleaning as hc


def make_rows(rows, ticker="AAA"):
    """rows: list of (ts, open, high, low, close, adj_close, volume)."""
    df = pd.DataFrame(rows, columns=["ts"] + hc.VALUE_COLS)
    df.insert(1, "ticker", ticker)
    return df


def five_days(ticker="AAA", start=100.0):
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    rows = [(d, start + i, start + i + 2, start + i - 2, start + i + 1, (start + i + 1) / 2, 1000)
            for i, d in enumerate(dates)]
    return make_rows(rows, ticker)


def reference_for(raw):
    """Reference table matching raw prices with an adjustment factor of exactly 0.5."""
    ref = hc.prepare_raw(raw)[["ticker", "ts", "close"]].drop_duplicates(["ticker", "ts"])
    ref["adj_close"] = ref["close"] * 0.5
    ref["ts"] = ref["ts"].dt.strftime("%Y-%m-%d")
    return ref


# --- tickers and duplicates -------------------------------------------------

@pytest.mark.parametrize("label", [
    '<a target="_blank"><span style="color: red">AAPL</span></a>',
    "  aapl ", "Aapl", "<b>AA&#80;L</b>",
])
def test_normalize_ticker_label(label):
    assert hc.normalize_ticker_label(label) == "AAPL"


def test_normalize_keeps_original_and_audits():
    raw = five_days().assign(ticker=["<b>AAA</b>", "aaa", "AAA", " AAA", "AAA"])
    prepared = hc.prepare_raw(raw)
    assert prepared["ticker"].eq("AAA").all()
    assert prepared["ticker_original"].tolist() == raw["ticker"].tolist()
    assert hc.ticker_audit(prepared)["rows"].sum() == 5


def test_split_duplicates_exact_vs_conflict():
    raw = five_days()
    exact_copy = raw.iloc[[0]].assign(ticker="<i>aaa</i>")       # same after normalization
    nan_copy = raw.iloc[[1]].assign(adj_close=np.nan)
    conflicting = raw.iloc[[2]].assign(close=999.0)
    prepared = hc.prepare_raw(pd.concat([raw.assign(adj_close=raw["adj_close"].where(raw.index != 1)),
                                         exact_copy, nan_copy, conflicting], ignore_index=True))
    unique, exact, conflicts = hc.split_duplicates(prepared)
    assert len(exact) == 2                  # NaN adj_close copies count as identical
    assert len(conflicts) == 2              # both versions of 2024-01-04 held back
    assert set(conflicts["ts"].dt.day) == {4}
    assert len(unique) == 4 and not unique.duplicated(hc.KEY).any()


# --- missing sessions -------------------------------------------------------

def test_missing_session_detects_one_removed_date():
    full = hc.prepare_raw(five_days())
    removed = full[full["ts"] != "2024-01-04"]
    missing = hc.find_missing_sessions(removed, full[hc.KEY])
    assert missing["ts"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-04"]
    summary = hc.gap_summary(removed, missing)
    assert summary["message"].iloc[0] == \
        "Ticker AAA is missing 1 trading days between 2024-01-02 and 2024-01-08"


def test_missing_session_zero_and_holidays_not_gaps():
    full = hc.prepare_raw(five_days())      # weekend 01-06/07 absent from reference too
    missing = hc.find_missing_sessions(full, full[hc.KEY])
    assert missing.empty
    assert hc.gap_summary(full, missing)["missing_days"].tolist() == [0]


# --- outlier screen ---------------------------------------------------------

def test_three_sigma_both_tails_ddof1_and_undefined():
    r = pd.Series([np.nan] + [0.0] * 30 + [0.5, -0.5] + [np.nan, 0.01, 0.01, 0.01])
    t = pd.Series(["A"] * 33 + ["B"] * 4)
    flags = hc.three_sigma_flags(r, t)
    assert flags.loc[[31, 32], "outlier"].all()
    assert flags["outlier"].sum() == 2
    assert flags.loc[0, "screen_status"] == "undefined_return"
    assert (flags.loc[34:, "screen_status"] == "undefined_or_zero_std").all()   # B has std 0
    # ddof=1: check the threshold against a hand computation
    a = r[:33].dropna()
    assert np.isclose(flags.loc[31, "zscore"], (0.5 - a.mean()) / a.std(ddof=1))


def test_simple_returns_requires_sorted():
    df = hc.prepare_raw(five_days())
    with pytest.raises(ValueError):
        hc.simple_returns(df.iloc[::-1])
    r = hc.simple_returns(df)
    assert np.isnan(r.iloc[0]) and np.isclose(r.iloc[1], 102 / 101 - 1)


# --- quality report ---------------------------------------------------------

def test_detect_issues_schema_and_multiple_issues_per_row():
    raw = five_days()
    raw.loc[2, ["high", "open"]] = [101.0, 0.0]     # high < close, open nonpositive & below low
    raw.loc[3, "volume"] = 0
    raw.loc[4, "adj_close"] = np.nan
    report = hc.detect_issues(hc.prepare_raw(raw))
    assert list(report.columns) == hc.REPORT_COLUMNS
    assert set(report["severity"]) <= {"high", "medium", "low"}
    on_day = set(report.loc[report["issue_date"] == "2024-01-04", "issue_type"])
    assert {"high_below_close", "nonpositive_price", "open_outside_range"} <= on_day
    assert "zero_volume" in set(report["issue_type"])
    assert "missing_adj_close" in set(report["issue_type"])


def test_detect_adj_close_mismatch_against_reference():
    raw = five_days()
    ref = reference_for(raw)
    raw.loc[1, "adj_close"] *= 20                  # inflated vendor value
    report = hc.detect_issues(hc.prepare_raw(raw), factors=hc.reference_adjustment_factors(ref))
    mism = report[report["issue_type"] == "adj_close_mismatch"]
    assert mism["issue_date"].tolist() == ["2024-01-03"]


# --- repairs ----------------------------------------------------------------

def test_repair_adj_close_missing_and_mismatch_keep_vendor():
    raw = five_days()
    ref = reference_for(raw)
    raw.loc[1, "adj_close"] = np.nan
    raw.loc[2, "adj_close"] = 9999.0
    out = hc.repair_adj_close(hc.prepare_raw(raw), hc.reference_adjustment_factors(ref))
    assert np.allclose(out["adj_close"], out["close"] * 0.5)
    assert out.loc[2, "adj_close_vendor"] == 9999.0
    assert out.loc[1, "repair_flags"] == "adj_close_filled_from_reference_factor"
    assert out.loc[2, "repair_flags"] == "adj_close_replaced_with_reference_factor"


def test_repair_adj_close_skips_when_raw_close_disagrees_with_reference():
    raw = five_days()
    ref = reference_for(raw)
    raw.loc[1, ["close", "adj_close"]] = [101.5, np.nan]   # close differs from reference
    out = hc.repair_adj_close(hc.prepare_raw(raw), hc.reference_adjustment_factors(ref))
    assert np.isnan(out.loc[1, "adj_close"]) and out.loc[1, "repair_flags"] == ""


def test_block_forward_fill_keeps_bounds_and_quarantines_leading():
    raw = five_days()
    raw.loc[0, "open"] = 0.0                       # leading invalid row: cannot fill
    raw.loc[3, "close"] = 500.0                    # close above high mid-series
    out, unresolved = hc.repair_invalid_raw_rows(hc.prepare_raw(raw))
    assert unresolved.tolist() == [True, False, False, False, False]
    assert out.loc[0, "open"] == 0.0               # quarantine sees the received value
    assert out.loc[3, hc.PRICE_COLS].tolist() == out.loc[2, hc.PRICE_COLS].tolist()
    assert "ohlc_forward_filled_from_previous_row" in out.loc[3, "repair_flags"]
    assert not hc.bounds_violation(out[~unresolved]).any()


def test_interpolate_inside_only():
    df = hc.prepare_raw(five_days())
    mask = pd.Series([False, False, True, False, True], index=df.index)
    out, unresolved = hc.interpolate_confirmed_errors(df, mask)
    expected = (df.loc[1, hc.PRICE_COLS] + df.loc[3, hc.PRICE_COLS]) / 2
    assert np.allclose(out.loc[2, hc.PRICE_COLS].astype(float), expected.astype(float))
    assert unresolved.tolist() == [False, False, False, False, True]   # trailing not extrapolated
    assert out.loc[4, "close"] == df.loc[4, "close"]


def test_repair_volume_unknown_vs_reported_zero():
    raw = five_days()
    raw.loc[1, "volume"] = np.nan
    raw.loc[2, "volume"] = 0
    out = hc.repair_volume(hc.prepare_raw(raw))
    assert out["volume"].dtype == "int64" and out.loc[1, "volume"] == 0
    assert out.loc[1, "repair_flags"] == "volume_missing_filled_0"
    assert out["zero_volume_flag"].tolist() == [False, False, True, False, False]


def test_outlier_class_confirmed_vs_unmatches_reference():
    n = 200
    close = 100.0 + 0.1 * np.sin(np.arange(n))      # quiet series...
    close[20:] += 30.0                              # ...with two level shifts, so each
    close[30:] -= 60.0                              # produces exactly one extreme return
    df = make_rows([(d.strftime("%Y-%m-%d"), c, c, c, c, c, 1)
                    for d, c in zip(pd.bdate_range("2024-01-01", periods=n), close)])
    ref = reference_for(df)
    ref.loc[30, "close"] = 100.0                    # reference disagrees with day 30's close
    out = hc.add_return_outlier_flags(hc.prepare_raw(df), hc.reference_adjustment_factors(ref))
    assert out.loc[20, "return_outlier_class"] == "matches_reference"
    assert out.loc[30, "return_outlier_class"] == "differs_from_reference"
    assert out.loc[~out["return_outlier"], "return_outlier_class"].eq("").all()
    no_ref = hc.add_return_outlier_flags(hc.prepare_raw(df))
    assert no_ref.loc[no_ref["return_outlier"], "return_outlier_class"].eq("differs_from_reference").all()


# --- validation and pipeline ------------------------------------------------

def test_validate_flags_each_problem():
    df = hc.prepare_raw(five_days())
    assert hc.validate_clean(df)["passed"].all()
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    bad.loc[1, "adj_close"] = np.nan
    bad.loc[2, "low"] = -1.0
    bad.loc[3, "volume"] = -5
    failures = hc.validate_clean(bad).set_index("check")["failures"]
    assert failures["duplicate (ticker, ts) keys"] == 1
    assert failures["missing price"] == 1
    assert failures["nonpositive price"] == 1
    assert failures["missing, non-finite or negative volume"] == 1
    with pytest.raises(ValueError):
        hc.assert_valid(bad)


def test_clean_ohlc_end_to_end_accounting():
    raw = pd.concat([five_days("AAA"), five_days("BBB", 50.0)], ignore_index=True)
    ref = reference_for(raw)
    raw = pd.concat([
        raw.assign(ticker=["<b>AAA</b>"] * 5 + ["bbb"] * 5),
        raw.iloc[[0]],                                          # exact duplicate after normalizing
        make_rows([("2024-01-09",) + (np.nan,) * 5 + (10,)], "AAA"),   # all OHLC missing
    ], ignore_index=True)
    raw.loc[1, "adj_close"] = np.nan                            # repaired from reference factor
    raw.loc[5, "open"] = 0.0                                    # BBB leading row -> quarantine
    raw.loc[7, "volume"] = np.nan                               # volume unknown -> 0 + flag

    result = hc.clean_ohlc(raw, ref)
    acct = result.row_accounting()["rows"]
    assert acct.tolist() == [12, 1, 1, 1, 9]
    assert result.quarantine["quarantine_reason"].str.contains("no earlier valid row").all()
    hc.assert_valid(result.clean)
    sql = hc.to_sql_frame(result.clean, hc.CLEAN_COLUMNS)
    assert sql["ts"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert set(sql["return_outlier"].unique()) <= {0, 1}
