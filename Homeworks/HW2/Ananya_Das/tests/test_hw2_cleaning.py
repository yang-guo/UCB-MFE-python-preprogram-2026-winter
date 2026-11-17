import sqlite3

import numpy as np
import pandas as pd
import pytest

import hw2_cleaning as hc


def bars(ticker, dates, closes, *, spread=0.01, volume=1_000, adj_factor=0.5, label=None):
    """Build valid daily bars around the given closes."""
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "ts": dates,
        "open": closes,
        "high": closes * (1 + spread),
        "low": closes * (1 - spread),
        "close": closes,
        "adj_close": closes * adj_factor,
        "volume": volume,
        "ticker": label or ticker,
    })


def days(n, start="2024-01-02"):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def with_rowid(frame):
    out = frame.reset_index(drop=True)
    out.insert(0, "source_rowid", np.arange(1, len(out) + 1))
    return out


def reference_for(frame):
    ref = frame.drop(columns=["source_rowid"], errors="ignore").copy()
    ref["ticker"] = ref["ticker"].map(hc.normalize_ticker)
    return ref.drop_duplicates(subset=["ticker", "ts"])


# ---------------------------------------------------------------- tickers

@pytest.mark.parametrize("label", [
    "AAPL", " aapl ", "<a target=\"_blank\"><span style=\"color: red\">AAPL</span></a>",
    "<span style=\"font-weight: bold; color: blue\"> aapl\t</span>", "&#65;APL",
])
def test_normalize_ticker_strips_markup_whitespace_and_case(label):
    assert hc.normalize_ticker(label) == "AAPL"


@pytest.mark.parametrize("label", [None, np.nan, "", "  ", "<span></span>"])
def test_normalize_ticker_returns_na_for_empty(label):
    assert hc.normalize_ticker(label) is pd.NA


def test_normalize_tickers_keeps_raw_label_and_audits_each_variant():
    frame = pd.DataFrame({"ticker": ["AAPL", "<b>AAPL</b>", " msft", "AAPL"], "ts": ["2024-01-02"] * 4})
    out, audit = hc.normalize_tickers(frame)
    assert out["ticker"].tolist() == ["AAPL", "AAPL", "MSFT", "AAPL"]
    assert out["ticker_raw"].tolist() == frame["ticker"].tolist()
    assert audit["n_rows"].sum() == 4
    row = audit.set_index("ticker_raw").loc["<b>AAPL</b>"]
    assert row["markup_removed"] and not row["case_changed"]
    row = audit.set_index("ticker_raw").loc[" msft"]
    assert row["whitespace_stripped"] and row["case_changed"]


# ------------------------------------------------------------- duplicates

def test_exact_duplicates_across_label_variants_are_removed_once():
    a = bars("AAPL", days(2), [10, 11])
    b = bars("AAPL", days(2), [10, 11], label="<b>AAPL</b>")
    norm, _ = hc.normalize_tickers(pd.concat([a, b], ignore_index=True))
    kept, removed = hc.split_exact_duplicates(norm)
    assert len(kept) == 2 and len(removed) == 2
    assert kept["duplicate_copies_removed"].tolist() == [1, 1]


def test_rows_differing_only_by_missing_value_are_not_exact_duplicates():
    a = bars("AAPL", days(1), [10])
    b = a.copy()
    b["adj_close"] = np.nan
    kept, removed = hc.split_exact_duplicates(pd.concat([a, b], ignore_index=True))
    assert len(removed) == 0
    unique, conflicts = hc.split_conflicting_keys(kept)
    assert len(unique) == 0 and len(conflicts) == 2


def test_clean_quarantines_every_row_of_a_conflicting_key():
    good = bars("AAPL", days(3), [10, 11, 12])
    clash = good.iloc[[1]].copy()
    clash["close"] = 10.9
    raw = with_rowid(pd.concat([good, clash], ignore_index=True))
    res = hc.clean_ohlc(raw, reference_for(good))
    assert set(res.quarantine["quarantine_reason"]) == {"conflicting_duplicate_key"}
    assert len(res.quarantine) == 2
    assert days(3)[1] not in res.cleaned["ts"].tolist()


# --------------------------------------------------------- missing values

def test_missing_value_summary_counts_each_column():
    frame = bars("AAPL", days(3), [10, 11, 12])
    frame.loc[0, "adj_close"] = np.nan
    frame.loc[[0, 1], "volume"] = np.nan
    summary = hc.missing_value_summary(frame)
    assert summary.loc["adj_close", "missing"] == 1
    assert summary.loc["volume", "missing"] == 2
    assert summary.loc["close", "missing"] == 0


def test_missing_date_detector_finds_one_deliberately_removed_date():
    calendar = bars("DEMO", days(5), [1, 2, 3, 4, 5])
    delivered = calendar.drop(index=2)
    missing = hc.find_missing_dates(delivered, calendar)
    assert missing["ts"].tolist() == [days(5)[2]]
    summary = hc.missing_dates_summary(delivered, calendar)
    assert summary.loc[0, "message"] == (
        f"Ticker DEMO is missing 1 trading days between {days(5)[0]} and {days(5)[-1]}")


def test_missing_date_detector_reports_zero_and_ignores_other_tickers():
    a = bars("A", days(3), [1, 2, 3])
    b = bars("B", days(4), [1, 2, 3, 4])
    calendar = pd.concat([a, b])
    assert hc.find_missing_dates(a, calendar).empty
    assert hc.missing_dates_summary(a, calendar)["missing_days"].tolist() == [0]


def test_weekend_holidays_are_not_counted_when_absent_from_reference():
    calendar = bars("DEMO", ["2024-07-03", "2024-07-05"], [1, 2])  # July 4 closed
    assert hc.find_missing_dates(calendar, calendar).empty


# ----------------------------------------------------------------- returns

def test_simple_returns_are_within_ticker_and_sorted_by_date():
    a = bars("A", days(3), [10, 11, 12.1])
    b = bars("B", days(2), [100, 50])
    frame = pd.concat([a.iloc[::-1], b], ignore_index=True)
    ret = hc.simple_returns(frame)
    by_key = dict(zip(zip(frame["ticker"], frame["ts"]), ret))
    assert np.isnan(by_key[("A", days(3)[0])])
    assert by_key[("A", days(3)[2])] == pytest.approx(0.1)
    assert np.isnan(by_key[("B", days(2)[0])])
    assert by_key[("B", days(2)[1])] == pytest.approx(-0.5)


def test_three_sigma_flags_both_tails_with_sample_std():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 200)
    rets[50], rets[120] = 0.2, -0.2
    closes = 100 * np.cumprod(np.r_[1, 1 + rets])
    frame = bars("A", days(len(closes)), closes)
    flags = hc.flag_return_outliers(frame)
    r = flags["return"].dropna()
    expected = (r - r.mean()).abs() > 3 * r.std(ddof=1)
    assert flags.loc[expected.index, "is_outlier"].tolist() == expected.tolist()
    flagged = flags.loc[flags["is_outlier"], "return"]
    assert (flagged > 0).any() and (flagged < 0).any()


def test_zero_std_and_insufficient_returns_never_flag():
    flat = bars("FLAT", days(5), [10] * 5)
    short = bars("SHORT", days(2), [10, 20])
    frame = pd.concat([flat, short], ignore_index=True)
    stats = hc.ticker_return_stats(frame)
    assert stats.loc["FLAT", "status"] == "zero_std"
    assert stats.loc["SHORT", "status"] == "insufficient_returns"
    assert not hc.flag_return_outliers(frame)["is_outlier"].any()


def test_undefined_returns_are_excluded():
    frame = bars("A", days(4), [10, 11, 12, 13])
    frame.loc[2, "close"] = 0.0
    ret = hc.simple_returns(frame)
    assert ret.isna().tolist() == [True, False, True, True]


# ------------------------------------------------------------ price checks

def test_invalid_prices_include_adjusted_close():
    frame = bars("A", days(4), [10, 11, 12, 13])
    frame.loc[0, "open"] = 0.0
    frame.loc[1, "close"] = -1.0
    frame.loc[2, "high"] = np.inf
    frame.loc[3, "adj_close"] = 0.0
    issues = hc.detect_invalid_prices(frame)
    assert sorted(issues["issue_type"]) == ["nonfinite_price", "nonpositive_price",
                                            "nonpositive_price", "nonpositive_price"]
    assert issues["description"].str.startswith("adj_close").any()


def test_ohlc_inconsistencies_each_type_and_adj_close_ignored():
    frame = bars("A", days(5), [10, 10, 10, 10, 10])
    frame.loc[0, "high"] = 9.5          # high < close (and high < open)
    frame.loc[1, "low"] = 10.5          # low > close (and low > open)
    frame.loc[2, "open"] = 12.0         # open above high only
    frame.loc[3, ["high", "low"]] = [9.0, 11.0]
    frame.loc[4, "adj_close"] = 50.0    # adjusted close is not bound by raw range
    issues = hc.detect_ohlc_inconsistencies(frame)
    got = set(zip(issues["issue_date"], issues["issue_type"]))
    d = days(5)
    assert (d[0], "high_below_close") in got
    assert (d[1], "low_above_close") in got
    assert (d[2], "open_outside_range") in got and (d[2], "high_below_close") not in got
    assert (d[3], "high_below_low") in got
    assert d[4] not in issues["issue_date"].tolist()


def test_zero_and_invalid_volume():
    frame = bars("A", days(3), [10, 11, 12])
    frame["volume"] = [0, -5, 100]
    issues = hc.detect_volume_issues(frame)
    assert dict(zip(issues["issue_date"], issues["issue_type"])) == {
        days(3)[0]: "zero_volume", days(3)[1]: "invalid_volume"}


def test_weekend_dates_flagged():
    frame = bars("A", ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"], [1, 2, 3, 4])
    assert hc.detect_weekend_dates(frame)["issue_date"].tolist() == ["2024-01-06", "2024-01-07"]


def test_quality_report_has_required_columns_and_multiple_issues_per_row():
    frame = bars("A", days(3), [10, 11, 12])
    frame.loc[1, "open"] = 0.0
    frame.loc[1, "volume"] = 0
    report = hc.build_quality_report(frame, frame)
    assert list(report.columns) == hc.ISSUE_COLUMNS
    on_day = report.loc[report["issue_date"] == days(3)[1], "issue_type"]
    assert {"nonpositive_price", "open_outside_range", "zero_volume"} <= set(on_day)
    assert set(report["severity"]) <= {"high", "medium", "low"}


# ---------------------------------------------------------------- spikes

def spike_series():
    closes = [10.0, 10.1, 10.2, 20.4, 10.3, 10.4, 10.5]
    frame = bars("T", days(7), closes)
    frame.loc[3, ["open", "high", "low"]] = 20.4
    frame.loc[3, "volume"] = 0
    return frame


def test_isolated_spike_is_detected_with_evidence():
    spikes = hc.find_isolated_price_spikes(spike_series())
    assert spikes["ts"].tolist() == [days(7)[3]]
    assert spikes.iloc[0]["prev_close"] == 10.2 and spikes.iloc[0]["next_close"] == 10.3


@pytest.mark.parametrize("change", ["has_volume", "no_reversal", "edge", "bad_neighbour"])
def test_spike_requires_all_evidence(change):
    frame = spike_series()
    if change == "has_volume":
        frame.loc[3, "volume"] = 500
    elif change == "no_reversal":
        frame.loc[4:, ["open", "high", "low", "close"]] *= 2
    elif change == "edge":
        frame = frame.iloc[3:].reset_index(drop=True)
    elif change == "bad_neighbour":
        frame.loc[4, "low"] = 50.0
    assert hc.find_isolated_price_spikes(frame).empty


def test_clean_interpolates_spike_and_rebuilds_adj_from_neighbour_factors():
    frame = spike_series()
    frame.loc[3, "adj_close"] = 20.4 * 0.9      # spiked factor shared with reference
    raw = with_rowid(frame)
    res = hc.clean_ohlc(raw, reference_for(frame))
    row = res.cleaned.set_index("ts").loc[days(7)[3]]
    assert row["close"] == pytest.approx(10.25)
    # each field is interpolated from the same field of the neighbours
    assert row[["open", "high", "low"]].tolist() == pytest.approx([10.25, 10.25 * 1.01, 10.25 * 0.99])
    assert row["low"] <= row["close"] <= row["high"]
    assert row["price_repair_method"] == "linear_interpolation_between_neighbours"
    assert row["adj_close"] == pytest.approx(10.25 * 0.5)
    assert row["adj_close_source"] == "neighbor_factor"
    audit = res.repair_audit
    assert set(audit.loc[audit["ts"] == days(7)[3], "column"]) == {"open", "high", "low", "close", "adj_close"}
    assert audit.loc[audit["column"] == "close", "original_value"].item() == 20.4


# --------------------------------------------------------- missing prices

def test_all_missing_rows_dropped_before_filling():
    frame = bars("A", days(3), [10, 11, 12])
    frame.loc[1, ["open", "high", "low", "close"]] = np.nan
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    assert res.counts["removed_all_prices_missing"] == 1
    assert days(3)[1] not in res.cleaned["ts"].tolist()


def test_forward_fill_stays_within_ticker_and_leading_gap_is_quarantined():
    a = bars("A", days(3), [10, 10.05, 10.1])
    b = bars("B", days(3), [50, 50, 50])
    ref = reference_for(pd.concat([a, b]))
    a.loc[1, "close"] = np.nan
    b.loc[0, "close"] = np.nan                 # no earlier B price; must not borrow A's
    res = hc.clean_ohlc(with_rowid(pd.concat([a, b], ignore_index=True)), ref)
    filled = res.cleaned.set_index(["ticker", "ts"]).loc[("A", days(3)[1])]
    assert filled["close"] == 10 and filled["price_repaired"]
    assert res.quarantine["quarantine_reason"].tolist() == ["unresolved_leading_gap"]
    assert res.quarantine["ticker"].tolist() == ["B"]


def test_forward_fill_that_breaks_today_range_is_quarantined():
    frame = bars("A", days(3), [10, 20, 21])
    ref = reference_for(frame)
    frame.loc[1, "close"] = np.nan             # carrying 10 into a 19.8-20.2 range is impossible
    res = hc.clean_ohlc(with_rowid(frame), ref)
    assert res.quarantine["quarantine_reason"].tolist() == ["forward_fill_violates_bounds"]


def test_nonpositive_open_on_first_row_is_quarantined_not_replaced_by_close():
    frame = bars("A", days(3), [10, 11, 12])
    frame.loc[0, "open"] = 0.0
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    q = res.quarantine.iloc[0]
    assert q["quarantine_reason"] == "unresolved_leading_gap" and q["open"] == 0.0
    assert days(3)[0] not in res.cleaned["ts"].tolist()


def test_ambiguous_bounds_violation_is_quarantined():
    frame = bars("A", days(3), [10, 11, 12])
    frame.loc[1, "high"] = 10.0
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    assert res.quarantine["quarantine_reason"].tolist() == ["ohlc_bounds_violation"]


def test_missing_volume_filled_with_zero_and_flagged():
    frame = bars("A", days(3), [10, 11, 12])
    frame["volume"] = frame["volume"].astype(float)
    frame.loc[1, "volume"] = np.nan
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    row = res.cleaned.set_index("ts").loc[days(3)[1]]
    assert row["volume"] == 0 and row["volume_filled"] and not row["zero_volume_flag"]
    assert res.repair_audit["method"].tolist() == ["fill_zero_flagged"]


# ---------------------------------------------------------- adjusted close

def adj_series(n=15):
    return bars("A", days(n), np.linspace(10, 11.4, n), adj_factor=0.5)


def test_missing_adj_close_uses_date_matched_reference_factor():
    frame = adj_series()
    ref = reference_for(frame)
    ref.loc[7, "adj_close"] = frame.loc[7, "close"] * 0.5004   # date-specific factor
    frame.loc[7, "adj_close"] = np.nan
    res = hc.clean_ohlc(with_rowid(frame), ref)
    row = res.cleaned.set_index("ts").loc[days(15)[7]]
    assert row["adj_close"] == pytest.approx(frame.loc[7, "close"] * 0.5004)
    assert row["adj_close_source"] == "reference_factor"
    assert row["adj_close_repair_reason"] == "missing"


def test_corrupt_adj_close_detected_and_replaced_from_reference():
    frame = adj_series()
    ref = reference_for(frame)
    frame.loc[5, "adj_close"] *= 14
    res = hc.clean_ohlc(with_rowid(frame), ref)
    row = res.cleaned.set_index("ts").loc[days(15)[5]]
    assert row["adj_close"] == pytest.approx(ref.loc[5, "adj_close"])
    assert row["adj_close_repair_reason"] == "corrupted_vendor_factor"


def test_matching_adj_close_is_left_alone():
    frame = adj_series()
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    assert (res.cleaned["adj_close_source"] == "vendor").all()
    assert res.repair_audit.empty


def test_missing_adj_not_bridged_across_a_factor_change_without_reference():
    frame = adj_series()
    frame.loc[8:, "adj_close"] = frame.loc[8:, "close"] * 0.52   # dividend between rows 7 and 8
    ref = reference_for(frame).drop(index=7)                    # no reference row for the gap
    frame.loc[7, "adj_close"] = np.nan
    res = hc.clean_ohlc(with_rowid(frame), ref)
    assert res.quarantine["quarantine_reason"].tolist() == ["adj_close_unresolved"]


def test_reference_factor_that_is_itself_a_spike_is_not_trusted():
    frame = adj_series()
    ref = reference_for(frame)
    ref.loc[6, "adj_close"] *= 1.3
    frame.loc[6, "adj_close"] = np.nan
    res = hc.clean_ohlc(with_rowid(frame), ref)
    row = res.cleaned.set_index("ts").loc[days(15)[6]]
    assert row["adj_close_source"] == "neighbor_factor"
    assert row["adj_close"] == pytest.approx(frame.loc[6, "close"] * 0.5)


def test_unconfirmed_edge_anomaly_is_kept_and_flagged():
    frame = adj_series()
    frame.loc[0, "adj_close"] = frame.loc[0, "close"] * 0.4
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    row = res.cleaned.set_index("ts").loc[days(15)[0]]
    assert row["adj_factor_review_flag"] and row["adj_close_source"] == "vendor"
    assert hc.validate_cleaned(res.cleaned)["passed"].all()


# --------------------------------------------------- validation & storage

def test_row_accounting_and_validation_on_mixed_delivery():
    a = adj_series()
    dup = a.iloc[[2, 3]].assign(ticker="<b>A</b>")
    frame = pd.concat([a, dup], ignore_index=True)
    frame.loc[0, "open"] = 0.0
    res = hc.clean_ohlc(with_rowid(frame), reference_for(a))
    c = res.counts
    assert c["input_rows"] == c["removed_exact_duplicates"] + c["removed_all_prices_missing"] \
        + c["quarantined"] + c["retained"]
    assert c["removed_exact_duplicates"] == 2 and c["quarantined"] == 1
    assert hc.validate_cleaned(res.cleaned)["passed"].all()


def test_validate_cleaned_catches_each_failure():
    frame = adj_series()
    frame["adj_factor_review_flag"] = False
    broken = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    broken.loc[1, "close"] = np.nan
    broken.loc[2, "high"] = broken.loc[2, "low"] - 1
    broken.loc[3, "ts"] = "01/05/2024"
    broken.loc[4, "volume"] = -1
    result = hc.validate_cleaned(broken).set_index("check")["passed"]
    for check in ["no missing prices", "unique (ticker, ts)", "low <= high",
                  "ts stored as ISO YYYY-MM-DD", "volume present, finite, >= 0"]:
        assert not result[check], check


def test_write_tables_refuses_source_tables(tmp_path):
    db = tmp_path / "t.db"
    sqlite3.connect(db).close()
    with pytest.raises(PermissionError):
        hc.write_tables(db, {"ohlc_hw": pd.DataFrame({"x": [1]})})
    hc.write_tables(db, {"ohlc_hw_cleaned": pd.DataFrame({"x": [1]})})
    assert hc.table_counts(db)["ohlc_hw_cleaned"] == 1


def test_readonly_connection_cannot_write(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE ohlc_hw (x)")
    with hc.connect_readonly(db) as con, pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO ohlc_hw VALUES (1)")


# ------------------------------------------------- regression: audit fixes

def test_duplicate_count_includes_copies_with_missing_adj_close():
    frame = bars("A", days(3), [10, 11, 12])
    frame.loc[1, "adj_close"] = np.nan
    copies = frame.iloc[[1, 1, 2]].assign(ticker="<b>A</b>")      # two NaN-adj copies, one normal
    raw = with_rowid(pd.concat([frame, copies], ignore_index=True))
    norm, _ = hc.normalize_tickers(raw)
    kept, removed = hc.split_exact_duplicates(norm)
    counts = kept.set_index("ts")["duplicate_copies_removed"]
    assert counts.to_dict() == {days(3)[0]: 0, days(3)[1]: 2, days(3)[2]: 1}
    assert counts.sum() == len(removed) == 3
    assert removed["duplicate_of_source_rowid"].tolist() == [2, 2, 3]


def test_clean_reconciles_every_removed_copy_with_a_kept_or_quarantined_row():
    frame = adj_series()
    frame.loc[4, "adj_close"] = np.nan
    frame.loc[0, "open"] = 0.0                                    # quarantined, and it has a copy
    copies = frame.iloc[[0, 4, 4, 9]]
    res = hc.clean_ohlc(with_rowid(pd.concat([frame, copies], ignore_index=True)), reference_for(adj_series()))
    assert res.counts["removed_exact_duplicates"] == 4
    assert res.cleaned["duplicate_copies_removed"].sum() + res.quarantine["duplicate_copies_removed"].sum() == 4
    assert res.quarantine["duplicate_copies_removed"].tolist() == [1]
    dupes = res.removed[res.removed["removal_reason"] == "exact_duplicate"]
    kept_ids = set(res.cleaned["source_rowid"]) | set(res.quarantine["source_rowid"])
    assert set(dupes["duplicate_of_source_rowid"]) <= kept_ids


def test_quarantine_preserves_unparseable_timestamp():
    frame = bars("A", days(3), [10, 11, 12])
    frame.loc[1, "ts"] = "31/02/2024 ??"
    res = hc.clean_ohlc(with_rowid(frame), reference_for(bars("A", days(3), [10, 11, 12])))
    q = res.quarantine.iloc[0]
    assert q["quarantine_reason"] == "invalid_date"
    assert q["ts_raw"] == "31/02/2024 ??" and pd.isna(q["ts"])
    assert q["volume"] == 1_000 and q["source_rowid"] == 2


def test_quarantine_and_removed_rows_keep_delivered_values():
    frame = bars("A", days(3), [10, 11, 12], label=" <i>a</i> ")
    frame.loc[0, "open"] = 0.0
    frame.loc[2, ["open", "high", "low", "close"]] = np.nan
    res = hc.clean_ohlc(with_rowid(frame), reference_for(bars("A", days(3), [10, 11, 12])))
    q, r = res.quarantine.iloc[0], res.removed.iloc[0]
    assert (q["ticker_raw"], q["ts_raw"], q["open"]) == (" <i>a</i> ", days(3)[0], 0.0)
    assert r["removal_reason"] == "all_raw_prices_missing" and pd.isna(r["duplicate_of_source_rowid"])
    assert list(res.quarantine.columns) == hc.QUARANTINE_COLUMNS
    assert list(res.removed.columns) == hc.REMOVED_COLUMNS


def test_interpolation_changes_only_masked_rows():
    frame = bars("A", days(6), [10, 11, 30, 12, 13, 14])
    frame.loc[4, "close"] = np.nan                                # an ordinary gap, not a confirmed error
    out, ok = hc.interpolate_confirmed_errors(frame, frame.index == 2)
    assert out.loc[2, "close"] == pytest.approx(11.5)             # midpoint of the neighbouring closes
    assert out.loc[2, "high"] == pytest.approx(11.5 * 1.01)
    assert np.isnan(out.loc[4, "close"])                          # left for the forward-fill step
    pd.testing.assert_frame_equal(out.drop(index=2), frame.drop(index=2))
    assert ok.tolist() == [False, False, True, False, False, False]


def test_interpolation_result_failing_bounds_is_reported():
    frame = bars("A", days(3), [10, 99, 10], spread=0.01)
    frame.loc[0, "open"] = 10.5                                   # inconsistent neighbour: open > high
    out, ok = hc.interpolate_confirmed_errors(frame, frame.index == 1)
    assert out.loc[1, "open"] > out.loc[1, "high"]
    assert not ok[1]


def test_clean_quarantines_spike_whose_interpolation_fails(monkeypatch):
    frame = spike_series()

    def edge_spike(f, **_):                                       # pretend the first row is a confirmed error
        first = hc._sorted(f).iloc[[0]].copy()
        return first.assign(prev_ts="-", next_ts=days(7)[1], prev_close=np.nan, next_close=10.1)

    monkeypatch.setattr(hc, "find_isolated_price_spikes", edge_spike)
    res = hc.clean_ohlc(with_rowid(frame), reference_for(frame))
    assert res.quarantine["quarantine_reason"].tolist() == ["interpolation_failed_validation"]
    assert res.quarantine.iloc[0]["close"] == 10.0                # original values kept
    assert not res.repair_audit["method"].eq("linear_interpolation_between_neighbours").any()


def test_spike_repair_does_not_interpolate_other_gaps():
    frame = spike_series()
    ref = reference_for(frame)
    frame.loc[5, "close"] = np.nan                                # ordinary missing close later in the series
    res = hc.clean_ohlc(with_rowid(frame), ref)
    audit = res.repair_audit.set_index(["ts", "column"])
    assert audit.loc[(days(7)[3], "close"), "method"] == "linear_interpolation_between_neighbours"
    assert audit.loc[(days(7)[5], "close"), "method"] == "forward_fill_within_ticker"
    assert res.cleaned.set_index("ts").loc[days(7)[5], "close"] == 10.3


def test_missing_dates_distinguish_withheld_from_source_gaps():
    calendar = bars("DEMO", days(5), [1, 2, 3, 4, 5])
    cleaned = calendar.drop(index=[1, 3])
    withheld = pd.DataFrame({"ticker": ["DEMO"], "ts": [days(5)[1]], "reason": ["unresolved_leading_gap"]})
    missing = hc.find_missing_dates(cleaned, calendar, withheld)
    assert dict(zip(missing["ts"], missing["cause"])) == {
        days(5)[1]: hc.WITHHELD_BY_CLEANING, days(5)[3]: hc.ABSENT_FROM_SOURCE}
    issues = hc.detect_missing_dates(cleaned, calendar, withheld).set_index("issue_date")["description"]
    assert "withheld by cleaning (unresolved_leading_gap)" in issues[days(5)[1]]
    assert "absent from the source" in issues[days(5)[3]]
    summary = hc.missing_dates_summary(cleaned, calendar, withheld).iloc[0]
    assert (summary["missing_days"], summary["absent_from_source"], summary["withheld_by_cleaning"]) == (2, 1, 1)
    assert summary["message"].endswith("(1 absent from source, 1 withheld by cleaning)")


def test_cleaning_result_lists_withheld_keys():
    frame = adj_series()
    frame.loc[0, "open"] = 0.0
    frame.loc[5, ["open", "high", "low", "close"]] = np.nan
    res = hc.clean_ohlc(with_rowid(pd.concat([frame, frame.iloc[[9]]], ignore_index=True)),
                        reference_for(adj_series()))
    withheld = res.withheld()
    assert set(zip(withheld["ts"], withheld["reason"])) == {
        (days(15)[0], "unresolved_leading_gap"), (days(15)[5], "all_raw_prices_missing")}
    missing = hc.find_missing_dates(res.cleaned, reference_for(adj_series()), withheld)
    assert (missing["cause"] == hc.WITHHELD_BY_CLEANING).all() and len(missing) == 2


# ------------------------------------------------- 3-sigma investigation

def noisy_series(ticker, n=150, seed=0, volume=1_000):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    return bars(ticker, days(n), closes, volume=volume)


def category_on(inv, ticker, ts):
    return inv.set_index(["ticker", "ts"]).loc[(ticker, ts), "category"]


def test_investigation_covers_exactly_the_flagged_rows_and_changes_nothing():
    frame = noisy_series("A")
    frame.loc[60:, ["open", "high", "low", "close", "adj_close"]] *= 1.12      # persistent jump
    before = frame.copy()
    inv = hc.investigate_return_outliers(frame)
    pd.testing.assert_frame_equal(frame, before)
    flagged = hc.flag_return_outliers(frame)
    assert set(zip(inv["ticker"], inv["ts"])) == set(zip(flagged.loc[flagged["is_outlier"], "ticker"],
                                                         flagged.loc[flagged["is_outlier"], "ts"]))
    assert category_on(inv, "A", days(150)[60]) == "persistent_move"
    assert set(inv["assessment"]) <= {"likely_data_error", "plausible_market_event", "unresolved"}


def test_low_volume_one_day_reversal_and_its_return_leg_are_likely_errors():
    frame = noisy_series("A")
    frame.loc[80, ["open", "high", "low", "close", "adj_close"]] *= [0.88, 0.88, 0.88, 0.88, 0.88]
    frame.loc[80, "volume"] = 0
    inv = hc.investigate_return_outliers(frame)
    assert category_on(inv, "A", days(150)[80]) == "low_volume_one_day_reversal"
    assert category_on(inv, "A", days(150)[81]) == "reversal_of_likely_error"
    assert (inv.loc[inv["category"].str.contains("error|reversal_of"), "assessment"] == "likely_data_error").all()


def test_same_reversal_on_normal_volume_is_not_called_an_error():
    frame = noisy_series("A")
    frame.loc[80, ["open", "high", "low", "close", "adj_close"]] *= 0.88
    inv = hc.investigate_return_outliers(frame)
    assert category_on(inv, "A", days(150)[80]) == "one_day_reversal"
    assert inv.set_index("ts").loc[days(150)[80], "assessment"] == "unresolved"


def test_market_wide_cluster_and_large_level_shift():
    frames = []
    for i, tk in enumerate(["A", "B", "C", "D", "E"]):
        f = noisy_series(tk, seed=i)
        f.loc[100:, ["open", "high", "low", "close", "adj_close"]] *= 0.85         # same-day drop everywhere
        frames.append(f)
    big = noisy_series("F", seed=9)
    big.loc[50:, ["open", "high", "low", "close", "adj_close"]] *= 0.4              # -60% and it persists
    inv = hc.investigate_return_outliers(pd.concat(frames + [big], ignore_index=True))
    assert all(category_on(inv, tk, days(150)[100]) == "market_wide_move" for tk in "ABCDE")
    assert category_on(inv, "F", days(150)[50]) == "large_level_shift"
    assert inv.set_index(["ticker", "ts"]).loc[("F", days(150)[50]), "assessment"] == "unresolved"


def test_raw_move_offset_by_adjustment_factor_change_is_unresolved():
    frame = noisy_series("A")
    frame.loc[70:, ["open", "high", "low", "close"]] *= 0.9                        # raw drops 10%...
    inv = hc.investigate_return_outliers(frame)                                    # ...adj_close does not
    row = inv.set_index("ts").loc[days(150)[70]]
    assert row["category"] == "adjustment_factor_change" and row["factor_change"] == pytest.approx(1 / 0.9 - 1)


def test_flat_bar_is_reported_but_does_not_change_the_category():
    frame = noisy_series("A")
    frame.loc[80, ["open", "high", "low", "close", "adj_close"]] *= 0.88    # reversal on normal volume
    flat = frame.copy()
    flat.loc[80, ["open", "high", "low"]] = flat.loc[80, "close"]           # same day recorded as a flat bar
    ranged = hc.investigate_return_outliers(frame).set_index("ts").loc[days(150)[80]]
    flattened = hc.investigate_return_outliers(flat).set_index("ts").loc[days(150)[80]]
    assert not ranged["flat_bar"] and flattened["flat_bar"]
    assert ranged["category"] == flattened["category"] == "one_day_reversal"
