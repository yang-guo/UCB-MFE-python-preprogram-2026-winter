"""Independently verify the completed, frozen-strategy HW4 accounting replay.

This audit imports no backtester, research, data-pipeline, or accounting-overlay
helpers. It uses the unmodified official DuckDB, the original frozen forecasts,
the dated corporate-action registry, source quote supplements, and CSV exports.
It independently rebuilds eligible monthly labels, the 42 training cutoffs,
every order and lot, daily cash/NAV, and the strategy and benchmark metrics.

Run from the submission directory::

    python audit_hw4.py --database PATH \\
        --output outputs_replay

Only a completed run can pass. FULL_ACCOUNTING_AUDIT.json is written inside
--output after every assertion succeeds; an incomplete run never receives a
successful full-period audit. External legal terms are documented in the source
registry. This script checks their dated numerical application and quote
lineage, not the legal authenticity of a webpage or the accuracy of its vendor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATABASE_SHA256 = "a028641d96c12e6f29d5dd34e3a6ebc059074dd38351648bf0d970b1dd4095b9"
PREDICTIONS_SHA256 = "fdb6fc0c8a176b72b00d8125e4350f98a32c1afd77f53b96929c96f70295ac82"
PROTOCOL_SHA256 = "e9c65fbe569e59d2d33904c7b7dc2b3bde76eb1f9c5f63bf97416abd0139b21f"
ORIGINAL_EVIDENCE_SHA256 = {
    "fit_audit.csv": "70d2747ee68c3373ebe48a735482045ae55cba8fd716944e612b9c50769276e7",
    "validation.csv": "ee303f63447396f67762a4d807e7e300ef596f4849d21058beeba28baef88601",
}
START = pd.Timestamp("2016-04-01")
END = pd.Timestamp("2026-09-30")
ANCHOR = pd.Timestamp("2016-03-31")
EXPECTED_SESSIONS = 2640

# Eligibility is reconstructed independently from the calendar, the quarterly
# membership record known at the signal close, and 64 consecutive valid prices.
# An unavailable entry/exit label remains an eligible forecast candidate.
LABEL_SQL = """
WITH cal AS (
    SELECT date, row_number() OVER (ORDER BY date) AS session_id,
           lag(date) OVER (ORDER BY date) AS signal_date
    FROM calendar WHERE date <= '2026-09-30'
), monthly AS (
    SELECT date AS entry_date, signal_date,
           row_number() OVER (
               PARTITION BY date_trunc('month', date) ORDER BY date
           ) AS month_position
    FROM cal
), schedule AS (
    SELECT m.entry_date, m.signal_date,
           (SELECT min(date) FROM cal
            WHERE date >= m.entry_date + INTERVAL '90 days') AS label_end
    FROM monthly m
    WHERE month_position = 1
      AND ((entry_date BETWEEN '2010-01-01' AND '2015-12-31')
           OR (entry_date BETWEEN '2016-04-01' AND '2026-09-30'))
), valid_px AS (
    SELECT p.symbol,p.date,p.adj_close,c.session_id,
           lag(c.session_id, 63) OVER (
               PARTITION BY p.symbol ORDER BY p.date
           ) AS old_session
    FROM prices p JOIN cal c USING(date)
    WHERE p.date >= '2009-01-01'
      AND isfinite(p.adj_close) AND p.adj_close > 0
)
SELECT s.signal_date,s.entry_date,s.label_end,p.symbol,
       ep.adj_close AS entry_price,xp.adj_close AS exit_price,
       xp.adj_close / ep.adj_close - 1 AS target
FROM schedule s
JOIN valid_px p ON p.date = s.signal_date
              AND p.session_id - p.old_session = 63
JOIN universe_membership u
  ON u.symbol = p.symbol
 AND u.quarter_start = date_trunc('quarter',s.entry_date)
 AND u.selection_date <= s.signal_date
LEFT JOIN valid_px ep ON ep.symbol=p.symbol AND ep.date=s.entry_date
LEFT JOIN valid_px xp ON xp.symbol=p.symbol AND xp.date=s.label_end
ORDER BY s.entry_date,p.symbol
"""


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value) if isinstance(value, Path) else value


class Checks:
    """Track numeric discrepancies while failing at the first material error."""

    def __init__(self):
        self.maximum_errors = {}
        self.checked_numbers = 0

    def close(self, name, actual, expected, *, atol=1e-7, rtol=1e-10):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        require(actual.shape == expected.shape, f"{name}: different shapes")
        equal_nan = np.isnan(actual) & np.isnan(expected)
        good = equal_nan | (np.isfinite(actual) & np.isfinite(expected)
                            & np.isclose(actual, expected, atol=atol, rtol=rtol))
        if not np.all(good):
            index = np.flatnonzero(~good.reshape(-1))[0]
            raise AssertionError(
                f"{name}: got {actual.reshape(-1)[index]!r}, "
                f"independent value {expected.reshape(-1)[index]!r}"
            )
        finite = np.isfinite(actual) & np.isfinite(expected)
        error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0
        self.maximum_errors[name] = max(error, self.maximum_errors.get(name, 0.0))
        self.checked_numbers += int(actual.size)


def read_frame(path, dates=()):
    frame = pd.read_csv(path)
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def compare_frames(checks, name, published, rebuilt, keys, dates=(), numeric=()):
    """Compare every reconstructed field, including blank event metadata."""
    require(len(published) == len(rebuilt), f"{name}: row count differs")
    require(not published.duplicated(keys).any(), f"{name}: duplicate record keys")
    require(not rebuilt.duplicated(keys).any(), f"{name}: independent duplicate keys")
    require(set(rebuilt.columns).issubset(published.columns), f"{name}: missing exported columns")
    published = published.sort_values(keys).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    for column in rebuilt.columns:
        a, b = published[column], rebuilt[column]
        if column in numeric:
            checks.close(f"{name}.{column}", a, b, atol=1e-5 if column in {
                "cash", "nav", "holdings_value", "stock_value", "cash_claim_value",
                "cash_change", "gross_value", "realized_pnl", "unrealized_pnl",
                "market_value", "entry_budget", "budget", "signal_cash", "signal_nav",
            } else 1e-8)
        elif column in dates:
            a, b = pd.to_datetime(a), pd.to_datetime(b)
            require(bool((a.eq(b) | (a.isna() & b.isna())).all()),
                    f"{name}.{column}: date mismatch")
        else:
            require(a.fillna("").astype(str).equals(b.fillna("").astype(str)),
                    f"{name}.{column}: value mismatch")


def check_model_evidence(connection, scores, output, evidence_dir, checks):
    """Rebuild training eligibility/outcome clocks; never refit or tune a model."""
    evidence_hashes = {}
    for name, expected_hash in ORIGINAL_EVIDENCE_SHA256.items():
        original = evidence_dir / name
        require(digest(original) == expected_hash, f"Original {name} changed")
        current = output / name
        require(current.is_file(), f"Missing model evidence export: {current}")
        require(digest(current) == expected_hash, f"Replay {name} differs from the original")
        evidence_hashes[name] = expected_hash
    fits = read_frame(evidence_dir / "fit_audit.csv", (
        "fit_date", "first_training_signal", "last_training_signal", "last_training_label_end",
    ))
    validation = read_frame(evidence_dir / "validation.csv", (
        "fit_date", "last_training_label_end", "last_validation_label_end",
    ))
    labels = connection.execute(LABEL_SQL).fetchdf()
    for col in ("signal_date", "entry_date", "label_end"):
        labels[col] = pd.to_datetime(labels[col]).astype("datetime64[ns]")
    require(not labels.duplicated(["entry_date", "symbol"]).any(), "Duplicate independent labels")
    require(not scores.duplicated(["entry_date", "symbol"]).any(), "Duplicate frozen forecasts")
    final = labels.loc[labels.entry_date.ge(START)].copy()
    key_cols = ["signal_date", "entry_date", "symbol"]
    a = scores[key_cols].sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    b = final[key_cols].sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    require(a.equals(b), "Frozen forecast universe differs from independently eligible quarterly members")
    require(len(scores) == 249796 and scores.entry_date.nunique() == 126,
            "Unexpected original forecast coverage")
    require(np.isfinite(scores.score).all(), "Non-finite frozen prediction")
    require(scores.fit_date.le(scores.signal_date).all(), "Prediction precedes its fitted model")
    require(len(fits) == 42 and fits.entry_quarter.is_unique, "Need the 42 original quarterly fits")
    finite = np.isfinite(labels.target)
    quarters = scores.entry_date.dt.to_period("Q").astype(str)
    for row in fits.itertuples():
        group = scores.loc[quarters.eq(row.entry_quarter)]
        require(len(group) > 0, f"No scores for audited quarter {row.entry_quarter}")
        asof = group.signal_date.min()
        require(asof == row.fit_date and group.fit_date.eq(asof).all(),
                f"{row.entry_quarter}: model fit is not the first quarterly signal")
        train = labels.loc[finite & labels.signal_date.lt(asof) & labels.label_end.lt(asof)]
        earlier = labels.loc[labels.signal_date.lt(asof)]
        require(len(train) == row.training_rows, f"{row.entry_quarter}: training count differs")
        require(train.signal_date.min() == row.first_training_signal
                and train.signal_date.max() == row.last_training_signal
                and train.label_end.max() == row.last_training_label_end,
                f"{row.entry_quarter}: independently reconstructed training dates differ")
        require(row.last_training_label_end < asof and row.last_training_signal < asof,
                f"{row.entry_quarter}: unmatured outcome or contemporaneous feature in training")
        require(len(earlier) - len(train) == row.earlier_rows_not_yet_usable,
                f"{row.entry_quarter}: purged/censored training count differs")
        require(row.alpha == 1000.0 and row.scored_rows == len(group),
                f"{row.entry_quarter}: model configuration or score count changed")
    for year in (2013, 2014, 2015):
        candidates = labels.loc[labels.entry_date.dt.year.eq(year)]
        asof = candidates.signal_date.min()
        train = labels.loc[finite & labels.signal_date.lt(asof) & labels.label_end.lt(asof)]
        valid = candidates.loc[np.isfinite(candidates.target)
                               & candidates.label_end.le("2015-12-31")]
        fold = validation.loc[validation.validation_year.eq(year)]
        require(len(fold) == 4 and (fold.model.eq("training_mean").sum() == 1),
                f"Incomplete fixed validation grid for {year}")
        require(set(fold.loc[fold.model.eq("ridge"), "alpha"]) == {10., 100., 1000.},
                f"Changed Ridge candidates for {year}")
        require(fold.fit_date.eq(asof).all()
                and fold.last_training_label_end.eq(train.label_end.max()).all()
                and fold.last_validation_label_end.eq(valid.label_end.max()).all(),
                f"{year}: validation timing differs from independent labels")
        for column, expected in {
            "training_rows": len(train), "eligible_validation_rows": len(candidates),
            "evaluated_validation_rows": len(valid),
            "unevaluated_validation_rows": len(candidates) - len(valid),
        }.items():
            require(fold[column].eq(expected).all(), f"{year}: {column} differs")
        dummy_rmse = np.sqrt(np.mean((valid.target.to_numpy() - train.target.mean()) ** 2))
        checks.close(f"validation.training_mean_rmse_{year}",
                     fold.loc[fold.model.eq("training_mean"), "rmse"].iloc[0], dummy_rmse,
                     atol=1e-10)
    require(validation.last_validation_label_end.le("2015-12-31").all(),
            "Validation evaluated an outcome beyond development")
    ranking = validation.loc[validation.model.eq("ridge")].groupby("alpha").rmse.mean()
    best = sorted(ranking.items(), key=lambda item: (item[1], item[0]))[0][0]
    require(best == 1000., "Frozen alpha is not the original equal-year RMSE selection")
    return {
        "original_evidence_sha256": evidence_hashes,
        "independently_rebuilt_eligible_monthly_labels": len(labels),
        "forecast_rows": len(scores), "monthly_forecast_dates": int(scores.entry_date.nunique()),
        "quarterly_fits": len(fits), "selected_alpha": best,
        "first_fit_date": fits.fit_date.min(), "last_fit_date": fits.fit_date.max(),
        "final_eligible_rows_with_censored_or_missing_targets": int((~np.isfinite(final.target)).sum()),
        "all_eligible_final_rows_predicted_without_outcome_filter": True,
        "all_training_labels_end_strictly_before_fit": True,
        "buffer_entry_rows_in_training": 0,
        "training_count_range": [int(fits.training_rows.min()), int(fits.training_rows.max())],
        "independent_training_mean_validation_rmse_verified": True,
        "validation_mean_rmse_by_alpha": {str(k): float(v) for k, v in ranking.items()},
        "model_audit_scope": (
            "Exact frozen forecast bytes and original validation/fit exports; independent "
            "monthly eligibility, membership, label availability, training counts and cutoffs. "
            "The audit does not refit Ridge or claim to independently re-create its coefficients."
        ),
    }


def first_present(record, names, required=True):
    for name in names:
        if name in record and pd.notna(record[name]):
            return record[name]
    if required:
        raise AssertionError(f"Source evidence missing one of {names}: {record.get('action_id', '')}")
    return None


def archived_quote_evidence(supplement_path, connection, checks):
    """Read archived observations independently of the supplement builder."""
    if not supplement_path.is_file():
        return {}, {"archived_source_files": {}, "observations_rechecked": 0}
    data = read_frame(supplement_path, ["date"])
    cache_dir = ROOT / "verification/market_data_sources"
    sources, hashes, overlaps = {}, {}, []
    for evidence_file, group in data.groupby("evidence_file"):
        path = cache_dir / evidence_file
        require(path.is_file() and path.resolve().parent == cache_dir.resolve(),
                f"Archived quote evidence is missing or outside the evidence directory: {evidence_file}")
        source_hash = digest(path)
        require(group.source_sha256.eq(source_hash).all(), f"Archived quote hash differs: {evidence_file}")
        hashes[evidence_file] = source_hash
        document = json.loads(path.read_text())
        require(group.source_symbol.nunique() == 1, "An archive is assigned several source symbols")
        source_symbol = group.source_symbol.iloc[0]
        if "chart" in document:
            result = document["chart"]["result"][0]
            require(result["meta"]["symbol"] == source_symbol, "Provider response symbol differs")
            days = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(
                "America/New_York").normalize().tz_localize(None)
            quote = result["indicators"]["quote"][0]
            adjusted = result["indicators"]["adjclose"][0]["adjclose"]
            require(len(days) == len(quote["close"]) == len(adjusted), "Provider arrays differ in length")
            require(days.is_unique, "Provider has duplicate dated observations")
            observed_rows = [
                {"date": day, "close": close, "adj_close": adj, "volume": volume}
                for day, close, adj, volume in zip(days, quote["close"], adjusted, quote["volume"])
            ]
            require(not result.get("events"), "Supplement source contains an unaccounted split/dividend event")
        else:
            require(document.get("source_symbol") == source_symbol
                    and group.source_url.eq(document["source_url"]).all(), "Issuer evidence identity differs")
            observed_rows = []
            for row in document["observations"]:
                day = pd.Timestamp(row["date"])
                require(row["closing_price_text"].endswith(f"Closing Price ${float(row['close']):.2f}"),
                        f"Issuer visible closing text differs: {day}")
                controls = row["selected_controls"]
                for expected in (f"year, Value: {day.year}", f"month, Value: {day.strftime('%b')}",
                                 f"day, Value: {day.day}"):
                    require(expected in controls, f"Issuer selected date differs: {day}")
                observed_rows.append({"date": day, "close": row["close"],
                                      "adj_close": row["close"], "volume": row["volume"]})
        for row in observed_rows:
            key = row["date"], source_symbol
            require(key not in sources, f"Duplicate archived source observation: {key}")
            require(row["close"] is not None and np.isfinite(row["close"]) and row["close"] > 0,
                    f"Invalid archived source observation: {key}")
            sources[key] = row
        for row in group.itertuples():
            key = pd.Timestamp(row.source_date), row.source_symbol
            require(row.date == key[0] and key in sources, "Supplement lacks its same-day archive observation")
            source = sources[key]
            checks.close("archive.source_close", row.source_close, source["close"], atol=1e-10)
            checks.close("archive.source_adj_close", row.source_adj_close, source["adj_close"], atol=1e-10)
            checks.close("archive.source_volume", row.source_volume, source["volume"], atol=0., rtol=0.)
        for symbol, subset in group.groupby("symbol"):
            require(subset.price_multiplier.nunique() == 1, "Changing supplemental multiplier inside a gap")
            factor = subset.price_multiplier.iloc[0]
            first_missing = subset.date.min()
            source_days = sorted(day for day, ticker in sources if ticker == source_symbol and day < first_missing)
            require(source_days, f"No pre-gap observations establish supplement units: {symbol}")
            pairs = connection.execute(
                "SELECT date,close,adj_close FROM prices WHERE symbol=? AND date BETWEEN ? AND ? ORDER BY date",
                [symbol, source_days[0].date(), source_days[-1].date()],
            ).fetchdf()
            count, disagreements = 0, []
            for row in pairs.itertuples():
                key = row.date, source_symbol
                if key not in sources or not np.isfinite(row.adj_close) or row.adj_close <= 0:
                    continue
                predicted_price = sources[key]["close"] * factor
                if symbol == "WOLF":
                    # Document the known source disagreement; do not invent a
                    # scale from it or rewrite the original source observation.
                    checks.close("archive.wolf_course_raw_equals_adjusted", row.close, row.adj_close, atol=0.)
                    if not np.isclose(row.adj_close, predicted_price, rtol=1e-10, atol=1e-10):
                        disagreements.append({"date": row.date, "course_adj_close": row.adj_close,
                                              "issuer_close": sources[key]["close"]})
                    else:
                        count += 1
                else:
                    checks.close(f"archive.{symbol}_pre_gap_overlap", row.adj_close,
                                 predicted_price, atol=1e-7, rtol=1e-6)
                    count += 1
            require(count >= (2 if symbol == "WOLF" else 3), f"Insufficient checked unit overlaps: {symbol}")
            if symbol == "WOLF":
                require(float(factor) == 1. and len(disagreements) == 1
                        and disagreements[0]["date"] == pd.Timestamp("2024-10-14"),
                        "WOLF source/basis conflict differs from its documented exception")
            overlaps.append({"symbol": symbol, "source_symbol": source_symbol,
                             "checked_matching_pre_gap_observations": count,
                             "price_multiplier": factor, "documented_source_disagreements": disagreements})
    return sources, {"archived_source_files": hashes, "observations_rechecked": len(data),
                     "independent_pre_gap_basis_checks": overlaps}


def validate_registry(registry, connection, checks, external_quotes=None):
    """Check all declared adjusted-unit transformations against original pairs."""
    events = registry.get("events", [])
    external_quotes = external_quotes or {}
    require(len({e["action_id"] for e in events}) == len(events), "Duplicate corporate action ID")
    verified = []

    def original_pair(symbol, date_value, allow_external=False):
        rows = connection.execute(
            "SELECT close,adj_close FROM prices WHERE symbol=? AND date=?",
            [symbol, pd.Timestamp(date_value).date()],
        ).fetchall()
        if not rows and allow_external:
            key = pd.Timestamp(date_value), symbol
            require(key in external_quotes, f"Missing archived alias basis observation: {symbol} {date_value}")
            item = external_quotes[key]
            return float(item["close"]), float(item["adj_close"])
        require(len(rows) == 1, f"Missing/duplicate original basis pair: {symbol} {date_value}")
        raw, adjusted = map(float, rows[0])
        require(np.isfinite([raw, adjusted]).all() and min(raw, adjusted) > 0,
                f"Invalid original adjustment reference: {symbol} {date_value}")
        return raw, adjusted

    for event in events:
        key, symbol = event["action_id"], event["symbol"]
        require(str(event.get("source_url", "")).startswith(("https://", "http://")),
                f"{key}: missing public source URL")
        effective = pd.Timestamp(event["effective_date"])
        require(effective <= END, f"{key}: event after study end")
        kind = event.get("event_type")
        reference = first_present(event, ["reference_price_date", "basis_date", "reference_date"])
        raw, adj = original_pair(symbol, reference)
        for column, expected in [("reference_raw_close", raw), ("old_close", raw),
                                 ("reference_adj_close", adj), ("old_adj_close", adj)]:
            if column in event:
                checks.close(f"registry.{column}", event[column], expected, atol=1e-10)
        if "cash_per_adjusted_unit" in event:
            require(pd.Timestamp(reference) < effective,
                    f"{key}: cash-unit reference must precede the event")
            raw_amount = first_present(event, ["cash_per_raw_share", "cash_per_raw_unit",
                                               "cash_per_actual_warrant", "cash_per_actual_note_calculated"])
            require(np.isfinite(raw_amount) and raw_amount >= 0, f"{key}: invalid sourced cash amount")
            checks.close("registry.cash_per_adjusted_unit", event["cash_per_adjusted_unit"],
                         float(raw_amount) * adj / raw, atol=1e-10)
            if "raw_shares_per_adjusted_unit" in event:
                checks.close("registry.raw_shares_per_adjusted_unit",
                             event["raw_shares_per_adjusted_unit"], adj / raw, atol=1e-12)
            if all(k in event for k in ("principal_per_note", "annual_coupon", "accrual_days")):
                amount = event["principal_per_note"] * (
                    1 + event["annual_coupon"] * event["accrual_days"] / 360)
                checks.close("registry.note_principal_and_30_360_interest", raw_amount, amount, atol=1e-10)
        elif kind in {"ticker_alias", "security_conversion"}:
            successor = event["successor_symbol"]
            successor_date = first_present(event, ["successor_reference_price_date", "reference_successor_date"], False)
            successor_date = reference if successor_date is None else successor_date
            new_raw, new_adj = original_pair(successor, successor_date, allow_external=(kind == "ticker_alias"))
            for column, expected in [
                ("reference_successor_raw_close", new_raw), ("successor_reference_raw_close", new_raw),
                ("new_close", new_raw), ("reference_successor_adj_close", new_adj),
                ("successor_reference_adj_close", new_adj), ("new_adj_close", new_adj),
            ]:
                if column in event and event[column] is not None:
                    checks.close(f"registry.{column}", event[column], expected, atol=1e-10)
            if kind == "ticker_alias":
                require(pd.Timestamp(reference) == pd.Timestamp(successor_date),
                        f"{key}: alias requires simultaneous reference observations")
                start = pd.Timestamp(event.get("overlay_start", event["effective_date"]))
                require(pd.Timestamp(reference) < start and effective <= start,
                        f"{key}: alias splice uses a future reference or precedes the rename")
                # Exact 1:1 symbol changes can have provider float32 rounding;
                # independent pre-gap observations also check this convention.
                checks.close("registry.alias_price_multiplier", event["price_multiplier"],
                             adj / new_adj, atol=1e-10, rtol=1e-6)
            else:
                require(pd.Timestamp(reference) < effective and pd.Timestamp(successor_date) < effective,
                        f"{key}: security conversion reference must precede the event")
                raw_ratio = first_present(event, ["raw_share_exchange_ratio", "actual_share_exchange_ratio",
                                                  "raw_exchange_ratio", "ratio_actual_shares"])
                require(float(raw_ratio) > 0, f"{key}: nonpositive sourced conversion ratio")
                expected = float(raw_ratio) * (adj / raw) / (new_adj / new_raw)
                checks.close("registry.units_per_adjusted_unit", event["units_per_adjusted_unit"],
                             expected, atol=1e-12)
        else:
            raise AssertionError(f"Unsupported source event: {key} ({kind})")
        verified.append({"action_id": key, "event_type": kind,
                         "symbol": symbol, "effective_date": effective,
                         "reference_price_date": pd.Timestamp(reference)})
    return verified


def assemble_prices(connection, calendar, scores, registry, supplement_path, lineage_path, checks):
    """Build only observed source prices; audit each overlay before adding it."""
    potential = scores.sort_values(["entry_date", "score", "symbol"], ascending=[True, False, True])
    potential = potential.groupby("entry_date", sort=False).head(20)
    symbols = set(potential.symbol)
    for event in registry["events"]:
        symbols.add(event["symbol"])
        if "successor_symbol" in event:
            symbols.add(event["successor_symbol"])
    supplement = read_frame(supplement_path, ["date"]) if supplement_path.is_file() else pd.DataFrame()
    if not supplement.empty:
        require(not supplement.duplicated(["symbol", "date"]).any(), "Duplicate supplemental observation")
        symbols.update(supplement.symbol)
    raw = connection.execute(
        "SELECT symbol,date,close,adj_close FROM prices "
        "WHERE symbol IN (SELECT unnest(?)) AND date BETWEEN ? AND ? ORDER BY date,symbol",
        [sorted(symbols), ANCHOR.date(), END.date()],
    ).fetchdf()
    require(not raw.duplicated(["symbol", "date"]).any(), "Duplicate original quote")
    valid = raw.date.isin(calendar) & np.isfinite(raw.adj_close) & raw.adj_close.gt(0)
    observed = {(r.date, r.symbol): float(r.adj_close) for r in raw.loc[valid].itertuples()}
    original = observed.copy()
    try:
        lineage = read_frame(lineage_path, ["date"])
    except pd.errors.EmptyDataError:
        lineage = pd.DataFrame(columns=["date", "symbol", "type"])
    require(not lineage.duplicated(["symbol", "date"]).any(), "Duplicate overlay lineage")
    require(lineage.date.isin(calendar).all(), "Overlay quote outside the study calendar")
    expected = {}
    events = {e["action_id"]: e for e in registry["events"]}
    # Enumerating all qualifying source observations also detects omitted or
    # fabricated lineage records, rather than merely checking exported rows.
    for event in registry["events"]:
        if event.get("event_type") != "ticker_alias":
            continue
        symbol, successor = event["symbol"], event["successor_symbol"]
        begin = pd.Timestamp(event.get("overlay_start", event["effective_date"]))
        finish = pd.Timestamp(event.get("overlay_end", END))
        multiplier = float(event["price_multiplier"])
        for day in calendar[(calendar >= begin) & (calendar <= finish)]:
            key, source_key = (day, symbol), (day, successor)
            if key in observed or source_key not in observed:
                continue
            source_price = observed[source_key]
            expected[key] = {
                "type": "verified_ticker_alias", "action_id": event["action_id"],
                "source_symbol": successor, "source_adj_close": source_price,
                "price_multiplier": multiplier, "adj_close": source_price * multiplier,
                "source_url": event["source_url"],
            }
            observed[key] = source_price * multiplier
    for record in supplement.to_dict("records"):
        key = (record["date"], record["symbol"])
        require(record["date"] in calendar, f"Supplement outside calendar: {key}")
        price = float(record["adj_close"])
        require(np.isfinite(price) and price > 0, f"Invalid supplement: {key}")
        require(str(record.get("source_url", "")).startswith(("https://", "http://")),
                f"Supplement lacks source URL: {key}")
        source_day = first_present(record, ["source_date", "observation_date"], False)
        if source_day is not None:
            require(pd.Timestamp(source_day) == record["date"], f"Supplement carries another day's price: {key}")
        source_close = first_present(record, ["source_close", "source_raw_close", "raw_close", "close"], False)
        multiplier = first_present(record, ["price_multiplier", "adjustment_multiplier", "raw_to_adjusted_factor"], False)
        require(source_close is not None and multiplier is not None,
                f"Supplement needs observed source close and explicit multiplier: {key}")
        checks.close("supplement.observed_price_times_multiplier", price,
                     float(source_close) * float(multiplier), atol=1e-8)
        if key in observed:
            checks.close("supplement.existing_quote_unchanged", observed[key], price, rtol=1e-8)
            continue
        expected[key] = record | {"type": "sourced_supplemental_observation"}
        observed[key] = price
    require(set(expected) == set(zip(lineage.date, lineage.symbol)),
            "Exported quote lineage does not exactly match independently sourced overlays")
    for row in lineage.to_dict("records"):
        key = row["date"], row["symbol"]
        require(key not in original, f"Overlay overwrites an original finite quote: {key}")
        source = expected[key]
        require(row["type"] == source["type"], f"Overlay type mismatch: {key}")
        for field in ("source_url", "action_id", "source_symbol"):
            if field in source and pd.notna(source[field]):
                require(str(row.get(field, "")) == str(source[field]), f"Overlay {field} mismatch: {key}")
        for field in ("adj_close", "source_adj_close", "price_multiplier", "source_close", "raw_close"):
            if field in source and pd.notna(source[field]):
                checks.close(f"overlay.{field}", row[field], source[field], atol=1e-8)
        require(np.isfinite(row["adj_close"]) and row["adj_close"] > 0, f"Invalid overlay value: {key}")
        if row["type"] == "verified_ticker_alias":
            require(row["action_id"] in events, f"Undocumented alias: {key}")
    return observed, {
        "original_valid_quotes_loaded": len(original), "overlay_rows": len(lineage),
        "overlay_rows_by_type": dict(Counter(lineage.type)),
        "supplemental_input_rows": len(supplement),
        "all_overlay_rows_individually_reconciled": True,
        "original_finite_prices_overwritten": 0,
        "overlay_prices_carried_from_other_dates": 0,
        "supplement_source_validation": (
            "Supplement CSV, observation dates, source URLs and numerical scaling checked; "
            "live external pages are not fetched by this deterministic offline audit."
        ),
    }


def reconstruct_accounting(calendar, scores, prices, registry, config):
    """Independent lot account built from frozen scores and source observations."""
    initial, fee_rate = float(config["initial_cash"]), config["fee_bps_per_side"] / 10000
    actions = defaultdict(dict)
    for event in registry["events"]:
        if "cash_per_adjusted_unit" not in event and event.get("event_type") != "security_conversion":
            continue
        pos = calendar.searchsorted(pd.Timestamp(event["effective_date"]))
        require(pos < len(calendar), "Event maps beyond study end")
        day = calendar[pos]
        require(event["symbol"] not in actions[day], "Two mandatory actions for one symbol/session")
        actions[day][event["symbol"]] = event
    groups = {day: group for day, group in scores.groupby("entry_date", sort=True)}
    lots, trades, orders = {}, [], []
    cash = initial
    history = [{"date": calendar[0], "cash": cash, "holdings_value": 0., "nav": cash,
                "n_open_lots": 0, "stale_lots": 0, "stock_value": 0., "cash_claim_value": 0.}]
    next_id = 1
    stale_marks, max_stale = 0, 0

    def trade_record(day, side, lot, price, *, fee=0., cash_change=0., realized=0.,
                     from_symbol="", from_units=np.nan, ratio=np.nan):
        return {
            "date": day, "side": side, "symbol": lot["symbol"], "lot_id": lot["lot_id"],
            "signal_date": lot["signal_date"], "entry_date": lot["entry_date"],
            "maturity_date": lot["maturity_date"], "scheduled_exit": lot["scheduled_exit"],
            "units": lot["units"], "price": price, "gross_value": lot["units"] * price,
            "fee": fee, "cash_change": cash_change, "realized_pnl": realized,
            "position_type": lot["position_type"], "action_id": lot["action_id"],
            "action_effective_date": lot["action_effective_date"], "source_url": lot["source_url"],
            "original_symbol": lot["original_symbol"], "from_symbol": from_symbol,
            "from_units": from_units, "units_per_adjusted_unit": ratio,
        }

    for day in calendar[1:]:
        previous = history[-1]
        # The independently reconstructed prior-close balance fixes this month's
        # ticket before any current-day maturity proceeds or corporate actions.
        candidates = groups.get(day)
        todays_orders = []
        if candidates is not None and previous["cash"] > 1e-8:
            require(candidates.signal_date.eq(previous["date"]).all(), "Signal is not the previous session")
            ticket = min(previous["cash"], previous["nav"] * config["cohort_fraction"]) / config["top_n"]
            if ticket > 1e-8:
                ranked = candidates.sort_values(["score", "symbol"], ascending=[False, True]).head(config["top_n"])
                for row in ranked.itertuples():
                    order = {
                        "order_id": len(orders), "signal_date": row.signal_date, "entry_date": day,
                        "symbol": row.symbol, "score": float(row.score), "budget": ticket,
                        "signal_cash": previous["cash"], "signal_nav": previous["nav"], "status": "planned",
                    }
                    orders.append(order)
                    todays_orders.append(order)
        for lot_id in list(lots):
            lot = lots[lot_id]
            event = actions[day].get(lot["symbol"]) if lot["position_type"] == "stock" else None
            if event is not None:
                require(lot["entry_date"] < day, "New entry incorrectly receives a historical action")
                old_symbol, old_units = lot["symbol"], lot["units"]
                lot.update(action_id=event["action_id"],
                           action_effective_date=pd.Timestamp(event["effective_date"]),
                           source_url=event["source_url"], stale_sessions=0)
                if "cash_per_adjusted_unit" in event:
                    lot.update(position_type="cash_claim", last_price=float(event["cash_per_adjusted_unit"]))
                    trades.append(trade_record(day, "CASH_MERGER_CONVERSION", lot, lot["last_price"],
                                               from_symbol=old_symbol, from_units=old_units, ratio=1.))
                else:
                    ratio = float(event["units_per_adjusted_unit"])
                    successor = event["successor_symbol"]
                    require((day, successor) in prices, f"Unobserved conversion quote: {successor} {day}")
                    lot.update(symbol=successor, units=old_units * ratio,
                               last_price=prices[day, successor], last_quote_date=day)
                    trades.append(trade_record(day, "SECURITY_CONVERSION", lot, lot["last_price"],
                                               from_symbol=old_symbol, from_units=old_units, ratio=ratio))
            if pd.notna(lot["scheduled_exit"]) and day == lot["scheduled_exit"]:
                claim = lot["position_type"] == "cash_claim"
                require(claim or (day, lot["symbol"]) in prices,
                        f"Exit has no same-session observation: lot {lot_id} {day}")
                price = lot["last_price"] if claim else prices[day, lot["symbol"]]
                gross = lot["units"] * price
                fee = gross * fee_rate
                proceeds = gross - fee
                cash += proceeds
                trades.append(trade_record(day, "CASH_CLAIM_RELEASE" if claim else "SELL", lot,
                                           price, fee=fee, cash_change=proceeds,
                                           realized=proceeds - lot["entry_budget"]))
                del lots[lot_id]
                continue
            require(pd.isna(lot["scheduled_exit"]) or lot["scheduled_exit"] > day,
                    f"Overdue lot remains open: {lot_id}")
            if lot["position_type"] == "cash_claim":
                require(lot["stale_sessions"] == 0, "A cash claim was treated as a stale stock quote")
                continue
            if (day, lot["symbol"]) in prices:
                lot.update(last_price=prices[day, lot["symbol"]], last_quote_date=day, stale_sessions=0)
            else:
                lot["stale_sessions"] += 1
                require(lot["stale_sessions"] <= config["max_stale_sessions"],
                        f"More than five stale sessions: lot {lot_id} {day}")
                stale_marks += 1
                max_stale = max(max_stale, lot["stale_sessions"])
        for order in todays_orders:
            key = day, order["symbol"]
            event = actions[day].get(order["symbol"])
            if event is not None:
                order["status"] = "cancelled_cash_action" if "cash_per_adjusted_unit" in event else "cancelled_security_action"
                continue
            if key not in prices:
                order["status"] = "cancelled_missing_quote"
                continue
            budget, price = order["budget"], prices[key]
            require(budget <= cash + 1e-7, f"Fixed ticket overspends cash on {day}")
            units = budget / (price * (1 + fee_rate))
            maturity = day + pd.Timedelta(days=90)
            exit_index = calendar.searchsorted(maturity)
            scheduled = calendar[exit_index] if exit_index < len(calendar) else pd.NaT
            lot = {
                "lot_id": next_id, "symbol": order["symbol"], "signal_date": order["signal_date"],
                "entry_date": day, "maturity_date": maturity, "scheduled_exit": scheduled,
                "units": units, "entry_price": price, "entry_budget": budget,
                "last_price": price, "last_quote_date": day, "stale_sessions": 0,
                "position_type": "stock", "action_id": "", "action_effective_date": pd.NaT,
                "source_url": "", "original_symbol": order["symbol"], "entry_units": units,
            }
            lots[next_id] = lot
            trades.append(trade_record(day, "BUY", lot, price,
                                       fee=units * price * fee_rate, cash_change=-budget))
            cash -= budget
            if -1e-7 < cash < 0:
                cash = 0.
            order["status"] = "filled"
            next_id += 1
        stock = sum(lot["units"] * lot["last_price"] for lot in lots.values() if lot["position_type"] == "stock")
        claims = sum(lot["units"] * lot["last_price"] for lot in lots.values() if lot["position_type"] == "cash_claim")
        require(cash >= -1e-7 and stock >= 0 and claims >= 0, f"Invalid long-only balance on {day}")
        require(np.isfinite(cash + stock + claims), f"Non-finite NAV on {day}")
        history.append({"date": day, "cash": cash, "holdings_value": stock + claims,
                        "nav": cash + stock + claims, "n_open_lots": len(lots),
                        "stale_lots": sum(lot["stale_sessions"] > 0 for lot in lots.values()),
                        "stock_value": stock, "cash_claim_value": claims})
    holdings = []
    for lot in lots.values():
        market_value = lot["units"] * lot["last_price"]
        holdings.append(lot | {"market_value": market_value,
                               "unrealized_pnl": market_value - lot["entry_budget"]})
    return (pd.DataFrame(history), pd.DataFrame(trades), pd.DataFrame(orders),
            pd.DataFrame(holdings), {"stale_lot_session_marks": stale_marks,
                                     "maximum_consecutive_stale_sessions": max_stale})


def independent_metrics(values):
    """Use initial principal, 2% risk-free, sample volatility, actual-day CAGR."""
    levels = np.asarray(values, dtype=float)
    require(np.isfinite(levels).all() and (levels >= 0).all(), "Invalid independent metric series")
    previous, current = levels[:-1], levels[1:]
    require(not ((previous == 0) & (current > 0)).any(), "NAV recovers from zero without external cash")
    returns = np.divide(current, previous, out=np.ones_like(current), where=previous != 0) - 1
    years = (END - START).days / 365.25
    volatility = np.std(returns, ddof=1)
    risk_free_daily = 1.02 ** (1 / 252) - 1
    return {
        "final_value": float(levels[-1]),
        "total_return": float(levels[-1] / levels[0] - 1),
        "cagr": float((levels[-1] / levels[0]) ** (1 / years) - 1),
        "max_drawdown": float(np.min(levels / np.maximum.accumulate(levels) - 1)),
        "sharpe": float(np.mean(returns - risk_free_daily) / volatility * np.sqrt(252)),
    }


def audit(database, output, registry_path=None, supplement_path=None,
          predictions_path=None, frozen_path=None, evidence_dir=None):
    database, output = Path(database).resolve(), Path(output).resolve()
    registry_path = Path(registry_path or ROOT / "corporate_actions.json").resolve()
    supplement_path = Path(supplement_path or ROOT / "supplemental_prices.csv").resolve()
    predictions_path = Path(predictions_path or ROOT / "outputs/frozen_predictions.csv.gz").resolve()
    frozen_path = Path(frozen_path or ROOT / "verification/FROZEN_STRATEGY.json").resolve()
    evidence_dir = Path(evidence_dir or ROOT / "verification/original_incomplete_attempt").resolve()
    success_path = output / "FULL_ACCOUNTING_AUDIT.json"
    if success_path.is_file():
        success_path.unlink()  # a failed rerun must not leave a stale success claim
    status = json.loads((output / "RUN_STATUS.json").read_text())
    require(status.get("complete") is True and status.get("status") == "completed",
            "Full accounting audit requires a completed final replay")
    require(status.get("last_valid_session") == str(END.date()), "Run does not reach the study end")
    config = json.loads((output / "run_config.json").read_text())
    frozen = json.loads(frozen_path.read_text())
    require(frozen["predictions_sha256"] == PREDICTIONS_SHA256, "Frozen forecast identity changed")
    raw_predictions = gzip.decompress(predictions_path.read_bytes())
    require(hashlib.sha256(raw_predictions).hexdigest() == PREDICTIONS_SHA256,
            "Original prediction bytes changed after accounting repair")
    require(len(raw_predictions) == frozen["predictions_bytes"] == 14134272, "Frozen forecast size changed")
    for key, original in frozen["original_run_config"].items():
        require(config.get(key) == original, f"Original strategy setting changed: {key}")
    require(config.get("protocol_sha256") == PROTOCOL_SHA256 and config.get("selected_alpha") == 1000.,
            "Original research protocol/alpha changed")
    require(config.get("accounting_repair_only") is True, "Replay is not identified as an accounting repair")
    require(config.get("frozen_predictions_sha256") == PREDICTIONS_SHA256
            and status.get("frozen_predictions_sha256") == PREDICTIONS_SHA256,
            "Replay is not linked to the original forecast bytes")
    require(digest(database) == DATABASE_SHA256, "Database differs from the published official file")
    registry = json.loads(registry_path.read_text())
    registry_hash = digest(registry_path)
    require(config.get("accounting_overlay_sha256") == registry_hash, "Source registry changed after the replay")
    exported_registry = json.loads((output / "corporate_action_evidence.json").read_text())
    require(exported_registry.pop("file_sha256", None) == registry_hash and exported_registry == registry,
            "Exported corporate-action evidence differs from the input registry")
    checks = Checks()
    scores = pd.read_csv(io.BytesIO(raw_predictions), parse_dates=["signal_date", "entry_date", "fit_date"])
    with duckdb.connect(str(database), read_only=True) as connection:
        calendar = pd.DatetimeIndex(connection.execute(
            "SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
            [ANCHOR.date(), END.date()],
        ).fetchdf().date).astype("datetime64[ns]")
        require(calendar.is_unique and calendar.is_monotonic_increasing
                and calendar[0] == ANCHOR and calendar[1] == START and calendar[-1] == END,
                "Unexpected official study calendar")
        require(len(calendar) - 1 == EXPECTED_SESSIONS, "Study does not cover exactly 2,640 sessions")
        model_evidence = check_model_evidence(connection, scores, output, evidence_dir, checks)
        external_quotes, archived_evidence = archived_quote_evidence(supplement_path, connection, checks)
        registry_evidence = validate_registry(registry, connection, checks, external_quotes)
        prices, price_evidence = assemble_prices(
            connection, calendar, scores, registry, supplement_path,
            output / "price_overlay_lineage.csv", checks,
        )
        benchmark_prices = connection.execute(
            "SELECT date,adj_close FROM benchmark WHERE symbol='^SP500TR' "
            "AND date BETWEEN ? AND ? ORDER BY date", [START.date(), END.date()],
        ).fetchdf().set_index("date").adj_close
        benchmark_prices.index = benchmark_prices.index.astype("datetime64[ns]")
    require(benchmark_prices.index.equals(calendar[1:]) and np.isfinite(benchmark_prices).all()
            and benchmark_prices.gt(0).all(), "Incomplete or invalid observed benchmark")
    history, trades, orders, positions, carry_evidence = reconstruct_accounting(
        calendar, scores, prices, registry, config,
    )
    all_dates = ["date", "signal_date", "entry_date", "maturity_date", "scheduled_exit",
                 "action_effective_date", "last_quote_date"]
    published_history = read_frame(output / "daily_nav.csv", all_dates)
    published_trades = read_frame(output / "trades.csv", all_dates)
    published_orders = read_frame(output / "orders.csv", all_dates)
    published_positions = read_frame(output / "open_positions.csv", all_dates)
    require(pd.DatetimeIndex(published_history.date).equals(calendar), "Daily NAV omits/adds study sessions")
    compare_frames(checks, "daily", published_history, history, ["date"], all_dates,
                   [c for c in history if c != "date"])
    trade_numbers = ["lot_id", "units", "price", "gross_value", "fee", "cash_change", "realized_pnl",
                     "from_units", "units_per_adjusted_unit"]
    compare_frames(checks, "trades", published_trades, trades, ["date", "lot_id", "side"], all_dates, trade_numbers)
    compare_frames(checks, "orders", published_orders, orders, ["order_id"], all_dates,
                   ["order_id", "score", "budget", "signal_cash", "signal_nav"])
    compare_frames(checks, "positions", published_positions, positions, ["lot_id"], all_dates,
                   ["lot_id", "units", "entry_price", "entry_budget", "last_price", "stale_sessions",
                    "market_value", "unrealized_pnl", "entry_units"])
    by_day = trades.groupby("date").cash_change.sum().reindex(calendar, fill_value=0.)
    checks.close("ledger.daily_cash_from_all_cash_changes", history.cash,
                 config["initial_cash"] + by_day.cumsum().to_numpy(), atol=1e-5)
    checks.close("ledger.terminal_profit", history.nav.iloc[-1] - config["initial_cash"],
                 trades.realized_pnl.sum() + positions.unrealized_pnl.sum(), atol=1e-5)
    converted = trades.loc[trades.side.isin(["CASH_MERGER_CONVERSION", "SECURITY_CONVERSION"])]
    checks.close("ledger.conversion_cash_fees_realized_are_zero",
                 converted[["cash_change", "fee", "realized_pnl"]].to_numpy(),
                 np.zeros((len(converted), 3)), atol=0., rtol=0.)
    benchmark = np.r_[config["initial_cash"],
                      config["initial_cash"] * benchmark_prices.to_numpy() / benchmark_prices.iloc[0]]
    metrics = {"strategy": independent_metrics(history.nav), "benchmark": independent_metrics(benchmark)}
    published_metrics = pd.read_csv(output / "metrics.csv").set_index("portfolio")
    require(set(published_metrics.index) == set(metrics), "Unexpected published metric portfolios")
    for portfolio, values in metrics.items():
        for metric, value in values.items():
            checks.close(f"metrics.{portfolio}.{metric}", published_metrics.loc[portfolio, metric], value,
                         atol=1e-7 if metric == "final_value" else 1e-11)
    comparison = read_frame(output / "comparison.csv", ["date"])
    require(pd.DatetimeIndex(comparison.date).equals(calendar), "Comparison calendars differ")
    checks.close("comparison.strategy", comparison.strategy, history.nav, atol=1e-5)
    checks.close("comparison.benchmark", comparison.benchmark, benchmark, atol=1e-5)
    diagnostics = json.loads((output / "backtest_diagnostics.json").read_text())
    require(diagnostics.get("complete") is True, "Accounting diagnostics are incomplete")
    for field, expected in {
        "buy_count": int(trades.side.eq("BUY").sum()), "sell_count": int(trades.side.eq("SELL").sum()),
        "cash_merger_conversion_count": int(trades.side.eq("CASH_MERGER_CONVERSION").sum()),
        "cash_claim_release_count": int(trades.side.eq("CASH_CLAIM_RELEASE").sum()),
        "security_conversion_count": int(trades.side.eq("SECURITY_CONVERSION").sum()),
        "open_lots": len(positions), "stale_lot_session_marks": carry_evidence["stale_lot_session_marks"],
        "terminal_stale_lots": int(positions.stale_sessions.gt(0).sum()),
        "terminal_cash_claim_lots": int(positions.position_type.eq("cash_claim").sum()),
    }.items():
        require(diagnostics[field] == expected, f"Diagnostic count differs: {field}")
    checks.close("diagnostics.total_fees", diagnostics["total_fees"], trades.fee.sum(), atol=1e-5)
    checks.close("diagnostics.realized_pnl", diagnostics["realized_pnl"], trades.realized_pnl.sum(), atol=1e-5)
    checks.close("diagnostics.unrealized_pnl", diagnostics["unrealized_pnl"], positions.unrealized_pnl.sum(), atol=1e-5)
    require(status.get("overlay_price_rows") == price_evidence["overlay_rows"], "Status overlay count differs")
    require(status.get("corporate_actions") == len(registry["events"]), "Status action count differs")
    files = ["RUN_STATUS.json", "run_config.json", "corporate_action_evidence.json",
             "price_overlay_lineage.csv", "daily_nav.csv", "trades.csv", "orders.csv",
             "open_positions.csv", "comparison.csv", "metrics.csv", "backtest_diagnostics.json"]
    report = {
        "audit_passed": True, "complete_final_period_verified": True,
        "audit_method": "Independent implementation; no submission portfolio/research/overlay helpers imported",
        "initial_anchor": ANCHOR, "period_start": START, "period_end": END,
        "market_sessions": len(calendar) - 1, "daily_rows_including_initial_anchor": len(calendar),
        "database_sha256": DATABASE_SHA256, "original_predictions_sha256": PREDICTIONS_SHA256,
        "protocol_sha256": PROTOCOL_SHA256, "accounting_registry_sha256": registry_hash,
        "prediction_bytes_unchanged": True, "original_strategy_settings_unchanged": True,
        "export_sha256": {name: digest(output / name) for name in files},
        "supplemental_prices_sha256": digest(supplement_path) if supplement_path.is_file() else None,
        "audit_script_sha256": digest(Path(__file__)),
        "model_and_membership": model_evidence, "source_registry_events_checked": registry_evidence,
        "price_lineage": price_evidence, "archived_quote_evidence": archived_evidence,
        "stale_mark_policy": carry_evidence,
        "orders": len(orders), "order_status_counts": dict(Counter(orders.status)),
        "trade_event_counts": dict(Counter(trades.side)), "terminal_open_lots": len(positions),
        "terminal_cash": history.cash.iloc[-1], "terminal_stock_value": history.stock_value.iloc[-1],
        "terminal_cash_claim_value": history.cash_claim_value.iloc[-1], "terminal_nav": history.nav.iloc[-1],
        "total_fees": trades.fee.sum(), "realized_pnl": trades.realized_pnl.sum(),
        "unrealized_pnl": positions.unrealized_pnl.sum(),
        "independent_metrics": metrics, "numeric_values_compared": checks.checked_numbers,
        "maximum_absolute_errors": checks.maximum_errors,
        "explicit_conventions_verified": [
            "Top 20 frozen scores with symbol tie-break; no forward outcome or quote-completeness filter",
            "Ticket min(prior-close cash, prior-close NAV/3)/20, fixed before entry-day cash changes",
            "Fractional adjusted-price units and 10 basis points each buy/exit",
            "Every lot's original 90-calendar-day maturity, rounded to the next market session",
            "Cash conversion has zero cash flow/fee/PnL; fixed claim stays ring-fenced to original maturity",
            "Mandatory security conversion preserves original cost and maturity and charges no conversion fee",
            "Same-session observed fills; existing stock marks carry at most five market sessions",
            "Terminal open lots marked without later prices or forced sale",
            "Independent full-period ^SP500TR, initial-capital drawdown anchor, 2% risk-free Sharpe",
        ],
        "interpretation_limits": [
            "Accounting repairs were researched after the initial missing-data stop; this is a repaired accounting replay of frozen predictions, not a fresh untouched holdout.",
            "Exact source data and declared transformations reconcile; the audit does not certify economic correctness of all positive source prices or point-in-time financial statement vintages.",
            "Recorded cash-claim lock, modeled exit fee, and fractional successor exposure are simulation conventions, not claims about actual broker settlement or mandatory cash-in-lieu handling.",
        ],
    }
    safe = json_safe(report)
    success_path.write_text(json.dumps(safe, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return safe


# Supporting-data restoration is deliberately independent of model/portfolio code.
def ensure_supporting_files(root=None):
    """Unpack a fresh snapshot once and verify its immutable supporting data."""
    root = Path(root).resolve() if root is not None else ROOT
    archive = root / "supporting_evidence.zip"
    restore_snapshot = not any(
        (root / "outputs" / name).exists()
        for name in ("RUN_STATUS.json", "run_config.json")
    )
    pending = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate members in supporting evidence")
        manifest = json.loads(bundle.read("MANIFEST.json"))
        entries = manifest["files"]
        if set(names) != {"MANIFEST.json", *entries}:
            raise ValueError("Supporting evidence does not match its manifest")
        for name, record in entries.items():
            relative = PurePosixPath(name)
            root_input = name in {"corporate_actions.json", "supplemental_prices.csv"}
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or (not root_input and (len(relative.parts) < 2
                        or relative.parts[0] not in {"outputs", "verification"}))
                    or relative.suffix not in {".csv", ".json", ".txt", ".gz", ".md", ".png"}):
                raise ValueError(f"Unexpected supporting path: {name}")
            info = bundle.getinfo(name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Expected a regular supporting file: {name}")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Supporting path escapes submission: {name}")
            payload = bundle.read(name)
            if (len(payload) != record["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != record["sha256"]):
                raise ValueError(f"Supporting evidence checksum mismatch: {name}")
            immutable = (root_input or relative.parts[0] == "verification"
                         or name == "outputs/frozen_predictions.csv.gz")
            if not immutable and not restore_snapshot:
                continue
            if target.exists():
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Existing supporting record has changed: {name}")
            else:
                pending.append((name, target, payload))
        # Install run metadata after the supporting tables. A later call
        # must not add historical proof to any output directory with metadata.
        pending.sort(key=lambda item: (
            item[0] in {"outputs/run_config.json", "outputs/RUN_STATUS.json"},
            item[0] == "outputs/RUN_STATUS.json", item[0],
        ))
        for name, target, payload in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as destination:
                destination.write(payload)
    return {"restored_files": len(pending), "restored_snapshot": restore_snapshot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_replay")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--model-evidence", type=Path)
    args = parser.parse_args()
    ensure_supporting_files()
    report = audit(args.database, args.output, args.registry, args.supplement,
                   args.predictions, args.frozen, args.model_evidence)
    print(json.dumps({
        "audit_passed": report["audit_passed"], "market_sessions": report["market_sessions"],
        "orders": report["orders"], "trade_event_counts": report["trade_event_counts"],
        "terminal_nav": report["terminal_nav"],
        "report": str(args.output.resolve() / "FULL_ACCOUNTING_AUDIT.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
