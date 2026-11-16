"""Prediction inputs for the selected fixed model, with no fitting operation.

Reuse the reviewed course data and documented price-unit corrections. Future
outcomes are audit columns only: they never determine eligibility or a score.
The old accounting/checking modules are reused with temporary output folders,
so this evaluation cannot overwrite another account's evidence.
"""

from contextlib import contextmanager
import json

import duckdb
import numpy as np
import pandas as pd

from hw4.data import holding_labels, stock_features
from hw4.settings import DATABASE, PROJECT
from rank_accounting_prices import (
    REPAIRS,
    normalize_accounting_quotes,
    normalize_prefix,
)
from research_models import add_market_features, eligible_panel


@contextmanager
def redirected_attribute(module, name, value):
    """Redirect a legacy output location for this single-process call only."""
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def corrected_feature_panel(context, features):
    """Apply established unit fixes consistently to features and audit labels.

    Forward normalization leaves observations before each seam unchanged. It has
    the same returns as the accounting normalization, which rescales the prefix.
    No imputed trade prices or additional model features are introduced here.
    """
    accounting, audit = normalize_accounting_quotes(context["quotes"])
    accounting, extra_audit = extra_price_repairs(accounting)
    audit += extra_audit
    for repair in audit:
        repair["scope"] = (
            "Selected pipeline: consistent unit repairs in derived features, "
            "audit labels and accounting. Pre-seam feature history is preserved; "
            "saved coefficients and trading rules are unchanged."
        )
    repairs = REPAIRS + extra_repair_specs()
    forward = context["quotes"].copy()
    panel = context["panel"].copy()
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for spec in repairs:
            symbol, seam = spec["symbol"], pd.Timestamp(spec["seam"])
            forward.loc[forward.index >= seam, symbol] *= spec["divisor"]
            np.testing.assert_allclose(
                forward[symbol].pct_change(fill_method=None),
                accounting[symbol].pct_change(fill_method=None),
                atol=1e-12,
                equal_nan=True,
            )
            pd.testing.assert_series_equal(
                forward.loc[forward.index < seam, symbol],
                context["quotes"].loc[forward.index < seam, symbol],
            )
            raw = con.execute(
                "SELECT symbol,date,close,volume FROM prices WHERE symbol=? "
                "AND date BETWEEN '2008-01-01' AND '2026-09-30' ORDER BY date",
                [symbol],
            ).df()
            raw["date"] = pd.to_datetime(raw.date).astype("datetime64[ns]")
            raw = raw.loc[raw.date.isin(forward.index)].copy()
            raw["adj_close"] = forward[symbol].reindex(raw.date).to_numpy()
            ix = panel.index[panel.symbol.eq(symbol)]
            dates = panel.loc[ix, "date"]
            revised = stock_features(raw, forward.index).reindex(dates)
            labels = holding_labels(dates, forward[symbol], forward.index)
            for column in revised:
                panel.loc[ix, column] = revised[column].to_numpy()
            for column in ["entry_date", "scheduled_exit", "label_end", "target"]:
                panel.loc[ix, column] = labels[column].to_numpy()
            panel.loc[ix, "eligible"] = (
                panel.loc[ix, "eligible_history"] & panel.loc[ix, "is_member"]
            )
    corrected = add_market_features(
        panel, forward, context["benchmark"], context["sectors"]
    )
    before = corrected.date.lt(min(s["seam"] for s in repairs))
    np.testing.assert_allclose(
        corrected.loc[before, features],
        context["panel"].loc[before, features],
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )
    return corrected, audit


def extra_repair_specs():
    return json.loads(
        (PROJECT / "research_events/fixed_anchor_price_repairs.json").read_text()
    )["repairs"]


def extra_price_repairs(quotes):
    """Check the source seam before applying each independently cited split."""
    q, audit = quotes.copy(), []
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for spec in extra_repair_specs():
            symbol = spec["symbol"]
            raw = con.execute(
                "SELECT date,close,adj_close FROM prices WHERE symbol=? AND date IN (?,?) ORDER BY date",
                [symbol, spec["prior_date"], spec["seam"]],
            ).df()
            assert len(raw) == 2
            np.testing.assert_allclose(
                raw.close, [spec["prior_raw_close"], spec["seam_raw_close"]]
            )
            np.testing.assert_allclose(raw.close, raw.adj_close, rtol=0, atol=1e-10)
            if "quote_evidence" in spec:
                evidence = json.loads((PROJECT / spec["quote_evidence"]).read_text())
                item = evidence["data"]["chart"]["result"][0]
                dates = (
                    pd.to_datetime(item["timestamp"], unit="s", utc=True)
                    .tz_convert("America/New_York")
                    .tz_localize(None)
                    .normalize()
                )
                provider = np.asarray(item["indicators"]["quote"][0]["close"])
                source = (
                    con.execute(
                        "SELECT date,close FROM prices WHERE symbol=? AND date BETWEEN ? AND ? ORDER BY date",
                        [symbol, dates.min(), dates.max()],
                    )
                    .df()
                    .set_index("date")
                    .close.reindex(dates)
                    .to_numpy()
                )
                np.testing.assert_allclose(
                    source / provider,
                    np.where(dates < spec["seam"], spec["divisor"], 1),
                    rtol=0.0001,
                    atol=0,
                )
            before = q[symbol].copy()
            q[symbol] = normalize_prefix(before, spec["seam"], spec["divisor"])
            audit.append(
                {
                    **spec,
                    "original_seam_return": float(
                        before.loc[spec["seam"]] / before.loc[spec["prior_date"]] - 1
                    ),
                    "normalized_seam_return": float(
                        q.loc[spec["seam"], symbol] / q.loc[spec["prior_date"], symbol]
                        - 1
                    ),
                }
            )
    return q, audit


def add_accounting_events(quotes, raw, events, sectors):
    """Use course successor quotes and preserve the original event policy."""
    additions = json.loads(
        (PROJECT / "research_events/fixed_anchor_events.json").read_text()
    )["events"]
    combined = events + additions
    assert len(combined) == len({e["id"] for e in combined})
    symbols = sorted({s for e in additions for s in [e["old"], e.get("new")] if s})
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        extra = con.execute(
            "SELECT symbol,date,close FROM prices WHERE symbol IN (SELECT unnest(?)) AND date<='2026-09-30'",
            [symbols],
        ).df()
    extra["date"] = pd.to_datetime(extra.date).astype("datetime64[ns]")
    extra = extra.pivot(index="date", columns="symbol", values="close").reindex(
        quotes.index
    )
    raw = raw.copy()
    quotes = quotes.copy()
    for symbol in symbols:
        if symbol in extra:
            raw[symbol] = extra[symbol]
    supplement = PROJECT / "research_events/fixed_anchor_supplemental_quotes.csv"
    if supplement.exists():
        supplied = pd.read_csv(supplement, parse_dates=["date"])
        assert not supplied.duplicated(["symbol", "date"]).any()
        assert supplied.date.isin(quotes.index).all()
        assert supplied.date.le("2026-09-30").all()
        for symbol, frame in supplied.groupby("symbol"):
            frame = frame.set_index("date")
            if symbol not in quotes:
                quotes[symbol] = np.nan
            assert quotes.loc[frame.index, symbol].isna().all()
            quotes.loc[frame.index, symbol] = frame.adj_close
            raw.loc[frame.index, symbol] = frame.close
    for event in additions:
        successor = event.get("new")
        if successor and successor not in sectors:
            sectors[successor] = sectors[event["old"]]
    assert all(s in quotes and s in sectors for s in symbols)
    quotes, audit = extra_price_repairs(quotes)
    return quotes, raw, combined, audit


def predict_selected(model, panel, configuration):
    """Predict with an already fitted object; retain unknown future targets."""
    p = eligible_panel(panel)
    candidates = p.loc[
        p.eligible
        & p.date.ge("2016-03-31")
        & p.entry_date.between("2016-04-01", "2026-09-30")
    ]
    columns = ["date", "symbol", "entry_date", "scheduled_exit", "target", "label_end"]
    scores = candidates[columns].copy()
    scores["score"] = model.predict(candidates[configuration["features"]])
    scores["fit_date"] = pd.Timestamp(configuration["fitted_asof"])
    if scores.empty or scores.duplicated(["date", "symbol"]).any():
        raise ValueError("Expected a nonempty forecast panel with unique keys.")
    if not np.isfinite(scores.score).all():
        raise ValueError("Every forecast must be finite.")
    if not (scores.fit_date < scores.date).all():
        raise ValueError("This fixed model must predate every forecast.")
    return scores
