"""Reviewed history and the selected rank features for the revised final run.

Historical price corrections are not extrapolated. Rebase each repaired prefix
to the source's last 2015 unit: this preserves every reviewed historical return
while leaving all later source quotes unchanged, without a unit-join jump.
"""

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from hw4.data import build_data, checksum, holding_labels, stock_features
from hw4.settings import DATABASE, PROJECT
from development_revision import remove_fluidigm_alias
from research_models import add_market_features
from research_session import load_context


def rebase_reviewed_prefix(original, reviewed, cutoff="2015-12-31"):
    """Pure change of adjusted-unit scale; never extend a price repair in time."""
    prefix = original.index[original.index <= pd.Timestamp(cutoff)]
    clean = reviewed.reindex(prefix)
    common = clean.notna() & original.loc[prefix].notna()
    anchor = prefix[common][-1]
    factor = original.loc[anchor] / clean.loc[anchor]
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid historical rebase anchor")
    result = original.copy()
    result.loc[prefix] = clean * factor
    np.testing.assert_allclose(result.loc[anchor], original.loc[anchor], atol=1e-10)
    pd.testing.assert_series_equal(
        result.loc[result.index > cutoff], original.loc[original.index > cutoff]
    )
    np.testing.assert_allclose(
        result.loc[prefix].pct_change(fill_method=None),
        clean.pct_change(fill_method=None),
        atol=1e-12,
        rtol=0,
        equal_nan=True,
    )
    return result, dict(
        anchor=str(anchor.date()),
        factor=float(factor),
        future_quotes_unchanged=True,
        reviewed_returns_preserved=True,
    )


def load_final_context():
    development, protected = load_context()
    identity = {
        **protected,
        "full_inputs_code": checksum(PROJECT / "rank_final_inputs.py"),
        "features_code": checksum(PROJECT / "research_models.py"),
        "end": "2026-09-30",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = PROJECT / f"outputs/rank_final/context_{key}.pkl"
    if cache.exists():
        return pd.read_pickle(cache), protected
    data = build_data(DATABASE, end="2026-09-30")
    registry = json.loads(
        (PROJECT / "data_review/development_price_repairs.json").read_text()
    )
    affected = sorted({r["symbol"] for r in registry["repairs"]})
    rebases = []
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for symbol in affected:
            revised, note = rebase_reviewed_prefix(
                data.quotes[symbol], development["quotes"][symbol]
            )
            data.quotes[symbol] = revised
            rebases.append(dict(symbol=symbol, **note))
            raw = con.execute(
                "SELECT symbol,date,close,volume FROM prices WHERE symbol=? AND date BETWEEN '2008-01-01' AND '2026-09-30' ORDER BY date",
                [symbol],
            ).df()
            raw["date"] = pd.to_datetime(raw.date).astype("datetime64[ns]")
            raw = raw.loc[raw.date.isin(data.quotes.index)].copy()
            raw["adj_close"] = revised.reindex(raw.date).to_numpy()
            indexes = data.panel.index[data.panel.symbol.eq(symbol)]
            dates = data.panel.loc[indexes, "date"]
            features = stock_features(raw, data.quotes.index).reindex(dates)
            labels = holding_labels(dates, revised, data.quotes.index)
            for col in features:
                data.panel.loc[indexes, col] = features[col].to_numpy()
            for col in ["entry_date", "scheduled_exit", "label_end", "target"]:
                data.panel.loc[indexes, col] = labels[col].to_numpy()
            data.panel.loc[indexes, "eligible"] = (
                data.panel.loc[indexes, "eligible_history"]
                & data.panel.loc[indexes, "is_member"]
            )
        sectors = (
            con.execute("SELECT symbol,sector FROM universe ORDER BY symbol")
            .df()
            .set_index("symbol")
            .sector
        )
    # Retain exactly the reviewed alias scope. Do not extend an identity policy
    # using future returns or assume every later issuer mapping is certified.
    data, aliases = remove_fluidigm_alias(data)
    panel = add_market_features(data.panel, data.quotes, data.benchmark, sectors)
    keys = ["date", "symbol"]
    old = development["panel"]
    old = old.loc[old.eligible & old.risk_screen_pass].set_index(keys).sort_index()
    new = (
        panel.loc[
            panel.eligible
            & panel.risk_screen_pass
            & panel.date.isin(old.index.get_level_values("date"))
        ]
        .set_index(keys)
        .sort_index()
    )
    pd.testing.assert_index_equal(old.index, new.index)
    rank_features = [c for c in old if c.startswith("rank_")]
    for col in rank_features:
        np.testing.assert_allclose(
            old[col], new[col], rtol=0, atol=1e-12, equal_nan=True
        )
    context = dict(
        panel=panel,
        quotes=data.quotes,
        benchmark=data.benchmark,
        sectors=sectors,
        rebases=rebases,
        development_candidate_feature_parity=True,
        historical_alias_exclusions=aliases,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(context, cache)
    return context, protected
