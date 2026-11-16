"""Evidence-backed price-unit normalization for the frozen account only.

The course series switches adjustment vintages on the dates below, which are
NOT the legal stock-split dates. Independent provider observations on both
sides confirm the scale changes. Normalize earlier prices to the later unit;
do not invent a corporate action on a vendor splice date or count a split twice.
Forecasts and all model inputs remain exactly as frozen. That leaves a disclosed
limitation: post-splice model features may contain these source discontinuities.
"""

import json

import duckdb
import numpy as np
import pandas as pd

from hw4.settings import DATABASE, PROJECT

REPAIRS = [
    dict(symbol="COKE", seam="2025-05-23", divisor=10.0,
         actual_split_trading_date="2025-05-27",
         source="https://investor.cokeconsolidated.com/news-releases/news-release-details/coca-cola-consolidateds-10-1-stock-split-finalized-shares-trade"),
    dict(symbol="TPL", seam="2025-11-21", divisor=3.0,
         actual_split_trading_date="2025-12-23",
         source="https://www.texaspacific.com/investors/sec-filings/all-sec-filings/content/0001811074-26-000018/tpl-20251231.htm"),
    dict(symbol="BKNG", seam="2026-03-26", divisor=25.0,
         actual_split_trading_date="2026-04-06",
         source="https://infomemo.theocc.com/infomemos?number=58460"),
    dict(symbol="REZI", seam="2026-07-29", divisor=1.437,
         actual_split_trading_date="2026-08-04", relative_tolerance=0.0006,
         source="https://investor.resideo.com/news/news-details/2026/Resideo-Technologies-Completes-Spin-Off-of-ADI-Global-Distribution/default.aspx",
         note="Distribution adjustment, not a legal 1.437-for-1 stock split. Issuer confirms ADIG spin-off completed August 3 (one ADIG share per two REZI). Provider encodes total-return adjustment as 1437:1000 on August 4. Seven pre-seam prices match this factor; later prices differ at most 0.051%. Preserve course adjusted-unit exposure, with no separate ADIG payout or shares added."),
]


def normalize_prefix(series, seam, divisor):
    """Use a known scale factor; preserve every within-segment return."""
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("A verified positive unit factor is required.")
    earlier = series.index < pd.Timestamp(seam)
    result = series.copy()
    result.loc[earlier] /= divisor
    np.testing.assert_allclose(
        result.loc[earlier].pct_change(fill_method=None),
        series.loc[earlier].pct_change(fill_method=None), equal_nan=True, atol=1e-12,
    )
    pd.testing.assert_series_equal(result.loc[~earlier], series.loc[~earlier])
    return result


def normalize_accounting_quotes(quotes):
    """Require independently saved observations around each reviewed seam."""
    q = quotes.copy()
    audit = []
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for spec in REPAIRS:
            symbol, seam, divisor = spec["symbol"], pd.Timestamp(spec["seam"]), spec["divisor"]
            path = PROJECT / f"research_events/price_unit_evidence/{symbol}.json"
            evidence = json.loads(path.read_text())
            item = evidence["data"]["chart"]["result"][0]
            dates = pd.to_datetime(item["timestamp"], unit="s", utc=True).tz_convert(
                "America/New_York"
            ).tz_localize(None).normalize()
            provider = pd.Series(item["indicators"]["quote"][0]["close"], index=dates)
            source = con.execute(
                "SELECT date,close FROM prices WHERE symbol=? AND date BETWEEN ? AND ? ORDER BY date",
                [symbol, dates.min(), dates.max()],
            ).df().set_index("date").close.reindex(dates)
            assert source.notna().all() and provider.notna().all()
            expected_ratio = np.where(dates < seam, divisor, 1.0)
            ratio = source.to_numpy() / provider.to_numpy()
            # Two decimal rounding / source-vintage differences are recorded.
            # REZI's four when-issued dates differ at most 0.051%; do not overwrite.
            np.testing.assert_allclose(
                ratio, expected_ratio, rtol=spec.get("relative_tolerance", 0.0001), atol=0
            )
            if symbol == "REZI":
                split = next(iter(item["events"]["splits"].values()))
                assert split["numerator"] / split["denominator"] == divisor
            assert (dates < seam).sum() >= 5 and (dates >= seam).sum() >= 5
            before = q[symbol].copy()
            q[symbol] = normalize_prefix(before, seam, divisor)
            prev = q.index[q.index.get_loc(seam) - 1]
            audit.append({
                **spec, "provider_url": evidence["source"],
                "checked_source_rows": len(dates),
                "maximum_raw_relative_error": float(np.max(abs(ratio / expected_ratio - 1))),
                "source_seam_return": float(before.loc[seam] / before.loc[prev] - 1),
                "normalized_seam_return": float(q.loc[seam, symbol] / q.loc[prev, symbol] - 1),
                "scope": "Derived accounting prices only; frozen predictions unchanged; residual dividend-vintage differences retained.",
            })
    return q, audit
