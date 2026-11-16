"""Read-only database audit and causal monthly feature/label construction.

Concepts adapted from Lectures 2, 4 and 7. The financial-event and as-of join
helpers below are explicitly reused from course_data.py (permitted by HW4).
The backtester is independently implemented in engine.py.

Important distinctions: a review flag is not a confirmed error; missing prices
stay missing; future outcomes are separate from model features and eligibility.
"""

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .settings import COURSE, PROJECT

LOG = logging.getLogger(__name__)
sys.path.insert(0, str(COURSE / "Lectures/modeling"))
from course_data import financial_events, join_financials  # noqa: E402


@dataclass
class ResearchData:
    """Daily adjusted quotes plus monthly decision rows; units are return fractions."""

    quotes: pd.DataFrame
    benchmark: pd.Series
    panel: pd.DataFrame
    audit: dict


def checksum(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_database(database):
    """Describe raw data, without using diagnostics to delete historical losers."""
    queries = {
        "prices": """SELECT count(*) AS rows, count(distinct symbol) AS stocks,
            min(date) AS first_date,max(date) AS last_date,
            count(*) FILTER (WHERE adj_close IS NULL OR NOT isfinite(adj_close) OR adj_close<=0) AS invalid_adjusted,
            count(*) FILTER (WHERE open<=0 OR high<=0 OR low<=0 OR close<=0) AS nonpositive_raw,
            count(*) FILTER (WHERE open<low OR open>high OR close<low OR close>high OR low>high) AS ohlc_flags,
            count(*) FILTER (WHERE volume=0) AS zero_volume FROM prices""",
        "off_calendar": "SELECT count(*) AS rows FROM prices p ANTI JOIN calendar c ON p.date=c.date",
        "duplicate_prices": "SELECT count(*) AS keys FROM (SELECT symbol,date FROM prices GROUP BY ALL HAVING count(*)>1)",
        "fundamentals": """SELECT count(*) AS rows,
            count(*) FILTER (WHERE filing_date IS NULL OR filing_date<=fiscal_date) AS invalid_filing_dates,
            count(*) FILTER (WHERE reported_currency IS NULL) AS unknown_currency,
            count(*) FILTER (WHERE total_assets<=0) AS nonpositive_assets FROM fundamentals""",
        "membership": """SELECT count(*) AS rows,count(distinct quarter_start) AS quarters,
            count(*) FILTER (WHERE selection_date>=effective_start OR shares_available_date>selection_date) AS timing_violations
            FROM universe_membership""",
        "suspicious_rank_inputs": """SELECT symbol,quarter_start,selection_close,selection_shares,universe_rank
            FROM universe_membership WHERE quarter_start='2010-01-01'
            AND (selection_close>100000 OR selection_shares>20000000000) ORDER BY symbol""",
    }
    with duckdb.connect(str(database), read_only=True) as con:
        results = {
            name: con.execute(sql).fetchdf().to_dict("records")
            for name, sql in queries.items()
        }
    if results["duplicate_prices"][0]["keys"]:
        raise ValueError("Conflicting price keys require investigation")
    if results["membership"][0]["timing_violations"]:
        raise ValueError("Universe timing is invalid")
    results["sha256"] = checksum(database)
    results["database"] = Path(database).name
    return results


def stock_features(prices, calendar):
    """Causal features on market sessions, never on a compressed observed-row index.

    Complete trailing 63 returns are required for entry eligibility. Missing
    long-horizon momentum is retained for training-only imputation. Zero volume
    or invalid raw close blocks a new signal on that date, not the full stock.
    """
    p = prices.set_index("date").reindex(calendar)
    px = p.adj_close.where(np.isfinite(p.adj_close) & p.adj_close.gt(0))
    ret = px.pct_change(fill_method=None)
    out = pd.DataFrame(index=calendar)
    out["momentum_63"] = px / px.shift(63) - 1
    out["momentum_252_21"] = px.shift(21) / px.shift(252) - 1
    out["volatility_63"] = ret.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)
    # Causal observability/risk screen, not a claim that flagged prices are errors.
    # Require a full year with positive recorded volume and no >2x/<0.5x jump.
    gross = px / px.shift(1)
    out["largest_log_move_252"] = (
        np.log(gross).abs().rolling(252, min_periods=252).max()
    )
    out["positive_volume_sessions_252"] = (
        p.volume.gt(0).rolling(252, min_periods=252).sum()
    )
    out["risk_screen_pass"] = (
        out.volatility_63.le(0.80)
        & out.largest_log_move_252.le(np.log(2))
        & out.positive_volume_sessions_252.eq(252)
    )
    out["eligible_history"] = (
        px.notna()
        & out.volatility_63.notna()
        & p.volume.gt(0)
        & np.isfinite(p.close)
        & p.close.gt(0)
    )
    out.index.name = "date"
    return out


def holding_labels(signal_dates, quotes, calendar, holding_days=90, max_delay=5):
    """Return actual entry/exit dates and gross simple returns; no invented fills.

    Missing entry or unobserved maturity leaves a censored target. Outcome fields
    are for training/evaluation only, never to select prediction candidates.
    """
    calendar = pd.DatetimeIndex(calendar)
    valid_quotes = quotes.dropna().sort_index()
    rows = []
    for signal in signal_dates:
        ix = calendar.searchsorted(signal, side="right")
        entry = calendar[ix] if ix < len(calendar) else pd.NaT
        due = entry + pd.Timedelta(days=holding_days)
        ep = quotes.get(entry, np.nan)
        end, xp = pd.NaT, np.nan
        if pd.notna(entry) and pd.notna(ep) and due <= calendar[-1]:
            pos = valid_quotes.index.searchsorted(due)
            if pos < len(valid_quotes):
                candidate = valid_quotes.index[pos]
                delay = calendar.get_loc(candidate) - calendar.searchsorted(due)
                if delay <= max_delay:
                    end, xp = candidate, valid_quotes.iloc[pos]
        rows.append((signal, entry, due, end, xp / ep - 1 if pd.notna(xp) else np.nan))
    return pd.DataFrame(
        rows, columns=["date", "entry_date", "scheduled_exit", "label_end", "target"]
    )


def build_data(database, end="2015-12-31", use_cache=True):
    """Batch daily stock history, retaining only month-end model rows in memory.

    Development calls do not load any price after end. Cache identity contains
    source bytes, the code and cutoff; cached results cannot silently survive a
    data-policy change. Only locally created caches are loaded.
    """
    database = Path(database)
    identity = (
        checksum(database)
        + checksum(__file__)
        + checksum(COURSE / "Lectures/modeling/course_data.py")
        + end
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache = PROJECT / "outputs/cache" / key
    if use_cache and (cache / "data.pkl").exists():
        LOG.info("Loading own cached data %s", key)
        return pd.read_pickle(cache / "data.pkl")
    audit = audit_database(database)
    with duckdb.connect(str(database), read_only=True) as con:
        benchmark = con.execute(
            "SELECT date,adj_close FROM benchmark WHERE date BETWEEN '2008-01-01' AND ? ORDER BY date",
            [end],
        ).fetchdf()
        universe = con.execute(
            "SELECT symbol,sector FROM universe ORDER BY symbol"
        ).fetchdf()
        membership = con.execute(
            "SELECT symbol,quarter_start,selection_date FROM universe_membership WHERE quarter_start<=?",
            [end],
        ).fetchdf()
        calendar = pd.DatetimeIndex(
            pd.to_datetime(benchmark.date).astype("datetime64[ns]")
        )
        benchmark = pd.Series(
            benchmark.adj_close.to_numpy(), index=calendar, name="benchmark"
        )
        if (
            not (np.isfinite(benchmark) & benchmark.gt(0)).all()
            or calendar.has_duplicates
        ):
            raise ValueError("Invalid benchmark")
        quotes = pd.DataFrame(
            np.nan, index=calendar, columns=universe.symbol, dtype=float
        )
        # Only actual month-end closes, not an incomplete final month.
        signal_dates = calendar[:-1][
            calendar[:-1].to_period("M") != calendar[1:].to_period("M")
        ]
        signal_dates = signal_dates[signal_dates >= "2009-12-01"]
        all_parts, histories = [], []
        symbols = universe.symbol.tolist()
        for offset in range(0, len(symbols), 64):
            batch = symbols[offset : offset + 64]
            raw = con.execute(
                """SELECT symbol,date,close,adj_close,volume FROM prices
                WHERE date BETWEEN '2008-01-01' AND ? AND symbol IN (SELECT unnest(?)) ORDER BY symbol,date""",
                [end, batch],
            ).fetchdf()
            financials = con.execute(
                """SELECT symbol,fiscal_date,period,filing_date,reported_currency,
                revenue,net_income,operating_income,total_assets,total_debt,cash_and_equivalents,
                free_cash_flow,operating_cash_flow FROM fundamentals
                WHERE fiscal_date<=? AND (filing_date<=? OR filing_date IS NULL)
                AND symbol IN (SELECT unnest(?))""",
                [end, end, batch],
            ).fetchdf()
            for col in ["fiscal_date", "filing_date"]:
                financials[col] = pd.to_datetime(financials[col]).astype(
                    "datetime64[ns]"
                )
            raw["date"] = pd.to_datetime(raw.date).astype("datetime64[ns]")
            raw = raw.loc[raw.date.isin(calendar)].copy()
            raw.loc[~np.isfinite(raw.adj_close) | raw.adj_close.le(0), "adj_close"] = (
                np.nan
            )
            parts = []
            for symbol, group in raw.groupby("symbol", sort=False):
                series = group.set_index("date").adj_close.reindex(calendar)
                quotes[symbol] = series
                feature = (
                    stock_features(group, calendar).reindex(signal_dates).reset_index()
                )
                feature["symbol"] = symbol
                label = holding_labels(signal_dates, series, calendar)
                feature = feature.merge(label, on="date", validate="one_to_one")
                feature = feature.loc[feature.entry_date.notna()]
                parts.append(feature)
                # Retrospective gap counts are diagnostics, not entry screens.
                valid = series.dropna()
                if len(valid):
                    within = series.loc[valid.index[0] : valid.index[-1]]
                    histories.append(
                        {
                            "symbol": symbol,
                            "missing_internal_sessions": int(within.isna().sum()),
                        }
                    )
            if parts:
                panel = pd.concat(parts, ignore_index=True)
                panel = join_financials(
                    panel,
                    financial_events(financials),
                    universe.loc[universe.symbol.isin(batch)],
                )
                all_parts.append(panel)
            LOG.info(
                "Prepared %d/%d securities through %s",
                min(offset + 64, len(symbols)),
                len(symbols),
                end,
            )
    panel = pd.concat(all_parts, ignore_index=True)
    panel["quarter_start"] = panel.entry_date.dt.to_period("Q").dt.start_time
    for col in ["quarter_start", "selection_date"]:
        membership[col] = pd.to_datetime(membership[col]).astype("datetime64[ns]")
    panel = panel.merge(
        membership, on=["symbol", "quarter_start"], how="left", validate="many_to_one"
    )
    panel["is_member"] = panel.selection_date.notna()
    if (
        not panel.loc[panel.is_member, "selection_date"]
        .le(panel.loc[panel.is_member, "date"])
        .all()
    ):
        raise ValueError("Future universe membership")
    panel["eligible"] = panel.eligible_history & panel.is_member
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    audit["loaded_through"] = end
    audit["monthly_rows"] = len(panel)
    audit["eligible_monthly_rows"] = int(panel.eligible.sum())
    audit["internal_gap_symbols"] = sum(
        x["missing_internal_sessions"] > 0 for x in histories
    )
    audit["eligible_censored_targets"] = int(
        (panel.eligible & panel.target.isna()).sum()
    )
    audit["history_diagnostics"] = histories
    result = ResearchData(quotes, benchmark, panel, audit)
    cache.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, cache / "data.pkl")
    (cache / "audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    return result
