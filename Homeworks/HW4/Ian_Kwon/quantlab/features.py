"""Point-in-time features.  Every feature is a registered ``FeatureSpec``.

Timing convention (same as Lectures 7-8): features are computed after the
close of signal date t from information available by that close; the
earliest entry is the next session's close.  Price windows end at t.
Financial statements enter at ``available_date`` (filing date + 1 calendar
day; see ``financial_events``).  Membership is that of the *entry* quarter,
selected at a close <= t.

Extreme values are never winsorized here.  Model inputs are cross-sectional
percentile ranks per date (``*_csr`` columns, see ``panel``), which bound the
influence of extreme values without deleting or altering the raw values,
which stay in the panel for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
STALE_FINANCIAL_DAYS = 550          # course rule: blank ratios older than this
PLACEHOLDER_AVAILABILITY_DAYS = 120  # filing_date <= fiscal_date: assume known 120 days after fiscal end
SECTOR_MIN_PEERS = 10


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    meaning: str
    rationale: str
    available: str
    missing: str = "left missing; model imputer uses the training median and adds an indicator"
    extremes: str = "kept raw; model sees the cross-sectional percentile rank"


REGISTRY: dict[str, FeatureSpec] = {}


def register(*specs: FeatureSpec):
    for spec in specs:
        if spec.name in REGISTRY:
            raise ValueError(f"Duplicate feature {spec.name}")
        REGISTRY[spec.name] = spec


PRICE_CLOSE_T = "close of signal date t (prices up to and including t)"
register(
    FeatureSpec("ret_21", "price", "21-session total return to t",
                "Short-term reversal: one-month winners tend to give back part of the move (liquidity provision).",
                PRICE_CLOSE_T, missing="missing if either endpoint has no quote (no price filling)"),
    FeatureSpec("ret_63", "price", "63-session total return to t", "Intermediate momentum / reversal.",
                PRICE_CLOSE_T, missing="missing if either endpoint has no quote"),
    FeatureSpec("mom_252_21", "price", "Return from t-252 to t-21 sessions (12-1 momentum)",
                "Classic cross-sectional momentum, skipping the reversal month.", PRICE_CLOSE_T,
                missing="missing if either endpoint has no quote"),
    FeatureSpec("vol_63", "price", "Annualized std of daily returns over 63 sessions",
                "Low-volatility anomaly; also a risk control.", PRICE_CLOSE_T,
                missing="requires >= 57 of 63 daily returns"),
    FeatureSpec("drawdown_252", "price", "Price / trailing 252-session max - 1",
                "Distance from 52-week high (anchoring; George-Hwang).", PRICE_CLOSE_T,
                missing="requires >= 200 quotes in the window"),
    FeatureSpec("log_dollar_volume_63", "liquidity", "log median(close x volume) over 63 sessions",
                "Liquidity / size; also used for the investability screen.", PRICE_CLOSE_T,
                missing="requires >= 40 observations"),
    FeatureSpec("log_mcap_proxy", "size", "log market-cap proxy from the entry quarter's selection",
                "Size effect.  Known at the selection close <= t.",
                "selection_date of the entry quarter (<= t by construction)",
                extremes="known proxy errors (e.g. APDN) are rank-bounded, not altered"),
)
FIN_TIME = "available_date = filing_date + 1 calendar day (placeholder filings: fiscal_date + 120 days)"
register(*[
    FeatureSpec(name, "financial", meaning, rationale, FIN_TIME,
                missing="missing if denominator <= 0, inputs missing, or statement older than 550 days")
    for name, meaning, rationale in [
        ("net_margin", "quarterly net income / revenue", "Profitability (quality)."),
        ("operating_roa", "quarterly operating income / total assets", "Profitability (quality)."),
        ("gross_profitability", "quarterly gross profit / total assets", "Novy-Marx gross profitability."),
        ("debt_to_assets", "total debt / total assets", "Leverage / distress risk."),
        ("cash_to_assets", "cash / total assets", "Financial slack."),
        ("fcf_to_assets", "free cash flow / total assets", "Cash-based profitability."),
        ("accruals_to_assets", "(net income - operating cash flow) / total assets",
         "Sloan accruals: high accruals predict lower returns."),
        ("book_to_market", "total equity / market-cap proxy of the entry quarter",
         "Value effect.  Proxy mixes vendor share/price bases: treat as noisy."),
    ]],
    *[FeatureSpec(name, "financial_trend", meaning, rationale,
                  FIN_TIME + "; the year-ago / trailing quarters must each have been available no later "
                  "than the current statement, otherwise missing",
                  missing="missing if the year-ago quarter (fiscal date within 20 days of one year earlier) "
                          "is absent, a required input is missing or <= 0, or the statement is older than 550 days")
      for name, meaning, rationale in [
          ("roa_yoy_change", "operating ROA minus operating ROA of the same quarter a year earlier",
           "Profitability trend: improving firms keep outperforming (slow reaction to fundamentals). "
           "Same-quarter comparison removes seasonality."),
          ("revenue_growth_yoy", "revenue / revenue of the same quarter a year earlier - 1",
           "Sales growth as a fundamental-momentum measure."),
          ("asset_growth_yoy", "total assets / total assets a year earlier - 1",
           "Asset-growth anomaly (Cooper-Gulen-Schill): fast balance-sheet growth predicts lower returns."),
          ("net_issuance_yoy", "log(shares outstanding / shares outstanding a year earlier)",
           "Net issuance anomaly (Pontiff-Woodgate): issuers underperform, repurchasers outperform.  Vendor "
           "share counts are on one split basis (checked on AAPL 2014), so the ratio is split-neutral; the "
           "few residual jumps near split ratios (about 0.3% of rows near x2) are kept and rank-bounded."),
          ("earnings_yield_ttm", "sum of the last four quarters' net income / market-cap proxy of the entry quarter",
           "Earnings-based value; less distorted by intangibles than book value.  Requires four quarters "
           "spanning 255-290 days.  Same proxy noise as book_to_market."),
      ]],
    FeatureSpec("filing_age_days", "financial", "days since the statement became available",
                "Information staleness; lets models discount old ratios.", FIN_TIME),
    FeatureSpec("availability_imputed", "financial", "1 if the statement's availability was imputed",
                "Transparency flag for the +120 day placeholder rule.", FIN_TIME, missing="0/1, never missing"),
)
SECTOR_RELATIVE = ["ret_21", "ret_63", "mom_252_21", "vol_63", "operating_roa", "gross_profitability",
                   "book_to_market"]
register(*[
    FeatureSpec(f"{base}_sr", "sector_relative", f"{base} minus the same-date median of eligible sector peers",
                "Separates stock-specific from common sector movement.",
                "same as the base feature; peers are the eligible cross-section of the same signal date",
                missing=f"missing if the sector has < {SECTOR_MIN_PEERS} eligible peers with the base feature")
    for base in SECTOR_RELATIVE
])

PRICE_FEATURES = ["ret_21", "ret_63", "mom_252_21", "vol_63", "drawdown_252"]
TREND_FEATURES = ["roa_yoy_change", "revenue_growth_yoy", "asset_growth_yoy", "net_issuance_yoy"]
MCAP_SCALED = ["book_to_market", "earnings_yield_ttm"]  # need the entry quarter's market-cap proxy (join time)
FINANCIAL_FEATURES = ["net_margin", "operating_roa", "gross_profitability", "debt_to_assets", "cash_to_assets",
                      "fcf_to_assets", "accruals_to_assets", *TREND_FEATURES, *MCAP_SCALED]
EVENT_RATIOS = [c for c in FINANCIAL_FEATURES if c not in MCAP_SCALED]  # computed per statement
YEAR_AGO_TOLERANCE_DAYS = 20     # same fiscal quarter a year earlier (52/53-week years shift a few days)
TTM_SPAN_DAYS = (255, 290)       # fiscal-date distance between the newest and oldest of four quarters


def feature_table() -> pd.DataFrame:
    """Documentation table generated from the registry (written into README/results)."""
    return pd.DataFrame([vars(s) for s in REGISTRY.values()])


# ---------------------------------------------------------------- price features

def _lagged(px: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(px, np.nan)
    out[lag:] = px[:-lag]
    return out


def price_feature_matrices(px: np.ndarray, close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    """Wide (session x symbol) price features.  Row t uses rows <= t only.

    ``px`` is the integrity-cleaned adjusted price with NaN gaps (never filled).
    Rolling statistics come from pandas with explicit ``min_periods``.
    """
    frame = pd.DataFrame(px)
    daily = frame / frame.shift(1) - 1  # NaN if either day lacks a quote
    feats = {
        "ret_21": px / _lagged(px, 21) - 1,
        "ret_63": px / _lagged(px, 63) - 1,
        "mom_252_21": _lagged(px, 21) / _lagged(px, 252) - 1,
        "vol_63": (daily.rolling(63, min_periods=57).std() * np.sqrt(TRADING_DAYS)).to_numpy(),
        "drawdown_252": (frame / frame.rolling(252, min_periods=200).max() - 1).to_numpy(),
    }
    dollar = pd.DataFrame(close * volume)
    med = dollar.rolling(63, min_periods=40).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["log_dollar_volume_63"] = np.where(med > 0, np.log(med), np.nan)
    feats["valid_returns_63"] = daily.notna().rolling(63, min_periods=1).sum().to_numpy()
    return feats


# ------------------------------------------------------------ financial features

def financial_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Statements dated by when an investor could first use them.

    Adapted from ``course_data.financial_events``.  Difference (decided with
    the user): placeholder rows with ``filing_date <= fiscal_date`` (23,661 in
    the full DB, filing date exactly equal to fiscal end) are not dropped; they
    become available ``fiscal_date + 120`` days, later than >99% of valid lags
    (median 38, p99 107), and carry ``availability_imputed = 1``.
    """
    f = fundamentals.loc[fundamentals.period.eq("quarter")].copy()
    valid = f.filing_date.gt(f.fiscal_date)
    f["availability_imputed"] = (~valid).astype(int)
    f["available_date"] = np.where(valid, f.filing_date + pd.Timedelta(days=1),
                                   f.fiscal_date + pd.Timedelta(days=PLACEHOLDER_AVAILABILITY_DAYS))
    f["available_date"] = pd.to_datetime(f.available_date)
    assets = f.total_assets.where(f.total_assets.gt(0))
    revenue = f.revenue.where(f.revenue.gt(0))
    f["net_margin"] = f.net_income / revenue
    f["operating_roa"] = f.operating_income / assets
    f["gross_profitability"] = f.gross_profit / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
    f = _trend_and_trailing(f)
    f[EVENT_RATIOS] = f[EVENT_RATIOS].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    # A late filing for an older fiscal period must not replace a newer known period.
    f = f.loc[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "filing_date", "available_date", "availability_imputed",
              "total_equity", "net_income_ttm", *EVENT_RATIOS]].reset_index(drop=True)


def _trend_and_trailing(f: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year and trailing-four-quarter values attached to each statement.

    Timing: a statement that becomes usable at ``available_date`` A may only be
    compared with other quarters that were themselves available by A.  Late
    filings of older quarters exist (506 research-mode rows were filed before
    the previous fiscal quarter's filing), so this is checked row by row
    instead of assumed; a comparison that fails it is left missing.
    """
    q = f.sort_values(["symbol", "fiscal_date"]).reset_index(drop=True)
    g = q.groupby("symbol", sort=False)
    # trailing four quarters of net income: all four present, regularly spaced, all known by A
    ni4 = sum(g.net_income.shift(k) for k in range(4))
    span = (q.fiscal_date - g.fiscal_date.shift(3)).dt.days
    older_available = pd.concat([g.available_date.shift(k) for k in (1, 2, 3)], axis=1)
    known = older_available.notna().all(axis=1) & older_available.max(axis=1).le(q.available_date)
    q["net_income_ttm"] = ni4.where(span.between(*TTM_SPAN_DAYS) & known)

    # same quarter one year earlier: nearest fiscal date within the tolerance
    prior = q[["symbol", "fiscal_date", "available_date", "operating_roa", "revenue", "total_assets",
               "shares_outstanding"]].rename(columns=lambda c: c if c == "symbol" else f"{c}_ya")
    q["_target"] = q.fiscal_date - pd.Timedelta(days=365)
    m = pd.merge_asof(q.sort_values("_target"), prior.sort_values("fiscal_date_ya"), left_on="_target",
                      right_on="fiscal_date_ya", by="symbol", direction="nearest",
                      tolerance=pd.Timedelta(days=YEAR_AGO_TOLERANCE_DAYS))
    ok = m.available_date_ya.notna() & m.available_date_ya.le(m.available_date)

    def pos(s):
        return s.where(s.gt(0))
    m["roa_yoy_change"] = (m.operating_roa - m.operating_roa_ya).where(ok)
    m["revenue_growth_yoy"] = (pos(m.revenue) / pos(m.revenue_ya) - 1).where(ok)
    m["asset_growth_yoy"] = (pos(m.total_assets) / pos(m.total_assets_ya) - 1).where(ok)
    m["net_issuance_yoy"] = np.log(pos(m.shares_outstanding) / pos(m.shares_outstanding_ya)).where(ok)
    return m.drop(columns=["_target", *[c for c in m.columns if c.endswith("_ya")]])


def join_financials(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join: the latest statement with available_date <= signal date."""
    left = panel.sort_values("date")
    right = events.sort_values("available_date")
    out = pd.merge_asof(left, right, left_on="date", right_on="available_date", by="symbol",
                        direction="backward", allow_exact_matches=True)
    stale = (out.date - out.fiscal_date).dt.days.gt(STALE_FINANCIAL_DAYS)
    out.loc[stale, EVENT_RATIOS + ["total_equity", "net_income_ttm"]] = np.nan
    out["filing_age_days"] = (out.date - out.available_date).dt.days
    out["availability_imputed"] = out.availability_imputed.fillna(0).astype(int)
    mcap = np.exp(out.log_mcap_proxy)
    out["book_to_market"] = (out.total_equity / mcap).where(mcap.gt(0))
    out["earnings_yield_ttm"] = (out.net_income_ttm / mcap).where(mcap.gt(0))
    if not (out.available_date.isna() | out.available_date.le(out.date)).all():
        raise AssertionError("A financial statement was joined before it was available")
    return out.drop(columns=["total_equity", "net_income_ttm"])


# ---------------------------------------------------------- cross-sectional ops

def sector_relative(panel: pd.DataFrame, columns, eligible_col="eligible", min_peers=SECTOR_MIN_PEERS):
    """x - median over same-date, same-sector *eligible* peers (full cross-section).

    Computed on the whole eligible cross-section of each date before any row
    reaches a model.  Ineligible rows get NaN (they are not candidates).
    Groups with fewer than ``min_peers`` valid values get NaN.
    """
    out = panel.copy()
    elig = out[eligible_col].to_numpy(bool)
    sub = out.loc[elig, ["date", "sector", *columns]]
    groups = sub.groupby(["date", "sector"], sort=False)
    for c in columns:
        med = groups[c].transform("median")
        cnt = groups[c].transform("count")
        val = (sub[c] - med).where(cnt.ge(min_peers))
        out[f"{c}_sr"] = np.nan
        out.loc[elig, f"{c}_sr"] = val.to_numpy()
    return out


def cross_sectional_rank(panel: pd.DataFrame, columns, eligible_col="eligible"):
    """Percentile rank within each date's eligible cross-section, centred at 0.

    Not a learned transform (no fitted parameters), so it cannot leak across
    time; it depends only on same-date eligible rows.
    """
    out = panel.copy()
    elig = out[eligible_col].to_numpy(bool)
    sub = out.loc[elig, ["date", *columns]]
    ranks = sub.groupby("date", sort=False)[list(columns)].rank(pct=True) - 0.5
    for c in columns:
        out[f"{c}_csr"] = np.nan
        out.loc[elig, f"{c}_csr"] = ranks[c].to_numpy()
    return out
