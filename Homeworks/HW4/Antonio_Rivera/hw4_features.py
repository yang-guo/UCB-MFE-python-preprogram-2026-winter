"""HW4 modules 2 + 4a: extra features on top of the course baseline.

Module 2
    A  gross_profitability = gross_profit / total_assets      (quality)
    B  asset_growth        = total_assets / assets one year earlier - 1 (investment)
    D  <feature>_rank      = same-date percentile among ELIGIBLE names only
Module 4a (financial block)
    sue              = (EPS - EPS same quarter last year) / std of the previous 8 such changes
    mcap             = close_t x shares_outstanding (latest filing available at t)
    V1 book_to_market  = total_equity / mcap
    V2 earnings_yield  = TTM net_income / mcap
    V3 cashflow_yield  = TTM operating_cash_flow / mcap
    V4 ebitda_to_ev    = TTM ebitda / (mcap + total_debt - cash_and_equivalents)
    V5 sales_to_price  = TTM revenue / mcap
    VC value_composite = mean of same-date eligible ranks of V1-V5 (>= 3 present)
    Only sue and value_composite go to the model; V1-V5 are diagnostics.

Data basis (checked on AAPL): `close`, `shares_outstanding`, and `eps` are all
restated for every later split, so close x shares is split-consistent. Names
with inconsistent bases (README: TSYS, APDN) are caught by plausibility bounds.
Fundamentals are per-quarter values (not year-to-date), so TTM = sum of 4.

Reused / adapted code (cited): Lectures/modeling/course_data.py
    - load_extra_fundamentals adapts the SQL pattern of load_data.
    - extra_financial_events follows the same filing rules as financial_events:
      quarterly rows only, filing_date > fiscal_date, usable from filing_date + 1 day,
      a late old fiscal period never replaces a newer one, stale after 550 days.

Timing: every feature on row (symbol, t) uses only information available at close t.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import hw4_data as hd

cd = hd.cd  # cited course helpers (imported once in hw4_data)

NEW_FINANCIAL = ["gross_profitability", "asset_growth"]
RAW_NUMERIC = cd.NUMERIC_FEATURES + NEW_FINANCIAL          # baseline + A + B (unchanged)
RANK_FEATURES = [f"{c}_rank" for c in RAW_NUMERIC]          # D (unchanged)

# Module 4a: per-filing fields carried through the as-of join.
EVENT_FIELDS = ["sue", "shares_outstanding", "total_equity", "total_debt",
                "cash_and_equivalents", "ttm_net_income", "ttm_ocf", "ttm_ebitda", "ttm_revenue"]
TTM_SOURCES = {"ttm_net_income": "net_income", "ttm_ocf": "operating_cash_flow",
               "ttm_ebitda": "ebitda", "ttm_revenue": "revenue"}
VALUE_METRICS = ["book_to_market", "earnings_yield", "cashflow_yield",
                 "ebitda_to_ev", "sales_to_price"]
MODEL_FINANCIAL_4A = ["sue", "value_composite"]             # what the model sees
PRICE_4B = ["beta_252", "idio_vol_63", "max_ret_21"]
BETA_WINDOW, BETA_MIN = 252, 189     # allow up to 25% missing days for beta
IDIO_WINDOW = 63                     # strict, like the course volatility_63
MAX_WINDOW = 21

STALE_DAYS = 550        # same staleness rule as the course financials
YEAR_AGO_TOL_DAYS = 20  # fiscal quarters drift (52/53-week years)
SUE_WINDOW, SUE_MIN = 8, 4           # previous changes used for sigma; minimum needed
TTM_SPAN_DAYS = (245, 295)           # fiscal end of 1st vs 4th quarter: 270 ± 25 days
VC_MIN_METRICS = 3

# Fixed plausibility bounds (set a priori from accounting logic, never fitted).
# Values outside are treated as vendor unit errors (e.g. POZN 2011-06-30, TSYS
# 2011–2013 mix units ~1000x) and blanked, not corrected.
#   quarterly gross profit cannot plausibly exceed 2x total assets, or lose > 1x;
#   assets falling > 90% or rising > 10x in one year is far likelier a unit error.
PLAUSIBLE = {"gross_profitability": (-1.0, 2.0), "asset_growth": (-0.9, 9.0)}
#   value, too cheap: book worth > 20x price, or annual earnings/cash flow > 1-2x
#   the whole market cap (or losses > 5x it) -> mcap too SMALL (e.g. TSYS, POZN).
#   value, too dear: price > 100x annual sales or > 200x book -> mcap too BIG.
#   Seen in the class DB (feature values only, no returns looked at): BKNG shares
#   restated for a later 25:1 split while close is not; APDN/PEGY/BESS closes
#   restated for reverse splits while shares are not. No corporate-action table
#   exists to repair this, so such rows are detected and blanked.
VALUE_PLAUSIBLE = {"book_to_market": (0.005, 20.0), "earnings_yield": (-5.0, 1.0),
                   "cashflow_yield": (-5.0, 2.0), "ebitda_to_ev": (-5.0, 2.0),
                   "sales_to_price": (0.01, 100.0)}

FUND_COLUMNS = ["gross_profit", "total_assets", "eps", "shares_outstanding", "total_equity",
                "total_debt", "cash_and_equivalents", "net_income", "operating_cash_flow",
                "ebitda", "revenue"]


# --- Financial events (one row per filing) ---------------------------------------

def load_extra_fundamentals(database=hd.CLASS_DB, end="2015-12-31", symbols=None):
    """Read only the columns we need; cut dates in SQL (adapted from load_data)."""
    sym_sql = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    params = [end, end] + ([list(symbols)] if symbols is not None else [])
    with duckdb.connect(str(Path(database)), read_only=True) as con:
        f = con.execute(f"""SELECT symbol, fiscal_date, period, filing_date,
                {", ".join(FUND_COLUMNS)}
            FROM fundamentals
            WHERE fiscal_date <= ? AND (filing_date <= ? OR filing_date IS NULL) {sym_sql}
            ORDER BY symbol, fiscal_date""", params).fetchdf()
    for col in ["fiscal_date", "filing_date"]:
        f[col] = pd.to_datetime(f[col]).astype("datetime64[ns]")
    return f


def _year_ago(f, cols):
    """Attach prior_<col> from the same symbol's quarter nearest fiscal_date - 365d (±20d).

    The prior record must itself have been public when this filing becomes usable.
    """
    left = f.assign(year_ago_target=f.fiscal_date - pd.Timedelta(days=365))
    right = (f[["symbol", "fiscal_date", "available_date"] + cols]
             .rename(columns={"fiscal_date": "prior_fiscal_date",
                              "available_date": "prior_available_date",
                              **{c: f"prior_{c}" for c in cols}})
             .sort_values("prior_fiscal_date"))
    out = pd.merge_asof(left.sort_values("year_ago_target"), right,
                        left_on="year_ago_target", right_on="prior_fiscal_date", by="symbol",
                        direction="nearest", tolerance=pd.Timedelta(days=YEAR_AGO_TOL_DAYS))
    unknown = ~out.prior_available_date.le(out.available_date)
    out.loc[unknown, [f"prior_{c}" for c in cols]] = np.nan
    return out.drop(columns=["year_ago_target", "prior_fiscal_date", "prior_available_date"])


def _prior_window_public(f, window):
    """True if each of the previous `window` filings (fiscal order) was public by this one.

    Guards rolling windows against a late-filed older quarter. `f` must be
    sorted by symbol, fiscal_date.
    """
    days = (f.available_date - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)
    latest_prior = days.groupby(f.symbol).transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).max())
    return latest_prior.isna() | latest_prior.le(days)


def extra_financial_events(fundamentals):
    """One row per (symbol, available_date) with A, B, SUE, TTM, and balance-sheet fields."""
    f = fundamentals.loc[fundamentals.period.eq("quarter")
                         & fundamentals.filing_date.gt(fundamentals.fiscal_date)].copy()
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    f["total_assets"] = f.total_assets.where(f.total_assets.gt(0))
    f["gross_profitability"] = f.gross_profit / f.total_assets

    # B and the SUE numerator both need the same quarter one year earlier.
    f = _year_ago(f, ["total_assets", "eps"])
    f["asset_growth"] = f.total_assets / f.prior_total_assets - 1

    # Rolling quantities are computed over each symbol's filings in fiscal order.
    f = f.sort_values(["symbol", "fiscal_date"]).reset_index(drop=True)
    g = f.groupby("symbol")

    # SUE: today's seasonal EPS change vs the spread of the previous 8 changes.
    f["eps_change"] = f.eps - f.prior_eps
    sigma = g.eps_change.transform(lambda s: s.shift(1).rolling(SUE_WINDOW, min_periods=SUE_MIN).std())
    f["sue"] = (f.eps_change / sigma.where(sigma.gt(0))).where(
        _prior_window_public(f, SUE_WINDOW))

    # TTM: 4 consecutive quarters, all public by this filing, spanning ~9 months of fiscal ends.
    span = (f.fiscal_date - g.fiscal_date.shift(3)).dt.days
    ttm_ok = span.between(*TTM_SPAN_DAYS) & _prior_window_public(f, 3)
    for out_col, src in TTM_SOURCES.items():
        f[out_col] = g[src].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(ttm_ok)

    f["shares_outstanding"] = f.shares_outstanding.where(f.shares_outstanding.gt(0))

    cols = NEW_FINANCIAL + ["sue"] + list(TTM_SOURCES)
    f[cols] = f[cols].replace([np.inf, -np.inf], np.nan)
    for col, (lo, hi) in PLAUSIBLE.items():
        f[f"{col}_blanked"] = f[col].notna() & ~f[col].between(lo, hi)
        f.loc[f[f"{col}_blanked"], col] = np.nan

    # Same dedupe rule as course financial_events: a late old period never wins.
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    f = f.loc[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    flags = [f"{c}_blanked" for c in PLAUSIBLE]  # audit columns: report counts, never model
    keep = ["symbol", "fiscal_date", "available_date"] + NEW_FINANCIAL + EVENT_FIELDS + flags
    return f[keep].reset_index(drop=True)


def join_extra_financials(panel, events):
    """Backward as-of join on available_date ≤ signal date, per symbol."""
    ev = events.drop(columns=[c for c in events if c.endswith("_blanked")]).rename(columns={
        "fiscal_date": "x_fiscal_date", "available_date": "x_available_date"})
    out = pd.merge_asof(panel.sort_values("date"), ev.sort_values("x_available_date"),
                        left_on="date", right_on="x_available_date", by="symbol",
                        direction="backward")
    stale = (out.date - out.x_fiscal_date).dt.days.gt(STALE_DAYS)
    out.loc[stale, NEW_FINANCIAL + EVENT_FIELDS] = np.nan
    assert (out.x_available_date.isna() | out.x_available_date.le(out.date)).all()
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


# --- Value metrics (need today's close) -----------------------------------------

def add_value_features(panel):
    """Market cap at close t, V1-V5, and plausibility blanking with audit flags.

    One mcap feeds all five metrics, so if ANY metric is implausible the mcap is
    untrustworthy: blank mcap and all of V1-V5 on that row (flag mcap_suspect).
    """
    out = panel.copy()
    out["mcap"] = out.close.where(out.close.gt(0)) * out.shares_outstanding
    ev = out.mcap + out.total_debt - out.cash_and_equivalents   # missing debt/cash -> NaN
    out["book_to_market"] = out.total_equity.where(out.total_equity.gt(0)) / out.mcap
    out["earnings_yield"] = out.ttm_net_income / out.mcap
    out["cashflow_yield"] = out.ttm_ocf / out.mcap
    out["ebitda_to_ev"] = out.ttm_ebitda / ev.where(ev.gt(0))
    out["sales_to_price"] = out.ttm_revenue / out.mcap
    out[VALUE_METRICS] = out[VALUE_METRICS].replace([np.inf, -np.inf], np.nan)
    for col, (lo, hi) in VALUE_PLAUSIBLE.items():
        out[f"{col}_blanked"] = out[col].notna() & ~out[col].between(lo, hi)
    out["mcap_suspect"] = out[[f"{c}_blanked" for c in VALUE_PLAUSIBLE]].any(axis=1)
    out.loc[out.mcap_suspect, ["mcap"] + VALUE_METRICS] = np.nan
    return out


# --- 4b: price block (market = ^SP500TR, returns through close t) ----------------

def load_market_returns(database=hd.CLASS_DB, end="2015-12-31"):
    """Daily ^SP500TR returns indexed by the market calendar (benchmark dates)."""
    with duckdb.connect(str(Path(database)), read_only=True) as con:
        b = con.execute("SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date",
                        [end]).fetchdf()
    b["date"] = pd.to_datetime(b.date).astype("datetime64[ns]")
    return b.set_index("date").adj_close.pct_change(fill_method=None).rename("mkt")


def price_block(prices, market_ret):
    """beta_252, idio_vol_63, max_ret_21 per (symbol, date); no price filling.

    beta     = cov(r_i, r_m) / var(r_m) over 252 sessions (>= 189 paired days)
    idio_vol = sqrt(var(r_i) - cov^2 / var(r_m)) * sqrt(252) over 63 sessions
               (= std of the 63-day OLS residuals, annualized; closed form, no loop)
    max_ret  = largest daily return over the last 21 sessions
    The market return is masked on the stock's missing days so both sides of
    every window use the same dates.
    """
    cal = market_ret.index
    frames = []
    for symbol, g in prices.groupby("symbol", sort=True):
        px = g.set_index("date").adj_close.reindex(cal)
        r = px.pct_change(fill_method=None)          # NaN across any gap
        m = market_ret.where(r.notna())
        rb, mb = r.rolling(BETA_WINDOW, min_periods=BETA_MIN), m.rolling(BETA_WINDOW, min_periods=BETA_MIN)
        beta = rb.cov(m) / mb.var()
        ri, mi = r.rolling(IDIO_WINDOW, min_periods=IDIO_WINDOW), m.rolling(IDIO_WINDOW, min_periods=IDIO_WINDOW)
        resid_var = (ri.var() - ri.cov(m) ** 2 / mi.var()).clip(lower=0)
        frames.append(pd.DataFrame({
            "symbol": symbol, "date": cal,
            "beta_252": beta.to_numpy(),
            "idio_vol_63": np.sqrt(resid_var * 252).to_numpy(),
            "max_ret_21": r.rolling(MAX_WINDOW, min_periods=MAX_WINDOW).max().to_numpy(),
        }).loc[px.notna().to_numpy()])
    out = pd.concat(frames, ignore_index=True)
    out[PRICE_4B] = out[PRICE_4B].replace([np.inf, -np.inf], np.nan)
    return out


# --- D + VC: same-date peer ranks ------------------------------------------------

def add_peer_ranks(panel, columns=RAW_NUMERIC):
    """Percentile rank (0, 1] of each column among eligible names on the same date.

    Ineligible rows get NaN ranks and never influence anyone else's rank.
    Missing raw values stay missing (the training-fitted imputer handles them).
    """
    out = panel.copy()
    ranks = (out.loc[out.eligible, ["date"] + list(columns)]
                .groupby("date")[list(columns)].rank(pct=True))
    for col in columns:
        out[f"{col}_rank"] = ranks[col]  # aligns on index; ineligible rows -> NaN
    return out


def add_value_composite(panel):
    """VC = mean of the eligible same-date ranks of V1-V5; needs >= 3 present."""
    out = add_peer_ranks(panel, VALUE_METRICS)
    r = out[[f"{c}_rank" for c in VALUE_METRICS]]
    out["value_composite"] = r.mean(axis=1).where(r.notna().sum(axis=1).ge(VC_MIN_METRICS))
    return out


# --- Full pipeline ---------------------------------------------------------------

def build_feature_panel(database=hd.CLASS_DB, end="2015-12-31", symbols=None, cross_sectional=True):
    """Module 1 panel + A + B + D + SUE + value + price block. Returns (panel, prices, calendar).

    symbols         optional ticker batch (all per-ticker steps; bounded memory)
    cross_sectional False skips same-date peer steps (D ranks, VC), which need every
                    ticker of a date at once; run them after concatenating batches.
    """
    panel, prices, calendar = hd.build_panel(database, end=end, symbols=symbols)
    events = extra_financial_events(load_extra_fundamentals(database, end=end, symbols=symbols))
    panel = join_extra_financials(panel, events)
    block = price_block(prices, load_market_returns(database, end=end))
    panel = panel.merge(block, on=["symbol", "date"], how="left", validate="one_to_one")
    panel = add_value_features(panel)
    if cross_sectional:
        panel = add_peer_ranks(panel)
        panel = add_value_composite(panel)
    return panel, prices, calendar


if __name__ == "__main__":
    ev = extra_financial_events(load_extra_fundamentals(hd.CLASS_DB))
    blanked = ev.loc[ev.filter(like="_blanked").any(axis=1)]
    print("filings blanked as implausible:", len(blanked), "of", len(ev))
    print(blanked[["symbol", "fiscal_date"]].groupby("symbol").size().to_dict())
    panel, _, _ = build_feature_panel(hd.CLASS_DB)
    elig = panel.loc[panel.eligible]
    print("rows", len(panel), "| eligible", len(elig))
    vflags = [f"{c}_blanked" for c in VALUE_METRICS]
    print("eligible rows failing each value bound:", elig[vflags].sum().to_dict())
    sus = elig.groupby("symbol").mcap_suspect.mean()
    print("share of eligible rows with mcap_suspect, by symbol (>0 only):")
    print(sus[sus > 0].round(3).sort_values(ascending=False).to_dict())
    print("missing share among eligible rows:")
    print(elig[NEW_FINANCIAL + MODEL_FINANCIAL_4A + PRICE_4B + VALUE_METRICS].isna().mean().round(3))
    print(elig[NEW_FINANCIAL + MODEL_FINANCIAL_4A + PRICE_4B + ["mcap"]].describe().T.round(3))
    print("median beta by symbol:", elig.groupby("symbol").beta_252.median().round(2).to_dict())
