"""Data loading, features, labels and the model for HW4.

The overall plan follows the Lecture 7/8 research notebook and
`Lectures/modeling/course_data.py`: features at the signal close, entry at the
next session's close, exit at the first close on/after 90 calendar days, and
financials usable from filing date + 1 day. The code below is my own. It works
on wide (date x symbol) arrays so the 3,849-stock database fits in memory and
runs in minutes. The parts I adapted from the lecture are marked "(lecture)".

Signals are made once a week, at the last session of each week (plus the
session just before the test starts, so the first buy is at the first test close).
"""

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[3]
FULL_DB = REPO / "Lectures/modeling/data/equities_top2000.duckdb"
CLASS_DB = REPO / "Lectures/modeling/data/equities_class.duckdb"

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5  # sessions; same label rule as the lecture

PRICE_FEATURES = ["mom_1m", "mom_3m", "mom_6m_skip1m", "mom_12m_skip1m", "vol_3m", "vol_12m",
                  "drawdown_12m", "max_ret_1m", "beta_12m", "idio_vol_12m", "log_dollar_volume"]
# Not a feature: log_size (close x reported shares). The audit shows the two use
# different split bases for many stocks, so it ranks penny stocks above Apple.
FUNDAMENTAL_FEATURES = ["roa", "accruals", "earnings_yield", "sales_yield", "fcf_yield",
                        "debt_to_assets", "cash_to_assets", "asset_growth", "sales_growth"]
FEATURES = PRICE_FEATURES + FUNDAMENTAL_FEATURES


# ---------------------------------------------------------------- loading
class MarketData:
    """Wide arrays on the market calendar. NaN = no quote. Never filled."""

    def __init__(self, database=FULL_DB, end="2026-09-30"):
        database = Path(database)
        if not database.is_file():
            raise FileNotFoundError(f"Download the database from bCourses to {database}")
        with duckdb.connect(str(database), read_only=True) as con:
            bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date",
                                [end]).fetchdf()
            self.calendar = pd.DatetimeIndex(pd.to_datetime(bench.date))
            self.benchmark = pd.Series(bench.adj_close.to_numpy(), index=self.calendar, name="^SP500TR")
            symbols = con.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchdf().symbol
            self.symbols = pd.Index(symbols)
            # Map symbols and dates to integer positions inside DuckDB (fast for 15.8M rows).
            con.register("sym", pd.DataFrame({"symbol": symbols, "j": np.arange(len(symbols))}))
            con.register("cal", pd.DataFrame({"date": self.calendar, "i": np.arange(len(self.calendar))}))
            off = con.execute("""SELECT count(*) FROM prices p WHERE p.date <= ?
                                 AND p.date NOT IN (SELECT CAST(date AS DATE) FROM cal)""", [end]).fetchone()[0]
            rows = con.execute("""SELECT c.i, s.j, p.close, p.adj_close, p.volume
                FROM prices p JOIN sym s USING (symbol) JOIN cal c ON CAST(c.date AS DATE) = p.date
                WHERE p.date <= ?""", [end]).fetchnumpy()
            self.membership = con.execute("SELECT * FROM universe_membership ORDER BY quarter_start, universe_rank").fetchdf()
            self.universe = con.execute("SELECT * FROM universe ORDER BY symbol").fetchdf()
            self.fundamentals = con.execute("SELECT * FROM fundamentals ORDER BY symbol, fiscal_date").fetchdf()

        shape = (len(self.calendar), len(self.symbols))
        adj_raw = np.full(shape, np.nan)
        self.close = np.full(shape, np.nan, dtype=np.float32)
        self.volume = np.full(shape, np.nan, dtype=np.float32)
        i, j = rows["i"].astype(int), rows["j"].astype(int)
        adj = np.asarray(rows["adj_close"], dtype=float)
        bad = ~np.isfinite(adj) | (adj <= 0)
        adj_raw[i[~bad], j[~bad]] = adj[~bad]
        self.close[i, j] = np.asarray(rows["close"], dtype=float)
        self.volume[i, j] = np.asarray(rows["volume"], dtype=float)
        self.quality = {"off_calendar_rows_dropped": int(off), "invalid_adj_close_rows_dropped": int(bad.sum())}
        if off or bad.any():
            warnings.warn(f"Quarantined rows: {self.quality}", stacklevel=2)
        # Everything downstream (features, labels, the backtester) uses the repaired prices.
        self.adj, self.repairs = clean_prices(adj_raw, self.close, self.volume, self.symbols, self.calendar,
                                              self.benchmark.pct_change().to_numpy())
        del adj_raw

        for c in ["quarter_start", "quarter_end", "selection_date"]:
            if c in self.membership:
                self.membership[c] = pd.to_datetime(self.membership[c])
        for c in ["fiscal_date", "filing_date"]:
            self.fundamentals[c] = pd.to_datetime(self.fundamentals[c])

    def prices_frame(self):
        return pd.DataFrame(self.adj, index=self.calendar, columns=self.symbols)


# ---------------------------------------------------------------- price repair
# No 3:2 here: a one-day 33% drop is too common in real crashes to call it a split.
SPLIT_RATIOS = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 100])


def clean_prices(adj, close, volume, symbols, calendar, market_returns):
    """Repair the four kinds of broken adjusted closes found in the audit.

    1. Zero-volume quote -> missing. No shares traded, so nobody could buy or
       sell at that price (AAM has $0.41 quotes with no volume next to real
       $20,000 trades).
    2. Frozen adjusted close -> missing. adj_close repeats while the raw close
       moves (CSC sits at $10.04 for over 1,000 days).
    3. One-day spike -> missing. A move of more than 3x that reverses at the
       next quote is a bad print.
    4. Unadjusted split -> rescale the earlier history. A one-day move within
       5% of a split ratio k, where share volume moves the other way by about
       k and stays there, is a split the data vendor missed (BKNG 25:1 and DD
       1:3 in 2026, ANET 4:1 in 2024). Earlier prices are divided by k, which
       is what adj_close is supposed to do. Days when the S&P 500 moved more
       than 4% are skipped, because crash-day volume can look like a split.

    Rules 3 and 4 look a few weeks past the event, so this is a one-time data
    repair, not something the strategy does on the day. That is fine because
    the facts it recovers (a split happened, a print was wrong) were public at
    the time. It never drops a stock or invents a price: it only blanks or
    rescales quotes. Returns the repaired array and a table of what changed.
    """
    adj = adj.copy()
    events = []
    calm_market = np.abs(np.nan_to_num(market_returns)) < 0.04
    for j in range(adj.shape[1]):
        rows = np.flatnonzero(np.isfinite(adj[:, j]))
        if len(rows) < 2:
            continue
        a, c, v = adj[rows, j], close[rows, j].astype(float), volume[rows, j].astype(float)

        # 1-2: quotes with no trading, and adjusted prices stuck while the raw close moves
        zero = v == 0
        frozen = np.r_[False, (a[1:] == a[:-1]) & (np.abs(c[1:] / c[:-1] - 1) > 1e-3)]
        drop = zero | frozen
        if zero.any():
            events.append((symbols[j], calendar[rows[zero][0]], "zero_volume_quote", int(zero.sum()), np.nan))
        if frozen.any():
            events.append((symbols[j], calendar[rows[frozen][0]], "frozen_adj_close", int(frozen.sum()), np.nan))
        rows, a, v = rows[~drop], a[~drop], v[~drop]
        if len(rows) < 3:
            adj[:, j] = np.nan
            adj[rows, j] = a
            continue

        # 3: a big move that the next quote undoes
        r = np.diff(np.log(a))
        spike = np.r_[False, (np.abs(r[:-1]) > np.log(3)) & (np.abs(r[:-1] + r[1:]) < np.log(1.5)), False]
        for k in np.flatnonzero(spike):
            events.append((symbols[j], calendar[rows[k]], "one_day_spike", 1, a[k] / a[k - 1]))
        rows, a, v = rows[~spike], a[~spike], v[~spike]

        # 4: splits the vendor did not adjust for. A crash can also halve a
        # price and raise volume, so a split must pass two volume tests:
        #   - share volume over the next months moves by about the split ratio, and
        #   - dollars traded on the day itself look normal (a crash day spikes).
        r = np.diff(np.log(a))
        for k in np.flatnonzero(np.abs(r) > np.log(1.4)) + 1:  # k = first quote after the jump
            size = np.exp(abs(r[k - 1]))
            ratio = SPLIT_RATIOS[np.argmin(np.abs(np.log(size / SPLIT_RATIOS)))]
            before, after = slice(max(0, k - 60), max(0, k - 5)), slice(k + 5, k + 60)
            if abs(np.log(size / ratio)) > np.log(1.05) or k - 5 < 20 or len(v[after]) < 20:
                continue
            if not (calm_market[rows[k - 1] + 1:rows[k] + 1]).all():
                continue
            if np.median(v[before]) <= 0 or np.median(v[after]) <= 0:
                continue
            volume_move = np.log(np.median(v[after]) / np.median(v[before]))
            dollars_today = a[k] * v[k] / np.median(a[before] * v[before])
            if (abs(volume_move + np.sign(r[k - 1]) * np.log(ratio)) < 0.4 * np.log(ratio)
                    and 0.4 < dollars_today < 2.5):
                factor = ratio if r[k - 1] > 0 else 1 / ratio
                a[:k] *= factor
                events.append((symbols[j], calendar[rows[k]], "unadjusted_split", 1, factor))

        adj[:, j] = np.nan
        adj[rows, j] = a
    repairs = pd.DataFrame(events, columns=["symbol", "date", "repair", "rows", "factor"])
    return adj, repairs


TEST_START = pd.Timestamp("2016-04-01")


def weekly_signals(calendar, start, end):
    """Last market session of each week: features at its close, entry next session.

    I also add the session before the test starts (Thu Mar 31, 2016), so the
    strategy's first purchase is at the same close as the benchmark's.
    """
    cal = pd.Series(calendar, index=calendar)
    last = set(cal.groupby(calendar.to_period("W-FRI")).max())
    if TEST_START in calendar:  # the class database ends in 2015
        last.add(calendar[calendar.get_loc(TEST_START) - 1])
    last = sorted(d for d in last if pd.Timestamp(start) <= d <= pd.Timestamp(end))
    # Need a next session to enter on.
    return pd.DatetimeIndex([d for d in last if calendar.get_loc(d) + 1 < len(calendar)])


# ---------------------------------------------------------------- features
def price_features(md, signal_dates):
    """Trailing price/volume features at each signal close (past data only)."""
    px = pd.DataFrame(md.adj, index=md.calendar, columns=md.symbols)
    ret = px.pct_change(fill_method=None)
    rows = md.calendar.get_indexer(signal_dates)
    out = {}

    def take(frame):  # keep only signal-date rows to save memory
        return frame.iloc[rows].to_numpy()

    out["mom_1m"] = take(px / px.shift(21) - 1)
    out["mom_3m"] = take(px / px.shift(63) - 1)
    out["mom_6m_skip1m"] = take(px.shift(21) / px.shift(126) - 1)
    out["mom_12m_skip1m"] = take(px.shift(21) / px.shift(252) - 1)
    out["vol_3m"] = take(ret.rolling(63, min_periods=60).std() * np.sqrt(252))
    out["vol_12m"] = take(ret.rolling(252, min_periods=200).std() * np.sqrt(252))
    out["drawdown_12m"] = take(px / px.rolling(252, min_periods=200).max() - 1)
    out["max_ret_1m"] = take(ret.rolling(21, min_periods=15).max())

    # Beta and idiosyncratic volatility against the S&P 500 TR, using only days the stock traded.
    mkt = md.benchmark.pct_change().to_numpy()[:, None]
    have = ret.notna()
    m = pd.DataFrame(np.where(have, mkt, np.nan), index=px.index, columns=px.columns)
    roll = dict(window=252, min_periods=200)
    mean_r, mean_m = ret.rolling(**roll).mean(), m.rolling(**roll).mean()
    cov = (ret * m).rolling(**roll).mean() - mean_r * mean_m
    var_m = (m * m).rolling(**roll).mean() - mean_m ** 2
    var_r = (ret * ret).rolling(**roll).mean() - mean_r ** 2
    beta = cov / var_m
    out["beta_12m"] = take(beta)
    out["idio_vol_12m"] = take(np.sqrt((var_r - beta ** 2 * var_m).clip(lower=0) * 252))
    del m, mean_r, mean_m, cov, var_m, var_r, beta

    # close and volume are both split-adjusted to the same basis (checked in the
    # audit: AAPL 2016 gives ~$2.8B a day), so their product is dollars traded.
    dv = pd.DataFrame(md.close.astype(float) * md.volume.astype(float), index=px.index, columns=px.columns)
    out["log_dollar_volume"] = take(np.log1p(dv.rolling(63, min_periods=40).mean()))
    out["dollar_volume_3m"] = take(dv.rolling(63, min_periods=40).mean())
    # Data-quality screens (trailing only), a backstop for what clean_prices
    # can't repair: frequent days with no trading, and one-day jumps so large
    # they are almost always a broken price series (e.g. two securities spliced).
    zero_vol = pd.DataFrame(np.isfinite(md.close) & (md.volume == 0), index=px.index, columns=px.columns)
    out["zero_volume_days_3m"] = take(zero_vol.rolling(63, min_periods=1).sum())
    out["max_abs_log_move_12m"] = take(np.log(px / px.shift(1)).abs().rolling(252, min_periods=1).max())
    out["raw_close"] = md.close[rows].astype(float)
    out["has_quote"] = np.isfinite(md.adj[rows])
    out["n_returns_3m"] = take(have.rolling(63, min_periods=1).sum())

    n_dates, n_sym = len(signal_dates), len(md.symbols)
    panel = pd.DataFrame({k: v.reshape(-1) for k, v in out.items()})
    panel.insert(0, "symbol", np.tile(md.symbols.to_numpy(), n_dates))
    panel.insert(0, "date", np.repeat(signal_dates.to_numpy(), n_sym))
    return panel


def add_membership(panel, md):
    """(lecture rule) Use the list for the quarter of the *next-session* entry,
    and only if that list was selected on or before the signal date."""
    cal = md.calendar
    entry = pd.Series(cal[cal.get_indexer(panel.date) + 1], index=panel.index)
    panel["entry_date"] = entry.to_numpy()
    panel["membership_quarter"] = entry.dt.to_period("Q").dt.start_time.to_numpy()
    mem = md.membership.rename(columns={"quarter_start": "membership_quarter"})
    keep = ["symbol", "membership_quarter", "selection_date", "universe_rank"]
    if "market_cap_proxy" in mem:
        keep.append("market_cap_proxy")
    panel = panel.merge(mem[keep], on=["symbol", "membership_quarter"], how="left", validate="many_to_one")
    panel["member"] = panel.universe_rank.notna() & panel.selection_date.le(panel.date)
    if "market_cap_proxy" in panel:
        panel["log_size"] = np.log(panel.market_cap_proxy.where(panel.market_cap_proxy > 0))
    else:
        panel["log_size"] = np.nan
    return panel


def fundamental_events(f):
    """Quarterly records -> features dated by when they became public.

    (lecture) Use only quarterly rows with filing date after fiscal end, from
    filing date + 1 day. A late old period never replaces a newer one.
    TTM sums need four quarters inside about one year.
    """
    f = f.loc[f.period.eq("quarter")].sort_values(["symbol", "fiscal_date"]).copy()
    f = f.drop_duplicates(["symbol", "fiscal_date"], keep="last")
    g = f.groupby("symbol", sort=False)
    span_ok = (f.fiscal_date - g.fiscal_date.shift(3)).dt.days.le(370)
    for col in ["net_income", "revenue", "operating_cash_flow", "free_cash_flow"]:
        f[f"{col}_ttm"] = g[col].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(span_ok)
    year_ago_ok = (f.fiscal_date - g.fiscal_date.shift(4)).dt.days.between(330, 400)
    assets = f.total_assets.where(f.total_assets > 0)
    f["asset_growth"] = (assets / g.total_assets.shift(4).where(lambda x: x > 0) - 1).where(year_ago_ok)
    f["sales_growth"] = (f.revenue / g.revenue.shift(4).where(lambda x: x > 0) - 1).where(year_ago_ok)
    f["roa"] = f.net_income_ttm / assets
    f["accruals"] = (f.net_income_ttm - f.operating_cash_flow_ttm) / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets

    f = f.loc[f.filing_date.gt(f.fiscal_date)].copy()
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    f = f.loc[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    cols = ["roa", "accruals", "debt_to_assets", "cash_to_assets", "asset_growth", "sales_growth",
            "net_income_ttm", "revenue_ttm", "free_cash_flow_ttm", "shares"]
    f = f.rename(columns={SHARES_COLUMN: "shares"}) if SHARES_COLUMN != "shares" else f
    f[cols] = f[cols].replace([np.inf, -np.inf], np.nan)
    return f[["symbol", "fiscal_date", "available_date"] + cols]


SHARES_COLUMN = "shares_outstanding"


def add_fundamentals(panel, events):
    """As-of join: the latest filing available on or before each signal date."""
    left = panel.sort_values("date")
    right = events.sort_values("available_date")
    out = pd.merge_asof(left, right, left_on="date", right_on="available_date", by="symbol",
                        direction="backward")
    assert (out.available_date.isna() | out.available_date.le(out.date)).all()
    stale = (out.date - out.fiscal_date).dt.days.gt(550)  # (lecture) blank very old reports
    mcap = out.raw_close * out.shares.where(out.shares > 0)
    out["earnings_yield"] = out.net_income_ttm / mcap
    out["sales_yield"] = out.revenue_ttm / mcap
    out["fcf_yield"] = out.free_cash_flow_ttm / mcap
    out.loc[stale, FUNDAMENTAL_FEATURES] = np.nan
    out[FUNDAMENTAL_FEATURES] = out[FUNDAMENTAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------- labels
def add_labels(panel, md):
    """Gross return from the next-session close to the first close on/after
    entry + 90 calendar days (at most 5 sessions late). Missing = censored."""
    cal = md.calendar
    out = []
    for date, grp in panel.groupby("date", sort=True):
        e = cal.get_loc(date) + 1
        due = cal[e] + pd.Timedelta(days=HOLDING_DAYS)
        k = cal.searchsorted(due)
        cols = md.symbols.get_indexer(grp.symbol)
        entry_px = md.adj[e, cols]
        exit_px = np.full(len(cols), np.nan)
        exit_i = np.full(len(cols), -1)
        for step in range(MAX_EXIT_DELAY + 1):
            if k + step >= len(cal):
                break
            px = md.adj[k + step, cols]
            fill = np.isnan(exit_px) & np.isfinite(px)
            exit_px[fill], exit_i[fill] = px[fill], k + step
        g = grp.copy()
        g["entry_price"] = entry_px
        g["label_end"] = pd.DatetimeIndex([cal[x] if x >= 0 else pd.NaT for x in exit_i])
        g["target"] = exit_px / entry_px - 1
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_panel(md, start="2009-12-01", end="2026-09-30"):
    signal_dates = weekly_signals(md.calendar, start, end)
    panel = price_features(md, signal_dates)
    panel = add_membership(panel, md)
    # Rows we can ever trade or train on: a dated member with a quote and 3 months of returns.
    panel["eligible"] = panel.member & panel.has_quote & panel.n_returns_3m.ge(60)
    panel = panel.loc[panel.eligible].copy()
    panel = add_fundamentals(panel, fundamental_events(md.fundamentals))
    panel = add_labels(panel, md)
    panel = panel.merge(md.universe[["symbol", "sector"]], on="symbol", how="left")
    panel["sector"] = panel.sector.fillna("Unknown")
    return panel


# ---------------------------------------------------------------- model
def rank_features(panel, features=FEATURES):
    """Per-date percentile ranks in [-0.5, 0.5] among that date's eligible stocks.

    This uses only same-date data, so nothing is learned from the future, and
    it makes features comparable across years. Missing stays missing.
    """
    ranked = panel.groupby("date")[features].rank(pct=True) - 0.5
    return ranked


def make_target(panel, kind):
    """What the model learns. 'rank' = per-date rank of the 90-day return;
    'excess' = return minus that date's median, clipped to [-1, 1]."""
    if kind == "rank":
        return panel.groupby("date").target.rank(pct=True) - 0.5
    med = panel.groupby("date").target.transform("median")
    return (panel.target - med).clip(-1, 1)


def make_model(name):
    if name == "ridge":
        return Ridge(alpha=1000.0)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                             min_samples_leaf=500, l2_regularization=1.0,
                                             early_stopping=False, random_state=0)
    raise ValueError(name)


def fit_predict(train, test, X_train, X_test, model_name, target_kind):
    y = make_target(train, target_kind)
    model = make_model(model_name)
    if model_name == "ridge":  # Ridge can't take NaN: use the middle rank (a fixed rule, not learned)
        X_train, X_test = X_train.fillna(0.0), X_test.fillna(0.0)
    model.fit(X_train, y)
    return model.predict(X_test), model


def walk_forward(panel, X, fit_dates, model_name, target_kind, train_start="2010-01-01"):
    """Fit at each fit date on outcomes that ended *before* it; predict until the next fit."""
    preds = pd.Series(np.nan, index=panel.index)
    fits = []
    for k, fit_date in enumerate(fit_dates):
        stop = fit_dates[k + 1] if k + 1 < len(fit_dates) else pd.Timestamp.max
        tr = panel.date.ge(train_start) & panel.label_end.lt(fit_date) & panel.target.notna()
        te = panel.date.ge(fit_date) & panel.date.lt(stop)
        if not te.any():
            continue
        assert panel.loc[tr, "label_end"].max() < panel.loc[te, "date"].min()
        p, _ = fit_predict(panel.loc[tr], panel.loc[te], X.loc[tr], X.loc[te], model_name, target_kind)
        preds.loc[te] = p
        fits.append({"fit_date": fit_date, "train_rows": int(tr.sum()),
                     "last_label_end": panel.loc[tr, "label_end"].max(), "predict_rows": int(te.sum())})
    return preds, pd.DataFrame(fits)


def signal_quality(panel, preds, top_n=50):
    """Per-signal-date rank IC and the top-N basket's average 90-day return vs all names (clipped)."""
    df = panel.loc[preds.notna() & panel.target.notna(), ["date", "target"]].assign(pred=preds)
    ic = df.groupby("date").apply(
        lambda g: g.pred.corr(g.target, method="spearman") if g.pred.nunique() > 1 else np.nan,
        include_groups=False).dropna()  # a month with identical predictions has no ranking to score
    # Clip each month's returns at its 1st/99th percentile so a broken price
    # (the data has 90-day "returns" above 100,000x) can't dominate the average.
    lo, hi = (df.groupby("date").target.transform("quantile", q) for q in (0.01, 0.99))
    df["target"] = df.target.clip(lo, hi)
    top = df.sort_values("pred", ascending=False).groupby("date").head(top_n).groupby("date").target.mean()
    allm = df.groupby("date").target.mean()
    return pd.Series({"signal_dates": len(ic), "mean_rank_IC": ic.mean(), "IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
                      "share_dates_IC>0": (ic > 0).mean(),
                      f"top{top_n}_avg_90d_return": top.mean(), "all_avg_90d_return": allm.mean(),
                      f"top{top_n}_minus_all": (top - allm).mean()})


# ---------------------------------------------------------------- strategy
class WeeklyTopN:
    """Each week, buy the top-N predicted stocks with 1/14 of NAV.

    A lot bought on Monday is due 90 days later and sells on the Monday of
    week 13, so about 13 weekly cohorts are open when the next order is
    planned. 1/14 per cohort keeps cash for that order, so the account stays
    roughly 93% invested. (With monthly cohorts of 1/3, the exiting cohort's
    cash came back a day or two *after* each month-end decision and sat idle
    for a month: 25% average cash in validation.)
    """

    def __init__(self, scores, sectors, top_n=25, cohort_fraction=1 / 14, name_cap=0.05,
                 sector_cap=0.30):
        self.scores = {d: g.set_index("symbol").score.sort_values(ascending=False)
                       for d, g in scores.groupby("date")}
        self.sectors = sectors
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.name_cap = name_cap
        self.sector_cap = sector_cap

    def __call__(self, account):
        scores = self.scores.get(account.date)
        if scores is None or not len(scores):
            return pd.Series(dtype=float)
        budget = min(account.cash, account.nav * self.cohort_fraction)
        each = budget / self.top_n
        held = account.holdings.to_dict()
        held_sector = account.holdings.groupby(self.sectors.reindex(account.holdings.index).fillna("Unknown")).sum().to_dict()
        orders = {}
        for symbol in scores.index:  # best first
            if len(orders) == self.top_n or budget <= 1e-6:
                break
            sector = self.sectors.get(symbol, "Unknown")
            room = min(self.name_cap * account.nav - held.get(symbol, 0.0),
                       self.sector_cap * account.nav - held_sector.get(sector, 0.0))
            amount = min(each, budget, max(room, 0.0))
            if amount < 0.5 * each:  # skip instead of buying a tiny sliver
                continue
            orders[symbol] = amount
            budget -= amount
            held_sector[sector] = held_sector.get(sector, 0.0) + amount
        return pd.Series(orders, dtype=float)
