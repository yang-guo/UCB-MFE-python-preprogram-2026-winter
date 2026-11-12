"""Load the HW4 DuckDB database into memory-friendly wide (date x symbol) arrays.

Design notes
------------
* The full database has ~15.8M price rows. Building a long pandas frame and
  pivoting it would need several GB, so we let DuckDB return integer row/column
  codes and fill float32 numpy matrices directly.
* Rows are the supplied market calendar (the ^SP500TR sessions). Price rows on
  dates outside that calendar (28 rows in the delivery) and nonpositive adjusted
  prices (275 BHRB rows) are quarantined, exactly like the lecture loader
  (Lectures/modeling/course_data.py::load_data). Gaps stay NaN; we never fill
  prices.
* Quarterly universe membership is turned into a dated boolean mask using the
  lecture convention (course_data.apply_membership): a signal after the close
  on day t is for an entry on day t+1, so it uses the quarter list of t+1.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = REPO_ROOT / "Lectures/modeling/data/equities_top2000.duckdb"
EARLY_MONTHLY_START = "2004-01-01"   # 5 years before the first feature date we use
CLASS_DB = REPO_ROOT / "Lectures/modeling/data/equities_class.duckdb"


@dataclass
class MarketData:
    calendar: pd.DatetimeIndex      # market sessions (rows of every matrix)
    symbols: pd.Index               # columns of every matrix
    adj_close: np.ndarray           # float32 [T, N]; NaN = no quote
    dollar_volume: np.ndarray       # float32 [T, N]; close * volume (split-invariant)
    member: np.ndarray              # bool [T, N]; in the universe for an entry on t+1
    sector: pd.Series               # snapshot sector by symbol (risk buckets only)
    benchmark: pd.Series            # ^SP500TR adjusted close on the calendar
    fundamentals: pd.DataFrame      # raw quarterly records (long)
    repairs: pd.DataFrame           # audit log of price repairs (see clean_prices)
    close_location: np.ndarray | None = None   # float32 [T, N]; (C-L)/(H-L) of the raw daily bar
    industry: pd.Series | None = None          # snapshot industry by symbol (peer groups)
    early_monthly: pd.DataFrame | None = None  # month-end adjusted closes before `start` (long lookbacks)
    dup_copy: np.ndarray | None = None         # bool [T, N]; a second ticker for the same company at t

    def loc_date(self, date):
        return self.calendar.get_loc(pd.Timestamp(date))


def load_market_data(database=DEFAULT_DB, start="2008-10-01", end="2026-09-30"):
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Download the HW4 database from bCourses to {database}")
    with duckdb.connect(str(database), read_only=True) as con:
        cal = con.execute("SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
                          [start, end]).fetchdf()
        calendar = pd.DatetimeIndex(pd.to_datetime(cal.date)).astype("datetime64[ns]")
        universe = con.execute("SELECT symbol, sector, industry FROM universe ORDER BY symbol").fetchdf()
        symbols = pd.Index(universe.symbol)
        T, N = len(calendar), len(symbols)

        # Integer codes are computed in SQL so Python never holds 10M+ strings.
        arrays = con.execute("""
            WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS t
                         FROM calendar WHERE date BETWEEN ? AND ?),
                 sym AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS n FROM universe)
            SELECT cal.t::INTEGER AS t, sym.n::INTEGER AS n, p.adj_close, p.close, p.volume::DOUBLE AS volume,
                   p.high, p.low
            FROM prices p JOIN cal USING (date) JOIN sym USING (symbol)
        """, [start, end]).fetchnumpy()
        off_calendar = con.execute("""SELECT count(*) FROM prices p WHERE date BETWEEN ? AND ?
            AND date NOT IN (SELECT date FROM calendar)""", [start, end]).fetchone()[0]

        # Month-end adjusted closes before the daily window, only for 2-5 year
        # lookbacks. Uses the last traded (volume > 0) positive quote of each month.
        early = con.execute("""SELECT symbol, last_day(date) AS month_end, arg_max(adj_close, date) AS px
            FROM prices WHERE date >= ? AND date < ? AND adj_close > 0 AND volume > 0
            GROUP BY 1, 2""", [EARLY_MONTHLY_START, start]).fetchdf()
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [start, end]).fetchdf()
        membership = con.execute("""SELECT quarter_start, selection_date, symbol FROM universe_membership
            WHERE quarter_start <= ?""", [end]).fetchdf()
        fundamentals = con.execute("""SELECT symbol, fiscal_date, period, filing_date, revenue, gross_profit,
            operating_income, net_income, operating_cash_flow, free_cash_flow, total_assets, total_debt,
            total_equity, cash_and_equivalents, dividends_paid, common_stock_issued,
            common_stock_repurchased, shares_outstanding, long_term_debt, short_term_debt,
            total_current_assets, total_current_liabilities, interest_expense FROM fundamentals WHERE period = 'quarter'""").fetchdf()

    t, n = arrays["t"], arrays["n"]
    adj = np.asarray(arrays["adj_close"], dtype=float)
    valid = np.isfinite(adj) & (adj > 0)
    print(f"Loaded {len(adj):,} on-calendar price rows; quarantined {int((~valid).sum())} invalid "
          f"adjusted prices and {off_calendar} off-calendar rows.")
    def wide(values, rows):
        a = np.full((T, N), np.nan, dtype=np.float32)
        a[t[rows], n[rows]] = values[rows]
        return a

    adj_close = wide(adj, valid)
    close = wide(np.asarray(arrays["close"], dtype=float), valid)
    volume = wide(np.asarray(arrays["volume"], dtype=float), valid)
    clv_long = close_location(arrays["close"], arrays["high"], arrays["low"])
    clv = wide(clv_long, valid)
    del arrays, adj, clv_long
    adj_close, repairs = clean_prices(adj_close, close, volume, calendar, symbols)
    # close*volume is the traded dollar amount that day (split-invariant when
    # both fields share a basis). Quarantined quotes have no dollar volume.
    dollar_volume = close * volume
    dollar_volume[np.isnan(adj_close)] = np.nan
    clv[np.isnan(adj_close)] = np.nan          # no location for quarantined / unconfirmed quotes
    del close, volume

    benchmark = pd.Series(bench.adj_close.to_numpy(), index=pd.to_datetime(bench.date).astype("datetime64[ns]"))
    assert benchmark.index.equals(calendar), "Benchmark must cover every calendar session"

    member = membership_mask(membership, calendar, symbols)
    sector = universe.set_index("symbol").sector.fillna("Unknown")
    industry = universe.set_index("symbol").industry.fillna("Unknown")
    early["month_end"] = pd.to_datetime(early.month_end).astype("datetime64[ns]")
    early_monthly = early.pivot(index="month_end", columns="symbol", values="px").reindex(columns=symbols)
    for column in ["fiscal_date", "filing_date"]:
        fundamentals[column] = pd.to_datetime(fundamentals[column]).astype("datetime64[ns]")
    dup_copy, dup_pairs = duplicate_copies(adj_close, symbols)
    print(f"Duplicate tickers: {len(dup_pairs)} candidate pairs; "
          f"{int(dup_copy.any(axis=0).sum())} tickers dropped as a copy on {int(dup_copy.sum()):,} stock-days.")
    return MarketData(calendar, symbols, adj_close, dollar_volume, member, sector, benchmark, fundamentals,
                      repairs, clv, industry, early_monthly, dup_copy)


def duplicate_copies(adj_close, symbols, window=63, min_share=0.95, min_active=20, move=0.003,
                     min_hits=40, decimals=5, abs_tol=2e-4, rel_tol=0.03):
    """Point-in-time detection of one company listed under two tickers.

    The database sometimes carries both the old and the new ticker of a
    renamed company with the same price history, and both can be universe
    members at once (e.g. DISCA/WBD, AXL/DCH, JCOM/ZD). At session t, two
    tickers count as the same company if their daily returns over the past
    `window` sessions match (|r_a - r_b| <= abs_tol + rel_tol * |r_a|; the
    two copies' adjustment factors can differ slightly) on at least
    `min_share` of the days either moved more than `move` (and there were at
    least `min_active` such days). The ticker with the earlier first quote is
    kept (alphabetical on a tie), so the choice never flips; the other is
    flagged as a copy at t.

    Candidate pairs are those with >= `min_hits` identical-return days (to
    `decimals` places) over
    the whole sample. This only limits which pairs are checked: a flag at t
    still requires the past-`window` rule, so it can miss a duplicate but
    never adds a flag the point-in-time rule would not.
    """
    px = adj_close.astype(np.float64)
    ret = np.full_like(px, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = px[1:] / px[:-1] - 1
    key = np.round(ret, decimals)
    t_idx, n_idx = np.nonzero(np.isfinite(ret) & (np.abs(ret) > move))
    k = key[t_idx, n_idx]
    order = np.lexsort((n_idx, k, t_idx))
    t_s, k_s, n_s = t_idx[order], k[order], n_idx[order]
    same = (t_s[1:] == t_s[:-1]) & (k_s[1:] == k_s[:-1])
    a, b = n_s[:-1][same], n_s[1:][same]                     # adjacent members of equal-return groups
    codes, hits = np.unique(np.minimum(a, b) * len(symbols) + np.maximum(a, b), return_counts=True)
    pairs = [(int(c // len(symbols)), int(c % len(symbols))) for c in codes[hits >= min_hits]]

    T, N = px.shape
    first = np.where(np.isfinite(px).any(axis=0), np.argmax(np.isfinite(px), axis=0), T)
    copy = np.zeros((T, N), dtype=bool)
    kernel = np.ones(window)
    for i, j in pairs:
        ri, rj = ret[:, i], ret[:, j]
        both = np.isfinite(ri) & np.isfinite(rj)
        active = both & ((np.abs(ri) > move) | (np.abs(rj) > move))
        match = active & (np.abs(ri - rj) <= abs_tol + rel_tol * np.abs(ri))
        n_act = np.convolve(active, kernel)[:T]                 # trailing window sums (causal)
        n_match = np.convolve(match, kernel)[:T]
        dup = (n_act >= min_active) & (n_match >= min_share * np.maximum(n_act, 1))
        keep_i = (first[i], symbols[i]) <= (first[j], symbols[j])
        copy[:, j if keep_i else i] |= dup
    return copy, pairs


SPLIT_RATIOS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 20, 25, 30, 40, 50, 100], dtype=float)


def close_location(close, high, low):
    """Where the close sits in the day's range: 0 = at the low, 1 = at the high.

    Uses one day's raw high/low/close, so split and dividend adjustment cancel.
    NaN when the range is empty (H <= L) or the bar is inconsistent (close
    outside [low, high], or any field missing).
    """
    close, high, low = (np.asarray(a, dtype=float) for a in (close, high, low))
    with np.errstate(invalid="ignore", divide="ignore"):
        loc = (close - low) / (high - low)
    ok = (high > low) & (close >= low) & (close <= high)
    return np.where(ok, loc, np.nan)


def clean_prices(px, close, volume, calendar, symbols, big_move=1.8, revert=1.3, confirm=1.25,
                 confirm_quotes=2, max_pending=5, split_tol=0.06):
    """Point-in-time repair of price-basis errors in the delivered adjusted prices.

    The vendor series contains (see README "Data repairs"):
      1. zero-volume quotes that belong to a different security (AAM, TIC);
      2. short spikes that revert (e.g. AAM/TIC interleavings, GCT 2022);
      3. stock splits that were never back-adjusted (NFLX, ORLY, BKNG, ...);
      4. breaks in the adjustment factor (KDP 2018-07-10: adj x6.9, close x1.1).

    Causality: every decision about the quote on day d uses raw data up to d
    only. A large move is *not* trusted on the day it appears; it is
    "unconfirmed" (treated as no quote: no fill, no mark, no feature) until
    later quotes resolve it, and it is only accepted from the resolving
    quote onward. So the cleaned value on any date is the same whether we
    clean the full history or only the history available on that date, up to
    a constant rescaling of earlier prices that leaves every return unchanged
    (checked in test_backtester.PriceRepairs.test_cleaning_is_prefix_invariant).

    Per stock, scanning quotes in date order against the last *accepted* quote L:
      * zero volume -> no quote (no trade happened that day);
      * factor break: close moves < 1.8x but adj/close changes by > 50% ->
        rescale earlier prices so the day's return equals the close-to-close
        return (same-day information only), accept;
      * move < 1.8x vs L -> accept;
      * move >= 1.8x -> unconfirmed. Each later quote q then either
          - returns within 30% of L: the move was a spike; unconfirmed quotes
            stay removed and q is accepted;
          - confirms: the first unconfirmed level holds within 25% for two
            more quotes. If the move is within 6% of an integer split ratio
            and the split-day volume moved the opposite way (>= 1.5x its prior
            20-day median for a forward split, <= 0.7x for a reverse split),
            earlier prices are rescaled by the ratio, as a vendor does for a
            split; otherwise it is a genuine move. q is accepted;
          - after 5 unresolved quotes, q is accepted as a genuine move.
    """
    px = px.copy()
    T, N = px.shape
    zero = (volume == 0) & np.isfinite(px)
    px[zero] = np.nan
    log = []
    n_zero = int(zero.sum())

    # Only stocks with at least one large raw move or factor change need the scan.
    prev = pd.DataFrame(px).ffill().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.abs(np.log(px / prev))
        fac = px / close
        prev_fac = pd.DataFrame(np.where(np.isfinite(px), fac, np.nan)).ffill().shift(1).to_numpy()
        fac_jump = np.abs(np.log(fac / prev_fac))
    candidates = np.isfinite(jump) & ((jump >= np.log(big_move)) | (fac_jump >= np.log(1.5)))
    columns = np.unique(np.nonzero(candidates)[1])
    del prev, jump, fac, prev_fac, candidates
    # Median volume over the 20 sessions *before* each day (causal).
    vol_med = pd.DataFrame(volume[:, columns]).rolling(20, min_periods=5).median().shift(1).to_numpy()

    def record(d, j, event, factor):
        log.append({"date": calendar[d], "symbol": symbols[j], "event": event, "factor": float(factor)})

    for c_i, j in enumerate(columns):
        p, cl, v = px[:, j], close[:, j], volume[:, j]
        obs = np.flatnonzero(np.isfinite(p))
        if len(obs) < 2:
            continue
        last = obs[0]          # index of the last accepted quote
        pending = []           # indices of unconfirmed quotes
        pending_raw = []       # their raw prices (the cleaned array holds NaN)
        for d in obs[1:]:
            if not pending:
                r = p[d] / p[last]
                rc = cl[d] / cl[last] if np.isfinite(cl[d]) and np.isfinite(cl[last]) and cl[last] > 0 else np.nan
                if np.isfinite(rc) and max(rc, 1 / rc) < big_move and abs(np.log(r / rc)) >= np.log(1.5):
                    p[:d] *= r / rc
                    record(d, j, "adjustment_factor_break", r / rc)
                    last = d
                elif max(r, 1 / r) < big_move:
                    last = d
                else:
                    pending, pending_raw = [d], [p[d]]
                    p_first = p[d]
                    vr = v[d] / vol_med[d, c_i] if np.isfinite(vol_med[d, c_i]) and vol_med[d, c_i] > 0 else np.nan
                    first_volume_ratio = vr
                    p[d] = np.nan  # unconfirmed: not usable on its own date
                continue
            q = p[d]
            if abs(np.log(q / p[last])) < np.log(revert):
                record(pending[0], j, "spike_removed", p_first / p[last])
                pending, last = [], d
            elif (len(pending) >= confirm_quotes
                  and all(abs(np.log(x / p_first)) < np.log(confirm) for x in pending_raw[1:] + [q])):
                r = p_first / p[last]
                big = max(r, 1 / r)
                k = SPLIT_RATIOS[np.argmin(np.abs(np.log(big / SPLIT_RATIOS)))]
                volume_ok = (r < 1 and first_volume_ratio >= 1.5) or (r > 1 and first_volume_ratio <= 0.7)
                if abs(np.log(big / k)) <= np.log(1 + split_tol) and volume_ok:
                    ratio = 1 / k if r < 1 else k
                    p[:d] *= ratio
                    record(pending[0], j, "split_adjusted", ratio)
                else:
                    record(pending[0], j, "large_move_confirmed", r)
                pending, last = [], d
            elif len(pending) >= max_pending:
                record(pending[0], j, "large_move_accepted_unresolved", q / p[last])
                pending, last = [], d
            else:
                pending.append(d)
                pending_raw.append(q)
                p[d] = np.nan
        for d in pending:          # still unresolved at the end of the data
            p[d] = np.nan
        px[:, j] = p
    repairs = pd.DataFrame(log, columns=["date", "symbol", "event", "factor"])
    print(f"Price repairs: {n_zero:,} zero-volume quotes removed; "
          + ", ".join(f"{k}={v}" for k, v in repairs.event.value_counts().items()))
    return px, repairs


def membership_mask(membership, calendar, symbols):
    """member[t, n] is True when symbol n may be *bought* at the close of t+1.

    Signal date t uses the quarter of the next session (entry date). The list
    was selected on `selection_date`, which must be <= t (checked below).
    """
    membership = membership.reset_index(drop=True)
    membership["quarter_start"] = pd.to_datetime(membership.quarter_start)
    membership["selection_date"] = pd.to_datetime(membership.selection_date)
    if membership.duplicated(["quarter_start", "symbol"]).any():
        raise ValueError("Duplicate quarterly membership rows")
    T = len(calendar)
    entry_dates = pd.Series(calendar[1:].append(pd.DatetimeIndex([pd.NaT])))
    entry_quarter = entry_dates.dt.to_period("Q").dt.start_time.to_numpy()
    mask = np.zeros((T, len(symbols)), dtype=bool)
    col = symbols.get_indexer(membership.symbol)
    for quarter, rows in membership.groupby("quarter_start"):
        t_rows = np.flatnonzero(entry_quarter == np.datetime64(quarter))
        if not len(t_rows):
            continue
        # Availability check: list selected no later than the first signal using it.
        assert rows.selection_date.max() <= calendar[t_rows[0]], quarter
        cols = col[rows.index.to_numpy()]
        mask[np.ix_(t_rows, cols[cols >= 0])] = True
    return mask
