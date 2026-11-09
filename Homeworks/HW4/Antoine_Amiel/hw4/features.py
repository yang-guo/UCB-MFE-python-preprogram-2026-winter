"""Causal features, dated eligibility and 90-day labels.

Every feature for signal date t uses prices through the close of t, filings
available by t (filing date + 1 day) and the quarterly list known at t.
Labels are the only place where future prices are read; they are used as
training targets after their holding period has ended, never as inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketData

PRICE_FEATURES = ["rev_5", "mom_21", "mom_63", "mom_126", "mom_252_21", "vol_63", "vol_252",
                  "drawdown_252", "max_ret_21", "beta_252", "idio_vol_252", "log_dollar_volume_63"]
FUND_FEATURES = ["gross_margin", "operating_margin", "roa", "roe", "accruals", "fcf_to_assets",
                 "debt_to_assets", "cash_to_assets", "sales_growth", "asset_growth",
                 "earnings_yield", "book_to_market", "fcf_yield", "filing_age"]
OTHER_FEATURES = ["log_size", "universe_rank_pct"]
FEATURES = PRICE_FEATURES + FUND_FEATURES + OTHER_FEATURES


def _rolling_mean(frame, window, min_periods):
    return frame.rolling(window, min_periods=min_periods).mean()


def _price_features_block(px: pd.DataFrame, dv: pd.DataFrame, bench_ret: pd.Series) -> dict:
    """Trailing price features for a block of stocks (float64 inside the block)."""
    ret = px.pct_change(fill_method=None)
    out = {
        "rev_5": px.pct_change(5, fill_method=None),
        "mom_21": px.pct_change(21, fill_method=None),
        "mom_63": px.pct_change(63, fill_method=None),
        "mom_126": px.pct_change(126, fill_method=None),
        "mom_252_21": px.shift(21) / px.shift(252) - 1,
        "vol_63": ret.rolling(63, min_periods=63).std() * np.sqrt(252),
        "vol_252": ret.rolling(252, min_periods=200).std() * np.sqrt(252),
        "drawdown_252": px / px.rolling(252, min_periods=126).max() - 1,
        "max_ret_21": ret.rolling(21, min_periods=21).max(),
    }
    # Beta and idiosyncratic volatility vs the benchmark over 252 sessions,
    # using only the sessions where the stock itself has a return.
    b = pd.DataFrame(np.where(ret.notna(), bench_ret.to_numpy()[:, None], np.nan),
                     index=ret.index, columns=ret.columns)
    mx, my = _rolling_mean(ret, 252, 200), _rolling_mean(b, 252, 200)
    cov = _rolling_mean(ret * b, 252, 200) - mx * my
    beta = cov / (_rolling_mean(b * b, 252, 200) - my * my)
    var_x = _rolling_mean(ret * ret, 252, 200) - mx * mx
    out["beta_252"] = beta
    out["idio_vol_252"] = np.sqrt((var_x - beta * cov).clip(lower=0)) * np.sqrt(252)
    out["log_dollar_volume_63"] = np.log(dv.rolling(63, min_periods=42).median())
    return out


def price_feature_matrices(md: MarketData, block: int = 400) -> dict[str, pd.DataFrame]:
    """Wide float32 matrices of trailing price features; gaps stay missing (no padding).

    Computed in blocks of columns to bound memory on the full database.
    """
    shape = md.adj.shape
    store = {name: np.full(shape, np.nan, dtype="float32") for name in PRICE_FEATURES}
    bench_ret = md.benchmark.pct_change()
    for start in range(0, shape[1], block):
        sl = slice(start, start + block)
        px = md.adj.iloc[:, sl].astype("float64")
        dv = md.dollar_volume.iloc[:, sl].astype("float64")
        for name, frame in _price_features_block(px, dv, bench_ret).items():
            store[name][:, sl] = frame.replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float32")
    return {name: pd.DataFrame(a, index=md.adj.index, columns=md.adj.columns) for name, a in store.items()}


def fundamental_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Quarterly statement ratios dated by when they became public.

    Availability rule from course_data.financial_events: only filings dated
    after fiscal period end, usable from filing date + 1 day, and an old
    fiscal period filed late never replaces a newer known one.
    """
    f = fundamentals.loc[fundamentals.filing_date.gt(fundamentals.fiscal_date)].copy()
    f = f.sort_values(["symbol", "fiscal_date", "filing_date"]).drop_duplicates(["symbol", "fiscal_date"], keep="last")
    g = f.groupby("symbol", sort=False)

    def ttm(col):
        total = g[col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        span = (f.fiscal_date - g.fiscal_date.shift(3)).dt.days
        return total.where(span.between(250, 300))  # four consecutive quarters

    def yoy(col):
        prev, gap = g[col].shift(4), (f.fiscal_date - g.fiscal_date.shift(4)).dt.days
        return (f[col] / prev - 1).where(gap.between(340, 390) & prev.gt(0))

    assets = f.total_assets.where(f.total_assets > 0)
    equity = f.total_equity.where(f.total_equity > 0)
    rev_ttm = ttm("revenue").where(lambda s: s > 0)
    ni_ttm, ocf_ttm, fcf_ttm = ttm("net_income"), ttm("operating_cash_flow"), ttm("free_cash_flow")
    f["gross_margin"] = ttm("gross_profit") / rev_ttm
    f["operating_margin"] = ttm("operating_income") / rev_ttm
    f["roa"] = ni_ttm / assets
    f["roe"] = ni_ttm / equity
    f["accruals"] = (ni_ttm - ocf_ttm) / assets
    f["fcf_to_assets"] = fcf_ttm / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["sales_growth"] = yoy("revenue")
    f["asset_growth"] = yoy("total_assets")
    f["ni_ttm"], f["fcf_ttm"], f["equity"] = ni_ttm, fcf_ttm, equity
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    newest = f.groupby("symbol").fiscal_date.cummax()
    f = f.loc[f.fiscal_date.eq(newest)].drop_duplicates(["symbol", "available_date"], keep="last")
    cols = ["symbol", "fiscal_date", "available_date", "gross_margin", "operating_margin", "roa", "roe",
            "accruals", "fcf_to_assets", "debt_to_assets", "cash_to_assets", "sales_growth",
            "asset_growth", "ni_ttm", "fcf_ttm", "equity"]
    out = f[cols].copy()
    num = out.columns[3:]
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


class FeatureStore:
    """Holds wide feature matrices and produces cross-sections on demand."""

    def __init__(self, md: MarketData, holding_days: int = 90, max_exit_delay: int = 5):
        self.md = md
        self.calendar = md.calendar
        self.symbols = md.adj.columns
        self.price = price_feature_matrices(md)
        self.events = fundamental_events(md.fundamentals)
        self._build_labels(holding_days, max_exit_delay)
        self.membership = md.membership.copy()
        self.membership["quarter_start"] = self.membership.quarter_start.astype("datetime64[ns]")

    # ----- labels: next-session entry, first quote on/after 90 calendar days
    def _build_labels(self, holding_days, max_exit_delay):
        cal, adj = self.calendar, self.md.adj.to_numpy()
        n = len(cal)
        has = np.isfinite(adj)
        # Row of the first quote at or after each row, looking at most 5 sessions ahead.
        next_quote = np.full(adj.shape, np.nan, dtype="float32")
        for k in range(max_exit_delay, -1, -1):
            shifted = np.full(adj.shape, False)
            shifted[: n - k] = has[k:]
            next_quote = np.where(shifted, (np.arange(n) + k)[:, None].astype("float32"), next_quote)
        entry_row = np.arange(n) + 1
        target = np.full(adj.shape, np.nan, dtype="float32")
        label_end_row = np.full(adj.shape, np.nan, dtype="float32")
        for i in range(n - 1):
            due = cal[entry_row[i]] + pd.Timedelta(days=holding_days)
            j = cal.searchsorted(due)
            if j >= n:
                continue
            exit_row = next_quote[j]
            ok = has[entry_row[i]] & np.isfinite(exit_row)
            rows = exit_row[ok].astype(int)
            cols = np.flatnonzero(ok)
            target[i, cols] = adj[rows, cols].astype("float64") / adj[entry_row[i], cols] - 1
            label_end_row[i, cols] = rows
        self.target = target
        self.label_end_row = label_end_row

    def cross_section(self, dates, max_rank: int | None = None) -> pd.DataFrame:
        """Eligible stocks on each signal date, with raw and rank-transformed features.

        Eligible = in the quarterly list of the *entry* quarter (next session),
        list known by the signal date, quoted at the signal close and with 63
        complete daily returns. Optional ``max_rank`` keeps only the largest
        names by the dated universe rank (known at the time).
        """
        dates = pd.DatetimeIndex(dates)
        rows = self.calendar.get_indexer(dates)
        if (rows < 0).any() or (rows + 1 >= len(self.calendar)).any():
            raise ValueError("signal dates must be sessions followed by another session")
        entry_q = self.calendar[rows + 1].to_period("Q").start_time
        frames = []
        vol = self.price["vol_63"].to_numpy()
        for d, r, q in zip(dates, rows, entry_q):
            m = self.membership.loc[self.membership.quarter_start.eq(q) & self.membership.selection_date.le(d)]
            cols = self.symbols.get_indexer(m.symbol)
            keep = (cols >= 0) & np.isfinite(vol[r, np.maximum(cols, 0)])
            m, cols = m.loc[keep], cols[keep]
            if max_rank is not None:
                sel = m.universe_rank.le(max_rank).to_numpy()
                m, cols = m.loc[sel], cols[sel]
            frame = pd.DataFrame({"date": pd.Timestamp(d), "symbol": m.symbol.to_numpy(),
                                  "universe_rank": m.universe_rank.to_numpy(),
                                  "cap_proxy": m.market_cap_proxy.to_numpy(),
                                  "sel_date": m.selection_date.to_numpy()})
            for name, mat in self.price.items():
                frame[name] = mat.iloc[r, cols].to_numpy()
            # Capitalization at the signal close: selection proxy moved with the price.
            sel_rows = self.calendar.get_indexer(frame.sel_date)
            px = self.md.adj.to_numpy()
            frame["mcap"] = frame.cap_proxy * px[r, cols] / px[sel_rows, cols]
            frame["target"] = self.target[r, cols]
            le = self.label_end_row[r, cols]
            frame["label_end"] = pd.NaT
            ok = np.isfinite(le)
            frame.loc[ok, "label_end"] = self.calendar[le[ok].astype(int)]
            frames.append(frame)
        panel = pd.concat(frames, ignore_index=True)
        panel = self._join_fundamentals(panel)
        return self._add_ranks(panel)

    def _join_fundamentals(self, panel):
        panel = panel.sort_values("date")
        ev = self.events.sort_values("available_date")
        out = pd.merge_asof(panel, ev, left_on="date", right_on="available_date", by="symbol",
                            direction="backward")
        stale = (out.date - out.fiscal_date).dt.days.gt(550)
        cols = ["gross_margin", "operating_margin", "roa", "roe", "accruals", "fcf_to_assets",
                "debt_to_assets", "cash_to_assets", "sales_growth", "asset_growth", "ni_ttm", "fcf_ttm", "equity"]
        out.loc[stale, cols] = np.nan
        mcap = out.mcap.where(out.mcap > 0)
        out["earnings_yield"] = out.ni_ttm / mcap
        out["book_to_market"] = out.equity / mcap
        out["fcf_yield"] = out.fcf_ttm / mcap
        out["filing_age"] = (out.date - out.available_date).dt.days.where(~stale)
        out["log_size"] = np.log(mcap)
        assert (out.available_date.isna() | out.available_date.le(out.date)).all()
        assert out.sel_date.le(out.date).all()
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _add_ranks(panel):
        """Same-date percentile ranks among the eligible cross-section (no fitting)."""
        g = panel.groupby("date")
        panel["universe_rank_pct"] = g.universe_rank.rank(pct=True)
        for c in PRICE_FEATURES + FUND_FEATURES + ["log_size"]:
            panel["r_" + c] = g[c].rank(pct=True)
        panel["r_universe_rank_pct"] = panel.universe_rank_pct
        panel["target_rank"] = g.target.rank(pct=True)
        panel["target_excess"] = panel.target - g.target.transform("mean")
        return panel


RANK_FEATURES = ["r_" + c for c in FEATURES]
