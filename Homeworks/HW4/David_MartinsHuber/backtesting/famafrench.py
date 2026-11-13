"""Fama-French five-factor construction for HW4 research.

The model is deliberately an *FF5-inspired* implementation.  The supplied
database does not contain the CRSP/Compustat fields required for an exact
replication, so the following approximations are used and documented here and
in ``docs/FamaFrench.md``:

* ``MKT`` is the cap-weighted return of the eligible investable universe.  No
  risk-free rate is subtracted (the strategy does not use one).
* ``SMB`` is the average of the size legs from the book-to-market,
  profitability and investment sorts, as in the five-factor model.
* ``HML`` uses ``book equity = total_equity``.  Preferred stock and deferred
  taxes are not available.
* ``RMW`` uses ``(operating_income - interest_expense) / book equity``.  SG&A
  is not a separate field; it is already embedded in operating income.
* ``CMA`` uses year-over-year growth in ``total_assets``.
* Breakpoints are all-market (median size, 30th/70th percentiles for the other
  characteristics).  Historical NYSE membership is not available.
* Accounting data is lagged six months and portfolios are formed monthly.
  The lecture's June annual formation is not replicated.
* Negative book equity is excluded by default.  Financial Services are
  excluded by default because their book equity and leverage are not
  comparable.
* Delisting returns are not handled here; the backtester is responsible for
  forced exits.

Point-in-time rules
-------------------
A filing is usable on ``filing_date + 1 day`` (the ``MarketData`` convention).
At formation date ``t`` the characteristics use the latest filing available by
``t - lag_months``.  Market equity is measured at ``t``.  The forward return is
from the ``t`` close to the next month-end close.  Future filings and future
prices therefore cannot change a formation-date observation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market_data import MarketData


FACTOR_COLUMNS = ("MKT", "SMB", "HML", "RMW", "CMA")
REQUIRED_FUNDAMENTALS = ("total_equity", "total_assets", "operating_income",
                         "interest_expense", "shares_outstanding")
CHARACTERISTIC_COLUMNS = (
    "date", "symbol", "ME", "BE", "BM", "OP", "INV", "FWD_RET",
    "is_member", "sector",
    "member_ok", "sector_ok", "be_present", "be_ok", "values_ok",
    "me_present", "me_ok", "eligible",
    "size_group", "bm_group", "op_group", "inv_group",
)
PORTFOLIO_COLUMNS = ("date", "characteristic", "portfolio", "return", "n", "weight_sum")
DIAGNOSTIC_COLUMNS = (
    "date", "universe", "with_price", "eligible", "excluded_member",
    "excluded_sector", "excluded_no_filing", "excluded_be",
    "excluded_missing", "excluded_no_price", "excluded_me",
    "with_forward_return", "market_n", "market_weight",
)


@dataclass(frozen=True)
class FactorResult:
    """Factor returns and the cross-section used to build them."""

    factor_returns: pd.DataFrame
    characteristics: pd.DataFrame
    portfolios: pd.DataFrame
    diagnostics: pd.DataFrame

    @property
    def coverage(self):
        """Universe, quoted, eligible, and forward-return counts."""
        return self.diagnostics.set_index("date")[
            ["universe", "with_price", "eligible", "with_forward_return"]]


def _month_ends(calendar):
    """Last trading session of each month in ``calendar``."""
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    if calendar.empty:
        return calendar
    frame = pd.DataFrame({"date": calendar})
    return pd.DatetimeIndex(
        frame.groupby(frame.date.dt.to_period("M")).date.max().to_numpy())


def _finite_mean(values):
    """Mean ignoring NaN; NaN when no finite value exists."""
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else np.nan


def _complete_mean(values):
    """Mean that is NaN unless every component is finite.

    Factor legs are defined as equal-weight averages of complete 2x3 cells.
    Silently dropping a missing cell would change the factor definition rather
    than merely reduce its sample.
    """
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if np.isfinite(array).all() else np.nan


class FamaFrench5:
    """Build monthly FF5 factors from a :class:`MarketData` object.

    Parameters
    ----------
    lag_months:
        Accounting lag applied to book equity, operating profitability and
        investment.  Default six months.
    exclude_sectors:
        Sectors removed from the eligible universe.  Default Financial
        Services.
    exclude_negative_be:
        Drop non-positive book equity before sorts.
    min_me:
        Optional minimum market equity.  ``None`` keeps every eligible name.
    weighting:
        ``"cap"`` for market-value-weighted portfolios, ``"equal"`` for
        equal-weighted portfolios (useful in tests and robustness checks).
    min_names:
        Minimum eligible names required to form a cross-section.  The default
        of six is the smallest universe that can populate every 2x3 cell.
    """

    def __init__(self, *, lag_months=6, exclude_sectors=("Financial Services",),
                 exclude_negative_be=True, min_me=None, weighting="cap",
                 min_names=6, breakpoints="market"):
        if not isinstance(lag_months, int) or lag_months < 0:
            raise ValueError("lag_months must be a nonnegative integer")
        if isinstance(exclude_sectors, str):
            raise TypeError("exclude_sectors must be a sequence of sector names")
        if weighting not in ("cap", "equal"):
            raise ValueError("weighting must be 'cap' or 'equal'")
        if not isinstance(min_names, int) or min_names < 6:
            raise ValueError("min_names must be an integer of at least 6")
        if min_me is not None and (not np.isfinite(min_me) or min_me < 0):
            raise ValueError("min_me must be None or a finite nonnegative amount")
        if breakpoints != "market":
            raise ValueError("Only all-market breakpoints are supported")
        self.lag_months = lag_months
        self.exclude_sectors = tuple(exclude_sectors)
        self.exclude_negative_be = bool(exclude_negative_be)
        self.min_me = min_me
        self.weighting = weighting
        self.min_names = min_names
        self.breakpoints = breakpoints

    # ------------------------------------------------------------------ API
    def build(self, data: MarketData, start=None, end=None) -> FactorResult:
        """Build factor returns for every month-end in ``[start, end]``.

        ``start`` and ``end`` filter the *formation* dates.  The lookback is
        taken from the full supplied history so a short requested window does
        not silently lose the accounting or investment history.
        """
        if not isinstance(data, MarketData):
            raise TypeError("data must be a MarketData")
        month_ends = _month_ends(data.calendar)
        if len(month_ends) < 2:
            raise ValueError("At least two month-end sessions are required")
        formation = month_ends
        if start is not None:
            formation = formation[formation >= pd.Timestamp(start)]
        if end is not None:
            formation = formation[formation <= pd.Timestamp(end)]
        if formation.empty:
            raise ValueError("No month-end formation dates in the requested range")

        # Keep the accounting and investment lookback, plus one month after the
        # last formation so its forward return is observable, without building
        # the entire 2000-2026 grid for a short research window. A trailing
        # formation with no next-month return is dropped below.
        first = month_ends.get_loc(formation[0])
        last = month_ends.get_loc(formation[-1])
        panel_dates = month_ends[max(0, first - (self.lag_months + 12)):
                                 min(len(month_ends) - 1, last + 1) + 1]
        characteristics = self._characteristics(data, panel_dates)
        characteristics = characteristics.loc[characteristics.date.isin(formation)].copy()

        factor_rows, portfolio_rows, diagnostic_rows, grouped = [], [], [], []
        for date, cross in characteristics.groupby("date", sort=True):
            factors, portfolios, diagnostics, cross = self._factor_row(cross)
            factor_rows.append({"date": date, **factors})
            portfolio_rows.extend(portfolios)
            diagnostic_rows.append({"date": date, **diagnostics})
            grouped.append(cross)

        factor_returns = pd.DataFrame(factor_rows, columns=["date", *FACTOR_COLUMNS])
        factor_returns = factor_returns.set_index("date").sort_index()
        # The final formation date has no next-month return (and usually no
        # next-quarter membership). Drop only trailing all-NaN rows so interior
        # short-history gaps remain visible.
        while len(factor_returns) and factor_returns.iloc[-1].isna().all():
            factor_returns = factor_returns.iloc[:-1]
        characteristics = (pd.concat(grouped, ignore_index=True)
                           .sort_values(["date", "symbol"]).reset_index(drop=True))
        characteristics = characteristics.reindex(columns=CHARACTERISTIC_COLUMNS)
        portfolios = pd.DataFrame(portfolio_rows, columns=PORTFOLIO_COLUMNS)
        diagnostics = pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
        return FactorResult(factor_returns, characteristics, portfolios, diagnostics)

    def exposures(self, data: MarketData, date) -> pd.DataFrame:
        """Characteristics and group assignments for one month-end."""
        date = pd.Timestamp(date)
        month_ends = _month_ends(data.calendar)
        if date not in month_ends:
            raise ValueError("date must be the last trading session of a month")
        position = month_ends.get_loc(date)
        panel_dates = month_ends[max(0, position - (self.lag_months + 12)):position + 1]
        cross = self._characteristics(data, panel_dates)
        cross = cross.loc[cross.date.eq(date)]
        if cross.empty:
            raise ValueError(f"No cross-section at {date.date()}")
        _, _, _, cross = self._factor_row(cross)
        return cross.reindex(columns=CHARACTERISTIC_COLUMNS).reset_index(drop=True)

    # ------------------------------------------------------- point-in-time
    def _characteristics(self, data: MarketData, month_ends):
        """Monthly, availability-dated cross-section before group sorts."""
        events = data.financial_events()
        missing = set(REQUIRED_FUNDAMENTALS) - set(events.columns)
        if missing:
            raise ValueError(
                "Fama-French factors require financial columns: "
                + ", ".join(sorted(missing)))
        # A stub filing missing a required balance-sheet field must not shadow
        # the last complete filing for the following months. Dropping it lets
        # the backward as-of join retain the prior known values.
        core = ["total_equity", "total_assets", "operating_income",
                "shares_outstanding"]
        events = events.loc[events[core].notna().all(axis=1)].copy()

        symbols = pd.Index(sorted(data.prices.symbol.unique()))
        grid = pd.MultiIndex.from_product(
            [month_ends, symbols], names=["date", "symbol"]).to_frame(index=False)
        # ``merge_asof`` requires the on-key to be sorted globally; the ``by``
        # argument then keeps each symbol's observations separate.
        grid = grid.sort_values(["date", "symbol"]).reset_index(drop=True)
        events = events.sort_values(["available_date", "symbol"]).reset_index(drop=True)
        panel = pd.merge_asof(
            grid, events, left_on="date", right_on="available_date",
            by="symbol", direction="backward")
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

        month_prices = data.prices.loc[data.prices.date.isin(month_ends)]
        close = (month_prices.pivot(index="date", columns="symbol", values="close")
                 .reindex(index=month_ends, columns=symbols))
        adjusted = (month_prices.pivot(index="date", columns="symbol", values="adj_close")
                    .reindex(index=month_ends, columns=symbols))
        close.index.name, close.columns.name = "date", "symbol"
        adjusted.index.name, adjusted.columns.name = "date", "symbol"
        close_long = close.stack().rename("close").reset_index()
        adjusted_long = adjusted.stack().rename("adj_close").reset_index()
        panel = panel.merge(close_long, on=["date", "symbol"], how="left")
        panel = panel.merge(adjusted_long, on=["date", "symbol"], how="left")

        if data.universe.symbol.duplicated().any():
            raise ValueError("universe must contain one row per symbol")
        sector = data.universe[["symbol", "sector"]].reset_index(drop=True)
        panel = panel.merge(sector, on="symbol", how="left")

        members = self._membership_by_month(data, month_ends)
        panel = panel.merge(members, on=["date", "symbol"], how="left")
        panel["is_member"] = panel.is_member.fillna(False).astype(bool)

        panel["ME"] = panel.shares_outstanding * panel.close

        def wide(column):
            result = (panel.pivot(index="date", columns="symbol", values=column)
                      .reindex(index=month_ends, columns=symbols).sort_index())
            result.index.name, result.columns.name = "date", "symbol"
            return result

        market_equity = wide("ME")
        book_equity = wide("total_equity")
        assets = wide("total_assets")
        operating_income = wide("operating_income")
        interest = wide("interest_expense")

        lag = self.lag_months
        book_equity_lag = book_equity.shift(lag)
        assets_lag = assets.shift(lag)
        assets_lag_year = assets.shift(lag + 12)
        operating_lag = operating_income.shift(lag)
        interest_lag = interest.shift(lag)

        with np.errstate(divide="ignore", invalid="ignore"):
            book_to_market = book_equity_lag.div(market_equity)
            investment = assets_lag.div(assets_lag_year).sub(1.0)
            profitability_with_interest = operating_lag.sub(interest_lag).div(book_equity_lag)
            profitability_without_interest = operating_lag.div(book_equity_lag)
        # Missing interest expense is common for financial-style statements;
        # fall back to operating income over book equity in that case.
        profitability = profitability_with_interest.where(
            interest_lag.notna(), profitability_without_interest)

        forward_return = adjusted.shift(-1).div(adjusted).sub(1.0)
        for frame in (book_to_market, investment, profitability, forward_return):
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)

        pieces = []
        for name, frame in [
            ("ME", market_equity), ("BE", book_equity_lag),
            ("BM", book_to_market), ("OP", profitability),
            ("INV", investment), ("FWD_RET", forward_return),
        ]:
            pieces.append(frame.stack().rename(name).reset_index())
        # Start from the full month-end grid so missing values are retained for
        # diagnostics; stacking a frame drops its own NaN cells.
        chars = panel[["date", "symbol"]].drop_duplicates().reset_index(drop=True)
        for piece in pieces:
            chars = chars.merge(piece, on=["date", "symbol"], how="left")
        chars = chars.merge(members, on=["date", "symbol"], how="left")
        chars["is_member"] = chars.is_member.fillna(False).astype(bool)
        chars = chars.merge(sector, on="symbol", how="left")
        return chars

    @staticmethod
    def _membership_by_month(data, month_ends):
        """Point-in-time execution-quarter membership for each month-end."""
        rows = []
        calendar = data.calendar
        for date in month_ends:
            position = calendar.get_loc(date)
            if position + 1 >= len(calendar):
                continue
            execution = calendar[position + 1]
            members = data.membership_at(date, execution)
            if members.empty:
                continue
            rows.append(pd.DataFrame({
                "date": date,
                "symbol": members.symbol.to_numpy(),
                "is_member": True,
            }))
        if not rows:
            return pd.DataFrame(columns=["date", "symbol", "is_member"])
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------- sorting
    def _flag_eligible(self, cross):
        cross = cross.copy()
        member = cross["is_member"].fillna(False).astype(bool)
        sector = cross["sector"]
        sector_ok = sector.notna() & ~sector.isin(self.exclude_sectors)
        be = pd.to_numeric(cross["BE"], errors="coerce")
        be_present = be.notna()
        be_ok = be.gt(0) if self.exclude_negative_be else be_present
        values = cross[["BM", "OP", "INV"]].apply(pd.to_numeric, errors="coerce")
        values_ok = pd.Series(
            np.isfinite(values.to_numpy()).all(axis=1), index=cross.index)
        me = pd.to_numeric(cross["ME"], errors="coerce")
        me_present = np.isfinite(me)
        me_ok = me_present & me.gt(0)
        if self.min_me is not None:
            me_ok &= me.ge(self.min_me)
        cross["ME"] = me
        cross["BE"] = be
        for column in ("BM", "OP", "INV"):
            cross[column] = values[column]
        cross["FWD_RET"] = pd.to_numeric(cross["FWD_RET"], errors="coerce")
        cross["member_ok"] = member
        cross["sector_ok"] = sector_ok
        cross["be_present"] = be_present
        cross["be_ok"] = be_ok
        cross["values_ok"] = values_ok
        cross["me_present"] = me_present
        cross["me_ok"] = me_ok
        cross["eligible"] = (member & sector_ok & be_present & be_ok
                             & values_ok & me_ok)
        return cross

    @staticmethod
    def _bucket(values, low, high, low_label, mid_label, high_label):
        # With tied breakpoints every name satisfies both tails. Treat a
        # degenerate sort as Neutral rather than letting the High mask swallow
        # the entire cross-section.
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            return pd.Series(mid_label, index=values.index, dtype=object)
        result = pd.Series(mid_label, index=values.index, dtype=object)
        result.loc[values.le(low)] = low_label
        result.loc[values.ge(high)] = high_label
        return result

    def _weighted_return(self, subset):
        subset = subset.loc[subset.FWD_RET.notna()]
        if subset.empty:
            return np.nan, 0, 0.0
        if self.weighting == "cap":
            weights = subset.ME.to_numpy(dtype=float)
        else:
            weights = np.ones(len(subset), dtype=float)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            return np.nan, 0, 0.0
        returns = subset.FWD_RET.to_numpy(dtype=float)
        return float(np.dot(weights, returns) / total), len(subset), total

    def _factor_row(self, cross):
        """One formation date: groups, portfolio returns and factor returns."""
        cross = self._flag_eligible(cross)
        eligible = cross.loc[cross.eligible]
        date = pd.Timestamp(cross["date"].iloc[0]) if "date" in cross.columns else pd.NaT

        member_sector = cross.member_ok & cross.sector_ok
        diagnostics = {
            "universe": len(cross),
            "with_price": int(cross.me_present.sum()),
            "eligible": len(eligible),
            "excluded_member": int((~cross.member_ok).sum()),
            "excluded_sector": int((cross.member_ok & ~cross.sector_ok).sum()),
            "excluded_no_filing": int((member_sector & ~cross.be_present).sum()),
            "excluded_be": int((member_sector & cross.be_present & ~cross.be_ok).sum()),
            "excluded_missing": int((member_sector & cross.be_present & cross.be_ok
                                     & ~cross.values_ok).sum()),
            "excluded_no_price": int((member_sector & cross.be_present & cross.be_ok
                                      & cross.values_ok & ~cross.me_present).sum()),
            "excluded_me": int((member_sector & cross.be_present & cross.be_ok
                                & cross.values_ok & cross.me_present
                                & ~cross.me_ok).sum()),
            "with_forward_return": int((cross.eligible & cross.FWD_RET.notna()).sum()),
            "market_n": 0,
            "market_weight": 0.0,
        }

        for column in ("size_group", "bm_group", "op_group", "inv_group"):
            cross[column] = None

        if len(eligible) < self.min_names:
            return ({factor: np.nan for factor in FACTOR_COLUMNS},
                    [], diagnostics, cross)

        size_cut = eligible.ME.median()
        bm_low, bm_high = eligible.BM.quantile([0.3, 0.7])
        op_low, op_high = eligible.OP.quantile([0.3, 0.7])
        inv_low, inv_high = eligible.INV.quantile([0.3, 0.7])

        idx = eligible.index
        cross.loc[idx, "size_group"] = np.where(eligible.ME.le(size_cut), "S", "B")
        cross.loc[idx, "bm_group"] = self._bucket(
            eligible.BM, bm_low, bm_high, "L", "N", "H")
        cross.loc[idx, "op_group"] = self._bucket(
            eligible.OP, op_low, op_high, "W", "N", "R")
        cross.loc[idx, "inv_group"] = self._bucket(
            eligible.INV, inv_low, inv_high, "C", "N", "A")

        eligible = cross.loc[cross.eligible]
        portfolios, returns = [], {}
        specifications = [
            ("BM", "bm_group", ("L", "N", "H")),
            ("OP", "op_group", ("W", "N", "R")),
            ("INV", "inv_group", ("C", "N", "A")),
        ]
        for characteristic, column, labels in specifications:
            for size in ("S", "B"):
                for label in labels:
                    key = f"{size}/{label}"
                    subset = eligible.loc[
                        eligible.size_group.eq(size) & eligible[column].eq(label)]
                    value, count, weight = self._weighted_return(subset)
                    returns[(characteristic, key)] = value
                    portfolios.append({
                        "date": date, "characteristic": characteristic,
                        "portfolio": key, "return": value, "n": count,
                        "weight_sum": weight})

        market, market_n, market_weight = self._weighted_return(eligible)
        smb_bm = (_complete_mean(returns[("BM", f"S/{x}")] for x in ("L", "N", "H"))
                  - _complete_mean(returns[("BM", f"B/{x}")] for x in ("L", "N", "H")))
        smb_op = (_complete_mean(returns[("OP", f"S/{x}")] for x in ("W", "N", "R"))
                  - _complete_mean(returns[("OP", f"B/{x}")] for x in ("W", "N", "R")))
        smb_inv = (_complete_mean(returns[("INV", f"S/{x}")] for x in ("C", "N", "A"))
                   - _complete_mean(returns[("INV", f"B/{x}")] for x in ("C", "N", "A")))
        factors = {
            "MKT": market,
            "SMB": _complete_mean([smb_bm, smb_op, smb_inv]),
            "HML": (_complete_mean([returns[("BM", "S/H")], returns[("BM", "B/H")]])
                    - _complete_mean([returns[("BM", "S/L")], returns[("BM", "B/L")]])),
            "RMW": (_complete_mean([returns[("OP", "S/R")], returns[("OP", "B/R")]])
                    - _complete_mean([returns[("OP", "S/W")], returns[("OP", "B/W")]])),
            "CMA": (_complete_mean([returns[("INV", "S/C")], returns[("INV", "B/C")]])
                    - _complete_mean([returns[("INV", "S/A")], returns[("INV", "B/A")]])),
        }
        diagnostics["market_n"] = market_n
        diagnostics["market_weight"] = market_weight
        return factors, portfolios, diagnostics, cross
