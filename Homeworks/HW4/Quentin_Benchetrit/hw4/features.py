"""Weekly signal dates, features, entry screens and 90-day labels.

A feature dated t uses prices up to the close of t and statements filed at
least one calendar day before t. Later prices are only read in
forward_returns, which builds the training target.

The filing-date handling in fundamental_events (availability one day after
filing, an old period filed late never replaces a newer one) and the
entry-quarter membership join follow financial_events and apply_membership in
Lectures/modeling/course_data.py.
"""

import numpy as np
import pandas as pd

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5          # sessions a label may wait for an exit quote
STALE_STATEMENT_DAYS = 365  # statements older than this are ignored

# Entry screens, all computed from data known at the signal close.
MIN_DOLLAR_VOLUME = 1_000_000   # median daily close x volume over 63 sessions
MIN_ACTIVE_DAYS = 60            # of the last 63 sessions, quoted and traded
MAX_DAILY_JUMP = np.log(2)      # largest one-day move (either way) over 252 sessions
MIN_TYPICAL_MOVE = 0.0025       # median absolute daily return over 21 sessions
MIN_VOL_RATIO = 0.35            # 21-day volatility relative to 252-day volatility

PRICE_FEATURES = [
    "ret_5", "ret_21", "ret_63", "ret_126", "mom_6_1", "mom_12_1",
    "vol_21", "vol_63", "vol_252", "vol_ratio", "beta", "idio_vol",
    "max_ret_21", "high_252", "dollar_volume", "volume_trend", "amihud",
]
STATEMENT_FEATURES = [
    "roa", "roe", "gross_profitability", "operating_margin", "gross_margin",
    "accruals", "asset_turnover", "debt_to_assets", "cash_to_assets",
    "rd_to_assets", "revenue_growth", "revenue_acceleration", "income_change",
    "margin_change", "asset_growth", "share_growth",
]
VALUATION_FEATURES = [
    "earnings_yield", "book_to_price", "sales_to_price", "cash_flow_yield",
    "fcf_yield", "ebitda_to_ev", "market_cap", "turnover", "report_age",
]
FEATURES = PRICE_FEATURES + STATEMENT_FEATURES + VALUATION_FEATURES


def weekly_schedule(calendar):
    """One entry per week on its last session, decided at the previous close.

    With a Friday entry the lot matures on a Thursday 90 days later, is sold at
    that close, and the cash is known in time for that evening's decision.
    """
    days = pd.Series(calendar, index=calendar)
    entries = days.groupby(calendar.to_period("W")).max()
    if calendar[-1].weekday() != 4:     # the file ends mid-week: that week is not over
        entries = entries.iloc[:-1]
    position = calendar.get_indexer(entries)
    keep = position > 0
    return pd.DataFrame({"date": calendar[position[keep] - 1],
                         "entry_date": calendar[position[keep]]})


def price_features(market, signal_dates):
    """Trailing price and volume statistics, one row per signal date and stock."""
    price = market.adj_close
    ret = price.pct_change(fill_method=None)
    dollar_volume = market.close * market.volume

    wide = {}
    for n in (5, 21, 63, 126):
        wide[f"ret_{n}"] = price / price.shift(n) - 1
    wide["mom_6_1"] = price.shift(21) / price.shift(126) - 1
    wide["mom_12_1"] = price.shift(21) / price.shift(252) - 1
    for n in (21, 63, 252):
        wide[f"vol_{n}"] = ret.rolling(n, min_periods=int(0.9 * n)).std() * np.sqrt(252)
    wide["vol_ratio"] = wide["vol_21"] / wide["vol_252"]

    # One-year beta to the benchmark and the volatility it leaves unexplained.
    market_ret = market.benchmark.pct_change()
    m = ret.notna().astype(float).replace(0.0, np.nan).mul(market_ret, axis=0)
    roll = dict(window=252, min_periods=200)
    mean_r, mean_m = ret.rolling(**roll).mean(), m.rolling(**roll).mean()
    cov = (ret * m).rolling(**roll).mean() - mean_r * mean_m
    var_m = (m * m).rolling(**roll).mean() - mean_m ** 2
    var_r = (ret * ret).rolling(**roll).mean() - mean_r ** 2
    wide["beta"] = cov / var_m
    wide["idio_vol"] = np.sqrt((var_r - wide["beta"] ** 2 * var_m).clip(lower=0) * 252)

    wide["max_ret_21"] = ret.rolling(21, min_periods=15).max()
    wide["high_252"] = price / price.rolling(252, min_periods=200).max() - 1
    wide["dollar_volume"] = dollar_volume.rolling(63, min_periods=50).median()
    wide["volume_trend"] = (dollar_volume.rolling(21, min_periods=15).mean()
                            / dollar_volume.rolling(252, min_periods=200).mean())
    wide["amihud"] = (ret.abs() / dollar_volume.where(dollar_volume > 0)).rolling(63, min_periods=50).mean()

    # Inputs to the entry screens.
    wide["close"] = market.close
    wide["active_days"] = (price.notna() & (market.volume > 0)).rolling(63).sum()
    wide["typical_move"] = ret.abs().rolling(21, min_periods=15).median()
    raw = market.raw_adj_close
    jump = np.log(raw / raw.ffill().shift(1)).abs()
    wide["worst_jump"] = jump.rolling(252, min_periods=1).max()

    index = pd.MultiIndex.from_product([signal_dates, price.columns], names=["date", "symbol"])
    columns = {name: frame.loc[signal_dates].to_numpy().ravel() for name, frame in wide.items()}
    return pd.DataFrame(columns, index=index).reset_index()


def fundamental_events(fundamentals):
    """One row per usable filing with trailing-year and year-on-year ratios."""
    f = fundamentals.sort_values(["symbol", "fiscal_date"]).reset_index(drop=True)
    by_stock = f.groupby("symbol", sort=False)

    def lag(column, k):
        return by_stock[column].shift(k)

    def days_back(k):
        return (f.fiscal_date - lag("fiscal_date", k)).dt.days

    # Only trust sums and comparisons when the quarters involved are consecutive.
    full_year = days_back(3).between(250, 300)
    year_ago = days_back(4).between(350, 380)

    ttm = {}
    for column in ["revenue", "gross_profit", "operating_income", "net_income", "ebitda",
                   "operating_cash_flow", "free_cash_flow", "rd_expense"]:
        total = f[column] + lag(column, 1) + lag(column, 2) + lag(column, 3)
        ttm[column] = total.where(full_year)

    assets = f.total_assets.where(f.total_assets > 0)
    equity = f.total_equity.where(f.total_equity > 0)
    sales = ttm["revenue"].where(ttm["revenue"] > 0)
    shares = f.shares_outstanding.where(f.shares_outstanding > 0)

    out = f[["symbol", "fiscal_date", "filing_date"]].copy()
    out["roa"] = ttm["net_income"] / assets
    out["roe"] = ttm["net_income"] / equity
    out["gross_profitability"] = ttm["gross_profit"] / assets
    out["operating_margin"] = ttm["operating_income"] / sales
    out["gross_margin"] = ttm["gross_profit"] / sales
    out["accruals"] = (ttm["net_income"] - ttm["operating_cash_flow"]) / assets
    out["asset_turnover"] = sales / assets
    out["debt_to_assets"] = f.total_debt / assets
    out["cash_to_assets"] = f.cash_and_equivalents / assets
    out["rd_to_assets"] = ttm["rd_expense"] / assets

    revenue = f.revenue.where(f.revenue > 0)
    f["revenue_growth"] = (revenue / lag("revenue", 4).where(lambda x: x > 0) - 1).where(year_ago)
    f["quarter_margin"] = f.gross_profit / revenue
    by_stock = f.groupby("symbol", sort=False)
    out["revenue_growth"] = f.revenue_growth
    out["revenue_acceleration"] = f.revenue_growth - lag("revenue_growth", 1)
    out["income_change"] = ((f.net_income - lag("net_income", 4)) / assets).where(year_ago)
    out["margin_change"] = (f.quarter_margin - lag("quarter_margin", 4)).where(year_ago)
    out["asset_growth"] = (assets / lag("total_assets", 4).where(lambda x: x > 0) - 1).where(year_ago)
    out["share_growth"] = (shares / lag("shares_outstanding", 4).where(lambda x: x > 0) - 1).where(year_ago)

    # Kept in dollars; divided by the signal-date market value in build_panel.
    out["shares"] = shares
    out["equity"] = f.total_equity
    out["net_debt"] = f.total_debt.fillna(0) - f.cash_and_equivalents.fillna(0)
    for name, column in [("earnings", "net_income"), ("sales", "revenue"), ("ebitda", "ebitda"),
                         ("cash_flow", "operating_cash_flow"), ("fcf", "free_cash_flow")]:
        out[name] = ttm[column]
    out = out.replace([np.inf, -np.inf], np.nan)

    out = out.loc[out.filing_date > out.fiscal_date].copy()
    out["available_date"] = out.filing_date + pd.Timedelta(days=1)
    out = out.sort_values(["symbol", "available_date", "fiscal_date"])
    newest = out.groupby("symbol").fiscal_date.cummax()
    out = out.loc[out.fiscal_date.eq(newest)].drop_duplicates(["symbol", "available_date"], keep="last")
    return out.drop(columns="filing_date")


def forward_returns(market, schedule):
    """Return from the entry close to the first quote on or after day 90.

    The exit may slip by at most MAX_EXIT_DELAY sessions; beyond that, or when
    the entry session has no quote, the label is left missing.
    """
    calendar = market.calendar
    prices = market.adj_close.to_numpy()
    n_days, n_stocks = prices.shape
    entry_pos = calendar.get_indexer(schedule.entry_date)
    due = schedule.entry_date + pd.Timedelta(days=HOLDING_DAYS)
    due_pos = calendar.searchsorted(due)

    benchmark = market.benchmark.to_numpy()
    target = np.full((len(schedule), n_stocks), np.nan)
    market_move = np.full((len(schedule), n_stocks), np.nan)
    exit_pos = np.full((len(schedule), n_stocks), -1)
    for k, (start, first_exit) in enumerate(zip(entry_pos, due_pos)):
        if first_exit >= n_days:
            continue    # matures after the data ends
        window = prices[first_exit:first_exit + MAX_EXIT_DELAY + 1]
        quoted = ~np.isnan(window)
        offset = quoted.argmax(axis=0)
        found = quoted.any(axis=0)
        exit_price = window[offset, np.arange(n_stocks)]
        target[k] = np.where(found, exit_price / prices[start] - 1, np.nan)
        market_move[k] = benchmark[np.minimum(first_exit + offset, n_days - 1)] / benchmark[start] - 1
        exit_pos[k] = np.where(found & ~np.isnan(prices[start]), first_exit + offset, -1)

    index = pd.MultiIndex.from_product([schedule.date, market.symbols], names=["date", "symbol"])
    labels = pd.DataFrame({"target": target.ravel(), "benchmark_target": market_move.ravel(),
                           "exit_pos": exit_pos.ravel()}, index=index).reset_index()
    labels["label_end"] = pd.NaT
    done = labels.exit_pos >= 0
    labels.loc[done, "label_end"] = calendar[labels.exit_pos[done]]
    labels["label_end"] = labels.label_end.astype("datetime64[ns]")
    return labels.drop(columns="exit_pos")


def build_panel(market, fundamentals, membership, first_signal="2010-01-01"):
    """Assemble features, screens and labels for every weekly signal date."""
    schedule = weekly_schedule(market.calendar)
    schedule = schedule.loc[schedule.date >= first_signal].reset_index(drop=True)

    panel = price_features(market, pd.DatetimeIndex(schedule.date))
    panel = panel.merge(forward_returns(market, schedule), on=["date", "symbol"])
    panel = panel.merge(schedule, on="date")

    # Membership of the quarter in which the entry falls; that list is fixed at
    # the last close of the previous quarter, so it is known at the signal.
    panel["quarter_start"] = panel.entry_date.dt.to_period("Q").dt.start_time
    panel = panel.merge(membership, on=["symbol", "quarter_start"], how="inner")
    assert (panel.selection_date <= panel.date).all()

    events = fundamental_events(fundamentals)
    panel = pd.merge_asof(panel.sort_values("date"), events.sort_values("available_date"),
                          left_on="date", right_on="available_date", by="symbol")
    assert (panel.available_date.isna() | (panel.available_date <= panel.date)).all()
    statement_columns = [c for c in events.columns if c not in ("symbol", "available_date")]
    stale = (panel.date - panel.fiscal_date).dt.days > STALE_STATEMENT_DAYS
    panel.loc[stale, statement_columns + ["available_date"]] = np.nan

    value = panel.close * panel.shares
    enterprise = (value + panel.net_debt).where(lambda x: x > 0)
    panel["earnings_yield"] = panel.earnings / value
    panel["book_to_price"] = panel.equity / value
    panel["sales_to_price"] = panel.sales / value
    panel["cash_flow_yield"] = panel.cash_flow / value
    panel["fcf_yield"] = panel.fcf / value
    panel["ebitda_to_ev"] = panel.ebitda / enterprise
    panel["market_cap"] = value
    panel["turnover"] = panel.dollar_volume / value
    panel["report_age"] = (panel.date - panel.available_date).dt.days

    # The last two screens target stocks that have stopped moving: cash-merger
    # targets pinned under the offer price and blank-check shells. Their upside
    # is capped and the file cannot tell what a holder is paid when they vanish.
    panel["eligible"] = (
        panel.close.notna()
        & (panel.active_days >= MIN_ACTIVE_DAYS)
        & (panel.dollar_volume >= MIN_DOLLAR_VOLUME)
        & (panel.worst_jump < MAX_DAILY_JUMP)
        & (panel.typical_move >= MIN_TYPICAL_MOVE)
        & ~(panel.vol_ratio < MIN_VOL_RATIO)
    )
    panel["sector"] = panel.symbol.map(market.sectors)

    # Rank each feature among the stocks that can be bought that day. Ranks are
    # immune to outliers and unit problems, and need nothing fitted on history.
    raw = panel[["beta", "vol_63", "vol_ratio", "dollar_volume", "market_cap"]].add_prefix("raw_")
    ranks = panel.loc[panel.eligible].groupby("date")[FEATURES].rank(pct=True)
    panel[FEATURES] = ranks.reindex(panel.index).astype("float32")
    panel = pd.concat([panel, raw], axis=1)

    keep = ["date", "symbol", "entry_date", "sector", "universe_rank", "eligible",
            "active_days", "typical_move", "worst_jump", "target", "benchmark_target",
            "label_end", "raw_beta", "raw_vol_63", "raw_vol_ratio", "raw_dollar_volume",
            "raw_market_cap"] + FEATURES
    return panel[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
