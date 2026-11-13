"""HW4: point-in-time stock-return research and 90-day portfolio backtest.

Run from the repository root with:
    uv run python Homeworks/HW4/Josie_Wu/hw4.py

The database is read-only. Results are written beside this script.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Lectures/modeling"))
from course_data import QuantileClipper

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "Lectures/modeling/data/equities_top2000.duckdb"
OUT = Path(__file__).resolve().parent / "results"
FEATURES = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252"]
STARTING_CASH = 1_000_000.0
FEE_BPS = 5.0
HOLDING_DAYS = 90
# Follow the monthly allocation example from Lecture 8.
COHORT_FRACTION = 1 / 3
TOP_N = 3
RISK_FREE_RATE = 0.02


def read_reference_data(database: Path):
    """Read the trading calendar, benchmark, and dated quarterly membership."""
    with duckdb.connect(str(database), read_only=True) as con:
        calendar = pd.to_datetime(con.execute("SELECT date FROM calendar ORDER BY date").fetchdf().date)
        benchmark = con.execute(
            "SELECT date, adj_close FROM benchmark WHERE symbol='^SP500TR' ORDER BY date"
        ).fetchdf()
        membership = con.execute(
            "SELECT quarter_start, selection_date, symbol FROM universe_membership ORDER BY quarter_start, symbol"
        ).fetchdf()
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()
    benchmark["date"] = pd.to_datetime(benchmark.date)
    membership["quarter_start"] = pd.to_datetime(membership.quarter_start)
    membership["selection_date"] = pd.to_datetime(membership.selection_date)
    if benchmark.empty or membership.empty:
        raise ValueError("Database is missing the benchmark or dated universe membership.")
    return pd.DatetimeIndex(calendar), benchmark, membership, symbols


def month_end_signal_dates(calendar: pd.DatetimeIndex, start="2010-01-01", end="2026-08-31"):
    """Return month-end market sessions that have a next-session execution date."""
    frame = pd.DataFrame({"date": calendar})
    ends = frame.groupby(frame.date.dt.to_period("M")).date.max().tolist()
    last = calendar[-1]
    return [pd.Timestamp(d) for d in ends if pd.Timestamp(start) <= d <= pd.Timestamp(end)
            and calendar.get_loc(d) + 1 < len(calendar)
            and calendar[calendar.get_loc(d) + 1] <= last]


def feature_values(price: pd.Series):
    """Calculate trailing features on the supplied market-session calendar."""
    momentum_21 = price / price.shift(21) - 1
    momentum_63 = price / price.shift(63) - 1
    momentum_252_21 = price.shift(21) / price.shift(252) - 1
    daily_return = price.pct_change(fill_method=None)
    volatility_63 = daily_return.rolling(63, min_periods=63).std() * np.sqrt(252)
    drawdown_252 = price / price.rolling(252, min_periods=252).max() - 1
    return pd.DataFrame({
        "momentum_21": momentum_21,
        "momentum_63": momentum_63,
        "momentum_252_21": momentum_252_21,
        "volatility_63": volatility_63,
        "drawdown_252": drawdown_252,
    })


def make_research_panel(database: Path, calendar, membership, symbols, batch_size=100):
    """Build monthly, membership-aware signals in small ticker batches.

    Only 2010-2015 observations receive labels. Later data is used only for
    causal features, simulated fills, and daily portfolio marks.
    """
    signal_dates = month_end_signal_dates(calendar)
    signal_positions = np.array([calendar.get_loc(d) for d in signal_dates], dtype=int)
    entry_dates = [calendar[i + 1] for i in signal_positions]
    first_dev = pd.Timestamp("2010-01-01")
    last_dev = pd.Timestamp("2015-12-31")
    members_by_quarter = {
        q: set(g.symbol) for q, g in membership.groupby("quarter_start", sort=False)
    }
    selection_by_quarter = membership.drop_duplicates("quarter_start").set_index("quarter_start").selection_date.to_dict()
    rows = []

    with duckdb.connect(str(database), read_only=True) as con:
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset:offset + batch_size]
            prices = con.execute(
                """SELECT symbol, date, adj_close FROM prices
                   WHERE symbol IN (SELECT unnest(?)) AND date <= ?
                   ORDER BY symbol, date""",
                [batch, calendar[-1].date()],
            ).fetchdf()
            prices["date"] = pd.to_datetime(prices.date)
            for symbol, group in prices.groupby("symbol", sort=False):
                raw = group.set_index("date").adj_close
                px = raw.reindex(calendar).astype(float)
                feats = feature_values(px)
                if not px.notna().any():
                    continue
                vals = feats.iloc[signal_positions]
                current = px.iloc[signal_positions]
                for j, signal_date in enumerate(signal_dates):
                    exec_date = entry_dates[j]
                    quarter = exec_date.to_period("Q").start_time
                    members = members_by_quarter.get(quarter, set())
                    if symbol not in members or selection_by_quarter.get(quarter, pd.Timestamp.max) > signal_date:
                        continue
                    if not np.isfinite(current.iloc[j]) or not np.isfinite(vals.iloc[j]["volatility_63"]):
                        continue
                    if not np.isfinite(vals.iloc[j]["drawdown_252"]):
                        continue
                    row = {"signal_date": signal_date, "entry_date": exec_date, "symbol": symbol,
                           "entry_price": np.nan, "label_end": pd.NaT, "target": np.nan}
                    row.update(vals.iloc[j].to_dict())
                    # Future prices are used only to construct the supervised target.
                    if first_dev <= signal_date <= last_dev:
                        entry_i = signal_positions[j] + 1
                        if entry_i < len(calendar) and pd.notna(px.iloc[entry_i]):
                            due = calendar[entry_i] + pd.Timedelta(days=HOLDING_DAYS)
                            due_i = int(calendar.searchsorted(due, side="left"))
                            # A target is usable only when a quote exists at the scheduled
                            # maturity session; do not substitute a later price.
                            if due_i < len(calendar) and pd.notna(px.iloc[due_i]):
                                row["entry_price"] = float(px.iloc[entry_i])
                                row["label_end"] = calendar[due_i]
                                row["target"] = float(px.iloc[due_i] / px.iloc[entry_i] - 1)
                    rows.append(row)
            del prices
            print(f"Processed {min(offset + batch_size, len(symbols)):,}/{len(symbols):,} symbols", flush=True)
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No eligible monthly stock observations were built.")
    panel["signal_date"] = pd.to_datetime(panel.signal_date)
    panel["entry_date"] = pd.to_datetime(panel.entry_date)
    panel["label_end"] = pd.to_datetime(panel.label_end)
    panel = panel.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    return panel


def new_model(alpha):
    """Fit train-only tail clipping, imputation, scaling, and Ridge regression."""
    return make_pipeline(QuantileClipper(lower=0.01, upper=0.99),
                         SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def fit_model(train, alpha):
    """Fit Ridge to training-only winsorized labels to limit bad-price outliers."""
    low, high = train.target.quantile([0.01, 0.99]).to_numpy()
    target = train.target.clip(lower=low, upper=high)
    model = new_model(alpha).fit(train[FEATURES], target)
    return model, float(low), float(high)


def rank_ic(frame, prediction_column="prediction"):
    values = []
    for _, group in frame.groupby("signal_date"):
        if len(group) >= 5 and group.target.nunique() > 1 and group[prediction_column].nunique() > 1:
            values.append(group[prediction_column].corr(group.target, method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def validate_models(panel):
    """Use one model per annual fold, fitted before its first validation signal."""
    alpha_rows = []
    for year in (2013, 2014, 2015):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        validation = panel.loc[panel.signal_date.between(year_start, year_end) & panel.target.notna()].copy()
        if validation.empty:
            continue
        # Fit after the previous year's final close, before scoring that close
        # for the first entry of the validation year.
        fit_date = panel.loc[panel.signal_date.lt(year_start), "signal_date"].max()
        if pd.isna(fit_date):
            raise ValueError(f"No prior-year signal date is available before validation in {year}.")
        train = panel.loc[
            panel.signal_date.ge("2010-01-01") & panel.signal_date.lt(fit_date)
            & panel.label_end.notna() & panel.label_end.lt(fit_date) & panel.target.notna()
        ].copy()
        if train.empty:
            raise ValueError(f"No completed training labels before validation in {year}.")
        assert train.label_end.max() < fit_date
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            model, y_low, y_high = fit_model(train, alpha)
            pred = model.predict(validation[FEATURES])
            err = validation.target.to_numpy() - pred
            metric_frame = validation[["signal_date", "symbol", "target"]].copy()
            metric_frame["prediction"] = pred
            alpha_rows.append({"year": year, "fit_date": fit_date, "alpha": alpha, "training_rows": len(train),
                               "validation_rows": len(validation), "rmse": float(np.sqrt(np.mean(err ** 2))),
                               "mae": float(np.mean(np.abs(err))), "mean_monthly_rank_ic": rank_ic(metric_frame),
                               "training_target_clip_low": y_low, "training_target_clip_high": y_high,
                               "training_targets_winsorized": int(((train.target < y_low) | (train.target > y_high)).sum())})
    metrics = pd.DataFrame(alpha_rows)
    if metrics.empty:
        raise ValueError("The development period did not produce validation folds.")
    mean = metrics.groupby("alpha").agg(mean_rmse=("rmse", "mean"), mean_rank_ic=("mean_monthly_rank_ic", "mean"))
    chosen_alpha = float(mean.mean_rmse.idxmin())
    predictions = []
    for year in (2013, 2014, 2015):
        validation = panel.loc[panel.signal_date.dt.year.eq(year) & panel.target.notna()].copy()
        if validation.empty:
            continue
        year_start = pd.Timestamp(f"{year}-01-01")
        fit_date = panel.loc[panel.signal_date.lt(year_start), "signal_date"].max()
        if pd.isna(fit_date):
            raise ValueError(f"No prior-year signal date is available before validation in {year}.")
        train = panel.loc[
            panel.signal_date.ge("2010-01-01") & panel.signal_date.lt(fit_date)
            & panel.label_end.notna() & panel.label_end.lt(fit_date) & panel.target.notna()
        ].copy()
        model, _, _ = fit_model(train, chosen_alpha)
        prior_signal = fit_date
        score_dates = [prior_signal] + sorted(validation.signal_date.unique().tolist())
        scoring = panel.loc[panel.signal_date.isin(score_dates)].copy()
        scoring["prediction"] = model.predict(scoring[FEATURES])
        predictions.append(scoring[["signal_date", "entry_date", "symbol", "drawdown_252", "prediction"]])
    scores = pd.concat(predictions, ignore_index=True).drop_duplicates(["signal_date", "symbol"], keep="last")
    return metrics, mean.reset_index(), chosen_alpha, scores


@dataclass
class Lot:
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    fee_paid: float
    due_date: pd.Timestamp
    due_session: pd.Timestamp
    signal_date: pd.Timestamp


class CashLedger:
    """Simple long-only cash ledger; each purchase is an independently dated lot."""
    def __init__(self, initial_cash=STARTING_CASH, fee_bps=FEE_BPS):
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots: list[Lot] = []
        self.fills: list[dict] = []

    def buy_budget(self, symbol, budget, price, date, signal_date, calendar):
        if budget <= 0 or price <= 0 or not np.isfinite([budget, price]).all():
            return
        units = budget / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        total = notional + fee
        if total > self.cash + 1e-6:
            raise ValueError("Purchase would exceed available cash.")
        self.cash = max(0.0, self.cash - total)
        due = pd.Timestamp(date) + pd.Timedelta(days=HOLDING_DAYS)
        due_i = int(calendar.searchsorted(due, side="left"))
        due_session = calendar[due_i] if due_i < len(calendar) else pd.NaT
        lot = Lot(symbol, units, pd.Timestamp(date), price, fee, due, due_session, pd.Timestamp(signal_date))
        self.lots.append(lot)
        self.fills.append({"date": date, "signal_date": signal_date, "symbol": symbol, "side": "BUY",
                           "price": price, "units": units, "notional": notional, "fee": fee,
                           "cash_change": -total, "cash_after": self.cash, "due_date": due,
                           "due_session": due_session,
                           "entry_price": price, "realized_pnl": 0.0})

    def sell_matured(self, date, quotes):
        remaining = []
        for lot in self.lots:
            price = quotes.get(lot.symbol)
            if pd.Timestamp(date) == lot.due_session and price is not None and np.isfinite(price) and price > 0:
                proceeds = lot.units * price
                fee = proceeds * self.fee_rate
                pnl = proceeds - fee - lot.units * lot.entry_price - lot.fee_paid
                self.cash += proceeds - fee
                self.fills.append({"date": date, "signal_date": lot.signal_date, "symbol": lot.symbol,
                                   "side": "SELL", "price": price, "units": lot.units, "notional": proceeds,
                                   "fee": fee, "cash_change": proceeds - fee, "cash_after": self.cash,
                                   "due_date": lot.due_date, "due_session": lot.due_session,
                                   "entry_price": lot.entry_price,
                                   "realized_pnl": pnl})
            else:
                remaining.append(lot)
        self.lots = remaining

    def mark(self, last_marks):
        holding_value = sum(lot.units * last_marks[lot.symbol] for lot in self.lots)
        return self.cash + holding_value, holding_value


def load_quotes(database: Path, symbols, start, end, batch_size=250):
    """Load only benchmark and securities that can actually be selected."""
    quotes = {}
    symbols = sorted(set(symbols))
    with duckdb.connect(str(database), read_only=True) as con:
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset:offset + batch_size]
            part = con.execute(
                """SELECT symbol, date, adj_close FROM prices
                   WHERE symbol IN (SELECT unnest(?)) AND date BETWEEN ? AND ?
                   ORDER BY date, symbol""", [batch, start.date(), end.date()]
            ).fetchdf()
            part["date"] = pd.to_datetime(part.date)
            for symbol, group in part.groupby("symbol", sort=False):
                quotes[symbol] = dict(zip(group.date, group.adj_close.astype(float)))
    return quotes


def selected_orders(scores, cash, nav):
    eligible = scores.sort_values(["prediction", "symbol"], ascending=[False, True]).head(TOP_N)
    if eligible.empty or cash <= 0 or nav <= 0:
        return {}
    cohort_budget = min(cash, nav * COHORT_FRACTION)
    each = cohort_budget / len(eligible)
    return {row.symbol: each for row in eligible.itertuples()}


def simulate(database, calendar, benchmark, scores, start, end):
    """Run daily close-to-close NAV with next-session entries and dated lot exits."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    test_days = calendar[(calendar >= start) & (calendar <= end)]
    if not len(test_days):
        raise ValueError(f"No market sessions in {start.date()}–{end.date()}.")
    prior_day = calendar[calendar.get_loc(test_days[0]) - 1]
    benchmark_series = benchmark.set_index("date").adj_close.astype(float).sort_index()
    scores = scores.copy()
    score_map = {d: g.copy() for d, g in scores.groupby("signal_date", sort=True)}
    potential = set()
    for signal_date, group in score_map.items():
        next_i = calendar.get_loc(signal_date) + 1
        if next_i < len(calendar) and start <= calendar[next_i] <= end:
            potential.update(selected_orders(group, STARTING_CASH, STARTING_CASH).keys())
    quote_data = load_quotes(database, potential, prior_day, end)
    first_benchmark = float(benchmark_series.loc[test_days[0]])
    ledger = CashLedger()
    marks = {}
    planned = {}
    daily = [{"date": prior_day, "cash": STARTING_CASH, "holdings_value": 0.0,
              "nav": STARTING_CASH, "open_lots": 0, "fees_paid_cumulative": 0.0}]
    for i, day in enumerate(calendar[(calendar >= prior_day) & (calendar <= end)]):
        today_quotes = {s: float(q[day]) for s, q in quote_data.items() if day in q and np.isfinite(q[day]) and q[day] > 0}
        ledger.sell_matured(day, today_quotes)
        if day in planned:
            signal_date, budgets = planned.pop(day)
            for symbol, budget in budgets.items():
                price = today_quotes.get(symbol)
                if price is not None:
                    ledger.buy_budget(symbol, budget, price, day, signal_date, calendar)
        for symbol, price in today_quotes.items():
            marks[symbol] = price
        nav, holding_value = ledger.mark(marks)
        fees_paid = sum(fill["fee"] for fill in ledger.fills)
        daily.append({"date": day, "cash": ledger.cash, "holdings_value": holding_value,
                      "nav": nav, "open_lots": len(ledger.lots), "fees_paid_cumulative": fees_paid})
        next_i = calendar.get_loc(day) + 1
        if next_i < len(calendar):
            group = score_map.get(day)
            if group is not None and start <= calendar[next_i] <= end:
                budgets = selected_orders(group, ledger.cash, nav)
                if budgets:
                    # Budgets are reserved from cash known at the signal close.
                    if sum(budgets.values()) > ledger.cash + 1e-6:
                        raise AssertionError("Planned orders exceed signal-close cash.")
                    planned[calendar[next_i]] = (day, budgets)
    history = pd.DataFrame(daily).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    history = history.loc[history.date.between(prior_day, end)].copy()
    history["benchmark_nav"] = STARTING_CASH * (1 - FEE_BPS / 10_000) * benchmark_series.reindex(history.date).ffill().to_numpy() / first_benchmark
    history.loc[history.date < test_days[0], "benchmark_nav"] = STARTING_CASH
    history["stale_open_lots"] = 0
    stale_value = 0.0
    last_date = pd.Timestamp(end)
    for lot in ledger.lots:
        last_quote_date = max((d for d in quote_data.get(lot.symbol, {}) if d <= last_date), default=None)
        age = (last_date - last_quote_date).days if last_quote_date is not None else 99999
        if age > 5:
            stale_value += lot.units * marks.get(lot.symbol, lot.entry_price)
    history.loc[history.index[-1], "stale_open_lots"] = sum(
        1 for lot in ledger.lots
        if (last_date - max((d for d in quote_data.get(lot.symbol, {}) if d <= last_date), default=last_date)).days > 5
    )
    history["strategy_return"] = history.nav.pct_change().fillna(0.0)
    history["benchmark_return"] = history.benchmark_nav.pct_change().fillna(0.0)
    return history, pd.DataFrame(ledger.fills), ledger.lots, stale_value


def performance(history):
    """Calculate requested daily NAV metrics using a 2% annual risk-free rate."""
    results = []
    for name, column in (("Strategy", "nav"), ("S&P 500 total return", "benchmark_nav")):
        nav = history[column].astype(float)
        returns = nav.pct_change().dropna()
        years = (history.date.iloc[-1] - history.date.iloc[0]).days / 365.25
        cagr = (nav.iloc[-1] / STARTING_CASH) ** (1 / years) - 1 if years > 0 else np.nan
        drawdown = nav / nav.cummax() - 1
        daily_rf = (1 + RISK_FREE_RATE) ** (1 / 252) - 1
        sharpe = ((returns - daily_rf).mean() / returns.std(ddof=1) * np.sqrt(252)
                  if len(returns) > 1 and returns.std(ddof=1) > 0 else np.nan)
        results.append({"portfolio": name, "final_value": nav.iloc[-1], "CAGR": cagr,
                        "max_drawdown": drawdown.min(), "Sharpe_2pct_rf": sharpe})
    return pd.DataFrame(results)


def accounting_checks(history, fills, open_lots):
    """Record mechanical checks for cash, long-only exposure, and lot timing."""
    buys = fills.loc[fills.side.eq("BUY")]
    sells = fills.loc[fills.side.eq("SELL")]
    ledger_rows = fills.copy()
    if len(ledger_rows):
        ledger_rows["cash_rebuilt"] = STARTING_CASH + ledger_rows.cash_change.cumsum()
        fill_cash_reconciles = bool(np.allclose(ledger_rows.cash_rebuilt, ledger_rows.cash_after,
                                                rtol=1e-9, atol=1e-5))
        daily_cash = ledger_rows.groupby("date").cash_change.sum().sort_index().cumsum() + STARTING_CASH
        history_cash = history.set_index("date").cash
        daily_cash_reconciles = bool(np.allclose(daily_cash.reindex(history_cash.index).ffill().fillna(STARTING_CASH),
                                                 history_cash, rtol=1e-9, atol=1e-5))
    else:
        fill_cash_reconciles = daily_cash_reconciles = True
    check = {
        "all_buy_signals_precede_fills": bool((buys.signal_date < buys.date).all()) if len(buys) else True,
        "all_sells_on_scheduled_maturity_session": bool((sells.date == sells.due_session).all()) if len(sells) else True,
        "no_negative_cash": bool((fills.cash_after >= -1e-6).all()) if len(fills) else True,
        "cash_rebuilds_from_fill_ledger": fill_cash_reconciles,
        "daily_cash_matches_fill_ledger": daily_cash_reconciles,
        "no_borrowing_or_short_lots": bool(all(lot.units > 0 for lot in open_lots)),
        "all_nav_equals_cash_plus_marked_holdings": bool(np.allclose(history.nav, history.cash + history.holdings_value, rtol=1e-9, atol=1e-5)),
        "test_end_within_cutoff": bool(history.date.max() <= pd.Timestamp("2026-09-30")),
        "buy_count": int(len(buys)),
        "sell_count": int(len(sells)),
        "open_lots_at_end": int(len(open_lots)),
    }
    if not all(v for k, v in check.items() if isinstance(v, bool)):
        raise AssertionError(f"Accounting/timing check failed: {check}")
    return pd.DataFrame([{"check": k, "result": v} for k, v in check.items()])


def save_data_quality_summary(panel):
    """Expose extreme adjusted-price labels rather than silently deleting them."""
    labels = panel.loc[panel.target.notna(), "target"]
    rows = [
        ("mature_development_labels", len(labels)),
        ("labels_with_return_over_100pct", int((labels > 1.0).sum())),
        ("labels_with_return_over_300pct", int((labels > 3.0).sum())),
        ("labels_with_return_below_minus_90pct", int((labels < -0.90).sum())),
        ("minimum_raw_label_return", float(labels.min())),
        ("median_raw_label_return", float(labels.median())),
        ("maximum_raw_label_return", float(labels.max())),
    ]
    pd.DataFrame(rows, columns=["measure", "value"]).to_csv(OUT / "data_quality_summary.csv", index=False)


def save_outputs(validation, alpha_summary, chosen_alpha, validation_scores, database,
                calendar, benchmark, panel, dev_only=False):
    OUT.mkdir(exist_ok=True)
    validation.to_csv(OUT / "model_validation.csv", index=False)
    alpha_summary.to_csv(OUT / "model_selection.csv", index=False)

    dev_start, dev_end = pd.Timestamp("2013-01-01"), pd.Timestamp("2015-12-31")
    dev_ridge_scores = validation_scores.copy()
    dev_momentum_scores = panel.loc[
        panel.signal_date.between("2012-12-01", dev_end)
        & panel.momentum_63.notna(),
        ["signal_date", "entry_date", "symbol", "momentum_63"],
    ].rename(columns={"momentum_63": "prediction"})

    development_rows = []
    for name, scores in (("Ridge prediction", dev_ridge_scores),
                         ("63-session momentum", dev_momentum_scores)):
        history, fills, lots, stale = simulate(
            database, calendar, benchmark, scores, dev_start, dev_end
        )
        row = performance(history).iloc[0].to_dict()
        row.update({
            "strategy": name,
            "total_fees": float(fills.fee.sum()) if len(fills) else 0.0,
            "average_invested_fraction": float((1 - history.cash / history.nav).mean()),
            "buys": int(fills.side.eq("BUY").sum()) if len(fills) else 0,
            "sells": int(fills.side.eq("SELL").sum()) if len(fills) else 0,
            "ending_open_lots": len(lots),
            "stale_open_lots": int(history.stale_open_lots.iloc[-1]),
            "stale_open_value": stale,
        })
        development_rows.append(row)
    development = pd.DataFrame(development_rows)
    development.to_csv(OUT / "development_strategy_comparison.csv", index=False)

    print("Selected Ridge alpha:", chosen_alpha)
    print("Development strategy comparison (2013–2015):")
    print(development.to_string(index=False))
    if dev_only or calendar[-1] < pd.Timestamp("2026-09-30"):
        return

    # Fit once after the buffer. Only labels already mature before this close
    # enter the fit; the model remains fixed throughout the final test.
    test_start, test_end = pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30")
    first_test_i = int(calendar.searchsorted(test_start, side="left"))
    fit_date = calendar[first_test_i - 1]
    train = panel.loc[
        panel.signal_date.ge("2010-01-01") & panel.signal_date.lt(fit_date)
        & panel.label_end.notna() & panel.label_end.lt(fit_date) & panel.target.notna()
    ].copy()
    assert train.label_end.max() < fit_date
    final_model, final_y_low, final_y_high = fit_model(train, chosen_alpha)

    final_rows = panel.loc[panel.signal_date.between(fit_date, "2026-08-31")].copy()
    final_rows["prediction"] = final_model.predict(final_rows[FEATURES])
    ridge_scores = final_rows[["signal_date", "entry_date", "symbol", "prediction"]]
    momentum_scores = panel.loc[
        panel.signal_date.between(fit_date, "2026-08-31")
        & panel.momentum_63.notna(),
        ["signal_date", "entry_date", "symbol", "momentum_63"],
    ].rename(columns={"momentum_63": "prediction"})

    final_histories = {}
    all_fills = []
    all_open_positions = []
    all_checks = []
    comparison_rows = []
    for name, scores in (("Ridge prediction", ridge_scores),
                         ("63-session momentum", momentum_scores)):
        history, fills, open_lots, stale_value = simulate(
            database, calendar, benchmark, scores, test_start, test_end
        )
        summary = performance(history).iloc[0].to_dict()
        summary.update({
            "portfolio": name,
            "total_fees": float(fills.fee.sum()) if len(fills) else 0.0,
            "average_invested_fraction": float((1 - history.cash / history.nav).mean()),
            "buys": int(fills.side.eq("BUY").sum()) if len(fills) else 0,
            "sells": int(fills.side.eq("SELL").sum()) if len(fills) else 0,
            "open_lots_at_end": len(open_lots),
            "stale_open_lots": int(history.stale_open_lots.iloc[-1]),
            "stale_open_value": stale_value,
        })
        comparison_rows.append(summary)
        if len(fills):
            fills.insert(0, "strategy", name)
        all_fills.append(fills)
        positions = pd.DataFrame([lot.__dict__ for lot in open_lots])
        if len(positions):
            positions.insert(0, "strategy", name)
        all_open_positions.append(positions)
        checks = accounting_checks(history, fills.drop(columns="strategy", errors="ignore"), open_lots)
        checks.insert(0, "strategy", name)
        all_checks.append(checks)
        final_histories[name] = history

    history = final_histories["Ridge prediction"].copy()
    momentum_history = final_histories["63-session momentum"]
    history = history.rename(columns={
        "nav": "ridge_nav", "cash": "ridge_cash",
        "holdings_value": "ridge_holdings_value", "open_lots": "ridge_open_lots",
        "fees_paid_cumulative": "ridge_fees_paid_cumulative",
    })
    momentum_history = momentum_history.rename(columns={
        "nav": "momentum_nav", "cash": "momentum_cash",
        "holdings_value": "momentum_holdings_value", "open_lots": "momentum_open_lots",
        "fees_paid_cumulative": "momentum_fees_paid_cumulative",
    })
    columns = ["date", "momentum_nav", "momentum_cash", "momentum_holdings_value",
               "momentum_open_lots", "momentum_fees_paid_cumulative", "stale_open_lots"]
    history = history.merge(momentum_history[columns], on="date", how="left")
    history = history.rename(columns={"stale_open_lots_x": "ridge_stale_open_lots",
                                     "stale_open_lots_y": "momentum_stale_open_lots"})
    history["ridge_return"] = history.ridge_nav.pct_change().fillna(0.0)
    history["momentum_return"] = history.momentum_nav.pct_change().fillna(0.0)

    benchmark_summary = performance(final_histories["Ridge prediction"]).iloc[1].to_dict()
    benchmark_summary.update({
        "portfolio": "S&P 500 total-return benchmark",
        "total_fees": FEE_BPS / 10_000 * STARTING_CASH,
        "average_invested_fraction": 1.0,
        "buys": 1, "sells": 0, "open_lots_at_end": 0, "stale_open_lots": 0,
        "stale_open_value": 0.0,
    })
    comparison = pd.DataFrame(comparison_rows + [benchmark_summary])
    comparison.to_csv(OUT / "portfolio_comparison.csv", index=False)
    history.to_csv(OUT / "daily_portfolio.csv", index=False, date_format="%Y-%m-%d")
    pd.concat(all_fills, ignore_index=True).to_csv(
        OUT / "fills.csv", index=False, date_format="%Y-%m-%d"
    )
    pd.concat(all_checks, ignore_index=True).to_csv(OUT / "accounting_checks.csv", index=False)
    pd.concat(all_open_positions, ignore_index=True).to_csv(
        OUT / "open_positions_at_end.csv", index=False
    )
    pd.DataFrame([{
        "training_rows": len(train), "fit_date": fit_date,
        "winsorized_target_low": final_y_low, "winsorized_target_high": final_y_high,
        "training_targets_winsorized": int(
            ((train.target < final_y_low) | (train.target > final_y_high)).sum()
        ),
        "fixed_model_for_test": True,
    }]).to_csv(OUT / "final_model_fit.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    series_to_plot = [
        ("Ridge predicted-return strategy", "ridge_nav", "tab:blue"),
        ("S&P 500 total-return benchmark", "benchmark_nav", "tab:green"),
    ]
    for label, column, color in series_to_plot:
        growth = history[column] / STARTING_CASH
        ax.plot(history.date, growth, label=label, linewidth=1.8, color=color)
        ax.scatter(history.date.iloc[-1], growth.iloc[-1], color=color, s=20, zorder=3)
        ax.annotate(f"{growth.iloc[-1]:.2f}×",
                    (history.date.iloc[-1], growth.iloc[-1]),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yscale("log")
    ax.axhline(1, color="gray", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set(title="Cumulative returns, April 2016–September 2026",
           xlabel="Date", ylabel="Growth of $1 (log scale, fees included)")
    ax.grid(which="major", alpha=0.25)
    ax.grid(which="minor", alpha=0.10)
    ax.legend(loc="upper left", frameon=True)
    fig.subplots_adjust(right=0.88)
    fig.savefig(OUT / "cumulative_returns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Selected Ridge alpha:", chosen_alpha)
    print("Final-model fit date:", fit_date.date(), "Training rows:", len(train))
    print("Final-test comparison:")
    print(comparison.to_string(index=False))
    print("Accounting checks:")
    print(pd.concat(all_checks, ignore_index=True).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Run the HW4 stock-prediction and backtesting workflow.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help="Course DuckDB file (read-only).")
    parser.add_argument("--dev-only", action="store_true", help="Run development validation only; useful with the class database.")
    args = parser.parse_args()
    if not args.database.is_file():
        raise FileNotFoundError("Database not found. Place the course database in Lectures/modeling/data/.")
    calendar, benchmark, membership, symbols = read_reference_data(args.database)
    print(f"Database securities: {len(symbols):,}; sessions: {len(calendar):,}; date range: {calendar[0].date()}–{calendar[-1].date()}")
    panel = make_research_panel(args.database, calendar, membership, symbols)
    print(f"Monthly stock observations: {len(panel):,}; mature development labels: {panel.target.notna().sum():,}")
    OUT.mkdir(exist_ok=True)
    save_data_quality_summary(panel)
    validation, alpha_summary, chosen_alpha, dev_scores = validate_models(panel)
    print("Chronological validation folds:")
    print(validation.to_string(index=False))
    save_outputs(validation, alpha_summary, chosen_alpha, dev_scores, args.database, calendar,
                 benchmark, panel, dev_only=args.dev_only)


if __name__ == "__main__":
    main()
