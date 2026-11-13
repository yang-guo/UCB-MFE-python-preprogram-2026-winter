"""Final test: the fixed strategy on April 1, 2016 to September 30, 2026.

Every setting below was chosen on development data (2010-2015) before this
script was first run. Do not change a setting after seeing these results.

Stages (each saves its output, so a rerun continues where it stopped):
  1. Features and 90-day targets for model D through 2026-09-30.
  2. Suspect-price rule: drop rows whose lookback has a one-day price ratio
     outside [0.2, 5]; blank targets whose holding window has one.
  3. Annual ridge fits. First fit on 2016-04-01, then every January 1. Each
     fit trains only on targets whose holding period ended before the fit.
  4. Backtests: the primary strategy, three declared sensitivity variants,
     and a zero-recovery stress case. Compare with $1,000,000 in ^SP500TR
     bought at the first test close.

Run from the repository root:
    .venv/Scripts/python.exe Homeworks/HW4/Mauricio_Gutierrez/run_final_test.py
"""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import gc
import json
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HW_DIR))
from feature_engineering import (  # noqa: E402
    D_RATIOS, PRICE_FEATURES, build_full_development_cache, clip_training_target,
    make_ridge_pipeline, prediction_metrics, process_peak_memory_mib, suspect_price_flags,
)
from strategy_backtest import (  # noqa: E402
    BacktestConfig, Backtester, InverseVolatilityStrategy, detailed_performance, reconcile_account,
)

# ---- Fixed settings (chosen in development) ---------------------------------
DB_PATH = HW_DIR / "outputs/data_cleaning/equities_top2000/equities_top2000_cleaned.duckdb"
OUTPUT = HW_DIR / "outputs/final_test"
DEV_FORECASTS = HW_DIR / "outputs/full_data_development/suspect_rule/forecasts.csv"
LOOKBACK_START = "2000-01-01"
TRAIN_START = "2010-01-01"
TEST_START = "2016-04-01"
TEST_END = "2026-09-30"
FEATURES = PRICE_FEATURES + D_RATIOS + ["filing_age_days"]   # Model D.
ALPHA = 100.0
FIT_DATES = [TEST_START] + [f"{year}-01-01" for year in range(2017, 2027)]
VOLATILITY_QUANTILE = 0.80
SUSPECT_RATIO = (0.2, 5.0)
MIN_CASH_WEIGHT = 0.05
MIN_SESSIONS_BETWEEN = 21
CONFIG = BacktestConfig(initial_cash=1_000_000, top_n=30, budget_fraction=1 / 3, holding_days=90,
                        stock_limit=0.30, sector_limit=0.50, fee_rate=0.001, risk_free_rate=0.02)
PRIMARY = "Strategy (largest 1,000)"
# name: largest allowed universe_rank. Only PRIMARY is the reported result;
# the others were declared as sensitivity checks before the run.
VARIANTS = {PRIMARY: 1000, "Sensitivity: largest 500": 500,
            "Sensitivity: largest 200": 200, "Sensitivity: all 2,000": 2000}
CACHE = OUTPUT / "features.duckdb"
FILTERED = OUTPUT / "filtered_features.duckdb"
FORECASTS = OUTPUT / "forecasts.parquet"


def stage_features():
    """Stage 1: same feature code as development, extended to the test end."""
    if CACHE.exists() and (OUTPUT / "stage1.done").exists():
        return
    CACHE.unlink(missing_ok=True)
    audit = build_full_development_cache(DB_PATH, CACHE, lookback_start=LOOKBACK_START,
                                         train_start=TRAIN_START, end=TEST_END, group_size=50,
                                         allow_final_test=True)
    (OUTPUT / "stage1.done").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print("Stage 1 features:", audit)


def stage_suspect_rule():
    """Stage 2: apply the suspect-price rule with past data only."""
    if FILTERED.exists() and (OUTPUT / "stage2.done").exists():
        return
    FILTERED.unlink(missing_ok=True)
    dropped_rows = blanked_targets = 0
    with duckdb.connect(str(DB_PATH), read_only=True) as src, \
            duckdb.connect(str(CACHE), read_only=True) as cache, duckdb.connect(str(FILTERED)) as dest:
        dest.execute("SET memory_limit='384MB'")
        calendar = pd.DatetimeIndex(src.execute(
            "SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
            [LOOKBACK_START, TEST_END]).fetchdf().date).astype("datetime64[ns]")
        symbols = cache.execute("SELECT DISTINCT symbol FROM D ORDER BY symbol").fetchnumpy()["symbol"].tolist()
        created = False
        for start in range(0, len(symbols), 50):
            group = symbols[start:start + 50]
            prices = src.execute(
                "SELECT symbol, date, adj_close FROM prices WHERE date BETWEEN ? AND ? "
                "AND symbol IN (SELECT unnest(?)) ORDER BY symbol, date", [LOOKBACK_START, TEST_END, group]).fetchdf()
            prices["date"] = pd.to_datetime(prices.date).astype("datetime64[ns]")
            rows = cache.execute("SELECT * FROM D WHERE symbol IN (SELECT unnest(?)) ORDER BY date, symbol",
                                 [group]).fetchdf()
            flags = suspect_price_flags(prices, calendar, rows)
            assert rows[["date", "symbol"]].equals(flags[["date", "symbol"]])
            blanked_targets += int((flags.suspect_label & rows.target.notna()).sum())
            # A contaminated outcome window never removes a live candidate: only its target.
            rows.loc[flags.suspect_label, "target"] = np.nan
            kept = rows.loc[~flags.suspect_lookback]
            dropped_rows += len(rows) - len(kept)
            dest.register("part", kept)
            dest.execute(("INSERT INTO D BY NAME " if created else "CREATE TABLE D AS ") + "SELECT * FROM part")
            dest.unregister("part")
            created = True
            del prices, rows, flags, kept
            gc.collect()
        dest.execute("CHECKPOINT")
    audit = {"rows_dropped_suspect_lookback": dropped_rows, "targets_blanked_suspect_label": blanked_targets}
    (OUTPUT / "stage2.done").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Stage 2 suspect rule:", audit)


def fit_and_forecast(con, first_signal, last_signal):
    """One fit: train on completed targets before first_signal, forecast the window."""
    columns = ", ".join(["date", "symbol", "target", "label_end"] + FEATURES)
    signals = con.execute(f"SELECT {columns} FROM D WHERE date BETWEEN ? AND ? ORDER BY date, symbol",
                          [first_signal, last_signal]).fetchdf()
    cutoff = signals.date.min()
    train = con.execute(f"SELECT {columns} FROM D WHERE date >= ? AND date < ? AND label_end < ? "
                        "AND isfinite(target) ORDER BY date, symbol", [TRAIN_START, cutoff, cutoff]).fetchdf()
    assert train.label_end.max() < cutoff and train.date.max() < cutoff
    target, clip = clip_training_target(train.target)
    model = make_ridge_pipeline(alpha=ALPHA, feature_names=FEATURES)
    model.fit(train[FEATURES], target)
    forecast = signals[["date", "symbol", "target", "label_end", "volatility_63"]].copy()
    forecast["prediction"] = model.predict(signals[FEATURES])
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = dict(zip(names, model.named_steps["model"].coef_))
    coefficients["intercept"] = float(model.named_steps["model"].intercept_)
    split = {"first_signal": cutoff, "last_signal": signals.date.max(), "training_rows": len(train),
             "last_training_outcome": train.label_end.max(), **clip}
    del train, signals
    gc.collect()
    return forecast, split, coefficients


def stage_forecasts():
    """Stage 3: pipeline check on development, then the test-period fits."""
    if FORECASTS.exists() and (OUTPUT / "stage3.done").exists():
        return
    with duckdb.connect(str(FILTERED), read_only=True) as con:
        if DEV_FORECASTS.is_file():
            check_against_development(con)
        else:
            print("Development forecasts not found: skipping the 2015 pipeline check.")
        fit_test_period(con)


def check_against_development(con):
    """Before opening the test period, reproduce the saved development forecasts.

    The development forecasts for 2015 were made with data ending in 2015. The
    same code, now reading data through 2026, must return the same numbers. A
    difference would mean later data leaks into earlier features.
    """
    check, _, _ = fit_and_forecast(con, "2015-01-01", "2015-12-31")
    saved = pd.read_csv(DEV_FORECASTS, usecols=["date", "symbol", "prediction", "validation_year"],
                        parse_dates=["date"])
    saved = saved.loc[saved.validation_year.eq(2015)]
    merged = saved.merge(check[["date", "symbol", "prediction"]], on=["date", "symbol"],
                         suffixes=("_saved", "_now"), validate="one_to_one")
    difference = (merged.prediction_saved - merged.prediction_now).abs().max()
    print(f"Pipeline check on 2015: {len(merged):,} of {len(saved):,} forecasts matched, "
          f"largest difference {difference:.2e}")
    assert len(merged) == len(saved) and difference < 1e-8, "Pipeline differs from development."
    # The only extra rows allowed are on 2015-12-31. Development stopped its calendar
    # there, so that session had no next session and could not be a signal date.
    extra = check.loc[~check.set_index(["date", "symbol"]).index.isin(saved.set_index(["date", "symbol"]).index)]
    assert extra.date.eq(pd.Timestamp("2015-12-31")).all(), "Unexpected extra rows before the development end."
    print(f"Extra rows only on 2015-12-31: {len(extra):,}")
    del extra
    del check, saved, merged
    gc.collect()


def fit_test_period(con):
    """Annual fits and forecasts for the test period; saves forecasts and scores."""
    parts, splits, coefficients, scores = [], [], [], []
    for k, first in enumerate(FIT_DATES):
        last = (pd.Timestamp(FIT_DATES[k + 1]) - pd.Timedelta(days=1)) if k + 1 < len(FIT_DATES) else pd.Timestamp(TEST_END)
        started = perf_counter()
        forecast, split, coefs = fit_and_forecast(con, first, str(last.date()))
        forecast["fit_date"] = pd.Timestamp(first)
        parts.append(forecast)
        splits.append({"fit_date": first, **split, "seconds": perf_counter() - started})
        coefficients.append({"fit_date": first, **coefs})
        scored = forecast.loc[forecast.target.notna()]
        scores.append({"fit_date": first, **prediction_metrics(scored, scored.prediction).to_dict(),
                       "scored_rows": len(scored), "forecast_rows": len(forecast)})
        print(f"Fit {first}: {split['training_rows']:,} training rows, "
              f"{len(forecast):,} forecasts, peak memory {process_peak_memory_mib():,.0f} MiB")
    forecasts = pd.concat(parts, ignore_index=True)
    assert forecasts.date.min() >= pd.Timestamp(TEST_START) and forecasts.date.max() <= pd.Timestamp(TEST_END)
    forecasts.to_parquet(FORECASTS, index=False)
    pd.DataFrame(splits).to_csv(OUTPUT / "training_splits.csv", index=False)
    pd.DataFrame(coefficients).set_index("fit_date").T.to_csv(OUTPUT / "ridge_coefficients.csv")
    pd.DataFrame(scores).to_csv(OUTPUT / "model_scores.csv", index=False)
    (OUTPUT / "stage3.done").write_text("done", encoding="utf-8")


def load_test_inputs():
    """Read only what the execution engine needs, bounded at the test end."""
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        prices = con.execute("SELECT symbol, date, adj_close FROM prices WHERE date BETWEEN ? AND ? "
                             "ORDER BY symbol, date", [TEST_START, TEST_END]).fetchdf()
        sessions = con.execute("SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
                               [TEST_START, TEST_END]).fetchdf()
        membership = con.execute(
            "SELECT symbol, quarter_start, quarter_end, selection_date, universe_rank FROM universe_membership "
            "WHERE quarter_start <= ? AND selection_date <= ? ORDER BY quarter_start, universe_rank",
            [TEST_END, TEST_END]).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe").fetchdf()
        benchmark = con.execute("SELECT date, adj_close FROM benchmark WHERE symbol = '^SP500TR' "
                                "AND date BETWEEN ? AND ? ORDER BY date", [TEST_START, TEST_END]).fetchdf()
    prices["date"] = pd.to_datetime(prices.date).astype("datetime64[ns]")
    benchmark["date"] = pd.to_datetime(benchmark.date).astype("datetime64[ns]")
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    calendar = pd.DatetimeIndex(pd.to_datetime(sessions.date)).astype("datetime64[ns]")
    valid = prices.date.isin(calendar) & np.isfinite(prices.adj_close) & prices.adj_close.gt(0)
    prices = prices.loc[valid].reset_index(drop=True)
    assert not prices.duplicated(["symbol", "date"]).any() and not universe.symbol.duplicated().any()
    assert prices.date.max() <= pd.Timestamp(TEST_END) and benchmark.date.max() <= pd.Timestamp(TEST_END)
    assert benchmark.set_index("date").adj_close.reindex(calendar).gt(0).all()
    return prices, calendar, membership, universe.set_index("symbol").sector.to_dict(), benchmark


def engine(config, recovery=1.0):
    return Backtester(config, InverseVolatilityStrategy(config), allow_stale_quotes=True,
                      no_quote_recovery=recovery, volatility_quantile=VOLATILITY_QUANTILE, schedule="cash",
                      min_cash_weight=MIN_CASH_WEIGHT, min_sessions_between=MIN_SESSIONS_BETWEEN,
                      suspect_ratio=SUSPECT_RATIO)


def timing_checks(result, calendar):
    """Evidence that no decision used a later price and the test window is respected."""
    trades, decisions = result.trades, result.decisions
    buys = trades.loc[trades.side.eq("BUY")]
    sells = trades.loc[trades.side.eq("SELL")]
    position = {date: i for i, date in enumerate(calendar)}
    waiting_lots = set(result.valuation_flags.lot_id)
    checks = {
        "first decision is the first test session": decisions.signal_date.min() == calendar[0],
        "every entry is the session after its signal":
            all(position[e] == position[s] + 1 for s, e in zip(buys.signal_date, buys.entry_date)),
        "no trade or mark after the test end": trades.date.max() <= calendar[-1] and result.daily.date.max() == calendar[-1],
        "every sale is on or after 90 calendar days": (sells.date >= sells.due_date).all(),
        # A later sale is allowed only for a lot that was waiting for a quote (flagged).
        "every sale is the first session on or after maturity, unless waiting for a quote":
            all(d == calendar[calendar.searchsorted(due)] or lot in waiting_lots
                for d, due, lot in zip(sells.date, sells.due_date, sells.lot_id)),
        "open lots mature after the test end or await a quote":
            bool((result.open_lots.due_date > calendar[-1]).all()
                 or set(result.open_lots.loc[result.open_lots.due_date <= calendar[-1], "lot_id"])
                 <= set(result.valuation_flags.lot_id)),
        "cash never negative": result.daily.cash.min() >= -1e-7,
        "final value equals cash plus open positions":
            bool(np.isclose(result.daily.nav.iloc[-1], result.daily.cash.iloc[-1] + result.open_lots.market_value.sum())),
    }
    return {name: "PASS" if bool(ok) else "FAIL" for name, ok in checks.items()}


def stage_backtests():
    """Stage 4: primary result, sensitivity variants, stress case, chart."""
    prices, calendar, membership, sectors, benchmark = load_test_inputs()
    forecasts = pd.read_parquet(FORECASTS, columns=["date", "symbol", "prediction", "volatility_63"])
    rows, navs, check_rows = [], {}, []
    for name, max_rank in VARIANTS.items():
        members = membership.loc[membership.universe_rank.le(max_rank)]
        result = engine(CONFIG).run(prices, calendar, forecasts, sectors, members, TEST_START, TEST_END)
        check_rows.append({"portfolio": name, **reconcile_account(result, calendar, CONFIG), **timing_checks(result, calendar)})
        curves, metrics = detailed_performance(result, benchmark, CONFIG)
        navs[name] = curves.Strategy
        rows.append({"portfolio": name, **metrics.loc["Strategy"].to_dict(),
                     "cash_at_end": result.daily.cash.iloc[-1],
                     "open_positions_at_end": result.open_lots.market_value.sum(),
                     "decisions": len(result.decisions), "purchases": int(result.trades.side.eq("BUY").sum()),
                     "sales": int(result.trades.side.eq("SELL").sum()), "open_lots_at_end": len(result.open_lots),
                     "cancelled_orders": int(result.orders.status.ne("filled").sum()),
                     "no_quote_exits": int(result.trades.no_quote_exit.fillna(False).astype(bool).sum()),
                     "flagged_valuations": len(result.valuation_flags),
                     "suspect_quotes_removed": len(result.suspect_quotes),
                     "exposure_breach_sessions": int(result.daily.exposure_breach.sum())})
        if name == PRIMARY:
            run_dir = OUTPUT / "primary"
            run_dir.mkdir(exist_ok=True)
            for field in ["daily", "trades", "orders", "decisions", "rejections", "open_lots",
                          "valuation_flags", "suspect_quotes"]:
                getattr(result, field).to_csv(run_dir / f"{field}.csv", index=False)
            stress = engine(CONFIG, recovery=0.0).run(prices, calendar, forecasts, sectors, members, TEST_START, TEST_END)
            reconcile_account(stress, calendar, CONFIG)
            _, stress_metrics = detailed_performance(stress, benchmark, CONFIG)
            rows.append({"portfolio": "Stress: no-quote exits worth zero", **stress_metrics.loc["Strategy"].to_dict(),
                         "no_quote_exits": int(stress.trades.no_quote_exit.fillna(False).astype(bool).sum())})
            del stress
        print(f"{name}: final value ${result.daily.nav.iloc[-1]:,.0f}")
        del result
        gc.collect()
    navs["S&P 500 TR"] = curves["S&P 500 TR"]
    rows.append({"portfolio": "S&P 500 TR", **metrics.loc["S&P 500 TR"].to_dict()})
    table = pd.DataFrame(rows).set_index("portfolio")
    table.to_csv(OUTPUT / "performance_comparison.csv")
    pd.DataFrame(check_rows).to_csv(OUTPUT / "accounting_and_timing_checks.csv", index=False)
    nav = pd.DataFrame(navs)
    nav.to_csv(OUTPUT / "daily_nav.csv")

    cumulative = nav / CONFIG.initial_cash - 1
    figure, axis = plt.subplots(figsize=(11, 5.5))
    cumulative[[PRIMARY, "S&P 500 TR"]].plot(ax=axis, linewidth=1.4, color=["#1f4e79", "#c0504d"])
    axis.set_title("Final test: cumulative return, April 1, 2016 to September 30, 2026")
    axis.set_ylabel("Cumulative return")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "cumulative_returns.png", dpi=130)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    cumulative.plot(ax=axis, linewidth=1.1)
    axis.set_title("Final test: primary strategy, declared sensitivity variants, and benchmark")
    axis.set_ylabel("Cumulative return")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "cumulative_returns_sensitivity.png", dpi=130)
    plt.close(figure)

    primary, index = table.loc[PRIMARY], table.loc["S&P 500 TR"]
    verdict = "beat" if primary.final_value > index.final_value else "did not beat"
    summary = (f"The strategy {verdict} the benchmark. Final value ${primary.final_value:,.0f} against "
               f"${index.final_value:,.0f}, a difference of ${primary.final_value - index.final_value:,.0f}. "
               f"CAGR {primary.CAGR:.2%} against {index.CAGR:.2%}; Sharpe {primary.daily_Sharpe:.3f} against "
               f"{index.daily_Sharpe:.3f}; maximum drawdown {primary.max_drawdown:.2%} against {index.max_drawdown:.2%}.")
    (OUTPUT / "summary.md").write_text(summary + "\n", encoding="utf-8")
    (OUTPUT / "settings.json").write_text(json.dumps({
        "test_start": TEST_START, "test_end": TEST_END, "train_start": TRAIN_START, "features": FEATURES,
        "alpha": ALPHA, "fit_dates": FIT_DATES, "training_target_clip_quantiles": [0.01, 0.99],
        "candidate_volatility_quantile": VOLATILITY_QUANTILE, "suspect_ratio": SUSPECT_RATIO,
        "schedule": "cash", "min_cash_weight": MIN_CASH_WEIGHT, "min_sessions_between": MIN_SESSIONS_BETWEEN,
        "primary": PRIMARY, "variants_max_universe_rank": VARIANTS, "config": asdict(CONFIG)}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    columns = ["final_value", "total_return", "CAGR", "max_drawdown", "annualized_volatility", "daily_Sharpe",
               "average_cash_weight", "fees_paid"]
    print(table[columns].round(4).to_string())
    print(pd.DataFrame(check_rows).set_index("portfolio").T.to_string())
    print(summary)


if __name__ == "__main__":
    began = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_features()
    stage_suspect_rule()
    stage_forecasts()
    stage_backtests()
    print(f"Total {perf_counter() - began:,.0f} seconds; peak memory {process_peak_memory_mib():,.0f} MiB")
