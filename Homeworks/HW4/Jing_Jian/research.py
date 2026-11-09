"""Full HW4 research: development selection, frozen specification, final test.

Run development first. Final evaluation refuses to run without the saved frozen
specification, and never changes its model/strategy based on final returns.
Permitted lecture helpers construct data/features; our engine books trades.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from backtester import DataGapError, StrategyConfig, entry_schedule, performance, run_backtest
from corporate_actions import load_actions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Lectures/modeling"))
from course_data import (  # noqa: E402
    NUMERIC_FEATURES, PRICE_FEATURES, apply_membership, financial_events,
    join_financials, load_data, load_membership, make_labels, make_pipeline,
    price_features,
)

DATABASE = ROOT / "Lectures/modeling/data/equities_top2000.duckdb"
RESULTS = HERE / "results"
OUTPUTS = HERE / "outputs/full"
SCREEN = {
    "max_universe_rank": 200, "min_raw_close": 5.0, "max_raw_close": 10_000.0,
    "max_volatility_63": 0.60, "max_abs_daily_return_63": 0.25,
    "complete_trailing_returns": 252,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_hashes():
    return {name: sha256(HERE / name) for name in ["research.py", "backtester.py"]} | {
        "course_data.py": sha256(ROOT / "Lectures/modeling/course_data.py")
    }


def load_calendar(database, end):
    with duckdb.connect(str(database), read_only=True) as con:
        benchmark = con.execute(
            "SELECT date,adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]
        ).fetchdf()
    benchmark.date = pd.to_datetime(benchmark.date).astype("datetime64[ns]")
    return pd.DatetimeIndex(benchmark.date), benchmark.set_index("date").adj_close


def screen_rows(panel, rule):
    """All restrictions depend only on the signal close and dated membership."""
    return (
        panel.eligible & panel.universe_rank.le(rule["max_universe_rank"])
        & panel.close.between(rule["min_raw_close"], rule["max_raw_close"])
        & panel.volatility_63.le(rule["max_volatility_63"])
        & panel.max_abs_daily_63.le(rule["max_abs_daily_return_63"])
        & panel.complete_returns_252
    )


def prepare_monthly(database, end, label_cutoff):
    """Build causal features in ticker batches, retain monthly decisions only.

    Final-stage labels are built for pre-2016 rows only, with price access cut
    off before the initial final signal. No final-period target is constructed.
    A full symbol inventory is used for reading, never as historical eligibility.
    """
    calendar, _ = load_calendar(database, end)
    calendar = calendar[calendar >= pd.Timestamp("2008-01-01")]
    entries = entry_schedule(calendar, "2010-01-01", end)
    signals = calendar[calendar.get_indexer(entries) - 1]
    membership = load_membership(database, end=end)
    with duckdb.connect(str(database), read_only=True) as con:
        symbols = [row[0] for row in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    pieces = []
    warning_messages = set()
    for offset in range(0, len(symbols), 50):
        batch = symbols[offset:offset + 50]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prices, fundamentals, universe, _ = load_data(database, end=end, symbols=batch)
        warning_messages.update(str(item.message) for item in caught)
        prices = prices.loc[prices.date.ge(calendar[0])].copy()
        if prices.empty:
            continue
        panel = price_features(prices, calendar)
        # Reindex before rolling: missing sessions cannot be bridged or filled.
        quality = []
        for symbol, group in prices.groupby("symbol", sort=True):
            daily = group.set_index("date").adj_close.reindex(calendar).pct_change(fill_method=None)
            quality.append(pd.DataFrame({
                "date": calendar, "symbol": symbol,
                "max_abs_daily_63": daily.abs().rolling(63, min_periods=63).max().to_numpy(),
                "complete_returns_252": daily.rolling(252, min_periods=252).count().eq(252).to_numpy(),
            }))
        panel = panel.merge(pd.concat(quality, ignore_index=True), on=["date", "symbol"], validate="one_to_one")
        panel = panel.loc[panel.date.isin(signals)].copy()
        if panel.empty:
            continue
        panel = join_financials(panel, financial_events(fundamentals), universe)
        panel = apply_membership(panel, membership, calendar)
        train_rows = panel.loc[panel.date.lt("2016-01-01")].copy()
        label_calendar = calendar[calendar <= pd.Timestamp(label_cutoff)]
        labels = make_labels(
            train_rows, prices.loc[prices.date.le(label_cutoff)], label_calendar,
            max_exit_delay=0,
        )
        # Keep final predictions entirely separate from future-return columns.
        future_rows = panel.loc[panel.date.ge("2016-01-01")].copy()
        pieces.extend([labels, future_rows])
        if offset % 500 == 0:
            print(f"  Features: {min(offset + 50, len(symbols))}/{len(symbols)} symbols", flush=True)
    panel = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, sorted(warning_messages)


def read_quotes(database, calendar, symbols):
    """Only read stocks ever selected by prior-close scores, then keep all quotes.

    This is a storage optimization, not a survival screen. It never changes
    scores or orders; missing future quotes remain NaN and trigger engine policy.
    """
    symbols = sorted(set(symbols))
    matrix = pd.DataFrame(np.nan, index=calendar, columns=symbols)
    if not symbols:
        return matrix
    with duckdb.connect(str(database), read_only=True) as con:
        for offset in range(0, len(symbols), 50):
            batch = symbols[offset:offset + 50]
            rows = con.execute("""SELECT date,symbol,adj_close FROM prices
                WHERE date BETWEEN ? AND ? AND symbol IN (SELECT unnest(?))
                AND isfinite(adj_close) AND adj_close>0 ORDER BY date,symbol""",
                [calendar[0], calendar[-1], batch]).fetchdf()
            rows.date = pd.to_datetime(rows.date).astype("datetime64[ns]")
            rows = rows.loc[rows.date.isin(calendar)]
            pivot = rows.pivot(index="date", columns="symbol", values="adj_close")
            matrix.loc[:, batch] = pivot.reindex(index=calendar, columns=batch).to_numpy()
    return matrix


def shortlisted_symbols(scores, top_n):
    ordered = scores.loc[scores.prediction.gt(0)].sort_values(
        ["date", "prediction", "symbol"], ascending=[True, False, True]
    )
    return ordered.groupby("date").head(top_n).symbol.unique()


def fit_and_score(panel, signal_dates, specification):
    fit_before = signal_dates[0]
    eligible = screen_rows(panel, specification["screen"])
    train = panel.loc[
        eligible & panel.date.ge("2010-01-01") & panel.date.lt("2016-01-01")
        & panel.date.lt(fit_before) & panel.label_end.lt(fit_before) & panel.target.notna()
    ].copy()
    decisions = panel.loc[eligible & panel.date.isin(signal_dates)].copy()
    features = specification["features"]
    # Learn target winsorization from the training set only; do not censor
    # validation/test investments based on future returns or label availability.
    lower, upper = train.target.quantile([0.01, 0.99])
    target = train.target.clip(lower, upper)
    model = make_pipeline(Ridge(alpha=specification["alpha"]), features, include_sector=False)
    model.fit(train[features], target)
    scores = decisions[["date", "symbol"]].copy()
    scores["prediction"] = model.predict(decisions[features])
    audit = {
        "fit_before_signal": str(fit_before.date()), "training_rows": len(train),
        "latest_training_signal": str(train.date.max().date()),
        "latest_training_outcome": str(train.label_end.max().date()),
        "target_clip_lower": float(lower), "target_clip_upper": float(upper),
        "eligible_decisions": len(decisions),
        "screened_decisions": int((panel.date.isin(signal_dates) & ~eligible).sum()),
    }
    assert train.label_end.max() < fit_before
    assert train.date.max() < pd.Timestamp("2016-01-01")
    return scores, audit


def simulate(database, scores, start, end, specification, corporate_actions=()):
    calendar, benchmark = load_calendar(database, end)
    entries = entry_schedule(calendar, start, end)
    first_signal = calendar[calendar.get_loc(entries[0]) - 1]
    calendar = calendar[calendar >= first_signal]
    config = StrategyConfig(**specification["portfolio"])
    symbols = set(shortlisted_symbols(scores, config.top_n))
    symbols.update(action["new_symbol"] for action in corporate_actions if action["kind"] in ("stock", "rename"))
    quotes = read_quotes(database, calendar, symbols)
    result = run_backtest(None, calendar, scores, start, end, config, price_matrix=quotes,
                          corporate_actions=corporate_actions)
    benchmark = benchmark.reindex(result.history.index)
    result.history["benchmark_nav"] = config.initial_cash * benchmark / benchmark.iloc[0]
    metrics = {"strategy": performance(result.history.nav, config.initial_cash),
               "benchmark": performance(result.history.benchmark_nav, config.initial_cash)}
    return result, metrics


def development_data_audit(database):
    with duckdb.connect(str(database), read_only=True) as con:
        counts = con.execute("""SELECT count(*) AS membership_rows,
            count(*) FILTER (WHERE selection_close>10000) AS selection_close_over_10000,
            count(*) FILTER (WHERE selection_shares>20000000000) AS selection_shares_over_20bn
            FROM universe_membership WHERE quarter_start<'2016-01-01'""").fetchdf().iloc[0].to_dict()
        extremes = con.execute("""WITH p AS (SELECT symbol,date,close,adj_close,
            adj_close/lag(adj_close) OVER (PARTITION BY symbol ORDER BY date)-1 AS daily_return
            FROM prices WHERE date<'2016-01-01')
            SELECT * FROM p WHERE isfinite(daily_return) ORDER BY abs(daily_return) DESC LIMIT 12""").fetchdf()
        ranks = con.execute("""SELECT quarter_start,symbol,selection_close,selection_shares,
            market_cap_proxy,universe_rank FROM universe_membership
            WHERE quarter_start<'2016-01-01' AND (selection_close>10000 OR selection_shares>20000000000)
            ORDER BY market_cap_proxy DESC LIMIT 15""").fetchdf()
    RESULTS.mkdir(parents=True, exist_ok=True)
    extremes.to_csv(RESULTS / "development_price_flags.csv", index=False)
    ranks.to_csv(RESULTS / "development_ranking_flags.csv", index=False)
    return {key: int(value) for key, value in counts.items()}


def development(database):
    print("Development: reading only observations through 2015-12-31", flush=True)
    data_audit = development_data_audit(database)
    panel, loader_warnings = prepare_monthly(database, "2015-12-31", "2015-12-31")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUTS / "development_panel.parquet", index=False)
    calendar, _ = load_calendar(database, "2015-12-31")
    entries = entry_schedule(calendar, "2013-01-01", "2015-12-31")
    signals = calendar[calendar.get_indexer(entries) - 1]
    trials = []
    for feature_name, features in [("price", PRICE_FEATURES), ("price_financial", NUMERIC_FEATURES)]:
        for top_n in [10, 20]:
            name = f"{feature_name}_top{top_n}"
            specification = {
                "name": name, "features": features, "alpha": 100,
                "screen": SCREEN, "portfolio": asdict(StrategyConfig(top_n=top_n)),
                "training_signals": "2010-01-01 through 2015-12-31; completed outcomes only",
                "fit_rule": "once before first evaluation signal; no final-period retraining",
                "target_winsorization": "training-only 1st/99th percentiles",
            }
            scores, fit_audit = fit_and_score(panel, signals, specification)
            trial = {"name": name, "specification": specification, "fit_audit": fit_audit}
            try:
                result, metrics = simulate(database, scores, "2013-01-01", "2015-12-31", specification)
                trial.update(status="completed", metrics=metrics,
                             stale_marks=len(result.diagnostics), trades=len(result.fills))
            except DataGapError as error:
                trial.update(status="data_gap", error=str(error))
            trials.append(trial)
            print(f"  {name}: {trial['status']}", flush=True)
    write_json(RESULTS / "development.json", {
        "period": "2013-01-02 through 2015-12-31", "database": database.name,
        "data_audit": data_audit, "loader_warnings": loader_warnings,
        "selection_rule": "highest development Sharpe; CAGR tie-break; no missing-quote failures",
        "trials": trials,
    })
    completed = [trial for trial in trials if trial["status"] == "completed"]
    if not completed:
        raise ValueError("All development runs have unresolved quote gaps; investigate before final")
    winner = max(completed, key=lambda trial: (trial["metrics"]["strategy"]["sharpe"],
                                               trial["metrics"]["strategy"]["cagr"]))
    specification = winner["specification"] | {
        "frozen_before_final": True, "selection_source": "development.json",
        "database_sha256": sha256(database), "code_sha256": code_hashes(),
        "development_winner": winner["name"],
    }
    write_json(RESULTS / "frozen_strategy.json", specification)
    print(f"Frozen strategy: {winner['name']}. Final period has not been evaluated.", flush=True)


def accounting_evidence(result, config):
    history, fills = result.history, result.fills
    buys = fills.loc[fills.side.eq("BUY")].set_index("lot_id")
    sells = fills.loc[fills.side.eq("SELL")].set_index("lot_id")
    settlements = fills.loc[fills.side.eq("SETTLEMENT")].set_index("lot_id")
    ledger_cash = (config.initial_cash - (buys.notional + buys.fee).sum()
                   + (sells.notional - sells.fee).sum() + settlements.notional.sum())
    np.testing.assert_allclose(history.nav, history.cash + history.market_value, atol=1e-7)
    np.testing.assert_allclose(ledger_cash, history.cash.iloc[-1], atol=1e-6)
    np.testing.assert_allclose(result.open_lots.market_value.sum() if len(result.open_lots) else 0,
                               history.market_value.iloc[-1], atol=1e-6)
    assert history.cash.ge(0).all()
    assert buys.signal_date.lt(buys.date).all()
    assert ((sells.date - buys.loc[sells.index, "date"]).dt.days >= 90).all()
    assert ((settlements.date - buys.loc[settlements.index, "date"]).dt.days >= 90).all()
    exits = pd.concat([sells, settlements])
    scheduled = pd.DatetimeIndex(buys.loc[exits.index, "date"] + pd.Timedelta(days=90))
    expected = history.index[history.index.searchsorted(scheduled)]
    assert (exits.date.to_numpy() == expected.to_numpy()).all()
    signed_flows = pd.Series(
        np.where(fills.side.eq("BUY"), -(fills.notional + fills.fee), fills.notional - fills.fee),
        index=fills.date,
    )
    cash_path = config.initial_cash + signed_flows.groupby(level=0).sum().reindex(history.index, fill_value=0).cumsum()
    np.testing.assert_allclose(cash_path, history.cash, atol=1e-6)
    assert len(buys) == len(exits) + len(result.open_lots)
    return {
        "nonnegative_cash": True, "daily_nav_reconciles": True,
        "cash_ledger_reconciles": True, "terminal_holdings_reconcile": True,
        "all_buy_signals_precede_fills": True, "all_sales_at_least_90_days_after_entry": True,
        "buy_count": len(buys), "sell_count": len(sells), "open_lots": len(result.open_lots),
        "total_fees": float(fills.fee.sum()), "terminal_cash": float(history.cash.iloc[-1]),
        "terminal_open_value": float(history.market_value.iloc[-1]),
        "stale_mark_rows": int(result.diagnostics.event.eq("stale_mark").sum()),
        "cancelled_entry_rows": int(result.diagnostics.event.eq("missing_entry_quote").sum()),
        "minimum_cash": float(history.cash.min()),
        "every_exit_on_exact_maturity_session": True,
        "daily_cash_ledger_reconciles": True,
        "lot_counts_reconcile": True,
        "mandatory_stock_conversions": int(result.action_ledger.event.eq("mandatory_stock_conversion").sum()),
        "mandatory_cash_settlements": int(result.action_ledger.event.eq("mandatory_cash_settlement").sum()),
        "matured_cash_settlement_lots": len(settlements),
        "symbol_change_events": int(result.action_ledger.event.isin(["mandatory_symbol_change", "entry_symbol_alias"]).sum()),
    }


def final(database):
    frozen_path = RESULTS / "frozen_strategy.json"
    specification = json.loads(frozen_path.read_text())
    if not specification["frozen_before_final"] or specification["database_sha256"] != sha256(database):
        raise ValueError("Frozen specification or database checksum is invalid")
    if specification["code_sha256"] != code_hashes():
        amendment = json.loads((RESULTS / "execution_amendment.json").read_text())
        if (amendment["original_code_sha256"] != specification["code_sha256"]
                or amendment["amended_code_sha256"] != code_hashes()
                or amendment["corporate_actions_py_sha256"] != sha256(HERE / "corporate_actions.py")
                or amendment["corporate_actions_json_sha256"] != sha256(HERE / "corporate_actions.json")):
            raise ValueError("Code differs from the recorded execution-only amendment")
    print("Final: applying frozen specification; no strategy selection", flush=True)
    # Initial orders use March 31 features; only outcomes completed before that
    # signal may train the fixed model. Financial/price features remain causal.
    panel, loader_warnings = prepare_monthly(database, "2026-09-30", "2016-03-30")
    calendar, _ = load_calendar(database, "2026-09-30")
    entries = entry_schedule(calendar, "2016-04-01", "2026-09-30")
    signals = calendar[calendar.get_indexer(entries) - 1]
    scores, fit_audit = fit_and_score(panel, signals, specification)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUTS / "final_scores.csv", index=False)
    write_json(RESULTS / "final_fit_audit.json", fit_audit)
    try:
        actions = load_actions(database)
        result, metrics = simulate(database, scores, "2016-04-01", "2026-09-30", specification, actions)
    except DataGapError as error:
        write_json(RESULTS / "final_data_gap.json", {
            "status": "unresolved", "error": str(error),
            "policy": "Stop; do not retune using final survival or fabricate a fill.",
        })
        raise
    evidence = accounting_evidence(result, StrategyConfig(**specification["portfolio"]))
    result.history.to_csv(RESULTS / "daily_portfolio.csv")
    result.fills.to_csv(OUTPUTS / "final_fills.csv", index=False)
    result.open_lots.to_csv(RESULTS / "open_lots.csv", index=False)
    result.diagnostics.to_csv(RESULTS / "diagnostics.csv", index=False)
    result.action_ledger.to_csv(RESULTS / "corporate_action_ledger.csv", index=False)
    write_json(RESULTS / "corporate_action_basis.json", actions)
    if (RESULTS / "final_data_gap.json").exists():
        gap = json.loads((RESULTS / "final_data_gap.json").read_text())
        gap["status"] = "resolved_by_documented_mandatory_corporate_actions"
        write_json(RESULTS / "final_data_gap.json", gap)
    summary = {
        "period": "2016-04-01 through 2026-09-30", "database": database.name,
        "database_sha256": specification["database_sha256"],
        "frozen_strategy_sha256": sha256(frozen_path), "strategy": specification["name"],
        "fit_audit": fit_audit, "metrics": metrics, "accounting": evidence,
        "loader_warnings": loader_warnings,
        "excess_final_dollars": metrics["strategy"]["final_value"] - metrics["benchmark"]["final_value"],
        "excess_cagr_percentage_points": 100 * (metrics["strategy"]["cagr"] - metrics["benchmark"]["cagr"]),
    }
    write_json(RESULTS / "final_summary.json", summary)
    print("Final run complete; accounting checks passed. Results saved under results/.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "final"])
    parser.add_argument("--database", type=Path, default=DATABASE)
    args = parser.parse_args()
    if args.stage == "development":
        if (RESULTS / "final_summary.json").exists() or (RESULTS / "final_data_gap.json").exists():
            parser.error("Final evaluation already attempted; do not overwrite the pre-test development freeze")
        development(args.database)
    else:
        final(args.database)
