"""Cache full-universe features, refit quarterly Ridge, and run HW4 simulation.

Walk-forward fitting follows Lecture 8/2_research_workflow.ipynb, using the
independent portfolio_backtest implementation for accounting and execution.
Generated databases and large outputs remain under ignored outputs/.
"""

from pathlib import Path
import argparse
import json
import shutil

import duckdb
import numpy as np
import pandas as pd

from feature_panel import (DEFAULT_DATABASE, FEATURES, load_reference_data,
                           load_symbol_batch, build_feature_batch)
from price_quality import screen_prices
from model_selection import make_model
from portfolio_backtest import simulate, UnreliableHeldPrice
from common_stock_screen import POLICY_VERSION, load_security_review


START, END = pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30")
ROOT = Path(__file__).parent


def append_frame(connection, table, frame, first):
    """Append a bounded batch without retaining the whole panel in RAM."""
    connection.register("batch_frame", frame)
    if first:
        connection.execute(f"CREATE TABLE {table} AS SELECT * FROM batch_frame")
    else:
        connection.execute(f"INSERT INTO {table} BY NAME SELECT * FROM batch_frame")
    connection.unregister("batch_frame")


def prepare_cache(database, cache_path, batch_size=100):
    """Persist causal features and full held-price history in a separate DB."""
    if cache_path.exists():
        with duckdb.connect(str(cache_path), read_only=True) as check:
            complete = check.execute("SELECT complete FROM cache_status").fetchone()[0]
            if not complete:
                raise ValueError("Incomplete feature cache; inspect it before rebuilding")
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    calendar, benchmark, membership, symbols = load_reference_data(database, end=str(END.date()))
    with duckdb.connect(str(cache_path)) as cache:
        cache.execute("SET memory_limit='1GB'")
        cache.execute("SET threads=2")
        cache.execute("CREATE TABLE cache_status AS SELECT false AS complete")
        append_frame(cache, "benchmark", benchmark, True)
        for offset in range(0, len(symbols), batch_size):
            batch_symbols = symbols[offset:offset + batch_size]
            panel = build_feature_batch(database, batch_symbols, calendar, benchmark, membership,
                                        end=str(END.date()))
            # Membership/quality eligibility is causal. Do not filter on target.
            selected = panel.loc[panel.eligible, ["date", "symbol", "label_end", "target", *FEATURES]].copy()
            selected[FEATURES] = selected[FEATURES].replace([np.inf, -np.inf], np.nan)
            prices, _, _ = load_symbol_batch(database, batch_symbols, end=str(END.date()))
            prices = prices.loc[prices.date.isin(calendar)]
            prices, audit = screen_prices(prices)
            first = offset == 0
            append_frame(cache, "features", selected, first)
            append_frame(cache, "quotes", prices[["date", "symbol", "adj_close"]], first)
            append_frame(cache, "quality_audit", audit, first)
            print(f"Cached {min(offset+batch_size,len(symbols))}/{len(symbols)} securities", flush=True)
        cache.execute("UPDATE cache_status SET complete=true")
        cache.execute("CHECKPOINT")


def quarterly_scores(cache_path, parameters, output, allowed_symbols):
    """Expand training quarterly and score monthly entry signals, ignoring labels."""
    output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(cache_path), read_only=True) as cache:
        calendar = pd.DatetimeIndex(cache.execute("SELECT date FROM benchmark ORDER BY date").fetchdf().date)
        test_days = calendar[(calendar >= START) & (calendar <= END)]
        month_first = pd.Series(test_days, index=test_days).groupby(test_days.to_period("M")).first()
        signal_dates = pd.DatetimeIndex([calendar[calendar.get_loc(day)-1] for day in month_first])
        fit_dates = [calendar[calendar.searchsorted(q.start_time)-1]
                     for q in pd.period_range(START, END, freq="Q")]
        columns = ", ".join(FEATURES)
        audits, predictions = [], []
        for index, asof in enumerate(fit_dates):
            next_fit = fit_dates[index+1] if index+1 < len(fit_dates) else END + pd.Timedelta(days=1)
            dates = signal_dates[(signal_dates >= asof) & (signal_dates < next_fit)]
            train = cache.execute(
                f"SELECT date,label_end,target,{columns} FROM features "
                "WHERE date < ? AND label_end < ? AND target IS NOT NULL AND isfinite(target) "
                "AND symbol IN (SELECT UNNEST(?))",
                [asof.to_pydatetime(), asof.to_pydatetime(), allowed_symbols],
            ).fetchdf()
            if train.empty:
                raise ValueError(f"No matured training observations at {asof}")
            model = make_model(parameters).fit(train[FEATURES], train.target)
            score_frame = cache.execute(
                f"SELECT date,symbol,{columns} FROM features WHERE date IN (SELECT UNNEST(?)) "
                "AND symbol IN (SELECT UNNEST(?)) ORDER BY date,symbol",
                [list(dates.to_pydatetime()), allowed_symbols],
            ).fetchdf()
            scored = score_frame[["date", "symbol"]].copy()
            scored["score"] = model.predict(score_frame[FEATURES]) if len(score_frame) else pd.Series(dtype=float)
            scored["fit_date"] = asof
            scored.to_csv(output / f"scores_{asof.date()}.csv", index=False)
            predictions.append(scored)
            audits.append(dict(fit_date=asof, training_rows=len(train),
                               last_training_signal=train.date.max(), last_label_end=train.label_end.max(),
                               scored_rows=len(scored)))
            pd.DataFrame(audits).to_csv(output / "fit_audit.csv", index=False)
            print(f"Quarter {index+1}/{len(fit_dates)}: fit {asof.date()}, {len(train):,} training rows, {len(scored):,} scores", flush=True)
            del train, model, score_frame
    return pd.concat(predictions, ignore_index=True)


def performance(series, start, end):
    """Measure return from initial capital, including first-session fees."""
    returns = series.pct_change(fill_method=None).dropna()
    daily_rf = 1.02 ** (1/252) - 1
    std = returns.std(ddof=1)
    years = (end-start).days / 365.25
    total_return = series.iloc[-1] / series.iloc[0] - 1
    return dict(final_value=float(series.iloc[-1]), total_return=total_return,
                CAGR=(1+total_return)**(1/years)-1,
                max_drawdown=(series/series.cummax()-1).min(),
                annualized_volatility=std*np.sqrt(252),
                Sharpe=(returns-daily_rf).mean()/std*np.sqrt(252) if std > 1e-12 else np.nan)


def save_charts(history, output, note="Database-only; no retrospective coverage exclusions"):
    """Plot returns, cash/NAV and per-date drawdown as in Lecture 8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"nav": "Quarterly Ridge", "benchmark_nav": "S&P 500 total return"}
    fig, ax = plt.subplots(figsize=(12, 6))
    ((history[["nav", "benchmark_nav"]]/1_000_000-1)*100).rename(columns=labels).plot(ax=ax)
    ax.set(title=f"HW4: cumulative return after transaction costs\n{note}",
           xlabel="Date", ylabel="Cumulative return (%)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "cumulative_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    (history.cash / history.nav * 100).plot(ax=ax, label="Quarterly Ridge: cash / NAV")
    ax.set(title=f"HW4: cash fraction\n{note}", xlabel="Date",
           ylabel="Cash / NAV (%)", ylim=(-2, 102))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cash_fraction.png", dpi=180)
    plt.close(fig)

    values = history[["nav", "benchmark_nav"]]
    drawdown = (values / values.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    drawdown.rename(columns=labels).plot(ax=ax)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(title=f"HW4: drawdown from running NAV peak\n{note}",
           xlabel="Date", ylabel="Drawdown (%)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "drawdown.png", dpi=180)
    plt.close(fig)


def export_submission_results(output, run_root=None):
    """Copy only the small report/evidence files into the submission folder."""
    run_root = output if run_root is None else run_root
    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    for name in [
        "performance.csv", "account_summary.csv", "portfolio_diagnostics.csv",
        "daily_nav.csv", "trades.csv", "cumulative_returns.png", "cash_fraction.png",
        "drawdown.png", "coverage_exclusions.json", "accounting_audit.json",
    ]:
        source = output / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    shutil.copy2(run_root / "predictions/fit_audit.csv", destination / "fit_audit.csv")
    for name in ["configuration.json", "validation_results.csv"]:
        shutil.copy2(ROOT / "outputs/development" / name, destination / name)
    config_path = destination / "configuration.json"
    config = json.loads(config_path.read_text())
    config["database"] = Path(config["database"]).name
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    source = ROOT / "outputs/accounting_checks.json"
    if source.is_file():
        shutil.copy2(source, destination / source.name)
    review = pd.read_csv(run_root / "security_type_review.csv")
    review.groupby(["common_stock_candidate", "reason"]).size().rename("Securities").reset_index().to_csv(
        destination / "security_type_summary.csv", index=False)
    for name in ["run_status.json", "unresolved_backtest.txt"]:
        path = run_root / name
        if path.is_file():
            shutil.copy2(path, destination / name)



def main():
    """Use frozen development choices; preserve failures rather than fake fills."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "final")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--reuse-predictions", action="store_true", help="Reuse complete predictions with matching frozen model configuration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "outputs/development/configuration.json").read_text())
    parameters = config["chosen_parameters"]
    if config.get("security_policy") != POLICY_VERSION:
        raise ValueError("Rerun model_selection.py for the current common-stock policy")
    review = load_security_review(args.database)
    allowed_symbols = review.loc[review.common_stock_candidate, "symbol"].tolist()
    review.to_csv(args.output / "security_type_review.csv", index=False)
    cache_path = args.output / "features.duckdb"
    prepare_cache(args.database, cache_path, args.batch_size)
    prediction_dir = args.output / "predictions"
    snapshot_path = prediction_dir / "configuration.json"
    snapshot = {"parameters": parameters, "features": FEATURES, "security_policy": POLICY_VERSION,
                "allowed_symbols": allowed_symbols}
    if args.reuse_predictions:
        if json.loads(snapshot_path.read_text()) != snapshot:
            raise ValueError("Saved prediction configuration differs; refit rather than reuse")
        fits = pd.read_csv(prediction_dir / "fit_audit.csv", parse_dates=["fit_date", "last_label_end"])
        if len(fits) != 42 or not (fits.last_label_end < fits.fit_date).all():
            raise ValueError("Expected 42 complete, maturity-checked quarterly fits")
        parts = []
        for row in fits.itertuples():
            part = pd.read_csv(prediction_dir / f"scores_{row.fit_date.date()}.csv", parse_dates=["date", "fit_date"])
            if len(part) != row.scored_rows or not part.fit_date.eq(row.fit_date).all():
                raise ValueError("Saved quarterly scores do not match fit audit")
            parts.append(part)
        scores = pd.concat(parts, ignore_index=True)
        if scores.duplicated(["date", "symbol"]).any():
            raise ValueError("Duplicate saved signal scores")
    else:
        scores = quarterly_scores(cache_path, parameters, prediction_dir, allowed_symbols)
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    with duckdb.connect(str(cache_path), read_only=True) as cache:
        benchmark = cache.execute("SELECT * FROM benchmark ORDER BY date").fetchdf()
        # Entries are score-selected, but quotes for existing positions remain.
        prices = cache.execute("SELECT * FROM quotes WHERE date>=? ORDER BY date,symbol", [START.to_pydatetime()]).fetchdf()
        audit = cache.execute("SELECT * FROM quality_audit").fetchdf()
    # Use all historically eligible scores. Missing future coverage never
    # removes a security or causes a reranking/restart of the experiment.
    report_output = args.output
    status = {"requested_start":str(START.date()), "requested_end":str(END.date()),
              "retrospective_coverage_exclusions":False, "data_source":"course database only"}
    try:
        result = simulate(prices, benchmark, scores, START, END, quality_audit=audit)
        status["status"] = "complete"
        (args.output / "unresolved_backtest.txt").unlink(missing_ok=True)
    except UnreliableHeldPrice as error:
        result = error.partial_result
        if result is None:
            raise
        status.update(status="incomplete", failed_date=str(error.failed_date.date()), reason=str(error))
        report_output = args.output / "partial"
        report_output.mkdir(exist_ok=True)
        (args.output / "unresolved_backtest.txt").write_text(str(error), encoding="utf-8")
        print("Evaluation incomplete:",error,flush=True)
    status["last_evaluated_date"] = str(result.history.index[-1].date())
    (args.output / "run_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    normal = result.trades.side.isin(["BUY", "SELL"])
    result.trades["source"] = result.trades["source"].astype("string")
    result.trades.loc[normal, "source"] = "course prices table"
    result.history.to_csv(report_output / "daily_nav.csv")
    result.trades.to_csv(report_output / "trades.csv", index=False)
    result.positions.to_csv(report_output / "open_positions.csv", index=False)
    result.diagnostics.to_csv(report_output / "diagnostics.csv", index=False)
    final = result.history.iloc[-1]
    realized = result.trades.realized_pnl.sum()
    open_cost = (result.positions.units*result.positions.entry_price).sum()
    unrealized = final.holdings_value-open_cost-result.positions.entry_fee.sum()
    account_summary = pd.Series({
        "cash": final.cash, "holdings_value": final.holdings_value, "nav": final.nav,
        "realized_pnl": realized, "unrealized_pnl": unrealized,
        "total_pnl": realized+unrealized, "total_fees": result.trades.fee.sum(),
    }, name="value")
    account_summary.rename_axis("metric").to_csv(report_output / "account_summary.csv")
    sessions = result.history.iloc[1:]
    diagnostics_summary = pd.Series({
        "average_cash_fraction": (sessions.cash/sessions.nav).mean(),
        "fees": sessions.fees.sum(),
        "gross_traded_over_average_nav": (result.trades.units*result.trades.price).sum()/sessions.nav.mean(),
        "open_lots_at_end": len(result.positions),
        "stale_marks": int(sessions.stale_marks.sum()),
    }, name="value")
    diagnostics_summary.rename_axis("metric").to_csv(report_output / "portfolio_diagnostics.csv")
    actual_start, actual_end = result.history.index[1], result.history.index[-1]
    comparison = pd.DataFrame({
        "quarterly Ridge": performance(result.history.nav, actual_start, actual_end),
        "S&P 500 total return": performance(result.history.benchmark_nav, actual_start, actual_end),
    }).T
    comparison.index.name = "portfolio"
    comparison.to_csv(report_output / "performance.csv")
    save_charts(result.history, report_output, note=("PARTIAL evaluation through " + status["last_evaluated_date"] if status["status"] == "incomplete" else "Database-only; no retrospective coverage exclusions"))
    export_submission_results(report_output, run_root=args.output)
    print(comparison.to_string(), flush=True)


if __name__ == "__main__":
    main()
