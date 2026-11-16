"""Independent ledger reconstruction and metric checks for the completed revision.

Uses recorded fills and mandatory transitions, not the simulator's state loop.
Checks every daily cash balance, holding quantity, receivable and NAV; quote
availability; inherited maturities; event conversion arithmetic; and metrics.
"""

import json

import duckdb
import numpy as np
import pandas as pd

from hw4.data import checksum
from hw4.settings import DATABASE, PROJECT
from run_rank_final import OUT as RESULTS, accounting_inputs
from rank_final_inputs import load_final_context
from research_session import save_json, verify_saved_sources


def independent_metrics(nav, rf=0.02):
    x = np.asarray(nav, dtype=float)
    returns = x[1:] / x[:-1] - 1
    years = (nav.index[-1] - nav.index[1]).total_seconds() / (365.25 * 86400)
    vol = np.sqrt(np.sum((returns - returns.mean()) ** 2) / (len(returns) - 1))
    return dict(
        final_value=x[-1],
        total_return=x[-1] / x[0] - 1,
        CAGR=(x[-1] / x[0]) ** (1 / years) - 1,
        max_drawdown=np.min(x / np.maximum.accumulate(x) - 1),
        annualized_volatility=vol * np.sqrt(252),
        Sharpe=(returns.mean() - ((1 + rf) ** (1 / 252) - 1)) / vol * np.sqrt(252),
    )


def verify(lag, quotes, raw, config, membership, scores, sectors, benchmark):
    status = json.loads((RESULTS / "latest_status.json").read_text())
    local = PROJECT / status["local_output"]
    history = pd.read_csv(local / "history.csv", index_col=0, parse_dates=True)
    fills = pd.read_csv(
        local / "fills.csv", parse_dates=["date", "signal_date", "maturity"]
    )
    events = pd.read_csv(
        local / "events.csv",
        parse_dates=["date", "original_maturity", "cash_available_date"],
    )
    settlements = pd.read_csv(local / "settlements.csv", parse_dates=["date"])
    lots = pd.read_csv(local / "open_lots.csv", parse_dates=["entry_date", "maturity"])
    status = json.loads((RESULTS / "latest_status.json").read_text())
    assert status["status"] == "complete"
    assert history.index[-1] == pd.Timestamp("2026-09-30")
    expected_dates = quotes.index[
        (quotes.index >= "2016-03-31") & (quotes.index <= "2026-09-30")
    ]
    assert history.index.as_unit("ns").equals(expected_dates.as_unit("ns"))
    assert not fills.duplicated(["lot_id", "side"]).any()
    buys = fills.loc[fills.side.eq("BUY")].set_index("lot_id")
    sells = fills.loc[fills.side.eq("SELL")]
    assert buys.signal_date.lt(buys.date).all()
    entry_rows = buys.reset_index()
    entry_rows["quarter_start"] = entry_rows.date.dt.to_period("Q").dt.start_time
    dated = entry_rows.merge(
        membership, on=["symbol", "quarter_start"], how="left", validate="many_to_one"
    )
    assert dated.selection_date.notna().all()
    assert dated.selection_date.le(dated.signal_date).all()
    assert dated.effective_start.le(dated.date).all()
    assert (
        quotes.index.get_indexer(buys.date) - 1
        == quotes.index.get_indexer(buys.signal_date)
    ).all()
    ranked = (
        scores.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
        .groupby("date")
        .head(20)
    )
    chosen_keys = pd.MultiIndex.from_frame(
        buys[["signal_date", "symbol"]].rename(columns={"signal_date": "date"})
    )
    assert chosen_keys.isin(pd.MultiIndex.from_frame(ranked[["date", "symbol"]])).all()
    entry_targets = scores.set_index(["date", "symbol"]).target.reindex(chosen_keys)
    assert (
        entry_targets.isna().any()
    )  # Missing future outcomes did not erase candidates.
    assert (buys.date + pd.Timedelta(days=90)).eq(buys.maturity).all()
    for sale in sells.itertuples():
        entry = buys.loc[sale.lot_id]
        assert sale.maturity == entry.maturity
        assert sale.date == quotes.loc[sale.maturity :, sale.symbol].dropna().index[0]
    for lot in lots.itertuples():
        assert lot.maturity == buys.loc[lot.lot_id, "maturity"]
        assert lot.maturity > history.index[-1]
    rows = quotes.index.get_indexer(fills.date)
    cols = quotes.columns.get_indexer(fills.symbol)
    assert (rows >= 0).all() and (cols >= 0).all()
    np.testing.assert_allclose(
        fills.price, quotes.to_numpy()[rows, cols], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(fills.notional, fills.units * fills.price, rtol=1e-12)
    np.testing.assert_allclose(fills.fee, fills.notional * 0.0005, rtol=0, atol=1e-9)
    desired_cash = np.where(
        fills.side.eq("BUY"), -fills.notional - fills.fee, fills.notional - fills.fee
    )
    np.testing.assert_allclose(fills.cash_change, desired_cash, rtol=0, atol=1e-9)
    movement = [
        fills.assign(delta=fills.units.where(fills.side.eq("BUY"), -fills.units))[
            ["date", "symbol", "delta"]
        ]
    ]
    catalog = {e["id"]: e for e in config["events"]}
    for ev in events.itertuples():
        spec = catalog[ev.event_id]
        basis = pd.Timestamp(spec["basis_date"])
        equivalent = (
            ev.old_units
            * quotes.loc[basis, ev.old_symbol]
            / raw.loc[basis, ev.old_symbol]
        )
        np.testing.assert_allclose(ev.actual_share_equivalent, equivalent, rtol=1e-12)
        np.testing.assert_allclose(
            ev.cash_entitlement,
            equivalent * spec.get("cash_per_share", 0),
            rtol=1e-12,
            atol=1e-10,
        )
        assert ev.original_maturity == buys.loc[ev.lot_id, "maturity"]
        assert (
            ev.date <= ev.original_maturity
        )  # All actual events preceded the planned exits.
        if spec.get("new"):
            new_basis = basis if spec["kind"] == "rename" else ev.date
            expected = (
                equivalent
                * spec["share_ratio"]
                * raw.loc[new_basis, spec["new"]]
                / quotes.loc[new_basis, spec["new"]]
            )
            np.testing.assert_allclose(ev.new_units, expected, rtol=1e-12)
            if spec["kind"] == "rename":
                # A rename changes representation, not value at its common basis date.
                np.testing.assert_allclose(
                    ev.old_units * quotes.loc[basis, ev.old_symbol],
                    ev.new_units * quotes.loc[basis, ev.new_symbol],
                    rtol=0.0001,
                )
        movement.append(
            pd.DataFrame(
                [dict(date=ev.date, symbol=ev.old_symbol, delta=-ev.old_units)]
            )
        )
        if ev.new_units:
            movement.append(
                pd.DataFrame(
                    [dict(date=ev.date, symbol=ev.new_symbol, delta=ev.new_units)]
                )
            )
        later_buys = buys.loc[buys.symbol.eq(ev.old_symbol) & buys.date.ge(ev.date)]
        assert later_buys.empty
    changes = (
        pd.concat(movement, ignore_index=True)
        .groupby(["date", "symbol"])
        .delta.sum()
        .unstack(fill_value=0)
    )
    units = changes.reindex(history.index, fill_value=0).cumsum()
    assert units.min().min() >= -1e-7
    marks = quotes[units.columns].ffill(limit=5).reindex(history.index)
    assert not ((units.abs() > 1e-7) & marks.isna()).any().any()
    holdings = (units * marks).sum(axis=1)
    trade_cash = (
        fills.groupby("date").cash_change.sum().reindex(history.index, fill_value=0)
    )
    settled = (
        settlements.groupby("date")
        .cash_change.sum()
        .reindex(history.index, fill_value=0)
    )
    cash = 1_000_000 + (trade_cash + settled).cumsum()
    created = (
        events.groupby("date")
        .cash_entitlement.sum()
        .reindex(history.index, fill_value=0)
    )
    claims = (created - settled).cumsum()
    np.testing.assert_allclose(cash, history.cash, rtol=0, atol=1e-6)
    np.testing.assert_allclose(holdings, history.holdings, rtol=0, atol=1e-6)
    np.testing.assert_allclose(claims, history.receivables, rtol=0, atol=1e-7)
    reconstructed = cash + holdings + claims
    np.testing.assert_allclose(reconstructed, history.nav, rtol=0, atol=1e-6)
    final_units = (
        lots.groupby("symbol").units.sum().reindex(units.columns, fill_value=0)
    )
    np.testing.assert_allclose(units.iloc[-1], final_units, rtol=0, atol=1e-7)
    for payment in settlements.itertuples():
        event = events.loc[
            events.event_id.eq(payment.event_id) & events.lot_id.eq(payment.lot_id)
        ].iloc[0]
        assert payment.date == quotes.index[quotes.index.get_loc(event.date) + lag]
        np.testing.assert_allclose(
            payment.cash_change, event.cash_entitlement, rtol=1e-12
        )
    # Reconstruct prior-close capacity independently of the order planner.
    # Entry caps permit market drift; they do not require premature sales.
    cap_rows = []
    for date, day_buys in buys.groupby("date"):
        prior = day_buys.signal_date.iloc[0]
        prior_value = units.loc[prior] * marks.loc[prior].fillna(0)
        nav_before, cash_before = history.loc[prior, ["nav", "cash"]]
        assert -day_buys.cash_change.sum() <= cash_before + 1e-6
        names = day_buys.groupby("symbol").notional.sum()
        assert (names <= (0.05 * nav_before - prior_value.reindex(names.index).fillna(0)).clip(lower=0) + 1e-6).all()
        prior_sector = prior_value.groupby(prior_value.index.map(sectors)).sum()
        new_sector = names.groupby(names.index.map(sectors)).sum()
        assert (new_sector <= (0.20 * nav_before - prior_sector.reindex(new_sector.index).fillna(0)).clip(lower=0) + 1e-6).all()
        cap_rows.append(dict(date=date, buys=len(day_buys), prior_cash=cash_before,
                             spending=-day_buys.cash_change.sum(), prior_nav=nav_before))
    pd.DataFrame(cap_rows).to_csv(RESULTS / "entry_cap_audit.csv", index=False)
    expected_benchmark = 1e6 * benchmark.reindex(history.index) / benchmark.loc["2016-04-01"]
    expected_benchmark.iloc[0] = 1e6
    np.testing.assert_allclose(expected_benchmark, history.benchmark_nav, rtol=0, atol=1e-6)
    weights = units * marks.fillna(0)
    weights = weights.div(history.nav, axis=0)
    sector_weights = weights.T.groupby(weights.columns.map(sectors)).sum().T
    # Historical aliases are a remaining source limitation, measured explicitly.
    alias_groups = {"Axon": ["AXON", "TASR"], "Azenta": ["AZTA", "BRKS"],
                    "Permian": ["PR", "CDEV"], "CSW": ["CSW", "CSWI"]}
    aliases = []
    for issuer, symbols in alias_groups.items():
        available = [s for s in symbols if s in weights]
        combined = weights[available].sum(axis=1)
        simultaneous = (weights[available] > 1e-9).sum(axis=1) > 1
        aliases.append(dict(issuer=issuer, simultaneous_sessions=int(simultaneous.sum()),
                            maximum_combined_weight=float(combined.max())))
    pd.DataFrame(aliases).to_csv(RESULTS / "known_alias_exposure.csv", index=False)
    sector_weights.to_csv(RESULTS / "daily_sector_weights.csv")
    # Inspect economically large moves without automatically replacing genuine returns.
    returns = quotes[units.columns].pct_change(fill_method=None).reindex(history.index)
    screen = (units.shift(1).fillna(0) > 1e-7) & returns.abs().gt(0.20)
    large = []
    for date, symbol in screen.stack()[lambda x: x].index:
        large.append(dict(date=date, symbol=symbol, price_return=returns.loc[date, symbol],
                          prior_weight=weights.shift(1).loc[date, symbol]))
    pd.DataFrame(large, columns=["date", "symbol", "price_return", "prior_weight"]).to_csv(
        RESULTS / "large_move_review.csv", index=False)
    calculated = pd.DataFrame(
        {
            "strategy": independent_metrics(history.nav),
            "benchmark": independent_metrics(history.benchmark_nav),
        }
    ).T
    reported = pd.read_csv(RESULTS / "performance.csv", index_col=0)
    np.testing.assert_allclose(
        calculated[reported.columns], reported, rtol=1e-12, atol=1e-9
    )
    return dict(
        entry_days_independently_cap_checked=len(cap_rows),
        maximum_recorded_name_weight=float(weights.max().max()),
        maximum_sector_weight=float(sector_weights.max().max()),
        large_moves_for_context_review=len(large),
        nav_rows=len(history),
        fills=len(fills),
        maturity_sales_verified=len(sells),
        dated_entry_memberships_verified=len(buys),
        entries_with_originally_unavailable_targets=int(entry_targets.isna().sum()),
        event_lots_verified=len(events),
        cash_settlements_verified=len(settlements),
        terminal_open_lots=len(lots),
        max_nav_reconstruction_error=float(abs(reconstructed - history.nav).max()),
        minimum_reconstructed_cash=float(cash.min()),
        unsettled_terminal_claims=float(claims.iloc[-1]),
        daily_metrics_independently_matched=True,
    )


def run():
    context, protected = load_final_context()
    quotes, raw, events, sectors, _, _ = accounting_inputs(context)
    scores = pd.read_csv(PROJECT / "outputs/rank_final/predictions.csv", parse_dates=["date"])
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        membership = con.execute(
            "SELECT symbol,quarter_start,selection_date,effective_start FROM universe_membership"
        ).df()
    fit = json.loads((RESULTS / "fit_audit.json").read_text())
    policy = json.loads((RESULTS / "frozen_config.json").read_text())
    status = json.loads((RESULTS / "latest_status.json").read_text())
    assert pd.Timestamp(fit["last_training_label"]) < pd.Timestamp(fit["fit_date"])
    assert pd.Timestamp(fit["last_training_signal"]) <= pd.Timestamp("2015-12-31")
    assert fit["policy_sha256"] == policy["sha256"] == status["policy_sha256"]
    assert checksum(PROJECT / "artifacts/rank_monthly_model.joblib") == fit["artifact_sha256"]
    assert checksum(PROJECT / "outputs/rank_final/predictions.csv") == fit["predictions_sha256"]
    protocol = json.loads((RESULTS / "protocols" / (status["protocol_sha256"] + ".json")).read_text())
    for name, digest in protocol["files"].items():
        assert checksum(PROJECT / name) == digest, name
    receipt = verify(2, quotes, raw, {"events": events}, membership, scores, sectors, context["benchmark"])
    receipt.update(protocol_sha256=status["protocol_sha256"],
                   model_and_forecasts_unchanged=True,
                   mandatory_event_assumptions_disclosed=True,
                   accounting_price_unit_repairs=4,
                   forecast_input_limitations_remain=True)
    verify_saved_sources(protected)
    save_json(RESULTS / "verification.json", receipt)
    # Accounting verified; source coverage and price quality are separate claims.
    status["verified"] = True
    save_json(RESULTS / "latest_status.json", status)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    run()
