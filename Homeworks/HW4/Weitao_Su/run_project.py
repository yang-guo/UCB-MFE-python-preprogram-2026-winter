"""Train the model, test the trades, and save the results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from hw4_data import FEATURES, database_path, load_actions, load_quotes, market_data, prepare_panel, save_ranking_flags
from backtester import simulate
from course_data import date_folds, make_pipeline, prediction_metrics

HERE = Path(__file__).resolve().parent
SETTINGS = {
    "initial_cash": 1000000., "fee_bps": 5., "top_n": 10,
    "minimum_price": 5., "minimum_dollar_volume_21": 10000000.,
    "maximum_volatility_63": .8,
}


def metrics(daily):
    years = (daily.date.iloc[-1] - daily.loc[~daily.initial, "date"].iloc[0]).days / 365.25
    result = {}
    for name, column in [("Strategy", "nav"), ("S&P 500 total return", "benchmark_nav")]:
        value = daily[column]
        returns = value.pct_change(fill_method=None).dropna()
        excess = returns - (1.02 ** (1 / 252) - 1)
        volatility = returns.std()
        result[name] = {
            "final_value": value.iloc[-1],
            "CAGR": (value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan,
            "max_drawdown": (value / value.cummax() - 1).min(),
            "Sharpe": np.sqrt(252) * excess.mean() / volatility if volatility > 1e-12 else np.nan,
        }
    return pd.DataFrame(result).T


def predict(model, panel, start, end, fit_date):
    rows = panel.loc[panel.entry_date.between(start, end)]
    assert rows.date.ge(pd.Timestamp(fit_date)).all()
    scores = rows[["date", "symbol", "entry_date"]].copy()
    scores["score"] = model.predict(rows[FEATURES])
    scores["fit_date"] = pd.Timestamp(fit_date)
    return scores


def save_results(run, benchmark, settings, validation, model_selection, trials):
    results, output = HERE / "results", HERE / "outputs"
    for name in ["daily", "trades", "orders", "events", "claims"]:
        run[name].to_csv(output / f"{name}.csv", index=False)
    dates = pd.DatetimeIndex([pd.Timestamp("2016-03-31"), *benchmark.loc["2016-04-01":].index])
    chart = pd.DataFrame({"date": dates, "initial": dates == dates[0]})
    chart["benchmark_nav"] = settings["initial_cash"] * chart.date.map(benchmark) / benchmark.loc["2016-04-01"]
    chart.loc[chart.initial, "benchmark_nav"] = settings["initial_cash"]
    chart["nav"] = chart.date.map(run["daily"].set_index("date").nav)
    if run["problem"]:
        summary = metrics(chart.assign(nav=chart.benchmark_nav))
        summary.loc["Strategy"] = np.nan
    else:
        summary = metrics(run["daily"])
    save_ranking_flags(database_path(), output / "ranking_input_flags.csv")

    def records(frame):
        return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))

    saved = {
        "settings": settings,
        "model_selection": records(model_selection),
        "validation": records(validation.reset_index(names="metric")),
        "strategy_selection": records(trials),
        "performance": records(summary.reset_index(names="portfolio")),
        "open_lots": records(run["open_lots"]),
        "open_rights": records(run["claims"]),
        "run": {"last_date": str(run["daily"].date.iloc[-1].date()),
                "sessions": len(run["daily"]) - 1, "trades": len(run["trades"]),
                "corporate_events": len(run["events"]), "problem": run["problem"],
                "ending_cash": float(run["daily"].cash.iloc[-1]),
                "ending_holdings": float(run["daily"].open_value.iloc[-1]),
                "ranking_flags": len(pd.read_csv(output / "ranking_input_flags.csv"))},
    }
    if run["problem"]:
        saved["partial_performance"] = records(metrics(run["daily"]).reset_index(names="portfolio"))
    (results / "summary.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(chart.date, chart.nav / settings["initial_cash"] - 1, label="Strategy")
    ax.plot(chart.date, chart.benchmark_nav / settings["initial_cash"] - 1, label="S&P 500 total return")
    ax.set_title("Apr 2016 - Sep 2026 | INBX contingent right valued at $0", fontsize=10)
    if run["problem"]:
        ax.set_title(f"Strategy stops {run['daily'].date.iloc[-1].date()}: a stock price is missing")
    ax.set(xlabel="Date", ylabel="Total return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "cumulative_returns.png", dpi=140)
    plt.close(fig)
    print(summary.to_string())
    print(run["problem"] or "Backtest completed.")


def main():
    database = database_path()
    results, output = HERE / "results", HERE / "outputs"
    results.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    settings = dict(SETTINGS)
    development = prepare_panel(database, "2016-03-31", settings)
    tuning = development.loc[development.date.between("2010-01-01", "2014-12-31")
                             & development.label_end.lt("2015-01-01") & development.target.notna()].reset_index(drop=True)
    search = GridSearchCV(
        make_pipeline(Ridge(), numeric_features=FEATURES, include_sector=False),
        {"model__alpha": [1, 100, 1000]},
        cv=date_folds(tuning, validation_years=(2013, 2014)),
        scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    search.fit(tuning[FEATURES], tuning.target)
    model_selection = pd.DataFrame({"alpha": search.cv_results_["param_model__alpha"],
                                    "RMSE": -search.cv_results_["mean_test_score"]})
    valid = development.loc[development.date.between("2015-01-01", "2015-12-31")
                            & development.label_end.lt("2016-01-01") & development.target.notna()]
    assert tuning.label_end.max() < valid.date.min()
    validation = pd.DataFrame({
        "Ridge": prediction_metrics(valid, search.predict(valid[FEATURES])),
        "Baseline": prediction_metrics(valid, np.full(len(valid), tuning.target.mean())),
    })

    benchmark = market_data(database, "2015-12-31")
    scores = predict(search.best_estimator_, development, "2015-01-01", "2015-12-31", "2014-12-31")
    quotes = load_quotes(database, scores.symbol.unique(), "2014-12-31", "2015-12-31")
    trials = []
    for number in [10, 20]:
        trial = simulate(quotes, benchmark, scores, "2015-01-01", "2015-12-31", dict(settings, top_n=number),
                         load_actions(database, "2015-12-31"))
        if not trial["problem"]:
            trials.append({"top_n": number, **metrics(trial["daily"]).loc["Strategy"].to_dict()})
    trials = pd.DataFrame(trials)
    if trials.empty:
        raise ValueError("Missing prices stopped both 2015 tests.")
    settings["top_n"] = int(trials.sort_values(["Sharpe", "top_n"], ascending=[False, False]).iloc[0].top_n)
    settings["alpha"] = float(search.best_params_["model__alpha"])
    train = development.loc[development.date.between("2010-01-01", "2015-12-31")
                            & development.label_end.lt("2016-03-31") & development.target.notna()]
    assert train.label_end.max() < pd.Timestamp("2016-03-31")
    model = clone(search.best_estimator_).fit(train[FEATURES], train.target)
    settings["last_training_outcome"] = str(train.label_end.max().date())
    # The model and strategy are fixed before loading final-test data.

    panel = prepare_panel(database, "2026-09-30", settings)
    scores = predict(model, panel, "2016-04-01", "2026-09-30", "2016-03-31")
    benchmark = market_data(database, "2026-09-30")
    actions = load_actions(database, "2026-09-30")
    symbols = set(scores.symbol) | {a["new_symbol"] for a in actions if a["kind"] != "cash"}
    quotes = load_quotes(database, sorted(symbols), "2016-03-31", "2026-09-30")
    run = simulate(quotes, benchmark, scores, "2016-04-01", "2026-09-30", settings,
                   actions)
    save_results(run, benchmark, settings, validation, model_selection, trials)


if __name__ == "__main__":
    main()
