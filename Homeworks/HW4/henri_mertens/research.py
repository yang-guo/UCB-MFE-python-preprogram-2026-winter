"""Development-only stock research with dated, purged validation.

Adapted data/feature helpers: Lectures/modeling/course_data.py.
No test-period prices or outcomes are loaded by this module's experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import duckdb
import joblib
import numpy as np
import optuna
import pandas as pd
from boruta import BorutaPy
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Lectures" / "modeling"))
import course_data as course

FEATURES = list(course.NUMERIC_FEATURES)
OUTPUT = HERE / "outputs"
DATABASE = ROOT / "Lectures/modeling/data/equities_top2000.duckdb"


def read_config():
    return json.loads((HERE / "research_config.json").read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def monthly_signals(calendar):
    """A signal at the previous close for the month's first session."""
    calendar = pd.DatetimeIndex(calendar)
    first = pd.Series(calendar).groupby(calendar.to_period("M")).first()
    positions = calendar.get_indexer(first)
    return calendar[positions[positions > 0] - 1]


def build_panel(end, start, labels=True):
    """Build in ticker batches; filter monthly rows before financial joins.

    Prices before start provide lookbacks. All SQL reads stop at end.
    Actual future entry/exit columns are isolated from predictor inputs.
    """
    if labels and pd.Timestamp(end) > pd.Timestamp("2015-12-31"):
        raise ValueError("Training/validation labels must remain pre-2016")
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        symbols = connection.execute(
            "SELECT symbol FROM universe ORDER BY symbol"
        ).fetchdf().symbol.tolist()
        calendar = pd.DatetimeIndex(connection.execute(
            "SELECT date FROM calendar WHERE date<=? ORDER BY date", [end]
        ).fetchdf().date.astype("datetime64[ns]"))
    signals = monthly_signals(calendar)
    membership = course.load_membership(DATABASE, end=end)
    pieces = []
    for offset in range(0, len(symbols), 100):
        batch = symbols[offset:offset + 100]
        prices, financials, universe, _ = course.load_data(
            DATABASE, end=end, symbols=batch
        )
        # Two years suffice for the longest 252-session feature lookback.
        warmup = pd.Timestamp(start) - pd.DateOffset(years=2)
        prices = prices.loc[prices.date.ge(warmup)].copy()
        local_calendar = calendar[calendar >= warmup]
        if prices.empty:
            continue
        panel = course.price_features(prices, local_calendar)
        panel = panel.loc[
            panel.date.isin(signals) & panel.date.ge(start)
        ].copy()
        if panel.empty:
            continue
        panel = course.apply_membership(panel, membership, calendar)
        panel = panel.loc[panel.eligible].copy()
        if panel.empty:
            continue
        panel = course.join_financials(
            panel, course.financial_events(financials), universe
        )
        if labels:
            panel = course.make_labels(panel, prices, calendar)
        columns = ["date", "symbol"] + FEATURES
        if labels:
            columns += ["entry_date", "label_end", "target"]
        pieces.append(panel[columns])
        print(f"Feature batch {offset + len(batch)}/{len(symbols)}", flush=True)
    result = pd.concat(pieces, ignore_index=True)
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert not result.duplicated(["date", "symbol"]).any()
    assert result.date.max() <= pd.Timestamp(end)
    return result, calendar


def purged_split(panel, calendar, year, outcome_cutoff, extra_sessions=5):
    """Expand training in date order and purge by actual label end.

    The additional five-session safety gap applies after the last admitted
    training outcome. There is no training block after validation, so a
    post-validation embargo would remove no additional training rows.
    Validation predictions retain censored rows; scoring alone masks them.
    """
    validation = panel.date.dt.year.eq(year)
    first_signal = panel.loc[validation, "date"].min()
    if pd.isna(first_signal):
        raise ValueError(f"No validation dates in {year}")
    position = calendar.searchsorted(first_signal)
    boundary = calendar[max(0, position - extra_sessions)]
    training = (
        panel.date.lt(first_signal)
        & panel.label_end.lt(boundary)
        & panel.target.notna()
        & panel.label_end.le(pd.Timestamp(outcome_cutoff))
    )
    train = panel.loc[training].copy()
    valid = panel.loc[validation].copy()
    assert train.label_end.max() < boundary <= valid.date.min()
    assert set(train.date).isdisjoint(valid.date)
    assert train.entry_date.le(train.label_end).all()
    return train, valid, boundary


def balanced_sample(frame, maximum, seed):
    """Deterministic date-balanced compute cap, independent of target value.

    Monthly observations reduce redundant overlapping labels. Sampling is
    confined to already eligible training rows; validation keeps all names.
    """
    if len(frame) <= maximum:
        return frame.sort_values(["date", "symbol"]).copy()
    groups = list(frame.groupby("date", sort=True))
    quota, remainder = divmod(maximum, len(groups))
    samples = []
    for number, (_, group) in enumerate(groups):
        count = min(len(group), quota + (number < remainder))
        samples.append(group.sample(n=count, random_state=seed + number))
    return pd.concat(samples).sort_values(["date", "symbol"]).copy()


class KalmanRegression(RegressorMixin, BaseEstimator):
    """Random-walk regression coefficients, learned only during training.

    One process-noise step per monthly training date. Same-date stocks
    enter as a joint linear measurement, avoiding arbitrary ticker order.
    Prediction uses the final filtered coefficients with no state updates.
    """

    def __init__(self, process_noise=1e-4, measurement_noise=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def fit(self, X, y, dates=None):
        design = np.column_stack([np.ones(len(X)), X])
        dates = np.arange(len(X)) if dates is None else np.asarray(dates)
        dimension = design.shape[1]
        covariance = np.eye(dimension)
        coefficient = np.zeros(dimension)
        for date in np.unique(dates):
            mask = dates == date
            block, values = design[mask], np.asarray(y)[mask]
            predicted = covariance + self.process_noise * np.eye(dimension)
            precision = np.linalg.inv(predicted)
            covariance = np.linalg.inv(
                precision + block.T @ block / self.measurement_noise
            )
            coefficient = covariance @ (
                precision @ coefficient
                + block.T @ values / self.measurement_noise
            )
        self.coef_ = coefficient
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.coef_


def fit_features(train, config, name):
    """Imputation, clipping, scaling and Boruta see training rows only."""
    training = balanced_sample(
        train, config["max_training_rows"], config["seed"]
    )
    clipping = course.QuantileClipper().fit(training[FEATURES])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    values = imputer.fit_transform(clipping.transform(training[FEATURES]))
    scaler = StandardScaler().fit(values)
    values = scaler.transform(values)
    target = training.target.to_numpy()
    # Financial return tails can be severe; learn this robust fit cap only
    # from the training sample. Validation outcomes are never winsorized.
    limits = np.quantile(target, [0.01, 0.99])
    clipped_target = np.clip(target, *limits)
    forest = RandomForestRegressor(
        n_estimators=config["boruta_trees"], max_depth=5,
        min_samples_leaf=20, n_jobs=2, random_state=config["seed"],
    )
    selector = BorutaPy(
        forest, n_estimators=config["boruta_trees"], perc=100,
        max_iter=config["boruta_iterations"], random_state=config["seed"],
        verbose=0,
    )
    selector.fit(values, clipped_target)
    selected = selector.support_.copy()
    # A predeclared fallback avoids inventing a feature when Boruta finds
    # no confirmed signal. It yields a training-mean forecast, not all inputs.
    target_mean = float(clipped_target.mean())
    target_scale = max(float(clipped_target.std()), 1e-8)
    status = pd.DataFrame({
        "feature": FEATURES, "confirmed": selected,
        "tentative": selector.support_weak_, "rank": selector.ranking_,
    })
    status.to_csv(OUTPUT / f"features_{name}.csv", index=False)
    bundle = {
        "clipping": clipping, "imputer": imputer, "scaler": scaler,
        "selected": selected, "target_mean": target_mean,
        "target_scale": target_scale, "training_rows": len(training),
        "latest_training_label": str(training.label_end.max()),
    }
    X = values[:, selected]
    y = (clipped_target - target_mean) / target_scale
    print(f"{name}: {len(train)} available / {len(training)} sampled, "
          f"Boruta confirmed {list(np.array(FEATURES)[selected])}", flush=True)
    return bundle, X, y, training.date.to_numpy()


def transform_features(bundle, frame):
    values = bundle["clipping"].transform(frame[FEATURES])
    values = bundle["imputer"].transform(values)
    return bundle["scaler"].transform(values)[:, bundle["selected"]]


def create_model(name, params, seed):
    if name == "elastic_net":
        return ElasticNet(**params, max_iter=4000, random_state=seed)
    if name == "xgboost":
        return XGBRegressor(
            **params, objective="reg:squarederror", tree_method="hist",
            n_jobs=2, random_state=seed,
        )
    if name == "catboost":
        return CatBoostRegressor(
            **params, loss_function="RMSE", thread_count=2,
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
    if name == "mlp":
        params = params.copy()
        params["hidden_layer_sizes"] = tuple(
            int(part) for part in params.pop("architecture").split(",")
        )
        return MLPRegressor(
            **params, max_iter=150, early_stopping=False, shuffle=False,
            batch_size=256, random_state=seed,
        )
    if name == "kalman":
        return KalmanRegression(**params)
    if name == "rbf_svr":
        return SVR(**params, kernel="rbf", cache_size=256)
    raise ValueError(name)


def suggest_parameters(trial, name):
    if name == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=80),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", .02, .15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 10, 100, log=True),
            "subsample": trial.suggest_float("subsample", .7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 80, 240, step=80),
            "depth": trial.suggest_int("depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", .02, .15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 100, log=True),
        }
    if name == "mlp":
        return {
            "architecture": trial.suggest_categorical("architecture", ["16", "32", "32,16"]),
            "alpha": trial.suggest_float("alpha", .001, 1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", .0003, .003, log=True),
        }
    if name == "kalman":
        return {
            "process_noise": trial.suggest_float("process_noise", 1e-7, .01, log=True),
            "measurement_noise": trial.suggest_float("measurement_noise", .1, 10, log=True),
        }
    return {
        "C": trial.suggest_float("C", .01, 10, log=True),
        "epsilon": trial.suggest_float("epsilon", .05, .5),
        "gamma": trial.suggest_float("gamma", .001, 1, log=True),
    }


def fit_candidate(name, params, fold, seed):
    bundle, X, y, dates = fold["prepared"]
    if X.shape[1] == 0:
        return None
    model = create_model(name, params, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=2):
            if name == "kalman":
                model.fit(X, y, dates=dates)
            else:
                model.fit(X, y)
    fold["convergence_warnings"] += sum(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    return model


def predict_candidate(model, bundle, frame):
    if model is None:
        return np.full(len(frame), bundle["target_mean"])
    return (
        model.predict(transform_features(bundle, frame))
        * bundle["target_scale"] + bundle["target_mean"]
    )


def score_predictions(frame, predictions, cutoff):
    valid = frame.target.notna() & frame.label_end.le(pd.Timestamp(cutoff))
    values = frame.loc[valid, ["date", "target"]].copy()
    values["prediction"] = np.asarray(predictions)[valid]
    correlations = []
    for _, group in values.groupby("date"):
        if group.prediction.nunique() > 1 and group.target.nunique() > 1:
            correlations.append(group.prediction.corr(group.target, method="spearman"))
    residual = values.prediction - values.target
    return {
        "rank_ic": float(np.mean(correlations)) if correlations else 0.0,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(abs(residual))),
        "scored_rows": len(values), "scored_months": len(correlations),
        "predicted_rows": len(frame),
    }


def prepare_fold(panel, calendar, year, cutoff, config, tag):
    train, valid, boundary = purged_split(
        panel, calendar, year, cutoff, config["purge_extra_sessions"]
    )
    prepared = fit_features(train, config, tag)
    audit = {
        "fold": tag, "first_validation_signal": str(valid.date.min()),
        "purge_boundary": str(boundary),
        "last_training_signal": str(train.date.max()),
        "last_training_outcome": str(train.label_end.max()),
        "training_rows_available": len(train),
        "training_rows_used": prepared[0]["training_rows"],
        "validation_rows": len(valid),
    }
    return {"prepared": prepared, "validation": valid,
            "audit": audit, "convergence_warnings": 0}


def run_development():
    config = read_config()
    OUTPUT.mkdir(exist_ok=True)
    frozen = OUTPUT / "frozen_experiment.json"
    if frozen.exists():
        raise RuntimeError("An experiment is already frozen; do not overwrite its history.")
    save_json(OUTPUT / "development_protocol.json", config)
    panel, calendar = build_panel(
        config["development_end"], config["development_start"]
    )
    panel.to_pickle(OUTPUT / "development_panel.pkl")
    folds = [prepare_fold(
        panel, calendar, year, config["tuning_label_cutoff"], config,
        f"inner_{year}",
    ) for year in config["inner_validation_years"]]
    selection = prepare_fold(
        panel, calendar, config["selection_year"],
        config["tuning_label_cutoff"], config, "selection_2014",
    )
    pd.DataFrame([f["audit"] for f in folds + [selection]]).to_csv(
        OUTPUT / "split_audit.csv", index=False
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parameters, tuning_rows, selection_rows = {}, [], []
    for name in config["models"]:
        print(f"Optuna: {name}, {config['optuna_trials_per_model']} trials", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=config["seed"], n_startup_trials=4
            ),
        )

        def objective(trial):
            params = suggest_parameters(trial, name)
            scores = []
            for fold in folds:
                model = fit_candidate(name, params, fold, config["seed"])
                predictions = predict_candidate(
                    model, fold["prepared"][0], fold["validation"]
                )
                metrics = score_predictions(
                    fold["validation"], predictions,
                    config["tuning_label_cutoff"],
                )
                scores.append(metrics["rank_ic"])
            trial.set_user_attr("fold_rank_ics", scores)
            return float(np.mean(scores))

        study.optimize(objective, n_trials=config["optuna_trials_per_model"])
        study.trials_dataframe().to_csv(OUTPUT / f"trials_{name}.csv", index=False)
        parameters[name] = study.best_params
        tuning_rows.append({"model": name, "tuning_rank_ic": study.best_value})
        model = fit_candidate(name, parameters[name], selection, config["seed"])
        predictions = predict_candidate(
            model, selection["prepared"][0], selection["validation"]
        )
        metrics = score_predictions(
            selection["validation"], predictions,
            config["selection_label_cutoff"],
        )
        selection_rows.append({"model": name, **metrics})
        print(f"2014 selection {name}: {metrics}", flush=True)
        save_json(OUTPUT / "best_parameters.json", parameters)
        pd.DataFrame(tuning_rows).to_csv(OUTPUT / "tuning_summary.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(OUTPUT / "model_selection.csv", index=False)
    ranking = pd.DataFrame(selection_rows).sort_values(
        ["rank_ic", "rmse", "model"], ascending=[False, True, True]
    )
    winner = ranking.iloc[0].model
    # Winner and trading rules are locked before inspecting 2015 outcomes.
    save_json(OUTPUT / "selected_before_confirmation.json", {
        "model": winner, "parameters": parameters[winner],
        "strategy": config["strategy"],
        "selection_rule": "highest 2014 monthly rank IC; RMSE/name break ties",
    })
    confirmation = prepare_fold(
        panel, calendar, config["confirmation_year"],
        config["selection_label_cutoff"], config, "confirmation_2015",
    )
    model = fit_candidate(winner, parameters[winner], confirmation, config["seed"])
    predictions = predict_candidate(
        model, confirmation["prepared"][0], confirmation["validation"]
    )
    check = confirmation["validation"][["date", "symbol", "target", "label_end"]].copy()
    check["prediction"] = predictions
    check.to_csv(OUTPUT / "confirmation_predictions.csv", index=False)
    metrics = score_predictions(
        confirmation["validation"], predictions, config["development_end"]
    )
    save_json(OUTPUT / "confirmation_metrics.json", {"model": winner, **metrics})
    # No 2016 outcomes, including the buffer, enter the final fit.
    final_training = panel.loc[
        panel.target.notna()
        & panel.label_end.le(pd.Timestamp(config["development_end"]))
    ].copy()
    prepared = fit_features(final_training, config, "final_pre2016")
    fold = {"prepared": prepared, "convergence_warnings": 0}
    final_model = fit_candidate(winner, parameters[winner], fold, config["seed"])
    joblib.dump({"model": final_model, "features": prepared[0]}, OUTPUT / "frozen_model.joblib")
    source_files = ["research.py", "research_config.json", "backtester.py"]
    source_hashes = {
        path: hashlib.sha256((HERE / path).read_bytes()).hexdigest()
        for path in source_files
    }
    freeze = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": winner, "parameters": parameters[winner],
        "config": config, "source_sha256": source_hashes,
        "model_sha256": hashlib.sha256((OUTPUT / "frozen_model.joblib").read_bytes()).hexdigest(),
        "latest_training_outcome": str(final_training.label_end.max()),
        "confirmation": metrics,
        "convergence_warnings": {
            **{f["audit"]["fold"]: f["convergence_warnings"] for f in folds + [selection, confirmation]},
            "final": fold["convergence_warnings"],
        },
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scikit-learn", "Boruta", "optuna", "xgboost", "catboost"]
        },
    }
    save_json(frozen, freeze)
    print(f"FROZEN: {winner}; 2015 confirmation {metrics}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["develop"])
    parser.parse_args()
    # Import under a stable module name so the custom estimator reloads in
    # notebooks as well as command-line scripts.
    from research import run_development as execute_development
    execute_development()
