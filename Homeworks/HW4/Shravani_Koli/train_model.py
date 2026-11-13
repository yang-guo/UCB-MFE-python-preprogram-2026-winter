
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = (
    Path(__file__).resolve().parents[3]
    / "Lectures"
    / "modeling"
    / "data"
    / "equities_top2000.duckdb"
)

FEATURES = ["return_20d", "return_60d", "volatility_20d"]
def fit_final_model(data):
    """
    Train the final Ridge model for the backtest.
    """

    data = data.copy()

    data["feature_date"] = pd.to_datetime(data["feature_date"])
    data["exit_date"] = pd.to_datetime(data["exit_date"])

    final_training = data[
        (data["feature_date"] <= pd.Timestamp("2015-12-31"))
        & (data["exit_date"] < pd.Timestamp("2016-04-01"))
    ].copy()

    if final_training.empty:
        raise ValueError("No eligible training observations.")

    feature_lower = final_training[FEATURES].quantile(0.01)
    feature_upper = final_training[FEATURES].quantile(0.99)

    X_train = final_training[FEATURES].clip(
        lower=feature_lower,
        upper=feature_upper,
        axis=1
    )

    target_lower = final_training["target_90d"].quantile(0.01)
    target_upper = final_training["target_90d"].quantile(0.99)

    y_train = final_training["target_90d"].clip(
        lower=target_lower,
        upper=target_upper
    )

    final_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0)
    )

    final_model.fit(X_train, y_train)

    return final_model, feature_lower, feature_upper

def main():
    print("Preparing training data...")

    from model import load_development_data

    data = load_development_data()

    print("Total development observations:", len(data))

    data["feature_date"] = pd.to_datetime(data["feature_date"])
    data["entry_date"] = pd.to_datetime(data["entry_date"])
    data["exit_date"] = pd.to_datetime(data["exit_date"])

    training_cutoff = pd.Timestamp("2014-01-01")

    train = data[
        (data["feature_date"] < training_cutoff)
        & (data["exit_date"] < training_cutoff)
    ].copy()

    validation = data[
        (data["feature_date"] >= training_cutoff)
        & (data["feature_date"] <= pd.Timestamp("2015-12-31"))
    ].copy()

    print("Training observations:", len(train))
    print("Validation observations:", len(validation))

    if not train.empty:
        print("Latest training exit:", train["exit_date"].max())

    if not validation.empty:
        print(
            "First validation feature date:",
            validation["feature_date"].min()
        )

    # Limit extreme training returns using training data only.
    lower_limit = train["target_90d"].quantile(0.01)
    upper_limit = train["target_90d"].quantile(0.99)

    training_target = train["target_90d"].clip(
        lower=lower_limit,
        upper=upper_limit
    )

    print("\nTraining target lower limit:", round(lower_limit, 4))
    print("Training target upper limit:", round(upper_limit, 4))

    # Calculate feature clipping limits using training data only.
    feature_lower = train[FEATURES].quantile(0.01)
    feature_upper = train[FEATURES].quantile(0.99)

    X_train = train[FEATURES].clip(
        lower=feature_lower,
        upper=feature_upper,
        axis=1
    )

    X_validation = validation[FEATURES].clip(
        lower=feature_lower,
        upper=feature_upper,
        axis=1
    )

    print("\nFEATURE CLIPPING LIMITS")
    print("Lower limits:")
    print(feature_lower)
    print("Upper limits:")
    print(feature_upper)

    # Compare Ridge regression models.
    print("\nCOMPARING RIDGE MODELS")

    for alpha in [0.1, 10.0, 100.0]:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )

        model.fit(X_train, training_target)

        predictions = model.predict(X_validation)

        correlation = pd.Series(
            predictions,
            index=validation.index
        ).corr(validation["target_90d"])

        print(f"\nAlpha: {alpha}")
        print(
            "Average prediction:",
            round(predictions.mean(), 4)
        )
        print(
            "Prediction standard deviation:",
            round(predictions.std(), 6)
        )
        print(
            "Validation correlation:",
            round(correlation, 6)
        )

    print("\nMODEL COMPARISON COMPLETE")

    # Use predictions from the final Ridge model (alpha=100).
    validation["predicted_return"] = predictions

    # Evaluate stock rankings.
    print("\nSTOCK RANKING DIAGNOSTICS")

    validation["prediction_group"] = pd.qcut(
        validation["predicted_return"].rank(method="first"),
        q=5,
        labels=["Lowest", "Low", "Middle", "High", "Highest"]
    )

    ranking_results = validation.groupby(
        "prediction_group",
        observed=True
    ).agg(
        average_predicted_return=("predicted_return", "mean"),
        average_actual_return=("target_90d", "mean"),
        stock_observations=("target_90d", "count")
    )

    print(ranking_results)

    print("\nSTOCK RANKING BY VALIDATION YEAR")

    validation["year"] = validation["feature_date"].dt.year

    yearly_results = validation.groupby(
        ["year", "prediction_group"],
        observed=True
    ).agg(
        average_actual_return=("target_90d", "mean"),
        stock_observations=("target_90d", "count")
    )

    print(yearly_results)

    # Original data-quality diagnostics.
    print("\nTraining target distribution:")
    print(
        train["target_90d"].describe(
            percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
        )
    )

    print("\nValidation target distribution:")
    print(
        validation["target_90d"].describe(
            percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
        )
    )

    print("\nTOP 10 EXTREME TRAINING RETURNS")

    columns = [
        "symbol",
        "feature_date",
        "entry_date",
        "exit_date",
        "target_90d"
    ]

    print(
        train.nlargest(10, "target_90d")[columns].to_string(
            index=False
        )
    )

    print("\nTRAINING RETURNS ABOVE 500%:")
    print((train["target_90d"] > 5).sum())

    print("\nTRAINING RETURNS ABOVE 100%:")
    print((train["target_90d"] > 1).sum())

    # Feature diagnostics.
    print("\nFEATURE DIAGNOSTICS")

    print("\nTraining feature statistics:")
    print(train[FEATURES].describe().to_string())

    print("\nValidation feature statistics:")
    print(validation[FEATURES].describe().to_string())

    print("\nMissing feature values:")
    print(train[FEATURES].isna().sum())

    print("\nFeature correlations with 90-day returns:")
    print(
        train[FEATURES + ["target_90d"]]
        .corr()["target_90d"]
    )


if __name__ == "__main__":
    main()
