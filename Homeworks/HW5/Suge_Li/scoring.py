"""Turn one validated feature row into a finite rank prediction."""

import math

import pandas as pd


def score_features(pipeline, payload: dict[str, float], feature_names: list[str]) -> float:
    """Score one row with the saved pipeline.

    Column names and order match training. ``ColumnTransformer`` selects by
    name, and a DataFrame is required because those names are not positions.
    """
    frame = pd.DataFrame([payload]).reindex(columns=feature_names)
    if int(frame.isna().to_numpy().sum()) != 0:
        raise ValueError("every feature must be present")
    prediction = float(pipeline.predict(frame)[0])
    if not math.isfinite(prediction):
        raise RuntimeError("model returned a non-finite prediction")
    return prediction
