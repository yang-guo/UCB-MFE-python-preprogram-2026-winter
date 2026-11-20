import pandas as pd


def make_feature_frame(values, features):
    return pd.DataFrame([values], columns=features, dtype="float64")


def predict_frame(bundle, frame):
    clipped = frame.clip(
        lower=bundle["feature_min"],
        upper=bundle["feature_max"],
        axis=1,
    )

    scaled = bundle["scaler"].transform(clipped)
    return bundle["model"].predict(scaled)
