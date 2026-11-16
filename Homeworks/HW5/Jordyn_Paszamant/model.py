"""Keep the HW4 feature ordering and learned clipping available to the pipeline."""

FEATURE_COLUMNS = (
    "momentum_21",
    "momentum_63",
    "momentum_126",
    "momentum_252",
    "volatility_21",
    "volatility_63",
    "ma_ratio_50",
    "ma_ratio_200",
)


def clip_features(frame, *, lower, upper):
    """Apply the saved training quantiles to the original named features."""
    return frame.loc[:, list(FEATURE_COLUMNS)].clip(lower=lower, upper=upper, axis=1)
