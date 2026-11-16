"""The service accepts prepared HW4 ranks, not raw market observations."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# pandas rank(pct=True) is average rank divided by the nonmissing peer count.
# Zero is impossible; one is valid. Strict numeric types reject strings/bools.
PercentileRank = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]


class InputData(BaseModel):
    """Exactly eight named features; explicit null preserves missing financials."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    rank_momentum_21: PercentileRank
    rank_momentum_63: PercentileRank
    rank_momentum_252_21: PercentileRank
    rank_volatility_63: PercentileRank
    rank_drawdown_252: PercentileRank
    rank_momentum_126_21: PercentileRank
    rank_operating_roa: PercentileRank | None = Field(
        description="Required key; null means unavailable quarterly operating income/assets."
    )
    rank_fcf_to_assets: PercentileRank | None = Field(
        description="Required key; null means unavailable quarterly free cash flow/assets."
    )


class OutputData(BaseModel):
    """Forecast is a simple return fraction over the HW4 holding period."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    prediction: float
    model_version: str = Field(min_length=1)
