"""The twelve HW4 inputs, in their original training order and units."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from model_components import FEATURE_COLUMNS

MAX_FILING_AGE_DAYS = 2**53 - 1


class InputData(BaseModel):
    """One already-computed HW4 feature row; all twelve keys are required."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: float = Field(
        gt=-1, description="Adjusted-close return over 21 market sessions, as a fraction."
    )
    momentum_63: float = Field(
        gt=-1, description="Adjusted-close return over 63 market sessions, as a fraction."
    )
    momentum_252_21: float | None = Field(
        gt=-1,
        description="P[t-21] / P[t-252] - 1; fraction, or null for insufficient history.",
    )
    volatility_63: float = Field(
        ge=0,
        description="63-session sample standard deviation of daily returns * sqrt(252).",
    )
    drawdown_252: float | None = Field(
        gt=-1,
        le=0,
        description="Adjusted close / 252-session rolling maximum - 1, or null.",
    )
    net_margin: float | None = Field(
        description="Quarterly net income / revenue; finite ratio or explicit null."
    )
    operating_roa: float | None = Field(
        description="Quarterly operating income / total assets; finite ratio or null."
    )
    debt_to_assets: float | None = Field(
        description="Total debt / total assets; finite ratio or null."
    )
    cash_to_assets: float | None = Field(
        description="Cash and equivalents / total assets; finite ratio or null."
    )
    fcf_to_assets: float | None = Field(
        description="Quarterly free cash flow / total assets; finite ratio or null."
    )
    accruals_to_assets: float | None = Field(
        description="(Quarterly net income - operating cash flow) / total assets, or null."
    )
    filing_age_days: int | None = Field(
        ge=0,
        le=MAX_FILING_AGE_DAYS,
        description=(
            "Whole calendar days since filing date + 1 calendar day, or null; "
            "at most 2**53 - 1 to preserve exact integers in the float64 feature matrix."
        ),
    )


def to_feature_frame(data: InputData | Mapping[str, Any]) -> pd.DataFrame:
    """Keep training names/order and turn explicit nulls into imputer-compatible NaN."""
    values = data.model_dump() if isinstance(data, InputData) else dict(data)
    return pd.DataFrame(
        [[values[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS, dtype=float
    )
