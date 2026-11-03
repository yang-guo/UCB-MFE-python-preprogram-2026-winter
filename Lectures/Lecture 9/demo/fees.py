"""Teaching example: fee on nonnegative share quantity; no currency rounding."""

from math import isfinite


def trading_fee(shares: float, price: float, fee_bps: float) -> float:
    if not all(isfinite(x) for x in (shares, price, fee_bps)):
        raise ValueError("Inputs must be finite")
    if shares < 0 or price <= 0 or fee_bps < 0:
        raise ValueError("Shares and fee must be nonnegative; price must be positive")
    return shares * price * fee_bps / 10_000
