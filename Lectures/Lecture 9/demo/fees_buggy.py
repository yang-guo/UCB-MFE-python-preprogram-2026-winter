"""Intentional classroom bug. Never used by the API or the default test suite."""


def trading_fee(shares: float, price: float, fee_bps: float) -> float:
    return shares * price * fee_bps / 100  # BUG: basis points are not percentages.
