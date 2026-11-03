import pytest

from fees import trading_fee


def test_known_fee(fee_function):
    # Independent expectation: 2 * $100 * 0.1% = $0.20.
    actual = fee_function(shares=2, price=100, fee_bps=10)
    assert actual == pytest.approx(0.20)


@pytest.mark.parametrize(
    "shares,price,bps,expected",
    [
        (0, 100, 10, 0),
        (2, 100, 0, 0),
        (0.5, 100, 10, 0.05),
    ],
)
def test_boundaries(shares, price, bps, expected):
    assert trading_fee(shares, price, bps) == pytest.approx(expected)


@pytest.mark.parametrize(
    "shares,price,bps",
    [
        (-1, 100, 10),
        (2, 0, 10),
        (2, -100, 10),
        (2, 100, -1),
        (float("nan"), 100, 10),
        (2, float("inf"), 10),
        (2, 100, float("nan")),
    ],
)
def test_invalid_inputs(shares, price, bps):
    with pytest.raises(ValueError):
        trading_fee(shares, price, bps)
