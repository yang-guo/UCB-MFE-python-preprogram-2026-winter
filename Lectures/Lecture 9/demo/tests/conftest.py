import pytest
from fastapi.testclient import TestClient

from app import app


def pytest_addoption(parser):
    parser.addoption("--fee-version", choices=["fixed", "buggy"], default="fixed")


@pytest.fixture
def fee_function(request):
    if request.config.getoption("--fee-version") == "buggy":
        from fees_buggy import trading_fee
    else:
        from fees import trading_fee
    return trading_fee


@pytest.fixture
def client():
    # The context manager runs startup/shutdown, just like a running service.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_input():
    return {"momentum_20d": 0.02, "volatility_20d": 0.15}
