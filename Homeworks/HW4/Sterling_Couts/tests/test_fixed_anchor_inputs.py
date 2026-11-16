"""Guarantee forecast causality and isolation of archived account evidence."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fixed_anchor_inputs import predict_selected, redirected_attribute


class AlreadyFitted:
    def fit(self, *args, **kwargs):
        raise AssertionError("Evaluation must never fit a model")

    def predict(self, x):
        assert list(x) == ["rank_signal"]
        return x.rank_signal.fillna(0.5).to_numpy() * 0.1


def example():
    panel = pd.DataFrame(dict(
        date=pd.to_datetime(["2016-03-31"] * 3), symbol=["A", "B", "C"],
        entry_date=pd.Timestamp("2016-04-01"), scheduled_exit=pd.Timestamp("2016-06-30"),
        target=[0.2, -0.1, np.nan], label_end=pd.to_datetime(["2016-06-30"] * 2 + [None]),
        eligible=True, risk_screen_pass=[True, True, False], rank_signal=[0.2, np.nan, 0.9],
    ))
    return panel, dict(features=["rank_signal"], fitted_asof="2012-12-31")


def test_only_inputs_determine_scores_not_future_outcomes():
    panel, config = example()
    before = predict_selected(AlreadyFitted(), panel, config)
    panel["target"] = np.nan
    panel["label_end"] = pd.NaT
    after = predict_selected(AlreadyFitted(), panel, config)
    pd.testing.assert_frame_equal(
        before.drop(columns=["target", "label_end"]),
        after.drop(columns=["target", "label_end"]),
    )
    assert after.symbol.tolist() == ["A", "B"]
    assert after.target.isna().all()
    np.testing.assert_allclose(after.score, [0.02, 0.05])


def test_reject_future_fit_date():
    panel, config = example()
    config["fitted_asof"] = "2017-01-01"
    with pytest.raises(ValueError, match="predate"):
        predict_selected(AlreadyFitted(), panel, config)


def test_restore_archived_location_even_on_failure():
    module = SimpleNamespace(OUT="archive")
    with pytest.raises(RuntimeError):
        with redirected_attribute(module, "OUT", "new_account"):
            assert module.OUT == "new_account"
            raise RuntimeError("Simulated accounting failure")
    assert module.OUT == "archive"
