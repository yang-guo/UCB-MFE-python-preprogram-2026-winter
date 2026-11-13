"""Causality and arithmetic checks for the HMM research helpers."""

import unittest

import numpy as np
import pandas as pd

from research import (
    MarketSeries,
    ResearchConfig,
    ffd,
    ffd_weights,
    fit_hmm,
    forward_return,
    run_market,
    select_d,
    volume_bars,
    walk_forward_ffd_states,
    walk_forward_states,
)


class FractionalDifferenceChecks(unittest.TestCase):
    def test_weights_follow_the_afml_recursion(self):
        weights = ffd_weights(0.5, threshold=1e-6)
        expected = [1.0, -0.5, -0.125, -0.0625, -0.0390625]
        np.testing.assert_allclose(weights[: len(expected)], expected, atol=1e-12)

    def test_ffd_is_causal(self):
        values = np.linspace(1.0, 2.0, 120)
        changed = values.copy()
        changed[80:] += 10.0
        first = ffd(values, 0.5, threshold=1e-6)
        second = ffd(changed, 0.5, threshold=1e-6)
        np.testing.assert_allclose(first[:80], second[:80], equal_nan=True)

    def test_select_d_returns_a_grid_member(self):
        rng = np.random.default_rng(3)
        log_price = np.cumsum(rng.normal(0, 0.01, 500))
        grid = [0.0, 0.5, 1.0]
        chosen, table = select_d(log_price, grid, threshold=1e-5)
        self.assertIn(chosen, grid)
        self.assertEqual(list(table["d"]), grid)


class VolumeBarChecks(unittest.TestCase):
    def test_bars_are_contiguous_and_use_the_bucket_target(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        price = pd.Series(100 * np.exp(np.linspace(0, 0.2, 300)), index=dates)
        volume = pd.Series(np.full(300, 100.0), index=dates)
        result = volume_bars(price, volume, median_window=20, bucket_multiple=5)
        bars = result.bars
        self.assertGreater(len(bars), 40)
        self.assertTrue((bars["days"] == 5).all())
        self.assertTrue(
            (bars["start_pos"].iloc[1:].to_numpy()
             == bars["end_pos"].iloc[:-1].to_numpy() + 1).all()
        )
        np.testing.assert_allclose(bars["bar_volume"], 500.0)
        expected = np.log(
            price.to_numpy()[bars["end_pos"].to_numpy()]
            / price.to_numpy()[bars["base_pos"].to_numpy()]
        )
        np.testing.assert_allclose(bars["log_return"], expected)

    def test_bar_base_and_median_metadata_match_the_target(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        price = pd.Series(100 * np.exp(np.linspace(0, 0.2, 300)), index=dates)
        volume = pd.Series(np.full(300, 100.0), index=dates)
        bars = volume_bars(price, volume, median_window=20, bucket_multiple=5,
                           target_prior_session=True).bars
        self.assertTrue((bars.base_pos == bars.start_pos - 1).all())
        np.testing.assert_allclose(bars.median_volume * 5, bars.target)
        self.assertTrue((bars.days == 5).all())

    def test_bar_target_uses_prior_session_volume(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        price = pd.Series(100 * np.exp(np.linspace(0, 0.2, 300)), index=dates)
        volume = pd.Series(np.full(300, 100.0), index=dates)
        changed = volume.copy()
        changed.iloc[25] = 1000.0   # a bar-start session with a volume spike
        first = volume_bars(price, volume, median_window=20, bucket_multiple=5,
                                               target_prior_session=True).bars
        second = volume_bars(price, changed, median_window=20, bucket_multiple=5,
                                                target_prior_session=True).bars
        first_bar = first.loc[first.start_pos.eq(25)].iloc[0]
        second_bar = second.loc[second.start_pos.eq(25)].iloc[0]
        self.assertEqual(first_bar.days, 5)
        self.assertEqual(first_bar.target, 500.0)
        self.assertEqual(second_bar.target, 500.0)  # prior median, not the 1000 spike
        self.assertEqual(second_bar.days, 1)         # the spike fills the bucket at once

    def test_future_volume_does_not_change_earlier_bars(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        price = pd.Series(100 * np.exp(np.linspace(0, 0.2, 300)), index=dates)
        volume = pd.Series(np.full(300, 100.0), index=dates)
        changed = volume.copy()
        changed.iloc[200:] *= 50
        first = volume_bars(price, volume, median_window=20, bucket_multiple=5).bars
        second = volume_bars(price, changed, median_window=20, bucket_multiple=5).bars
        cutoff = dates[199]
        np.testing.assert_allclose(
            first.loc[first["end"] <= cutoff, "log_return"],
            second.loc[second["end"] <= cutoff, "log_return"],
        )

    def test_future_prices_do_not_change_earlier_bars(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        price = pd.Series(100 * np.exp(np.linspace(0, 0.2, 300)), index=dates)
        volume = pd.Series(np.full(300, 100.0), index=dates)
        changed = price.copy()
        changed.iloc[200:] *= 2
        first = volume_bars(price, volume, median_window=20, bucket_multiple=5).bars
        second = volume_bars(changed, volume, median_window=20, bucket_multiple=5).bars
        cutoff = dates[199]
        np.testing.assert_allclose(
            first.loc[first["end"] <= cutoff, "log_return"],
            second.loc[second["end"] <= cutoff, "log_return"],
        )


class WalkForwardChecks(unittest.TestCase):
    def test_states_before_a_future_shock_are_unchanged(self):
        rng = np.random.default_rng(11)
        values = rng.normal(0, 0.01, 220)
        dates = pd.bdate_range("2020-01-01", periods=220)
        kwargs = dict(min_train=60, refit_every=20, n_init=1, n_iter=50, seed=2)
        first = walk_forward_states(values, dates, **kwargs)
        changed = values.copy()
        changed[150:] += 5.0
        second = walk_forward_states(changed, dates, **kwargs)
        np.testing.assert_array_equal(
            first.states.to_numpy()[:150], second.states.to_numpy()[:150]
        )

    def test_forward_return_uses_the_next_90_calendar_days(self):
        dates = pd.bdate_range("2020-01-01", periods=200)
        price = pd.Series(np.arange(1, 201, dtype=float), index=dates)
        result = forward_return(price, [dates[0], dates[-1]], days=90)
        target = dates[1] + pd.Timedelta(days=90)
        j = dates.searchsorted(target, side="left")
        self.assertAlmostEqual(result.iloc[0], price.iloc[j] / price.iloc[1] - 1)
        self.assertTrue(np.isnan(result.iloc[1]))


class FFDWalkForwardChecks(unittest.TestCase):
    def test_daily_sample_matches_daily_dates(self):
        rng = np.random.default_rng(5)
        values = np.cumsum(rng.normal(0, 0.01, 900))
        dates = pd.bdate_range("2015-01-01", periods=900)
        result = walk_forward_ffd_states(
            values,
            dates,
            min_train=120,
            refit_every=40,
            d_refit_every=80,
            n_init=2,
            n_iter=50,
            threshold=1e-3,
            d_grid=[0.5, 1.0],
        )
        self.assertEqual(len(result.states), len(values))
        self.assertTrue(result.states.index.equals(dates))
        self.assertTrue(result.states.iloc[:120].eq(-1).all())
        self.assertTrue(result.states.iloc[120:].ge(0).any())

    def test_future_daily_prices_do_not_change_earlier_states(self):
        rng = np.random.default_rng(13)
        values = np.cumsum(rng.normal(0, 0.01, 900))
        dates = pd.bdate_range("2015-01-01", periods=900)
        kwargs = dict(
            min_train=120,
            refit_every=40,
            d_refit_every=80,
            n_init=2,
            n_iter=50,
            threshold=1e-3,
            d_grid=[0.5, 1.0],
            seed=3,
        )
        first = walk_forward_ffd_states(values, dates, **kwargs)
        changed = values.copy()
        changed[600:] += 0.05
        second = walk_forward_ffd_states(changed, dates, **kwargs)
        np.testing.assert_array_equal(
            first.states.to_numpy()[:600], second.states.to_numpy()[:600]
        )


class PipelineLookaheadChecks(unittest.TestCase):
    def test_future_prices_and_volume_do_not_change_earlier_pipeline_states(self):
        rng = np.random.default_rng(21)
        dates = pd.bdate_range("2018-01-01", periods=420)
        price = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))),
            index=dates,
        )
        volume = pd.Series(rng.lognormal(15, 0.3, len(dates)), index=dates)
        config = ResearchConfig(
            median_window=20,
            bucket_multiple=5.0,
            volume_min_train=40,
            volume_refit_every=10,
            ffd_min_train=60,
            ffd_refit_every=20,
            ffd_d_refit_every=40,
            n_init=3,
            n_iter=50,
            ffd_threshold=1e-3,
            ffd_d_grid=(0.5, 1.0),
        )
        first = run_market(MarketSeries("synthetic", price, volume), config)

        changed_price = price.copy()
        changed_volume = volume.copy()
        changed_price.iloc[300:] *= 1.01
        changed_volume.iloc[300:] *= 1.1
        second = run_market(
            MarketSeries("synthetic", changed_price, changed_volume), config
        )

        cutoff = dates[299]
        first_volume = first.volume.states.states
        second_volume = second.volume.states.states
        np.testing.assert_array_equal(
            first_volume.loc[first_volume.index <= cutoff].to_numpy(),
            second_volume.loc[second_volume.index <= cutoff].to_numpy(),
        )
        first_ffd = first.ffd.states.states
        second_ffd = second.ffd.states.states
        np.testing.assert_array_equal(
            first_ffd.loc[first_ffd.index <= cutoff].to_numpy(),
            second_ffd.loc[second_ffd.index <= cutoff].to_numpy(),
        )


class HMMFitChecks(unittest.TestCase):
    def test_isolated_large_last_bar_does_not_break_the_fit(self):
        rng = np.random.default_rng(0)
        values = np.r_[rng.normal(0, 0.009, 29), 0.12]
        fit = fit_hmm(values, n_states=3, n_init=3, seed=7, n_iter=100)
        self.assertTrue(np.isfinite(fit.log_likelihood))
        np.testing.assert_allclose(fit.model.transmat_.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
