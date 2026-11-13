"""Pairs_Tilt research: common-clock volume bars, clusters, ADF, Kalman.

Design decisions
----------------
* The cap-weighted eligible universe's own volume bars define the common
  sampling clock.  A bar closes when the portfolio's cumulative volume reaches
  ten times its trailing 90-session median daily volume.  Every stock's
  adjusted close is sampled at the same bar ends, so pair observations are
  aligned.
* Cointegration is retested on a rebalance calendar, not on every completed
  bar: candidate decision dates come every ``rebalance_days`` calendar days
  (or an explicit ``signal_dates`` list) and each date uses the latest
  completed bar end at or before it.  ``rebalance_days=None`` restores the
  legacy behaviour of testing every completed bar.
* Clusters are refreshed on a fixed bar cadence (roughly semiannual at the
  10x bar size) with Ward hierarchical clustering on robust-standardized
  ``log ME, BM, OP, INV``; sector dummies are optional and off by default.
* Within a cluster, candidate pairs are pre-filtered by return correlation,
  tested with the Engle-Granger ``coint`` statistic on the log-price residual,
  and ranked by FDR-adjusted p-value.  A Kalman filter with a random-walk beta supplies the
  current hedge ratio and spread z-score.
* Only long-only relative-value signals are emitted: the cheap leg is the
  long candidate; the expensive leg is not purchased.

This module is intentionally independent of the execution engine.  The
strategy in ``strategies/pairs.py`` consumes :class:`PairsResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numbers
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from backtesting import FactorResult, MarketData
from research.bars import VolumeBars, volume_bars


@dataclass(frozen=True)
class PairsConfig:
    """Tunable research settings with conservative defaults."""

    bar_median_window: int = 90
    bar_multiple: float = 10.0
    cluster_refresh_bars: int = 13          # ~semiannual at 10x bars
    min_cluster_size: int = 5
    min_clusters: int = 5
    max_clusters: int = 25
    silhouette_sample: int = 500
    pair_lookback_bars: int = 104
    min_history_bars: int = 60
    # Cointegration is retested every ``rebalance_days`` calendar days on the
    # latest completed bar.  ``None`` tests every completed bar (legacy).
    rebalance_days: int | None = 10
    candidate_pairs_per_cluster: int = 20
    max_pairs_per_cluster: int = 3
    adf_pvalue: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 26.0
    entry_z: float = 2.0
    max_stale_bars: int = 0
    kalman_process_var: float = 1e-6
    kalman_obs_var: float = 1e-3
    # Sector is a current vendor snapshot, not a point-in-time field.  Leave
    # dummies off by default so historical clusters do not use a later
    # classification; enable only with an explicit point-in-time sector map.
    sector_dummies: bool = False


@dataclass(frozen=True)
class PairsResult:
    """Point-in-time pair signals and the data used to build them."""

    signals: pd.DataFrame
    clusters: pd.DataFrame
    bars: pd.DataFrame
    coverage: pd.DataFrame
    bar_clock: VolumeBars


def point_in_time_eligible(chars):
    """Eligibility without the static sector snapshot.

    ``FamaFrench5.eligible`` includes ``sector_ok`` from ``universe.sector``,
    which is a current vendor snapshot.  For pair research we use the
    point-in-time membership/accounting/price flags instead; sector is not a
    dated field and must not define historical eligibility.
    """
    flags = {"member_ok", "be_present", "be_ok", "values_ok", "me_ok"}
    if flags.issubset(chars.columns):
        return (chars.member_ok & chars.be_present & chars.be_ok
                & chars.values_ok & chars.me_ok)
    return chars.eligible.eq(True)


def _monthly_characteristics(factors: FactorResult):
    chars = factors.characteristics
    required = {"date", "symbol", "ME", "BM", "OP", "INV", "eligible", "sector"}
    missing = required - set(chars.columns)
    if missing:
        raise ValueError("factor characteristics missing: " + ", ".join(sorted(missing)))
    return chars


def _portfolio_series(data: MarketData, factors: FactorResult):
    """Daily cap-weighted price and volume of the eligible universe."""
    chars = _monthly_characteristics(factors)
    me = chars.pivot(index="date", columns="symbol", values="ME")
    eligible = (chars.assign(pairs_eligible=point_in_time_eligible(chars))
                .pivot(index="date", columns="symbol", values="pairs_eligible"))
    me = me.reindex(data.calendar).ffill()
    eligible_daily = (eligible.reindex(data.calendar).ffill()
                      .fillna(False).astype(bool))
    me = me.where(eligible_daily)
    weights = me.div(me.sum(axis=1), axis=0)

    prices = data.price_panel("adj_close")
    returns = prices.pct_change(fill_method=None)
    # Use prior-close weights for the current return; same-day ME is an
    # end-of-day quantity and must not weight that day's return.
    lagged = weights.shift(1)
    available = returns.notna() & lagged.notna()
    portfolio_return = ((returns * lagged).sum(axis=1)
                        / lagged.where(available).sum(axis=1))
    volume = data.volume_panel()
    portfolio_volume = ((volume * lagged).sum(axis=1)
                        / lagged.where(volume.notna()).sum(axis=1))
    portfolio_return = portfolio_return.fillna(0.0)
    price = (1.0 + portfolio_return).cumprod()
    valid = weights.notna().any(axis=1)
    return price.loc[valid], portfolio_volume.loc[valid]


def _cluster_cross_section(cross, config: PairsConfig):
    """Ward clusters for one point-in-time cross-section."""
    frame = cross.copy()
    for column in ("ME", "BM", "OP", "INV"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["log_me"] = np.log(frame["ME"].where(frame["ME"].gt(0)))
    numeric = ["log_me", "BM", "OP", "INV"]
    values = frame[numeric].replace([np.inf, -np.inf], np.nan)
    # Ratios can be many orders of magnitude; signed log compresses tails
    # before robust scaling.  Keep log_me on its natural log scale.
    for column in ("BM", "OP", "INV"):
        values[column] = np.sign(values[column]) * np.log1p(values[column].abs())
    # Heavy tails make the 1%/99% quantiles outliers themselves.  Bound by
    # median +/- 5 IQR, falling back to the quantiles when the IQR is zero.
    median = values.median()
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = (q3 - q1)
    lower = (median - 5.0 * iqr).where(iqr.gt(0), values.quantile(0.01))
    upper = (median + 5.0 * iqr).where(iqr.gt(0), values.quantile(0.99))
    values = values.clip(lower, upper, axis=1)
    values = values.fillna(values.median()).fillna(0.0)
    scaled = RobustScaler().fit_transform(values)
    parts = [scaled]
    if config.sector_dummies:
        sectors = frame["sector"].fillna("Unknown").astype(str)
        dummies = pd.get_dummies(sectors, prefix="sector", dtype=float)
        parts.append(dummies.to_numpy())
    matrix = np.concatenate(parts, axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if len(frame) < 2 * config.min_cluster_size:
        return pd.Series(0, index=frame.index, name="cluster")

    upper = min(config.max_clusters, max(2, len(frame) // config.min_cluster_size))
    if upper < 2:
        return pd.Series(0, index=frame.index, name="cluster")
    fits = []
    for count in range(2, upper + 1):
        labels = AgglomerativeClustering(
            n_clusters=count, linkage="ward").fit_predict(matrix)
        if np.unique(labels).size < 2:
            continue
        sample = min(len(matrix), max(2, config.silhouette_sample))
        fits.append((silhouette_score(
            matrix, labels, sample_size=sample if sample < len(matrix) else None,
            random_state=0), labels))
    if not fits:
        return pd.Series(0, index=frame.index, name="cluster")
    fits.sort(key=lambda item: item[0], reverse=True)
    fallback = None
    for _, labels in fits:
        merged = _merge_small_clusters(labels, matrix, config.min_cluster_size)
        if fallback is None:
            fallback = merged
        if np.unique(merged).size >= 2:
            return pd.Series(merged, index=frame.index, name="cluster")
    return pd.Series(fallback if fallback is not None else 0,
                     index=frame.index, name="cluster")


def _merge_small_clusters(labels, matrix, min_size):
    """Merge Ward splinters into the nearest large cluster."""
    labels = np.asarray(labels, dtype=int).copy()
    while True:
        counts = np.bincount(labels)
        small = [cluster for cluster, count in enumerate(counts) if 0 < count < min_size]
        large = [cluster for cluster, count in enumerate(counts) if count >= min_size]
        if not small:
            break
        if not large:
            return np.zeros(len(labels), dtype=int)
        centroids = {cluster: matrix[labels == cluster].mean(axis=0)
                     for cluster in large}
        for cluster in small:
            rows = labels == cluster
            centroid = matrix[rows].mean(axis=0)
            nearest = min(large, key=lambda item: float(
                np.linalg.norm(centroid - centroids[item])))
            labels[rows] = nearest
    return pd.factorize(labels)[0]


def _ols_residual(y: pd.Series, x: pd.Series):
    design = sm.add_constant(x.to_numpy(dtype=float), has_constant="add")
    model = sm.OLS(y.to_numpy(dtype=float), design).fit()
    residual = y.to_numpy(dtype=float) - design @ model.params
    return model.params, residual


def _half_life(residual):
    series = pd.Series(residual)
    if len(series) < 6 or not np.isfinite(series.to_numpy()).all():
        return np.inf
    if float(np.ptp(series.to_numpy())) <= 1e-12:
        return np.inf
    lag = series.shift(1).dropna()
    change = series.diff().dropna()
    if len(lag) < 5 or float(np.ptp(lag.to_numpy())) <= 0:
        return np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(change.to_numpy(), sm.add_constant(lag.to_numpy())).fit()
    if len(model.params) < 2:
        return np.inf
    slope = model.params[1]
    if not np.isfinite(slope) or slope >= 0:
        return np.inf
    return float(-np.log(2.0) / slope)


def _kalman_hedge_ratio(y, x, config: PairsConfig):
    """Random-walk beta with an OLS prior; returns alpha, beta, one-step z.

    The prior matters: starting beta at 1.0 with log prices near 2-7 makes the
    first innovation drag beta toward zero, and the small process variance
    cannot recover.  Initialise from the window OLS and use a diffuse prior.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 4 or float(np.ptp(y)) <= 0 or float(np.ptp(x)) <= 0:
        raise ValueError("Kalman input has no variation")
    design = sm.add_constant(x, has_constant="add")
    fitted = sm.OLS(y, design).fit()
    prior = fitted.params
    if len(prior) < 2 or not np.isfinite(prior).all():
        raise ValueError("Kalman prior is not finite")
    theta = np.asarray(prior, dtype=float)
    covariance = np.eye(2) * 1e2
    process = np.diag([1e-6, config.kalman_process_var])
    residual_variance = float(np.var(fitted.resid, ddof=1))
    observation = (residual_variance if np.isfinite(residual_variance)
                   and residual_variance > 0 else config.kalman_obs_var)
    observation = max(observation, 1e-12)
    last_error, last_variance = np.nan, np.nan
    for target, regressor in zip(y, x):
        covariance = covariance + process
        row = np.array([1.0, float(regressor)])
        prediction = float(row @ theta)
        variance = float(row @ covariance @ row) + observation
        error = float(target) - prediction
        gain = covariance @ row / variance
        theta = theta + gain * error
        covariance = (np.eye(2) - np.outer(gain, row)) @ covariance
        last_error, last_variance = error, variance
    if not np.isfinite(last_variance) or last_variance <= 0:
        return float(theta[0]), float(theta[1]), 0.0
    return float(theta[0]), float(theta[1]), float(last_error / np.sqrt(last_variance))


class PairsResearch:
    """Build point-in-time pair signals from a ``MarketData`` object."""

    def __init__(self, config: PairsConfig | None = None):
        self.config = config or PairsConfig()
        if not isinstance(self.config.silhouette_sample, int) or self.config.silhouette_sample < 2:
            raise ValueError("silhouette_sample must be an integer of at least 2")
        rebalance = self.config.rebalance_days
        if rebalance is not None and (
                isinstance(rebalance, bool)
                or not isinstance(rebalance, numbers.Integral)
                or rebalance < 1):
            raise ValueError("rebalance_days must be None or a positive integer")

    def _decision_dates(self, bar_ends, signal_dates=None, start=None, end=None):
        """Candidate decision dates, before they are mapped to bar ends."""
        if signal_dates is not None:
            dates = pd.DatetimeIndex(
                pd.to_datetime(np.atleast_1d(signal_dates)))
            return dates[dates.notna()].unique().sort_values()
        if self.config.rebalance_days is None:
            return bar_ends
        first = pd.Timestamp(start) if start is not None else bar_ends[0]
        last = pd.Timestamp(end) if end is not None else bar_ends[-1]
        if last < first:
            return pd.DatetimeIndex([])
        return pd.date_range(first, last, freq=f"{self.config.rebalance_days}D")

    @staticmethod
    def _selected_bars(bar_ends, decision_dates):
        """(full-index position, bar end) for each decision date, causally.

        Each decision date uses the latest completed bar end at or before it;
        dates before the first bar and repeats of the same bar are skipped.
        """
        chosen, previous = [], None
        for date in decision_dates:
            position = int(bar_ends.searchsorted(date, side="right")) - 1
            if position < 0:
                continue
            bar_end = bar_ends[position]
            if bar_end == previous:
                continue
            previous = bar_end
            chosen.append((position, bar_end))
        return chosen

    def build(self, data: MarketData, factors: FactorResult,
              start=None, end=None, *, signal_dates=None) -> PairsResult:
        if not isinstance(data, MarketData):
            raise TypeError("data must be a MarketData")
        if not isinstance(factors, FactorResult):
            raise TypeError("factors must be a FactorResult")

        price, volume = _portfolio_series(data, factors)
        if start is not None:
            price = price.loc[price.index >= pd.Timestamp(start)]
            volume = volume.loc[volume.index >= pd.Timestamp(start)]
        if end is not None:
            price = price.loc[price.index <= pd.Timestamp(end)]
            volume = volume.loc[volume.index <= pd.Timestamp(end)]
        if len(price) <= self.config.bar_median_window:
            raise ValueError("Not enough daily history for the volume-bar clock")
        clock = volume_bars(
            price, volume,
            median_window=self.config.bar_median_window,
            bucket_multiple=self.config.bar_multiple,
            target_prior_session=True)
        bar_ends = pd.DatetimeIndex(clock.bars["end"])
        if len(bar_ends) < self.config.min_history_bars:
            raise ValueError("Not enough completed volume bars for pair research")

        daily_prices = data.price_panel("adj_close")
        # Do not forward-fill a delisted or missing name across bar ends;
        # an absent observation simply drops that pair observation.
        log_prices = np.log(daily_prices.reindex(bar_ends))
        log_prices.index.name = "bar_end"

        chars = _monthly_characteristics(factors)
        chars = chars.assign(pairs_eligible=point_in_time_eligible(chars))
        decisions = self._decision_dates(bar_ends, signal_dates, start, end)
        refresh = None
        signal_rows, cluster_rows, coverage_rows = [], [], []
        for position, bar_end in self._selected_bars(bar_ends, decisions):
            # Refresh on the true bar clock, not the filtered loop counter, so
            # coalescing decision dates cannot shift the cluster cadence.  Seed
            # the first selected bar when no cluster state exists yet.
            if (refresh is None or position == 0
                    or position % self.config.cluster_refresh_bars == 0):
                available = chars.loc[chars.date.le(bar_end) & chars.pairs_eligible]
                if not available.empty:
                    latest = available.date.max()
                    cross = available.loc[available.date.eq(latest)].drop_duplicates("symbol")
                    cross = cross.set_index("symbol")
                    labels = _cluster_cross_section(cross, self.config)
                    refresh = (bar_end, labels)
                    for symbol, cluster in labels.items():
                        cluster_rows.append(dict(date=bar_end, symbol=symbol,
                                                 cluster=int(cluster)))
            if refresh is None:
                continue
            _, labels = refresh
            current = chars.loc[chars.date.le(bar_end) & chars.pairs_eligible]
            if current.empty:
                eligible_now = set()
            else:
                latest_known = current.date.max()
                eligible_now = set(current.loc[
                    current.date.eq(latest_known), "symbol"])
            signals, tested, eligible = self._pairs_at(
                bar_end, labels, log_prices, eligible_now)
            signal_rows.extend(signals)
            coverage_rows.append(dict(date=bar_end, universe=len(eligible_now),
                                      clusters=int(labels.nunique()),
                                      pairs_tested=tested,
                                      pairs_eligible=eligible))

        signals = pd.DataFrame(signal_rows, columns=[
            "date", "pair_id", "leg_a", "leg_b", "long_symbol", "cluster",
            "beta", "alpha", "zscore", "adf_p", "adf_p_adj", "half_life",
            "correlation"])
        clusters = pd.DataFrame(cluster_rows, columns=["date", "symbol", "cluster"])
        coverage = pd.DataFrame(coverage_rows, columns=[
            "date", "universe", "clusters", "pairs_tested", "pairs_eligible"])
        bars = (log_prices.stack().dropna().rename("log_price")
                .reset_index().rename(columns={"bar_end": "date"}))
        return PairsResult(signals, clusters, bars, coverage, clock)

    def _pairs_at(self, bar_end, labels: pd.Series,
                  log_prices: pd.DataFrame, eligible_now):
        config = self.config
        candidates = []
        tested = 0
        for cluster, members in labels.groupby(labels):
            names = []
            for symbol in sorted(members.index):
                if symbol not in eligible_now or symbol not in log_prices.columns:
                    continue
                history = log_prices.loc[:bar_end, symbol].tail(
                    config.pair_lookback_bars)
                if history.notna().sum() >= config.min_history_bars:
                    names.append(symbol)
            if len(names) < 2:
                continue
            window = log_prices.loc[:bar_end, names].tail(config.pair_lookback_bars)
            # Pairwise-complete correlation: one sparse name must not shrink
            # every other pair's sample.
            returns = window.diff().iloc[1:]
            if returns.notna().sum().sum() < 10:
                continue
            correlation = returns.corr(min_periods=5)
            pairs = []
            for i, first in enumerate(names):
                for second in names[i + 1:]:
                    value = correlation.loc[first, second]
                    if np.isfinite(value):
                        pairs.append((abs(value), first, second, value))
            pairs.sort(reverse=True)
            for _, first, second, corr in pairs[:config.candidate_pairs_per_cluster]:
                pair = window[[first, second]].dropna()
                if len(pair) < config.min_history_bars:
                    continue
                # Both legs must actually quote at the signal bar; otherwise a
                # dead leg silently shortens the window and the pair can still
                # be "selected" on an old spread.
                last_pos = log_prices.index.get_loc(pair.index[-1])
                signal_pos = log_prices.index.get_loc(bar_end)
                if signal_pos - last_pos > config.max_stale_bars:
                    continue
                y, x = pair[first], pair[second]
                if float(np.ptp(y.to_numpy())) <= 0 or float(np.ptp(x.to_numpy())) <= 0:
                    continue
                try:
                    _, residual = _ols_residual(y, x)
                    maxlag = max(1, min(4, len(pair) // 10))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pvalue = coint(y.to_numpy(dtype=float),
                                       x.to_numpy(dtype=float), trend="c",
                                       maxlag=maxlag, autolag="aic")[1]
                    if not np.isfinite(pvalue):
                        continue
                    alpha, beta, zscore = _kalman_hedge_ratio(
                        y.to_numpy(), x.to_numpy(), config)
                except Exception:
                    continue
                tested += 1
                half_life = _half_life(residual)
                candidates.append(dict(
                    date=bar_end, pair_id=f"{first}|{second}",
                    leg_a=first, leg_b=second, cluster=int(cluster),
                    beta=beta, alpha=alpha, zscore=zscore, adf_p=pvalue,
                    half_life=half_life, correlation=corr))

        if not candidates:
            return [], tested, 0
        pvalues = [row["adf_p"] for row in candidates]
        rejected, adjusted, _, _ = multipletests(
            pvalues, alpha=config.adf_pvalue, method="fdr_bh")
        selected = []
        for row, keep, adj in zip(candidates, rejected, adjusted):
            if not keep or adj >= config.adf_pvalue:
                continue
            if not (config.min_half_life <= row["half_life"] <= config.max_half_life):
                continue
            if abs(row["zscore"]) < config.entry_z:
                continue
            row["adf_p_adj"] = float(adj)
            row["long_symbol"] = (row["leg_a"] if row["zscore"] < 0
                                  else row["leg_b"])
            selected.append(row)
        selected.sort(key=lambda row: (row["adf_p_adj"], -abs(row["zscore"])))
        # Keep the strongest pairs per cluster; the execution strategy applies
        # the one-pair-per-stock greedy rule across clusters.
        kept = []
        for cluster in sorted({row["cluster"] for row in selected}):
            group = [row for row in selected if row["cluster"] == cluster]
            kept.extend(group[:config.max_pairs_per_cluster])
        return kept, tested, len(kept)
