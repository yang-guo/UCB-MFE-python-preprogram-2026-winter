"""Portfolio analysis from daily NAV, after fees and including open lots.

Alpha is a daily excess-return regression intercept multiplied by 252.
VaR is a historical one-session loss quantile, not a guaranteed loss bound.
"""

import numpy as np
import pandas as pd


class BacktestAnalytics:
    """Read a BacktestResult without changing its account or history.

    The initial pre-trade capital row is retained, including first-entry fees
    in returns. Benchmark investment starts at the first test close.
    """

    def __init__(self, result, annual_rf=0.02, sessions_per_year=252):
        if not np.isfinite(annual_rf) or annual_rf <= -1:
            raise ValueError("annual_rf must be finite and greater than -1")
        if not isinstance(sessions_per_year, int) or sessions_per_year < 1:
            raise ValueError("sessions_per_year must be a positive integer")
        nav = result.history[["nav", "benchmark_nav"]].copy()
        if (not isinstance(nav.index, pd.DatetimeIndex) or len(nav) < 2
                or nav.index.hasnans or nav.index.has_duplicates
                or not nav.index.is_monotonic_increasing):
            raise ValueError("NAV needs at least two unique, increasing dates")
        if not np.isfinite(nav.to_numpy()).all() or nav.le(0).any().any():
            raise ValueError("NAVs must be finite and strictly positive; do not fill gaps")
        self.nav = nav
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self.returns = nav.pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(self.returns.to_numpy()).all():
            raise ValueError("Derived daily returns are not finite")
        self.sessions_per_year = sessions_per_year
        self.daily_rf = (1 + annual_rf) ** (1 / sessions_per_year) - 1

    def summary(self):
        """Metrics for both portfolios; returns/risks are decimal fractions.

        VaR_95_daily and VaR_99_daily are -q_05 and -q_01 of historical daily
        returns, with linear quantile interpolation. Positive means a loss;
        negative values are retained if the quantile represents a gain.
        """
        years = (self.nav.index[-1] - self.nav.index[0]).total_seconds() / (365.25 * 86400)
        market_excess = self.returns.benchmark_nav - self.daily_rf
        market_variance = market_excess.var(ddof=1)
        reports = {}
        for name in self.nav:
            nav, returns = self.nav[name], self.returns[name]
            excess = returns - self.daily_rf
            sigma = returns.std(ddof=1)
            beta = alpha = np.nan
            if len(returns) >= 2 and market_variance > 1e-24:
                beta = excess.cov(market_excess) / market_variance
                # Arithmetic annualization of the OLS intercept, not CAGR.
                alpha = (excess.mean() - beta * market_excess.mean()) * self.sessions_per_year
            reports[name] = {
                "final_value": nav.iloc[-1],
                "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
                "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
                "max_drawdown": (nav / nav.cummax() - 1).min(),
                "Sharpe": (np.sqrt(self.sessions_per_year) * excess.mean() / sigma
                           if sigma > 1e-12 else np.nan),
                "annualized_volatility": sigma * np.sqrt(self.sessions_per_year),
                "annualized_alpha": alpha,
                "beta": beta,
                "VaR_95_daily": -returns.quantile(0.05, interpolation="linear"),
                "VaR_99_daily": -returns.quantile(0.01, interpolation="linear"),
                "daily_observations": len(returns),
            }
        report = pd.DataFrame(reports).rename(columns={
            "nav": "Strategy", "benchmark_nav": "S&P 500 total return"
        })
        # NaN is intentional for undefined ratios (constant or short samples).
        if np.isinf(report.to_numpy()).any():
            raise ValueError("Derived performance metrics are not finite")
        required = ["final_value", "total_return", "CAGR", "max_drawdown", "VaR_95_daily", "VaR_99_daily"]
        if not np.isfinite(report.loc[required].to_numpy()).all():
            raise ValueError("Derived performance metrics are not finite")
        return report

    def comparison(self):
        """Benchmark outperformance in dollars and percentage-point units."""
        total_returns = self.nav.iloc[-1] / self.nav.iloc[0] - 1
        difference = self.nav.nav.iloc[-1] - self.nav.benchmark_nav.iloc[-1]
        with np.errstate(over="ignore", invalid="ignore"):
            difference_pp = 100 * (total_returns.nav - total_returns.benchmark_nav)
        if (not np.isfinite(total_returns.to_numpy()).all()
                or not np.isfinite([difference, difference_pp]).all()):
            raise ValueError("Benchmark comparison arithmetic is not finite")
        return pd.Series({
            "strategy_final_value": self.nav.nav.iloc[-1],
            "benchmark_final_value": self.nav.benchmark_nav.iloc[-1],
            "final_value_difference": difference,
            "total_return_difference_pp": difference_pp,
            "beat_benchmark": bool(difference > 0),
        })

    def cumulative_returns(self):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cumulative = self.nav.div(self.nav.iloc[0]).sub(1)
        if not np.isfinite(cumulative.to_numpy()).all():
            raise ValueError("Cumulative-return arithmetic is not finite")
        return cumulative

    def plot_cumulative_returns(self, ax=None):
        """Return an axes with both required cumulative-return curves."""
        ax = self.cumulative_returns().rename(columns={
            "nav": "Strategy", "benchmark_nav": "S&P 500 total return"
        }).plot(ax=ax, title="Strategy versus benchmark", ylabel="Cumulative return")
        return ax
