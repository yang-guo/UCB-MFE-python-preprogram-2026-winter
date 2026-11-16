"""Independent arithmetic for the monthly financial-delay review.

These helpers are diagnostic only. They do not change the predictor, eligibility,
prices, or trade engine. Availability is filing + one day + the stated extra lag.
"""

import numpy as np
import pandas as pd


def direct_financial_record(records, signal, extra_days):
    """Select the latest known fiscal quarter by filtering raw records directly.

    This intentionally does not call the course event or as-of-join functions.
    A late filing for an older quarter cannot displace a newer known quarter.
    Missing or logically invalid filing dates are excluded, never guessed.
    """
    if not isinstance(extra_days, int) or extra_days < 0:
        raise ValueError("Extra delay must be a nonnegative integer number of days.")
    signal = pd.Timestamp(signal)
    available = records.filing_date + pd.Timedelta(days=extra_days + 1)
    valid = records.loc[
        records.period.eq("quarter")
        & records.filing_date.gt(records.fiscal_date)
        & available.le(signal)
    ]
    result = dict(
        fiscal_date=pd.NaT,
        filing_date=pd.NaT,
        available_date=pd.NaT,
        operating_roa=np.nan,
        fcf_to_assets=np.nan,
        stale=False,
    )
    if valid.empty:
        return result
    record = valid.sort_values(["fiscal_date", "filing_date"], kind="stable").iloc[-1]
    result.update(
        fiscal_date=record.fiscal_date,
        filing_date=record.filing_date,
        available_date=record.filing_date + pd.Timedelta(days=extra_days + 1),
    )
    result["stale"] = (signal - record.fiscal_date).days > 550
    if (
        not result["stale"]
        and pd.notna(record.total_assets)
        and record.total_assets > 0
    ):
        for output, numerator in [
            ("operating_roa", "operating_income"),
            ("fcf_to_assets", "free_cash_flow"),
        ]:
            ratio = record[numerator] / record.total_assets
            result[output] = ratio if np.isfinite(ratio) else np.nan
    return result


def drawdown_episode(nav):
    """Return the worst close-to-close drawdown and its peak/trough/recovery.

    Use the last occurrence of the prior peak; recovery means reaching that NAV
    again. An unrecovered episode is explicitly left without a recovery date.
    """
    if nav.empty or nav.isna().any() or not nav.gt(0).all():
        raise ValueError("Drawdown needs a nonempty, finite, positive NAV series.")
    if not np.isfinite(nav).all() or not nav.index.is_monotonic_increasing:
        raise ValueError("NAV must be finite and in date order.")
    drawdown = nav / nav.cummax() - 1
    trough = drawdown.idxmin()
    before = nav.loc[:trough]
    peak_value = before.max()
    peak = before.loc[before.eq(peak_value)].index[-1]
    recovered = nav.loc[trough:].ge(peak_value)
    recovery = recovered.loc[recovered].index[0] if recovered.any() else pd.NaT
    return dict(
        max_drawdown=float(drawdown.loc[trough]),
        peak_date=peak,
        trough_date=trough,
        recovery_date=recovery,
        peak_nav=float(peak_value),
        trough_nav=float(nav.loc[trough]),
    )


def audit_financial_sample(panel, fundamentals, extra_days, size=100):
    """Compare deterministic sampled joins with raw-record filtering/arithmetic."""
    sampled = panel.sample(n=min(size, len(panel)), random_state=731)
    groups = {symbol: group for symbol, group in fundamentals.groupby("symbol")}
    empty = fundamentals.iloc[:0]
    evidence = []
    for item in sampled.itertuples():
        expected = direct_financial_record(
            groups.get(item.symbol, empty), item.date, extra_days
        )
        for column in ["fiscal_date", "filing_date", "available_date"]:
            value = getattr(item, column)
            assert (pd.isna(value) and pd.isna(expected[column])) or value == expected[
                column
            ]
        for column in ["operating_roa", "fcf_to_assets"]:
            np.testing.assert_allclose(
                getattr(item, column),
                expected[column],
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )
        evidence.append(
            dict(
                symbol=item.symbol,
                date=item.date,
                extra_days=extra_days,
                **expected,
                checked=True,
            )
        )
    return pd.DataFrame(evidence)
