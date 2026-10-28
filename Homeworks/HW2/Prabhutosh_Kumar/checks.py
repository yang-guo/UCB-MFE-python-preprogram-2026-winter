"""Deterministic injected examples; no synthetic results are course-data output."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import (clean_data, diagnose, normalize, missing_dates, valid_prices,
                      calculate_returns, calculate_volatility,
                      calculate_max_drawdown, calculate_sharpe_ratio, metrics)


def fixture():
    """Build two isolated ticker histories with an explicit reference calendar."""
    records = []
    for ticker, level in [('AAA', 100.), ('BBB', 200.)]:
        for date in pd.bdate_range('2024-01-01', periods=40):
            records.append(dict(ticker=ticker, ts=date, open=level, high=level+2,
                                low=level-2, close=level, adj_close=level*.5, volume=1000.))
    return pd.DataFrame(records)


def check_cleaning():
    """Exercise required missing/invalid cases, normalization and quarantine."""
    ref = fixture()
    dirty = ref.copy()
    dirty.loc[1, 'ticker'] = ' <b>aaa</b> '
    dirty.loc[1, 'open'] = np.nan
    dirty.loc[2, 'adj_close'] = np.nan
    dirty.loc[3, 'volume'] = np.nan
    dirty.loc[4, ['open', 'high', 'low', 'close']] = np.nan
    dirty.loc[5, 'high'] = 1  # impossible bounds
    dirty.loc[6, 'close'] = 0
    dirty.loc[7, 'adj_close'] = np.inf  # trusted factor repairs this
    dirty.loc[8, 'low'] = np.inf
    dirty.loc[9, 'volume'] = -1
    dirty.loc[10, 'volume'] = 0
    dirty.loc[40, 'open'] = np.nan  # BBB leading gap cannot borrow AAA
    dirty = pd.concat([dirty, dirty.iloc[[11]]], ignore_index=True)
    conflict = dirty.iloc[[12]].copy()
    conflict['volume'] = 5
    dirty = pd.concat([dirty, conflict], ignore_index=True).drop(index=13)
    clean, before, after, removed, quarantine = clean_data(dirty, ref)
    assert valid_prices(clean).all()
    assert not clean.duplicated(['ticker', 'ts']).any()
    assert len(clean) + len(removed) + len(quarantine) == len(dirty)
    assert clean.repair_flags.str.contains('ffill_open').any()
    assert clean.repair_flags.str.contains('unknown_volume_filled_zero').sum() == 1
    assert clean.repair_flags.str.contains('adj_close_from_reference_date_factor').sum() == 2
    assert ((clean.ticker == 'BBB') & (clean.ts == ref.loc[40, 'ts'])).sum() == 0
    assert {'missing_value', 'invalid_value', 'ohlc_bounds', 'duplicate_key', 'zero_volume', 'missing_date'} <= set(before.issue_type)
    assert len(missing_dates(normalize(ref.drop(index=13)), normalize(ref))) == 1
    assert not missing_dates(normalize(ref), normalize(ref))
    # Both tails: +100% and -50%, with enough zero returns for 3σ screening.
    jump = pd.concat([ref.iloc[:40]] * 3, ignore_index=True)
    jump['ts'] = pd.bdate_range('2024-01-01', periods=len(jump))
    jump.loc[20, ['open', 'close', 'adj_close']] = [200, 200, 100]
    jump.loc[20, 'high'] = 202
    report = diagnose(normalize(jump), normalize(ref.iloc[:40]))
    assert report.issue_type.eq('extreme_return').sum() == 2
    assert not diagnose(normalize(ref), normalize(ref)).issue_type.eq('extreme_return').any()
    # Filling yesterday's open outside today's range is quarantined after repair.
    bounded = ref.iloc[:2].copy()
    bounded.loc[1, ['open', 'high', 'low', 'close', 'adj_close']] = [np.nan, 52, 48, 50, 25]
    retained, _, _, _, quarantined = clean_data(bounded, bounded.fillna({'open': 50}))
    assert len(retained) == 1 and len(quarantined) == 1
    # No trusted factor: a raw repair cannot establish adjusted consistency.
    retained, _, _, _, quarantined = clean_data(ref.iloc[:2].assign(open=[100, np.nan]), ref.iloc[:1])
    assert len(retained) == 1 and len(quarantined) == 1


def check_metrics():
    """Verify recovery, warm-ups, denominators, zero risk and month boundaries."""
    prices = pd.Series([100., 80., 100., 100.])
    returns = calculate_returns(prices)
    assert np.isclose(returns.iloc[1], np.log(.8))
    assert np.allclose(calculate_max_drawdown(prices), [0, -.2, 0, 0])
    assert np.isnan(calculate_sharpe_ratio(pd.Series([0., 0., 0.])))
    assert np.isnan(calculate_sharpe_ratio(pd.Series([.1])))
    rolling = calculate_volatility(calculate_returns(pd.Series(np.exp(np.arange(25) * .01))))
    assert rolling.iloc[:20].isna().all() and rolling.iloc[20:].notna().all()
    summary = metrics(pd.DataFrame({'adj_close': prices}))
    assert np.isclose(summary['total_return'], 0)
    assert np.isclose(summary['positive_days'], 1/3)
    assert np.isclose(summary['max_drawdown'], -.2)
    spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('03_run_all_stocks.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.analysis_window('2024-12-06') == ('2024-06-01', '2024-11-30')
    assert module.analysis_window('2024-12-31') == ('2024-06-01', '2024-11-30')
    assert module.analysis_window('2024-01-31') == ('2023-07-01', '2023-12-31')


if __name__ == '__main__':
    check_cleaning()
    check_metrics()
    print('All injected checks passed.')
