"""Shared, offline data-quality and risk-report functions for Homework 2."""
from pathlib import Path
import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import PercentFormatter

RAW = ['open', 'high', 'low', 'close']
PRICES = RAW + ['adj_close']
ISSUES = ['ticker', 'issue_type', 'issue_date', 'description', 'severity']


def normalize(data):
    """Normalize vendor ticker labels before grouping; retain original labels."""
    frame = data.copy()
    frame['original_ticker'] = frame['ticker']
    frame['ticker'] = frame['ticker'].map(
        lambda x: re.sub(r'<[^>]*>', '', str(x)).strip().upper() if pd.notna(x) else None)
    frame['ts'] = pd.to_datetime(frame['ts'], errors='coerce').dt.normalize()
    for column in PRICES + ['volume']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame.sort_values(['ticker', 'ts'], kind='stable').reset_index(drop=True)


def missing_dates(frame, reference):
    """Find absent rows using each ticker's supplied reference history/calendar."""
    result = []
    for ticker, group in reference.groupby('ticker'):
        expected = pd.DatetimeIndex(group['ts'].dropna().unique())
        actual = pd.DatetimeIndex(frame.loc[frame.ticker.eq(ticker), 'ts'].dropna().unique())
        for date in expected.difference(actual):
            result.append((ticker, 'missing_date', date, 'Date present in supplied ohlc calendar', 'medium'))
    return result


def diagnose(frame, reference):
    """Report multiple issues per row, including a two-tail sample-std screen."""
    issues = missing_dates(frame, reference)
    def add(row, kind, detail, severity='high'):
        issues.append((row.ticker, kind, row.ts, detail, severity))
    for row in frame.itertuples():
        if pd.isna(row.ticker) or row.ticker == '' or pd.isna(row.ts):
            add(row, 'invalid_key', 'Missing ticker or invalid date')
        for column in PRICES + ['volume']:
            value = getattr(row, column)
            if pd.isna(value):
                add(row, 'missing_value', f'{column} is missing', 'medium')
            elif not np.isfinite(value) or value < 0 or (column in PRICES and value == 0):
                add(row, 'invalid_value', f'{column}={value}')
        if row.volume == 0:
            add(row, 'zero_volume', 'Review possible stale data; zero alone is not proof', 'low')
        if not (row.low <= row.open <= row.high and row.low <= row.close <= row.high):
            if all(pd.notna(getattr(row, col)) for col in RAW):
                add(row, 'ohlc_bounds', 'Require low <= open/close <= high')
    for ticker, group in frame.groupby('ticker'):
        group = group.sort_values('ts')
        returns = group.close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        std = returns.std(ddof=1)
        if pd.notna(std) and std > 0:
            flagged = (returns - returns.mean()).abs() > 3 * std
            for idx in group.index[flagged]:
                row = group.loc[idx]
                add(row, 'extreme_return', f'Simple raw-close return {returns.loc[idx]:.6g}; review, not deletion', 'medium')
    duplicates = frame.duplicated(['ticker', 'ts'], keep=False)
    for row in frame.loc[duplicates].itertuples():
        add(row, 'duplicate_key', 'Repeated normalized (ticker, ts); distinguish exact/conflicting records')
    return pd.DataFrame(issues, columns=ISSUES)


def valid_prices(frame):
    """Check finite positive prices and raw OHLC bounds, row by row."""
    return (np.isfinite(frame[PRICES]).all(axis=1) & frame[PRICES].gt(0).all(axis=1)
            & frame.low.le(frame.open) & frame.open.le(frame.high)
            & frame.low.le(frame.close) & frame.close.le(frame.high))


def clean_data(raw, reference):
    """Conservatively repair missing prices; quarantine ambiguous observations.

    Reference values supply ONLY date-specific adjustment factors and calendars.
    Conflicting keys, impossible raw bounds and unresolved gaps are quarantined.
    Extreme returns are retained as uncertain review flags, never called news/splits.
    """
    frame, ref = normalize(raw), normalize(reference)
    before = diagnose(frame, ref)
    frame['source_row'] = frame.index
    frame['repair_flags'] = ''
    removed, quarantined = [], []
    def remove(mask, reason, destination):
        nonlocal frame
        part = frame.loc[mask].copy()
        part['reason'] = reason
        destination.append(part)
        frame = frame.loc[~mask].copy()
    remove(frame.duplicated(['ticker', 'ts'] + PRICES + ['volume']), 'exact duplicate after normalization', removed)
    remove(frame[RAW].isna().all(axis=1), 'all raw prices missing before fill', removed)
    remove(frame.ticker.isna() | frame.ticker.eq('') | frame.ts.isna(), 'invalid key', quarantined)
    remove(frame.duplicated(['ticker', 'ts'], keep=False), 'conflicting normalized key; no arbitrary winner', quarantined)
    # Only unambiguous reference rows with valid raw/adjusted close define factors.
    ref = ref.loc[~ref.duplicated(['ticker', 'ts'], keep=False)].copy()
    ref['factor'] = ref.adj_close / ref.close
    ref = ref.loc[np.isfinite(ref.factor) & ref.factor.gt(0) & ref.close.gt(0)]
    frame = frame.merge(ref[['ticker', 'ts', 'factor']], on=['ticker', 'ts'], how='left', validate='one_to_one')
    # Sequential filling uses only previously validated rows, never another ticker
    # or a row quarantined for an invalid range/unknown adjustment convention.
    accepted, rejected = [], []
    for ticker, group in frame.groupby('ticker', sort=True):
        previous = None
        for _, source in group.sort_values('ts').iterrows():
            row = source.copy()
            flags = []
            for col in RAW:
                if pd.isna(row[col]) and previous is not None:
                    row[col] = previous[col]
                    flags.append(f'ffill_{col}')
            if pd.isna(row.volume):
                row.volume = 0.0
                flags.append('unknown_volume_filled_zero')
            raw_valid = (np.isfinite(row[RAW].astype(float)).all() and (row[RAW].astype(float) > 0).all()
                         and row.low <= row.open <= row.high and row.low <= row.close <= row.high)
            if raw_valid and pd.notna(row.factor):
                expected = row.close * row.factor
                if pd.isna(row.adj_close) or not np.isclose(row.adj_close, expected, rtol=1e-6, atol=1e-8):
                    row.adj_close = expected
                    flags.append('adj_close_from_reference_date_factor')
            needs_factor = any(flag.startswith('ffill_') for flag in flags) or pd.isna(row.adj_close)
            volume_valid = pd.notna(row.volume) and np.isfinite(row.volume) and row.volume >= 0
            adjusted_valid = pd.notna(row.adj_close) and np.isfinite(row.adj_close) and row.adj_close > 0
            row.repair_flags = ';'.join(flags)
            if not raw_valid or not adjusted_valid or not volume_valid or (needs_factor and pd.isna(row.factor)):
                row['reason'] = 'unresolved raw bounds/price, volume, or date-specific adjustment factor'
                rejected.append(row)
            else:
                accepted.append(row)
                previous = row
    cleaned = pd.DataFrame(accepted, columns=frame.columns).drop(columns='factor')
    if rejected:
        quarantined.append(pd.DataFrame(rejected).drop(columns='factor'))
    after = diagnose(cleaned, ref)
    extreme_keys = set(zip(after.loc[after.issue_type.eq('extreme_return'), 'ticker'],
                           after.loc[after.issue_type.eq('extreme_return'), 'issue_date']))
    cleaned['extreme_return_review'] = [(t, d) in extreme_keys for t, d in zip(cleaned.ticker, cleaned.ts)]
    cleaned['adjustment_factor'] = cleaned.adj_close / cleaned.close
    assert valid_prices(cleaned).all()
    assert not cleaned.duplicated(['ticker', 'ts']).any()
    assert (np.isfinite(cleaned.volume) & cleaned.volume.ge(0)).all()
    removed = pd.concat(removed, ignore_index=True)
    quarantine = pd.concat(quarantined, ignore_index=True)
    assert len(raw) == len(cleaned) + len(removed) + len(quarantine)
    return cleaned, before, after, removed, quarantine


def save_cleaned(cleaned, db_path):
    """Write only the derived table, with ISO dates; leave ohlc_hw unchanged."""
    output = cleaned.copy()
    output['ts'] = output.ts.dt.strftime('%Y-%m-%d')
    with sqlite3.connect(db_path) as conn:
        output.to_sql('ohlc_hw_cleaned', conn, if_exists='replace', index=False)
        conn.execute('CREATE UNIQUE INDEX cleaned_key ON ohlc_hw_cleaned(ticker, ts)')


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily log returns from adjusted prices without gap filling."""
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate annualized sample volatility; preserve the rolling warm-up."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)


def calculate_max_drawdown(prices: pd.Series) -> pd.Series:
    """Return recoverable per-date drawdown from the running price peak."""
    return prices / prices.cummax() - 1


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = .02) -> float:
    """Course approximation using annualized mean log return and sample std."""
    returns = returns.dropna()
    std = returns.std(ddof=1)
    return float((returns.mean() * 252 - risk_free_rate) / (std * np.sqrt(252))) if len(returns) >= 2 and std > 0 else np.nan


def load_stock(db_path, ticker, start_date, end_date):
    """Validate inputs and query cleaned observations with bound parameters."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start > end or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError('Supply a ticker and valid start_date <= end_date')
    if not Path(db_path).is_file():
        raise FileNotFoundError(db_path)
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query('SELECT * FROM ohlc_hw_cleaned WHERE ticker = ? AND ts BETWEEN ? AND ? ORDER BY ts',
                                  conn, params=(ticker.strip().upper(), start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    if len(frame) < 2:
        raise ValueError(f'{ticker}: need at least two prices; found {len(frame)}')
    frame['ts'] = pd.to_datetime(frame.ts)
    if not valid_prices(frame).all() or frame.ts.duplicated().any():
        raise ValueError('Cleaned data failed price/key validation')
    return frame.set_index('ts')


def metrics(frame):
    """Return named numeric outputs (fractions, not formatted percentages)."""
    prices = frame.adj_close
    returns = calculate_returns(prices).dropna()
    vol = calculate_volatility(calculate_returns(prices))
    drawdown = calculate_max_drawdown(prices)
    return {'total_return': float(prices.iloc[-1] / prices.iloc[0] - 1),
            'annualized_return': float(np.expm1(returns.sum() * 252 / len(returns))),
            'annualized_volatility': float(returns.std(ddof=1) * np.sqrt(252)),
            'sharpe_ratio': calculate_sharpe_ratio(returns), 'max_drawdown': float(drawdown.min()),
            'best_day': float(returns.max()), 'worst_day': float(returns.min()),
            'positive_days': float(returns.gt(0).mean()),
            'average_up_day': float(returns[returns > 0].mean()),
            'average_down_day': float(returns[returns < 0].mean()),
            'avg_daily_return': float(returns.mean()), 'avg_volatility': float(vol.mean())}


def plot_report(frame, ticker, start_date, end_date):
    """Draw a 16x12 risk dashboard, including retrospective volatility colors."""
    prices = frame.adj_close
    returns = calculate_returns(prices)
    vol = calculate_volatility(returns)
    dd = calculate_max_drawdown(prices)
    mean_vol = vol.mean()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    from matplotlib.dates import date2num
    x = date2num(frame.index.to_pydatetime())
    points = np.column_stack([x, prices.to_numpy()])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = ['gray' if pd.isna(v) else ('#d64550' if v > mean_vol else '#21865a') for v in vol.iloc[1:]]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.6))
    ax.autoscale_view()
    ax.xaxis_date()
    ax.plot(frame.index, prices.rolling(50, min_periods=50).mean(), color='#253b69', label='50-day MA')
    ax.legend()
    ax.set(title=f'{ticker} Price and 50-day MA', ylabel='Adjusted price ($)')
    if len(frame) < 50:
        ax.text(.02, .95, '50-day MA unavailable: fewer than 50 prices', transform=ax.transAxes, va='top')
    ax = axes[0, 1]
    ax.bar(frame.index, returns, color=np.where(returns >= 0, '#21865a', '#d64550'), width=1)
    ax.axhline(0, color='black', lw=.7)
    ax.set(title=f'{ticker} Daily Returns', ylabel='Daily log return')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 0]
    ax.plot(frame.index, vol, color='#253b69')
    if pd.notna(mean_vol):
        ax.axhline(mean_vol, color='black', ls='--', label='Mean rolling volatility')
        ax.fill_between(frame.index, vol, mean_vol, where=vol > mean_vol, color='#d64550', alpha=.25)
        ax.fill_between(frame.index, vol, mean_vol, where=vol < mean_vol, color='#21865a', alpha=.25)
        ax.legend()
    ax.set(title=f'{ticker} Rolling Volatility (20-day)', ylabel='Annualized volatility')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 1]
    for scale in np.linspace(.2, 1, 5):
        ax.fill_between(frame.index, dd * scale, 0, color='#db4261', alpha=.13)
    ax.plot(frame.index, dd, color='#a92445', lw=1)
    ax.scatter([dd.idxmin()], [dd.min()], color='#7b1230', zorder=3, label=f'Max drawdown: {dd.min():.2%}')
    ax.legend()
    ax.set(title=f'{ticker} Maximum Drawdown', ylabel='Drawdown')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel('Observation date')
        ax.tick_params(axis='x', rotation=25)
    fig.suptitle(f'{ticker} Risk Analysis Report - {start_date} to {end_date}', fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig
