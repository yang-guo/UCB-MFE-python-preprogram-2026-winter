"""Build weekly signals, choose a model in development, and run a locked test."""
import argparse
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import CAPITAL, HOLD, hand_checks, metrics, simulate

FEATURES = ['ret21', 'ret63', 'ret126', 'ret252skip21', 'vol63',
            'op_margin', 'cashflow_assets', 'debt_assets']
DEV_END = pd.Timestamp('2015-12-31')
VALID_START = pd.Timestamp('2014-01-01')
TEST_START = pd.Timestamp('2016-04-01')
TEST_END = pd.Timestamp('2026-09-30')
MODELS = ('ridge', 'tree', 'momentum')


def load(database, end):
    with duckdb.connect(str(database), read_only=True) as db:
        sessions = pd.DatetimeIndex(db.execute(
            'select date from calendar where date <= ? order by date', [end]
        ).df().date).as_unit('ns')
        prices = db.execute('''
            select p.date, p.symbol, p.adj_close from prices p
            inner join calendar c using (date)
            where p.date <= ? and p.adj_close > 0 and isfinite(p.adj_close)
        ''', [end]).df().pivot(index='date', columns='symbol', values='adj_close')
        prices = prices.reindex(sessions)
        filings = db.execute('''
            select symbol, fiscal_date, filing_date, operating_income, revenue,
                   operating_cash_flow, total_assets, total_debt
            from fundamentals
            where period = 'quarter' and filing_date > fiscal_date
              and filing_date < ?
            order by symbol, filing_date, fiscal_date
        ''', [end]).df()
        members = db.execute('''
            select quarter_start, selection_date, effective_start, symbol
            from universe_membership where effective_start <= ?
        ''', [end]).df()
        benchmark = db.execute('''
            select date, adj_close from benchmark
            where symbol = '^SP500TR' and date <= ? order by date
        ''', [end]).df().set_index('date').adj_close
    return prices, filings, members, benchmark


def financials(filings, signals):
    """Use the latest fiscal period known by each signal close."""
    columns = FEATURES[-3:]
    if filings is None:
        return pd.DataFrame(np.nan, index=signals, columns=columns)
    filings = filings.copy()
    filings['available'] = filings.filing_date + pd.Timedelta(days=1)
    # A later filing of an older fiscal quarter cannot displace a newer quarter.
    filings = filings[filings.fiscal_date.eq(filings.fiscal_date.cummax())]
    dated = pd.merge_asof(pd.DataFrame({'signal': signals}), filings,
                          left_on='signal', right_on='available')
    assert (dated.available.isna() | dated.available.le(dated.signal)).all()
    revenue = dated.revenue.where(dated.revenue > 0)
    assets = dated.total_assets.where(dated.total_assets > 0)
    ratios = pd.DataFrame(dict(op_margin=dated.operating_income / revenue,
                              cashflow_assets=dated.operating_cash_flow / assets,
                              debt_assets=dated.total_debt / assets))
    ratios.loc[(dated.signal - dated.fiscal_date).dt.days > 550] = np.nan
    return ratios.set_axis(signals)


def build_panel(prices, filings, members):
    dates = prices.index
    weeks = dates.to_period('W-SUN')
    first_of_week = np.flatnonzero(~weeks.duplicated())
    entry_idx = first_of_week[(first_of_week > 0) &
                              (dates[first_of_week] >= '2010-01-01')]
    # Include the initial test close even if it falls midweek.
    if TEST_START in dates:
        entry_idx = np.union1d(entry_idx, [dates.get_loc(TEST_START)])
    entries = dates[entry_idx]
    signals = dates[entry_idx - 1]
    quarters = entries.to_period('Q').start_time
    filing_groups = dict(tuple(filings.groupby('symbol', sort=False)))
    member_groups = dict(tuple(members.groupby('symbol', sort=False)))

    parts = []
    for symbol in prices:
        if symbol not in member_groups:
            continue
        p = prices[symbol]
        returns = p / p.shift(1) - 1
        vol = returns.rolling(63).std()
        member = member_groups[symbol].set_index('quarter_start').reindex(quarters)
        eligible = ((member.selection_date.to_numpy() <= signals.to_numpy()) &
                    (member.effective_start.to_numpy() <= entries.to_numpy()) &
                    np.isfinite(vol.iloc[entry_idx - 1].to_numpy()))
        if not eligible.any():
            continue
        idx = entry_idx[eligible]
        signal = dates[idx - 1]
        entry = dates[idx]
        part = pd.DataFrame(index=signal)
        for window in (21, 63, 126):
            part[f'ret{window}'] = (p / p.shift(window) - 1).iloc[idx - 1]
        part['ret252skip21'] = (p.shift(21) / p.shift(252) - 1).iloc[idx - 1]
        part['vol63'] = vol.iloc[idx - 1]
        part[FEATURES[-3:]] = financials(filing_groups.get(symbol), signal)
        quoted = p.dropna()
        maturity = entry + HOLD
        exit_pos = quoted.index.searchsorted(maturity)
        exits = quoted.reindex(quoted.index.append(pd.DatetimeIndex([pd.NaT])))
        exit_date = exits.index[exit_pos]
        label = exits.to_numpy()[exit_pos] / p.iloc[idx].to_numpy() - 1
        assert (exit_date.isna() | (exit_date >= maturity)).all()
        part['target'] = label
        part['exit'] = exit_date
        part['entry'] = entry
        part['symbol'] = symbol
        parts.append(part.rename_axis('signal').reset_index())
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def predict(train, rows, model):
    if model == 'momentum':
        return rows.ret63.to_numpy()
    if model == 'ridge':
        estimator = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                  StandardScaler(), Ridge(alpha=100))
    else:
        estimator = HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.06,
            min_samples_leaf=100, l2_regularization=5, early_stopping=False)
    # Limit extreme outcomes' influence on estimation. Raw labels are kept for evaluation.
    estimator.fit(train[FEATURES], train.target.clip(-1, 2))
    return estimator.predict(rows[FEATURES])


def evaluate(prices, panel, benchmark, start, end, model):
    sessions = prices.loc[start:end].index
    train = panel.loc[(panel.entry < start) & (panel.entry <= DEV_END) &
                      (panel.exit < start) & panel.target.notna()]
    assert len(train) and train.exit.max() < start
    rows = panel.loc[panel.entry.between(start, end)].copy()
    rows['score'] = predict(train, rows, model)
    rows = rows.sort_values(['entry', 'score', 'symbol'], ascending=[True, False, True])
    rankings = {day: (g.signal.iloc[0], g.symbol.tolist())
                for day, g in rows.groupby('entry')}
    daily, trades, lots = simulate(prices, sessions, rankings)
    b = benchmark.reindex(sessions)
    assert (np.isfinite(b) & (b > 0)).all()
    daily['benchmark'] = b / b.iloc[0] * CAPITAL
    paired = rows.dropna(subset=['target', 'score'])
    ic = paired.groupby('entry')[['target', 'score']].apply(
        lambda group: group.target.corr(group.score, method='spearman')
        if len(group) >= 3 else np.nan).mean()
    stats = dict(strategy=metrics(daily.nav), benchmark=metrics(daily.benchmark),
                 training_rows=len(train), latest_training_exit=str(train.exit.max().date()),
                 scored_rows=len(rows), realized_labels=len(paired),
                 mean_weekly_rank_ic=float(ic))
    return stats, daily, trades, lots


def plot(daily, output):
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    ax.plot(daily.index, daily.nav / CAPITAL - 1, label='Strategy (last-observed marks)',
            color='#135c79', linewidth=1.7)
    ax.plot(daily.index, daily.benchmark / CAPITAL - 1, label='S&P 500 Total Return',
            color='#a94829', linewidth=1.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='Cumulative return | April 2016 to September 2026',
           ylabel='Return from $1 million')
    ax.grid(axis='y', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['checks', 'develop', 'test'])
    parser.add_argument('--db', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.mode == 'checks':
        hand_checks()
        return
    if not args.db or not args.db.is_file():
        parser.error('Supply an existing course database with --db.')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / 'development.json'
    if args.mode == 'test':
        config = json.loads(lock.read_text())
        if config['database'] != args.db.name:
            parser.error('Use the same database for development and test.')
        winner = config['selected']
        end = TEST_END
    else:
        end = DEV_END
    prices, filings, members, benchmark = load(args.db, end)
    assert prices.index[-1] == end, 'Database has a different end date.'
    panel = build_panel(prices, filings, members)
    if args.mode == 'develop':
        results = {model: evaluate(prices, panel, benchmark, VALID_START, end,
                                   model)[0] for model in MODELS}
        winner = max(MODELS, key=lambda model: results[model]['strategy']['sharpe'])
        result = dict(database=args.db.name, selected=winner, validation=results)
        lock.write_text(json.dumps(result, indent=2) + '\n')
    else:
        result, daily, trades, lots = evaluate(prices, panel, benchmark, TEST_START,
                                               end, winner)
        buys = trades.loc[trades.side.eq('buy')]
        sells = trades.loc[trades.side.eq('sell')]
        sold = sells.merge(panel[['symbol', 'entry', 'exit', 'target']],
                           on=['symbol', 'entry'], validate='one_to_one')
        assert len(sold) == len(sells) and sold.date.eq(sold.exit).all()
        entry_quotes = buys.set_index(['symbol', 'entry']).price
        observed_returns = sold.price / entry_quotes.reindex(
            pd.MultiIndex.from_frame(sold[['symbol', 'entry']])
        ).to_numpy() - 1
        np.testing.assert_allclose(observed_returns, sold.target, atol=1e-12)
        result['accounting'] = dict(buys=len(buys), sales=len(sells),
                                    unfilled_buys=int(trades.side.eq('unfilled_buy').sum()),
                                    fees=float(trades.fee.sum()),
                                    end_cash=float(daily.cash.iloc[-1]),
                                    end_positions=int(daily.positions.iloc[-1]),
                                    end_overdue=int(daily.overdue.iloc[-1]),
                                    max_daily_stale_value=float(daily.stale_value.max()),
                                    days_with_stale=int(daily.stale.gt(0).sum()),
                                    sales_after_first_maturity_session=int(
                                        (prices.index.searchsorted(sells.maturity) <
                                         prices.index.get_indexer(sells.date)).sum()))
        if len(lots):
            stale = lots.last_quote < end
            result['end_stale_value'] = float(lots.loc[stale, 'value'].sum())
            result['end_value_zero_for_stale'] = float(
                daily.nav.iloc[-1] - result['end_stale_value'])
        daily.to_csv(args.out / 'daily.csv')
        trades.to_csv(args.out / 'orders_and_fills.csv', index=False)
        lots.to_csv(args.out / 'open_lots.csv', index=False)
        plot(daily, args.out / 'cumulative_returns.png')
        (args.out / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
