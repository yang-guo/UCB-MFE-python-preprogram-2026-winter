"""Report building blocks: summary table and figures for a Backtest (see trading.py)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_summary_table(backtest, rf=0.02):
    """Portfolio & S&P rows: CAGR, Vol, IR, Sharpe, Max Drawdown, Annual Turnover, Annual TC."""
    name, tri, daily = backtest.portfolio, backtest.total_return, backtest.daily
    rets = tri.pct_change().dropna()
    years = (tri.index[-1] - tri.index[0]).days / 365.25
    vol = rets.std() * np.sqrt(252)
    active = rets[name] - rets['sp500']
    return pd.DataFrame({
        'CAGR':            tri.iloc[-1] ** (1 / years) - 1,
        'Vol':             vol,
        'IR':              pd.Series({name: active.mean() * 252 / (active.std() * np.sqrt(252)), 'sp500': np.nan}),
        'Sharpe':          (rets.mean() * 252 - rf) / vol,
        'Max Drawdown':    (tri / tri.cummax() - 1).min(),
        'Annual Turnover': pd.Series({name: ((daily['buy_value'] + daily['sell_value']) / (2 * daily['nav'])).sum() / years,
                                      'sp500': np.nan}),
        'Annual TC':       pd.Series({name: (daily['tc'] / daily['nav']).sum() / years, 'sp500': np.nan}),
    }).rename(index={'sp500': 'S&P 500'})


SUMMARY_FMT = {'CAGR': '{:.2%}', 'Vol': '{:.2%}', 'IR': '{:.2f}', 'Sharpe': '{:.2f}',
               'Max Drawdown': '{:.2%}', 'Annual Turnover': '{:.1%}', 'Annual TC': '{:.2%}'}
FINAL_VALUE_FMT = {'Starting Value': '${:,.0f}', 'Final Value': '${:,.0f}', 'Cash': '${:,.0f}',
                   'Open Positions': '${:,.0f}', '# Positions': '{:,.0f}', 'Total Return': '{:.2%}'}
BAD_DAY_FMT = {'date': '{:%Y-%m-%d}', 'nav_ret': '{:.2%}', 'symbol': '{}', 'contribution': '{:.2%}',
               'shares_held': '{:,.0f}', 'price_prev': '{:,.2f}', 'price': '{:,.2f}', 'stock_ret': '{:.2%}'}


def create_final_value_table(backtest):
    """Portfolio & S&P rows: starting value, final value (cash + open positions), total return.
    Starting value comes from the Portfolio; the S&P is grown from the same starting value."""
    name, tri, daily = backtest.portfolio, backtest.total_return, backtest.daily
    start_value = backtest.trader.portfolio.start_cash
    final_nav, final_cash = daily['nav'].iloc[-1], daily['cash'].iloc[-1]
    sp_final = start_value * tri['sp500'].iloc[-1] / tri['sp500'].iloc[0]
    return pd.DataFrame({
        'Starting Value': [start_value, start_value],
        'Final Value':    [final_nav, sp_final],
        'Cash':           [final_cash, np.nan],
        'Open Positions': [final_nav - final_cash, np.nan],
        '# Positions':    [(backtest.holdings > 0).sum(), np.nan],
        'Total Return':   [final_nav / start_value - 1, sp_final / start_value - 1],
    }, index=[name, 'S&P 500'])


def plot_summary_table(summary, title, fmt=SUMMARY_FMT):
    """Render a table as a matplotlib figure. fmt maps column -> format string."""
    cells = summary.apply(lambda col: col.map(lambda v: '–' if pd.isna(v) else fmt[col.name].format(v)))
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis('off')
    table = ax.table(cellText=cells.values, rowLabels=cells.index, colLabels=cells.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d0d0d0')
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f0efec')
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
    return fig


def plot_total_return(backtest):
    """Total return index: portfolio vs S&P 500 (highlighted)."""
    name, tri = backtest.portfolio, backtest.total_return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(tri.index, tri[name], lw=1.8, color='#2a78d6', label=name)
    ax.plot(tri.index, tri['sp500'], lw=2.8, color='black', label='S&P 500', zorder=5)
    ax.axhline(1, color='gray', lw=0.8, ls='--')
    ax.set_title(f'Total Return: {name} vs S&P 500', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylabel('Growth of $1')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    return fig


def plot_active_return(backtest):
    """Cumulative active return: strategy total return index / S&P total return index - 1."""
    name, tri = backtest.portfolio, backtest.total_return
    active = tri[name] / tri['sp500'] - 1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(active.index, active, lw=1.8, color='#2a78d6', label=f'{name} − S&P 500')
    ax.fill_between(active.index, active, 0, where=active >= 0, color='#2a78d6', alpha=0.12)
    ax.fill_between(active.index, active, 0, where=active < 0, color='#e34948', alpha=0.12)
    ax.axhline(0, color='black', lw=1)
    ax.set_title(f'Cumulative Active Return: {name} vs S&P 500', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylabel('Cumulative active return')
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    return fig


def plot_worst_bad_day(backtest):
    """Table of the stocks behind the single worst NAV day, with a note on the total bad-day count."""
    bad = backtest.bad_days
    n_bad = 0 if bad.empty else bad['date'].nunique()
    if bad.empty:
        fig, ax = plt.subplots(figsize=(11, 1.5))
        ax.axis('off')
        ax.set_title('Worst Bad Day', fontsize=13, fontweight='bold', loc='left')
        ax.text(0, 0.5, 'No bad days: no daily NAV move exceeded the threshold.', fontsize=11)
        return fig
    worst_date = bad.loc[bad['nav_ret'].abs().idxmax(), 'date']
    worst = bad[bad['date'] == worst_date].set_index('symbol')
    fig = plot_summary_table(worst.drop(columns=['date', 'nav_ret']),
                             f'Worst Bad Day: {worst_date:%Y-%m-%d} (NAV {worst["nav_ret"].iloc[0]:+.2%})',
                             fmt=BAD_DAY_FMT)
    fig.text(0.01, -0.05, f'Total bad days in backtest: {n_bad}', fontsize=11, style='italic')
    return fig
