"""Independent round-two analytics audit; no production or shared-test edits."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import os
import warnings
import numpy as np
import pandas as pd

hw4 = Path.cwd()
base = hw4 / 'David_MartinsHuber' if (hw4/'David_MartinsHuber').is_dir() else hw4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting import BacktestAnalytics


def fixture(strategy, benchmark, dates=None):
    rows = len(strategy) + 1
    if dates is None:
        dates = pd.bdate_range('2018-01-01', periods=rows)
    capital = 1_000_000.
    nav = np.empty((rows, 2))
    nav[0] = capital
    # Separate scalar compounding oracle rather than library NAV operations.
    for i, (s,b) in enumerate(zip(strategy, benchmark), start=1):
        nav[i,0] = nav[i-1,0] * (1 + s)
        nav[i,1] = nav[i-1,1] * (1 + b)
    return SimpleNamespace(history=pd.DataFrame(nav, columns=['nav','benchmark_nav'], index=dates))


def q(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1) * probability
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return ordered[lo] + (position-lo)*(ordered[hi]-ordered[lo])


def must_reject(fn, name):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f'{name}: nonfinite derived arithmetic was not rejected')

rng=np.random.default_rng(831829)
max_alpha_error=0.
for trial in range(300):
    n=int(rng.integers(5,300))
    market=rng.normal(.0004,.015,n)
    strat=.0002 + .8*market + rng.normal(0,.009,n)
    rf=float(rng.choice([-.05,0,.02,.2]))
    freq=int(rng.choice([252,260,365]))
    result=fixture(strat,market)
    # Use the actual observed compounded NAVs to account for representation.
    sr=np.array([result.history.nav.iloc[i]/result.history.nav.iloc[i-1]-1 for i in range(1,n+1)])
    mr=np.array([result.history.benchmark_nav.iloc[i]/result.history.benchmark_nav.iloc[i-1]-1 for i in range(1,n+1)])
    daily_rf=math.expm1(math.log1p(rf)/freq)
    alpha,beta=np.linalg.lstsq(np.column_stack([np.ones(n),mr-daily_rf]),sr-daily_rf,rcond=None)[0]
    analysis=BacktestAnalytics(result,annual_rf=rf,sessions_per_year=freq)
    report=analysis.summary()
    np.testing.assert_allclose(report.loc['beta','Strategy'],beta,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(report.loc['annualized_alpha','Strategy'],alpha*freq,atol=1e-12,rtol=1e-10)
    max_alpha_error=max(max_alpha_error,abs(report.loc['annualized_alpha','Strategy']-alpha*freq))
    for name,r,values in [('Strategy',sr,result.history.nav.tolist()),('S&P 500 total return',mr,result.history.benchmark_nav.tolist())]:
        mean=sum(r)/n
        sd=math.sqrt(sum((x-mean)**2 for x in r)/(n-1))
        peak=values[0]
        worst=0.
        for v in values:
            peak=max(peak,v)
            worst=min(worst,v/peak-1)
        years=(result.history.index[-1]-result.history.index[0]).total_seconds()/31557600
        np.testing.assert_allclose(report.loc['Sharpe',name],math.sqrt(freq)*(mean-daily_rf)/sd,atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(report.loc['max_drawdown',name],worst,atol=1e-12)
        np.testing.assert_allclose(report.loc['CAGR',name],math.expm1(math.log(values[-1]/values[0])/years),atol=1e-12,rtol=1e-10)
        for prob,key in [(.05,'VaR_95_daily'),(.01,'VaR_99_daily')]:
            np.testing.assert_allclose(report.loc[key,name],-q(r,prob),atol=1e-12)
    assert report.loc['daily_observations','Strategy']==n
print('PASS: 300 independent OLS/manual quantile/drawdown/Sharpe/log-CAGR cases; maximum annual-alpha error',max_alpha_error)

# Known degenerate and short histories preserve only intentionally undefined metrics.
a=BacktestAnalytics(fixture([0,0,0],[0,0,0]))
s=a.summary()
assert np.isnan(s.loc['Sharpe','Strategy']) and np.isnan(s.loc['beta','Strategy'])
assert s.loc['annualized_volatility','Strategy']==0
assert s.loc['VaR_95_daily','Strategy']==0
one=BacktestAnalytics(fixture([.01],[.02])).summary()
assert np.isnan(one.loc['beta','Strategy']) and np.isnan(one.loc['Sharpe','Strategy'])
np.testing.assert_allclose(one.loc['VaR_99_daily','Strategy'],-.01,atol=1e-12)

# Input NAV stays untouched; errors never silently remove observations.
for kind in ['nan','zero','negative','inf','unsorted','duplicate']:
    r=fixture([.01,.02,.03],[.03,.02,.01])
    if kind=='nan': r.history.iloc[1,0]=np.nan
    elif kind=='zero': r.history.iloc[1,0]=0
    elif kind=='negative': r.history.iloc[1,0]=-1
    elif kind=='inf': r.history.iloc[1,0]=np.inf
    elif kind=='unsorted': r.history=r.history.iloc[::-1]
    else: r.history.index=[r.history.index[0]]*len(r.history)
    must_reject(lambda:BacktestAnalytics(r),kind)

# Original overflow defect and new fixed metric paths.
dates=pd.bdate_range('2020-01-01',periods=3)
r=SimpleNamespace(history=pd.DataFrame({'nav':[1e-300,1e300,1e300],'benchmark_nav':[100.,101.,102.]},index=dates))
must_reject(lambda:BacktestAnalytics(r),'daily-return overflow')
r.history['nav']=[1e-200,1.,1e200]
a=BacktestAnalytics(r)
with np.errstate(all='ignore'):
    must_reject(a.summary,'summary overflow')
    must_reject(a.comparison,'total-return comparison overflow')
    must_reject(a.cumulative_returns,'cumulative overflow')
r.history['nav']=[100.,1e100,1e200]
a=BacktestAnalytics(r)
with np.errstate(all='ignore'):
    must_reject(a.summary,'annualization overflow')
print('PASS: original derived-overflow correction, short samples, and malformed input cases')

# Compare displayed/report quantities using a small independently hand-computed case.
r=fixture([.1,-.1],[0,.02])
a=BacktestAnalytics(r,annual_rf=0)
c=a.comparison()
np.testing.assert_allclose(c.final_value_difference,-30000.,atol=1e-8)
np.testing.assert_allclose(c.total_return_difference_pp,-3.,atol=1e-10)
assert not c.beat_benchmark
os.environ.setdefault('MPLCONFIGDIR','/tmp/hw4-round2-analytics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ax=a.plot_cumulative_returns()
np.testing.assert_allclose(ax.lines[0].get_ydata(),[0,.1,-.01],atol=1e-12)
np.testing.assert_allclose(ax.lines[1].get_ydata(),[0,0,.02],atol=1e-12)
plt.close(ax.figure)
print('PASS: benchmark comparison and plotted curve arrays')

# New independently found arithmetic boundary: validation must include pp scaling.
r=SimpleNamespace(history=pd.DataFrame({'nav':[1.,1e307], 'benchmark_nav':[1.,1.]},index=pd.date_range('2020-01-01',periods=2)))
a=BacktestAnalytics(r)
with np.errstate(all='ignore'):
    must_reject(a.comparison,'percentage-point comparison overflow')
print('PASS: every derived comparison field is finite or explicitly rejected')
