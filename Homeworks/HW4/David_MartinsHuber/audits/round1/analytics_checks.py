import sys, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting import BacktestAnalytics

def quantile(values, probability):
    ordered=sorted(values)
    pos=(len(ordered)-1)*probability
    lo=math.floor(pos); hi=math.ceil(pos)
    return ordered[lo]+(pos-lo)*(ordered[hi]-ordered[lo])

def drawdown(values):
    peak=values[0]; worst=0
    for v in values:
        peak=max(peak,v)
        worst=min(worst, v/peak-1)
    return worst

def make(s,b,dates=None):
    dates=pd.bdate_range('2015-01-01',periods=len(s)+1) if dates is None else dates
    values=np.vstack(([100.,100.],100*np.cumprod(1+np.column_stack((s,b)),axis=0)))
    return SimpleNamespace(history=pd.DataFrame(values,index=dates,columns=['nav','benchmark_nav']))

rng=np.random.default_rng(8462)
max_error=0
for case in range(250):
    n=int(rng.integers(3,400))
    b=rng.normal(.0005,.012,n)
    s=.0001+1.4*b+rng.normal(0,.008,n)
    annual_rf=float(rng.choice([-.05,0,.02,.12]))
    frequency=int(rng.choice([252,260,365]))
    result=make(s,b)
    a=BacktestAnalytics(result,annual_rf=annual_rf,sessions_per_year=frequency)
    actual=a.summary()
    reconstructed_s=result.history.nav.to_numpy()[1:]/result.history.nav.to_numpy()[:-1]-1
    reconstructed_b=result.history.benchmark_nav.to_numpy()[1:]/result.history.benchmark_nav.to_numpy()[:-1]-1
    daily_rf=math.expm1(math.log1p(annual_rf)/frequency)
    intercept,slope=np.linalg.lstsq(np.column_stack((np.ones(n),reconstructed_b-daily_rf)),reconstructed_s-daily_rf,rcond=None)[0]
    np.testing.assert_allclose(actual.loc['beta','Strategy'],slope,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(actual.loc['annualized_alpha','Strategy'],intercept*frequency,rtol=1e-10,atol=1e-12)
    for key,returns in [('Strategy',reconstructed_s),('S&P 500 total return',reconstructed_b)]:
        nav=result.history.iloc[:,0 if key=='Strategy' else 1].to_numpy()
        sd=math.sqrt(sum((x-sum(returns)/n)**2 for x in returns)/(n-1))
        np.testing.assert_allclose(actual.loc['Sharpe',key],math.sqrt(frequency)*(sum(returns)/n-daily_rf)/sd,rtol=1e-10,atol=1e-12)
        for prob,label in [(.05,'VaR_95_daily'),(.01,'VaR_99_daily')]:
            np.testing.assert_allclose(actual.loc[label,key],-quantile(returns,prob),atol=1e-12)
        np.testing.assert_allclose(actual.loc['max_drawdown',key],drawdown(nav),atol=1e-12)
        years=(result.history.index[-1]-result.history.index[0]).total_seconds()/31557600
        growth=math.expm1(math.log(nav[-1]/nav[0])/years)
        np.testing.assert_allclose(actual.loc['CAGR',key],growth,rtol=1e-10,atol=1e-12)
    max_error=max(max_error,abs(actual.loc['annualized_alpha','Strategy']-intercept*frequency))
print('250 independently calculated OLS/quantile/drawdown/Sharpe/CAGR cases passed; max alpha error',max_error)

# Sparse history: the class documents it cannot infer absent market sessions.
short=make([.01],[.02])
a=BacktestAnalytics(short)
assert np.isnan(a.summary().loc['beta','Strategy'])
assert np.isnan(a.summary().loc['Sharpe','Strategy'])
assert abs(a.summary().loc['VaR_95_daily','Strategy']+.01)<1e-12

# Exact long-horizon growth, avoiding an implementation-provided CAGR oracle.
r=SimpleNamespace(history=pd.DataFrame({'nav':[100.,121.],'benchmark_nav':[100.,100.]},index=pd.to_datetime(['2020-01-01','2022-01-01'])))
expected=1.21**(365.25/731)-1
assert abs(BacktestAnalytics(r).summary().loc['CAGR','Strategy']-expected)<1e-12

# Input errors must not be silently skipped; original frames remain unchanged.
for kind in ['missing','zero','negative','inf','duplicate','unsorted']:
    r=make([.01,.02,.03],[.02,.01,.04])
    if kind=='missing': r.history.iloc[1,0]=np.nan
    elif kind=='zero': r.history.iloc[1,0]=0
    elif kind=='negative': r.history.iloc[1,0]=-1
    elif kind=='inf': r.history.iloc[1,0]=np.inf
    elif kind=='duplicate': r.history.index=[r.history.index[0]]*len(r.history)
    elif kind=='unsorted': r.history=r.history.iloc[::-1]
    try: BacktestAnalytics(r)
    except ValueError: pass
    else: raise AssertionError(kind+' accepted')
print('Short-sample, calendar-time, and malformed-input checks passed')

# Accepted finite NAV values can overflow derived daily returns.
r=SimpleNamespace(history=pd.DataFrame({'nav':[1e-300,1e300,1e300],'benchmark_nav':[100.,101.,102.]},index=pd.bdate_range('2020-01-01',periods=3)))
with np.errstate(all='ignore'):
    a=BacktestAnalytics(r)
    print('Extreme finite NAV derived returns:',a.returns.nav.tolist())
    print('Extreme finite NAV summary:',a.summary()['Strategy'].to_dict())
