"""Independent round-two shared-clock and portfolio audit. No production edits."""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base))
from backtesting import (Strategy,PortfolioOptimizer,ShadowPortfolioBacktester,
    MinimumVarianceOptimizer,PortfolioStrategy,EqualWeightOptimizer,DecisionContext,Portfolio,AllocationContext,Phase,ShadowSpec)

rng=np.random.default_rng(71302)
error=0
for _ in range(200):
    samples=rng.normal(size=(80,2)) @ np.array([[.014,-.007],[.005,.022]])
    cap=rng.uniform(.5,1); shrink=rng.uniform()
    c=np.cov(samples,rowvar=False,ddof=1)
    cross=c[0,1]*(1-shrink)
    expected=np.clip((c[1,1]-cross)/(c[0,0]+c[1,1]-2*cross),1-cap,cap)
    dates=pd.bdate_range('2020-01-01',periods=len(samples))
    context=AllocationContext(dates[-1],dates[-1]+pd.Timedelta(days=1),Portfolio(100,100,100,{}),pd.DataFrame(samples,index=dates,columns=['a','b']))
    actual=MinimumVarianceOptimizer(80,10,shrink,cap).weights(context)
    error=max(error,abs(actual['a']-expected)); assert abs(actual['a']-expected)<1e-6
print('PASS 200 independent capped covariance-shrinkage scalar optimizer oracles; max error',error)

class Sensitive(Strategy):
    def __init__(self,name,symbol): super().__init__(name); self.symbol=symbol
    def propose_buys(self,c):
        assert c.prices.index.max()<=c.signal_date
        frac=.3 if c.prices[self.symbol].dropna().iloc[-1]>100 else .1
        budget=c.portfolio.available_cash*frac
        return {self.symbol:budget} if budget>1e-8 else {}
class FollowRecent(PortfolioOptimizer):
    def weights(self,c):
        assert c.shadow_returns.empty or c.shadow_returns.index.max()<=c.signal_date
        if c.shadow_returns.empty: return {'a':.5,'b':.5}
        winner=c.shadow_returns.tail(3).sum().idxmax()
        return {name:float(name==winner) for name in ['a','b']}

dates=pd.bdate_range('2020-01-02',periods=120)
x=np.arange(len(dates))
wide=pd.DataFrame({'A':100*np.exp(.001*x+.03*np.sin(x/5)),
                   'B':100*np.exp(-.0001*x+.02*np.cos(x/7))},index=dates)
# Missing entry/mark observations stay missing; normal gap is only one session.
wide.loc[dates[[6,17,61]],'A']=np.nan
wide.loc[dates[[10,47,80]],'B']=np.nan
prices=wide.rename_axis('date').reset_index().melt('date',var_name='symbol',value_name='adj_close').dropna()
bench=pd.DataFrame({'date':dates,'adj_close':100*np.exp(.0002*x)})
membership=pd.DataFrame([dict(symbol=s,quarter_start=q,selection_date=q-pd.Timedelta(days=1))
                        for q in pd.to_datetime(['2020-01-01','2020-04-01']) for s in ['A','B']])
fee=.0007
start_index=20
runner=ShadowPortfolioBacktester(prices,bench,membership,[Sensitive('a','A'),Sensitive('b','B')],FollowRecent(),initial_cash=1000,shadow_initial_cash=1000,fee_bps=7)
actual=runner.run(dates[start_index],dates[-1],shadow_start=dates[1])

# Completely separate accounts: lists of (symbol, units, due) and scalar cash.
# Expected NAV never reads actual fills, account methods, or actual shadow NAV.
accounts={name:{'cash':1000.,'lots':[],'marks':{},'nav':[1000.]} for name in ['a','b','main']}
weights=[]
def complete(account,day,orders):
    observed=wide.loc[day].dropna().to_dict(); account['marks'].update(observed)
    kept=[]
    for s,q,due in account['lots']:
        if day>=due and s in observed: account['cash']+=q*observed[s]*(1-fee)
        else: kept.append((s,q,due))
    account['lots']=kept
    for s,budget in orders.items():
        if s not in observed:continue
        q=budget/(1+fee)/observed[s]
        account['cash']-=budget
        account['lots'].append((s,q,day+pd.Timedelta(days=90)))
    account['nav'].append(account['cash']+sum(q*account['marks'][s] for s,q,due in account['lots']))

for i,day in enumerate(dates[1:],start=1):
    proposal={}
    for name,symbol in [('a','A'),('b','B')]:
        signal_price=wide.loc[:dates[i-1],symbol].dropna().iloc[-1]
        proposal[name]={symbol:accounts[name]['cash']*(.3 if signal_price>100 else .1)}
    main_orders={}
    if i>=start_index:
        if len(accounts['a']['nav'])<=1:w={'a':.5,'b':.5}
        else:
            recent={k:sum((np.array(v['nav'])[1:]/np.array(v['nav'])[:-1]-1)[-3:]) for k,v in accounts.items() if k!='main'}
            winner=max(['a','b'],key=lambda k:recent[k]);w={k:float(k==winner) for k in ['a','b']}
        for k in ['a','b']:
            for s,budget in proposal[k].items():
                main_orders[s]=main_orders.get(s,0)+accounts['main']['nav'][-1]*w[k]*(budget/accounts[k]['nav'][-1])
        total=sum(main_orders.values());factor=min(1,accounts['main']['cash']/total) if total else 1
        main_orders={s:v*factor for s,v in main_orders.items() if v*factor>0}
        weights.append(w)
    for k in ['a','b']:complete(accounts[k],day,proposal[k])
    if i>=start_index:complete(accounts['main'],day,main_orders)

for name in ['a','b']:
    np.testing.assert_allclose(actual.shadows[name].history.nav,accounts[name]['nav'],rtol=1e-11,atol=1e-8)
np.testing.assert_allclose(actual.portfolio.history.nav,accounts['main']['nav'],rtol=1e-11,atol=1e-8)
pivot=actual.allocations.pivot(index='signal_date',columns='component_id',values='weight')
np.testing.assert_allclose(pivot[['a','b']],pd.DataFrame(weights)[['a','b']])
print('PASS independent shadow/main cash, fees, 90-day lots, missing quotes, price-sensitive decisions, return-sensitive weights over 119 sessions')

# Disturb future data in a way the sensitive decision functions would notice.
cutoff=dates[55]
p2=prices.copy();p2.loc[p2.date.gt(cutoff),'adj_close']*=10
b2=bench.copy();b2.loc[b2.date.gt(cutoff),'adj_close']*=2
second=ShadowPortfolioBacktester(p2,b2,membership,[Sensitive('a','A'),Sensitive('b','B')],FollowRecent(),initial_cash=1000,shadow_initial_cash=1000,fee_bps=7).run(dates[start_index],dates[-1],shadow_start=dates[1])
for name in ['a','b']:
    pd.testing.assert_frame_equal(actual.shadows[name].history.loc[:cutoff],second.shadows[name].history.loc[:cutoff])
pd.testing.assert_frame_equal(actual.portfolio.history.loc[:cutoff],second.portfolio.history.loc[:cutoff])
for attr in ['allocations','contributions']:
    a,b=getattr(actual,attr),getattr(second,attr)
    pd.testing.assert_frame_equal(a.loc[a.signal_date.le(cutoff)].reset_index(drop=True),b.loc[b.signal_date.le(cutoff)].reset_index(drop=True))
print('PASS sensitive future-price and benchmark perturbation invariance')

# Coverage defaults and deliberate partial runs have the same shared effective end.
def make():return ShadowPortfolioBacktester(prices,bench,membership,[Sensitive('a','A'),Sensitive('b','B')])
e=make()
try:e.run(dates[5],dates[-1]+pd.Timedelta(days=365),shadow_start=dates[1])
except ValueError as exc: assert 'coverage' in str(exc)
else:raise AssertionError('incomplete requested coverage silently accepted')
assert e.phase==Phase.FAILED
with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter('always')
    partial=make().run(dates[5],dates[-1]+pd.Timedelta(days=365),shadow_start=dates[1],allow_partial=True)
assert len(notices)==3
for r in [partial.portfolio,*partial.shadows.values()]:
    assert r.coverage.range_clipped
    assert r.coverage.effective_end==dates[-1]
    assert r.coverage.requested_end==dates[-1]+pd.Timedelta(days=365)
print('PASS strict coordinated coverage and explicit partial shared coverage metadata/warnings')

# Independent chronological training oracle with different per-component windows.
class MonthlyFitted(Sensitive):
    def __init__(self,name,window):
        super().__init__(name,'A');self.window=window;self.fit_month=None;self.observed=[]
    def needs_refit(self,c): return c.signal_date.to_period('M')!=self.fit_month
    def refit(self,data,c):
        keep=data.audit.date.ge(c.signal_date-pd.Timedelta(days=self.window))
        self.mean=float(data.y.loc[keep].mean())
        self.fit_month=c.signal_date.to_period('M')
        self.observed.append((c.signal_date,data.audit.loc[keep].copy(),data.y.loc[keep].copy()))
    def propose_buys(self,c):
        budget=c.portfolio.available_cash*(.2 if self.mean>0 else .05)
        return {'A':budget} if budget>1e-8 else {}
training_dates=pd.date_range('2018-01-01',dates[-1],freq='7D')
training=pd.DataFrame({'date':training_dates,'symbol':'A','available_date':training_dates,
                       'label_end':training_dates+pd.Timedelta(days=90),
                       'target':np.where(training_dates.month%2==0,.1,-.05),'feature':np.sin(np.arange(len(training_dates)))})
def fitted_run(frame):
    r=ShadowPortfolioBacktester(prices,bench,membership,
        [ShadowSpec(MonthlyFitted('short',180),frame,('feature',)),
         ShadowSpec(MonthlyFitted('long',365),frame,('feature',))])
    output=r.run(dates[20],dates[-1],shadow_start=dates[1])
    return r,output
fitted,fitted_result=fitted_run(training)
counts={}
for name,window in [('short',180),('long',365)]:
    counts[name]=[]
    for cutoff_date,received,targets in fitted._shadows[name].strategy.observed:
        expected=training.loc[training.date.ge(cutoff_date-pd.Timedelta(days=window)) & training.date.le(cutoff_date)
                              & training.available_date.le(cutoff_date) & training.label_end.lt(cutoff_date)]
        pd.testing.assert_frame_equal(received,expected[['date','symbol','available_date','label_end']])
        pd.testing.assert_series_equal(targets,expected.target)
        counts[name].append(len(received))
assert counts['short']!=counts['long']
changed_training=training.copy()
changed_training.loc[changed_training.label_end.ge(cutoff),'target']=-100
_,fitted_changed=fitted_run(changed_training)
pd.testing.assert_frame_equal(fitted_result.portfolio.history.loc[:cutoff],fitted_changed.portfolio.history.loc[:cutoff])
for name in ['short','long']:
    pd.testing.assert_frame_equal(fitted_result.shadows[name].history.loc[:cutoff],fitted_changed.shadows[name].history.loc[:cutoff])
print('PASS independent matured-label/date oracle for distinct180/365-day fitting windows and future-label invariance')

# Closed-form full deployment across huge but individually finite accounts.
a=PortfolioStrategy('parent',EqualWeightOptimizer())
d=pd.bdate_range('2020-01-01',periods=2)
a.observe(d[-1],pd.DataFrame({'a':[1e200,1e200]},index=d),{'a':{'X':1e200}},{'a':0})
c=DecisionContext(d[-1],d[-1]+pd.Timedelta(days=1),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),Portfolio(1e200,1e200,1e200,{}),0)
with np.errstate(all='ignore'):out=a.propose_buys(c)
assert out=={'X':1e200}, ('finite full deployment must remain finite and invested',out)
print('PASS finite-scale full-deployment arithmetic')

# Numeric data-boundary expectation: never expose overflowed returns to optimizer.
class ObserveNumeric(PortfolioOptimizer):
    def __init__(self):self.calls=0
    def weights(self,c):self.calls+=1;return {'s':1.0}
a=PortfolioStrategy('p',ObserveNumeric())
c=DecisionContext(d[-1],d[-1]+pd.Timedelta(days=1),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),Portfolio(100,100,100,{}),0)
try:
    with np.errstate(all='ignore'):
        a.observe(d[-1],pd.DataFrame({'s':[1e-200,1e200]},index=d),{'s':{}},{'s':0})
        a.propose_buys(c)
except ValueError:pass
else:raise AssertionError('nonfinite shadow returns were passed to optimizer')
assert a.optimizer.calls==0
assert a._cutoff is None and a._navs is None
print('PASS derived shadow-return numeric validation precedes optimizer invocation')

# Reject an invalid extension without corrupting an already-published prefix.
a=PortfolioStrategy('p',ObserveNumeric())
d3=pd.bdate_range('2020-01-01',periods=3)
prior=pd.DataFrame({'s':[100.,1e-200]},index=d3[:2])
a.observe(d3[1],prior,{'s':{}},{'s':7})
old_navs=a._navs.copy();old_returns=a._returns.copy()
bad=pd.DataFrame({'s':[100.,1e-200,1e200]},index=d3)
try:
    with np.errstate(all='ignore'):a.observe(d3[2],bad,{'s':{'X':1}},{'s':8})
except ValueError:pass
else:raise AssertionError('invalid extension accepted')
pd.testing.assert_frame_equal(a._navs,old_navs)
pd.testing.assert_frame_equal(a._returns,old_returns)
assert a._cutoff==d3[1] and a._versions=={'s':7} and a._proposals=={'s':{}}
print('PASS invalid extension preserves earlier published NAV/return/proposal/version state')
