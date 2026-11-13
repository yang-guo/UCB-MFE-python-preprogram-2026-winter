import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from backtesting import *
from backtesting.contracts import DecisionContext

# Analytic two-asset minimum-variance oracle, including caps, not code mirror.
rng=np.random.default_rng(65102)
max_error=0
for trial in range(200):
    R=rng.normal(size=(40,2)) @ np.array([[.01,.006],[0,.02]])
    cap=rng.uniform(.5,1)
    shrink=rng.uniform()
    cov=np.cov(R, rowvar=False, ddof=1)
    cov[0,1]*=1-shrink; cov[1,0]*=1-shrink
    # derivative of w^2 v1 + (1-w)^2 v2 + 2w(1-w)c = 0
    oracle=np.clip((cov[1,1]-cov[0,1])/(cov[0,0]+cov[1,1]-2*cov[0,1]),1-cap,cap)
    dates=pd.bdate_range('2020-01-01',periods=40)
    c=AllocationContext(dates[-1], dates[-1]+pd.Timedelta(days=1), Portfolio(100,100,100,{}),pd.DataFrame(R,index=dates,columns=['a','b']))
    w=MinimumVarianceOptimizer(40,20,shrink,cap).weights(c)
    max_error=max(max_error, abs(w['a']-oracle))
    assert abs(w['a']-oracle)<2e-6, (trial,w,oracle)
print('200 analytic capped minimum-variance cases passed; maximum weight error',max_error)

# Independently calculated contribution scaling with a cash-constrained main account.
dates=pd.bdate_range('2020-01-01',periods=2)
nav=pd.DataFrame({'s1':[100.,100.], 's2':[100.,100.]},index=dates)
a=PortfolioStrategy('parent',EqualWeightOptimizer())
a.observe(dates[-1],nav, {'s1':{'A':40},'s2':{'B':60}}, {'s1':2,'s2':5})
c=DecisionContext(dates[-1],dates[-1]+pd.Timedelta(days=1),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),Portfolio(10,10,100,{'Z':90}),0)
assert a.propose_buys(c)=={'A':4,'B':6}
assert [r['model_version'] for r in a.contributions]==[2,5]
print('Hand-calculated cash-constrained merge passed')

class SensitiveStrategy(Strategy):
    def __init__(self,name,symbol):
        super().__init__(name); self.symbol=symbol
    def propose_buys(self,c):
        assert c.prices.index.max()<=c.signal_date
        # Last released observation materially changes proposal fraction.
        fraction=.8 if c.prices[self.symbol].iloc[-1]>100 else .1
        budget=c.portfolio.available_cash*fraction
        return {self.symbol:budget} if budget>1e-8 else {}
class SensitiveOptimizer(PortfolioOptimizer):
    def weights(self,c):
        assert not len(c.shadow_returns) or c.shadow_returns.index.max()<=c.signal_date
        if not len(c.shadow_returns): return {'s1':.5,'s2':.5}
        winner=c.shadow_returns.tail(3).sum().idxmax()
        return {s:float(s==winner) for s in c.shadow_returns.columns}
dates=pd.bdate_range('2020-01-02',periods=30)
prices=pd.concat([pd.DataFrame({'date':dates,'symbol':'A','adj_close':100+np.arange(30)}),pd.DataFrame({'date':dates,'symbol':'B','adj_close':100-np.arange(30)})])
benchmark=pd.DataFrame({'date':dates,'adj_close':100.})
membership=pd.DataFrame({'symbol':['A','B'],'quarter_start':pd.to_datetime(['2020-01-01']*2),'selection_date':pd.to_datetime(['2019-12-31']*2)})
cutoff=dates[14]
def run(p,b,m):
    return ShadowPortfolioBacktester(p,b,m,[SensitiveStrategy('s1','A'),SensitiveStrategy('s2','B')],SensitiveOptimizer(),initial_cash=1000,shadow_initial_cash=1000).run(dates[1],dates[-1])
x=run(prices,benchmark,membership)
p2=prices.copy(); p2.loc[p2.date.gt(cutoff),'adj_close']*=7
b2=benchmark.copy(); b2.loc[b2.date.gt(cutoff),'adj_close']*=2
# Future calendar extension and new stock observations must not affect prefix.
extra=pd.Timestamp('2020-03-01')
p2=pd.concat([p2,pd.DataFrame({'date':[extra],'symbol':['FUTURE'],'adj_close':[20.]})],ignore_index=True)
b2=pd.concat([b2,pd.DataFrame({'date':[extra],'adj_close':[200.]})],ignore_index=True)
m2=pd.concat([membership,pd.DataFrame({'symbol':['FUTURE'],'quarter_start':pd.to_datetime(['2020-01-01']),'selection_date':[dates[20]]})],ignore_index=True)
y=run(p2,b2,m2)
for attr in ['allocations','contributions']:
    p,q=getattr(x,attr),getattr(y,attr)
    pd.testing.assert_frame_equal(p.loc[p.signal_date.le(cutoff)].reset_index(drop=True),q.loc[q.signal_date.le(cutoff)].reset_index(drop=True))
pd.testing.assert_frame_equal(x.portfolio.history.loc[:cutoff],y.portfolio.history.loc[:cutoff])
for name in x.shadows: pd.testing.assert_frame_equal(x.shadows[name].history.loc[:cutoff],y.shadows[name].history.loc[:cutoff])
print('Sensitive-strategy and sensitive-optimizer future price/calendar/membership perturbation passed')

# Public API edge: future rewrite must be rejected even when it is a different frame.
try:
    a.observe(dates[-1],nav.assign(s1=[100,200]), {'s1':{},'s2':{}},{'s1':2,'s2':5})
except ValueError: pass
else: raise AssertionError('accepted a rewritten history')
print('Independent checks passed')
