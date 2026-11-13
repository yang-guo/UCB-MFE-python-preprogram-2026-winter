import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from backtesting import Backtester, Strategy

class Planned(Strategy):
    def __init__(self, plans):
        super().__init__('independent_plan')
        self.plans = plans
        self.contexts = []
    def propose_buys(self, c):
        assert c.prices.index.max() <= c.signal_date
        assert c.membership.selection_date.le(c.signal_date).all()
        self.contexts.append((c.signal_date, c.execution_date))
        return self.plans.get(c.execution_date, {})

for seed in range(12):
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range('2019-10-01', periods=190)
    symbols=['A','B','C']
    wide=pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.02,(len(dates),3)),axis=0))*20,index=dates,columns=symbols)
    # Isolated missing observations: audit the gap policy without forcing stop.
    for j in range(3):
        for i in range(11+j, len(dates), 31): wide.iloc[i,j]=np.nan
    prices=wide.rename_axis('date').reset_index().melt('date',var_name='symbol',value_name='adj_close').dropna()
    bench=pd.DataFrame({'date':dates,'adj_close':np.linspace(100,130,len(dates))})
    quarters=pd.date_range('2019-10-01','2020-07-01',freq='QS')
    members=pd.DataFrame([dict(symbol=s,quarter_start=q,selection_date=q-pd.Timedelta(days=1)) for q in quarters for s in symbols])
    plans={d:{symbols[i%3]:75.0} for i,d in enumerate(dates[1:]) if i%7==0}
    fee=.0017
    engine=Backtester(prices,bench,members,Planned(plans),initial_cash=10000,fee_bps=fee*10000)
    actual=engine.run(dates[1],dates[-1])
    # Independent ledger oracle: do not call OMS or rely on its fill ledger.
    cash=10000.; lots=[]; marks={}; values=[cash]
    for day in dates[1:]:
        observed=wide.loc[day].dropna().to_dict()
        marks.update(observed)
        survivors=[]
        for sym,units,due in lots:
            if day>=due and sym in observed:
                cash+=units*observed[sym]*(1-fee)
            else: survivors.append((sym,units,due))
        lots=survivors
        for sym,budget in plans.get(day,{}).items():
            if sym not in observed: continue
            units=budget/(observed[sym]*(1+fee))
            cash-=budget
            lots.append((sym,units,day+pd.Timedelta(days=90)))
        values.append(cash+sum(units*marks[sym] for sym,units,due in lots))
    np.testing.assert_allclose(actual.history.nav,values,rtol=1e-12,atol=1e-8)
    assert actual.history.index[-1]==dates[-1]
print('PASS: 12 independent stochastic ledger cases, 189 sessions each; missing entries/marks, multiple lots, maturity, fees, NAV')

# Point-in-time publication and refit boundary oracle.
class FitObserver(Planned):
    def needs_refit(self,c): return True
    def refit(self,data,c):
        assert data.audit.label_end.lt(c.signal_date).all()
        assert data.audit.available_date.le(c.signal_date).all()
        assert data.audit.date.le(c.signal_date).all()
        assert c.fundamentals.filing_date.add(pd.Timedelta(days=1)).le(c.signal_date).all()
        self.seen=list(data.y)

p=prices; b=bench; m=members
panel=pd.DataFrame({'date':pd.to_datetime(['2019-06-01','2019-07-01','2019-08-01']), 'symbol':'A',
'available_date':pd.to_datetime(['2019-06-01','2019-07-01','2019-08-01']),
'label_end':pd.to_datetime(['2019-09-01','2019-10-01','2020-12-01']),'target':[1.,2.,999.], 'feature':[0.,1.,999.]})
f=pd.DataFrame({'symbol':'A','fiscal_date':pd.to_datetime(['2019-03-01','2019-06-01']), 'filing_date':pd.to_datetime(['2019-04-01','2020-12-01'])})
e=Backtester(p,b,m,FitObserver({}),fundamentals=f,training_panel=panel,feature_columns=['feature'])
e.begin(dates[1],dates[3]); e.prepare()
assert e.strategy.seen==[1.0], e.strategy.seen # label_end == initial signal date is deliberately unavailable.
e.advance(); e.prepare(); assert e.strategy.seen==[1.,2.]
print('PASS: strict label-boundary and future filing exclusion')

# Public incremental API invalid-call behavior.
for method in ['prepare','advance','finish']:
    e=Backtester(p,b,m,Planned({}))
    try: getattr(e,method)()
    except Exception as exc: print('LIFECYCLE',method,type(exc).__name__,str(exc))
# Caller asks for coverage beyond source; current behavior silently intersects.
e=Backtester(p,b,m,Planned({}))
r=e.run(dates[1],dates[-1]+pd.Timedelta(days=365))
print('COVERAGE requested_end', dates[-1]+pd.Timedelta(days=365),'actual_end',r.history.index[-1])
