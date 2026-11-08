import sys
sys.path.insert(0, '.'); sys.path.insert(0, '../../../Lectures/modeling')
import numpy as np, pandas as pd
from hw4.data import load_market_data, REPO
from hw4.features import FeatureStore
import course_data as cd
md = load_market_data(REPO/'Lectures/modeling/data/equities_class.duckdb', end='2015-12-31')
fs = FeatureStore(md)
dates = md.calendar[(md.calendar>='2010-01-04')&(md.calendar<='2015-08-01')][::7]
mine = fs.cross_section(dates)
ref = cd.prepare_panel()
ref = ref[ref.eligible]
m = mine.merge(ref, on=['date','symbol'], suffixes=('','_ref'))
print('rows mine', len(mine), 'ref eligible on these dates', ref.date.isin(dates).sum(), 'matched', len(m))
print('target max abs diff', np.nanmax(np.abs(m.target - m.target_ref)), 'nan mismatch', (m.target.isna()!=m.target_ref.isna()).sum())
print('label_end mismatch', (m.label_end != m.label_end_ref).sum())
for a,b in [('mom_21','momentum_21'),('mom_63','momentum_63'),('mom_252_21','momentum_252_21'),('vol_63','volatility_63')]:
    print(a, np.nanmax(np.abs(m[a]-m[b])))
