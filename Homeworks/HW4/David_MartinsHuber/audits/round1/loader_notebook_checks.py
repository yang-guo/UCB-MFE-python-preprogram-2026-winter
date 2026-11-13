from pathlib import Path
from types import SimpleNamespace
import ast
import importlib.util
import sys
import warnings
import duckdb
import numpy as np
import pandas as pd
import nbformat

hw4=Path(__file__).resolve().parents[2]
submission=hw4/'David_MartinsHuber'
sys.path.insert(0,str(submission))
from data_loading import load_data,load_membership
from strategies.demo import MonthlyMomentum
from backtesting import DecisionContext,Portfolio
lecture_path=hw4/'../../Lectures/modeling/course_data.py'
lecture_text=lecture_path.read_text()
ours_text=(submission/'data_loading.py').read_text()
def function_sources(text):
 return {node.name:ast.get_source_segment(text,node) for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}
a,b=function_sources(lecture_text),function_sources(ours_text)
for name in ['load_data','load_membership']:
 assert a[name]==b[name],name
print('Verbatim loader functions: PASS')

spec=importlib.util.spec_from_file_location('reference_course_data',lecture_path)
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
database=submission/'data/equities_top2000.duckdb'
with duckdb.connect(str(database),read_only=True) as c:
 before=c.execute("SELECT count(*) FROM prices WHERE symbol IN ('BHRB','AAPL') AND date <= '2001-12-31' AND (adj_close IS NULL OR NOT isfinite(adj_close) OR adj_close<=0)").fetchone()[0]
with warnings.catch_warnings(record=True) as notices:
 warnings.simplefilter('always')
 ours=load_data(database,end='2001-12-31',symbols=['BHRB','AAPL'])
 expected=reference.load_data(database,end='2001-12-31',symbols=['BHRB','AAPL'])
for x,y in zip(ours,expected): pd.testing.assert_frame_equal(x,y)
assert np.isfinite(ours[0].adj_close).all() and ours[0].adj_close.gt(0).all()
assert all(frame.date.max()<=pd.Timestamp('2001-12-31') for frame in [ours[0],ours[3]])
assert set(ours[0].symbol)<= {'AAPL','BHRB'}
with duckdb.connect(str(database),read_only=True) as c:
 after=c.execute("SELECT count(*) FROM prices WHERE symbol IN ('BHRB','AAPL') AND date <= '2001-12-31' AND (adj_close IS NULL OR NOT isfinite(adj_close) OR adj_close<=0)").fetchone()[0]
assert before==after
membership=load_membership(database,end='2001-12-31')
pd.testing.assert_frame_equal(membership,reference.load_membership(database,end='2001-12-31'))
assert membership.quarter_start.le(pd.Timestamp('2001-12-31')).all()
print(f'Real database small slice: PASS; {len(ours[0])} retained quotes; {before} raw invalid observations unchanged; exact lecture output parity')

# Independent strategy oracle: eligible C has +100%, A/B tie +50%; D is
# excluded by incomplete window, and E by missing universe membership.
prices=pd.DataFrame({'A':[10,12,15],'B':[20,25,30],'C':[10,13,20],
                     'D':[10,np.nan,100],'E':[10,50,100]},index=pd.bdate_range('2020-01-29',periods=3))
ctx=DecisionContext(pd.Timestamp('2020-01-31'),pd.Timestamp('2020-02-03'),prices,
                    pd.DataFrame({'symbol':['A','B','C','D']}),pd.DataFrame(),Portfolio(70,60,300,{}),0)
s=MonthlyMomentum('audit',lookback=2,top_n=2)
assert s.propose_buys(ctx)=={'C':30.,'A':30.},s.propose_buys(ctx)
assert s.propose_buys(SimpleNamespace(**{**ctx.__dict__,'execution_date':pd.Timestamp('2020-01-31')}))=={}
assert s.propose_buys(SimpleNamespace(**{**ctx.__dict__,'prices':prices.iloc[:2]}))=={}
assert s.propose_buys(SimpleNamespace(**{**ctx.__dict__,'membership':pd.DataFrame({'symbol':[]})}))=={}
print('Demo strategy hand-calculated ranking/tie/universe/cash/schedule/warmup: PASS (5 cases)')
for path in submission.rglob('*.ipynb'):
 if '.ipynb_checkpoints' in path.parts:continue
 nb=nbformat.read(path,as_version=4)
 nbformat.validate(nb)
 for cell in nb.cells:
  if cell.cell_type=='code':ast.parse(cell.source)
print('All notebook formats and Python cell syntax: PASS')
