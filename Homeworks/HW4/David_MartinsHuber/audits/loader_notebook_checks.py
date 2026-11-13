"""Independent loader/demo/notebook verification. No production writes."""
from pathlib import Path
from types import SimpleNamespace
import ast
import contextlib
import importlib.util
import io
import os
import sys
import warnings
import duckdb
import numpy as np
import pandas as pd
import nbformat

hw4 = Path(__file__).resolve().parents[2]
submission = hw4 / 'David_MartinsHuber'
sys.path.insert(0, str(submission))
from data_loading import load_data, load_membership
from strategies.demo import MonthlyMomentum
from backtesting import DecisionContext, Portfolio

lecture_path = hw4 / '../../Lectures/modeling/course_data.py'
def sources(text):
    return {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)}
original, local = sources(lecture_path.read_text()), sources((submission / 'data_loading.py').read_text())
for name in ['load_data', 'load_membership']:
    assert original[name] == local[name], name
print('PASS: both loader functions match lecture source verbatim')
spec = importlib.util.spec_from_file_location('audit_reference_course_data', lecture_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
database = submission / 'data/equities_top2000.duckdb'
with duckdb.connect(str(database), read_only=True) as con:
    count_sql = """SELECT count(*) FROM prices WHERE symbol IN ('BHRB','AAPL')
    AND date <= '2001-12-31' AND (adj_close IS NULL OR NOT isfinite(adj_close) OR adj_close<=0)"""
    before = con.execute(count_sql).fetchone()[0]
with warnings.catch_warnings(record=True):
    warnings.simplefilter('always')
    actual = load_data(database, end='2001-12-31', symbols=['BHRB','AAPL'])
    expected = reference.load_data(database, end='2001-12-31', symbols=['BHRB','AAPL'])
for a, b in zip(actual, expected):
    pd.testing.assert_frame_equal(a, b)
assert len(actual[0]) == 725
assert np.isfinite(actual[0].adj_close).all() and actual[0].adj_close.gt(0).all()
assert set(actual[0].symbol) <= {'AAPL', 'BHRB'}
assert actual[0].date.max() <= pd.Timestamp('2001-12-31')
with duckdb.connect(str(database), read_only=True) as con:
    assert con.execute(count_sql).fetchone()[0] == before == 275
membership = load_membership(database, end='2001-12-31')
pd.testing.assert_frame_equal(membership, reference.load_membership(database, end='2001-12-31'))
assert membership.quarter_start.le(pd.Timestamp('2001-12-31')).all()
print('PASS: real read-only AAPL/BHRB slice: 725 retained quotes, 275 invalid raw rows unchanged, exact lecture-output parity')

# Independently specified ranking oracle: C doubles, A and B both gain50%;
# D's incomplete window and E's absence from membership exclude them.
prices = pd.DataFrame({'A':[10,12,15], 'B':[20,25,30], 'C':[10,13,20],
                       'D':[10,np.nan,100], 'E':[10,50,100]}, index=pd.bdate_range('2020-01-29',periods=3))
ctx = DecisionContext(pd.Timestamp('2020-01-31'), pd.Timestamp('2020-02-03'), prices,
                      pd.DataFrame({'symbol':['A','B','C','D']}), pd.DataFrame(),
                      Portfolio(70,60,300,{}), 0)
strategy = MonthlyMomentum('audit',lookback=2,top_n=2)
assert strategy.propose_buys(ctx) == {'C':30.,'A':30.}
assert strategy.propose_buys(SimpleNamespace(**{**ctx.__dict__, 'execution_date':pd.Timestamp('2020-01-31')})) == {}
assert strategy.propose_buys(SimpleNamespace(**{**ctx.__dict__, 'prices':prices.iloc[:2]})) == {}
assert strategy.propose_buys(SimpleNamespace(**{**ctx.__dict__, 'membership':pd.DataFrame({'symbol':[]})})) == {}
assert strategy.propose_buys(SimpleNamespace(**{**ctx.__dict__, 'portfolio':Portfolio(0,0,300,{})})) == {}
print('PASS: demo ranking/tie/completeness/membership/cash/schedule/warmup oracles')

for path in submission.rglob('*.ipynb'):
    if '.ipynb_checkpoints' in path.parts: continue
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code': ast.parse(cell.source)
notebook = nbformat.read(submission/'Backtesting.ipynb', as_version=4)
code = '\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
assert "Homeworks/HW4/David_MartinsHuber/data/equities_top2000.duckdb" in code
assert "TEST_START, TEST_END = '2016-04-01', '2026-09-30'" in code
assert "data = MarketData.load(DATABASE, end=DATA_END)" in code
assert 'display(bundle.portfolio.coverage)' in code
assert 'Backtester progress:' in code
readme = (submission/'README.md').read_text()
assert 'data/equities_top2000.duckdb' in readme and 'April 2016' in readme
assert 'end="2026-09-30"' in readme
assert 'allow_partial' in (submission/'docs/Documentation.md').read_text()
print('PASS: notebook format/syntax and explicit database/date/coverage setup match instructions')

# Execute the actual interactive synthetic notebook, including discovered suite.
# No database backtest and no networked Jupyter kernel are launched.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hw4-round2-loader-mpl')
import matplotlib
matplotlib.use('Agg')
nb = nbformat.read(submission/'audits/Testing.ipynb', as_version=4)
namespace = {'__name__':'__main__'}
old_cwd = Path.cwd()
out = io.StringIO()
try:
    os.chdir(submission/'audits')
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and cell.source.strip():
                exec(compile(cell.source, f'Testing.ipynb:cell_{index}', 'exec'), namespace)
    assert namespace['test_result'].wasSuccessful()
    print('PASS: every Testing.ipynb code cell executed; discovered regression tests:', namespace['test_result'].testsRun)
except Exception:
    print(out.getvalue())
    raise
finally:
    os.chdir(old_cwd)
