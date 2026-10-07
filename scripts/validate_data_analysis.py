"""Execute data-analysis modules in fresh kernels using a disposable database copy.

Run from the project environment:
    uv run python scripts/validate_data_analysis.py [--core-only] [--output-dir PATH]

No original database, notebook output, credentials, or user kernel settings are changed.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = [
    'Lecture 2/2_pandas_to_portfolio.ipynb',
    'Lecture 2/3_visualize_and_clean.ipynb',
    'Lecture 4/1_sql_and_data_quality.ipynb',
    'Lecture 4/2_stock_analysis_template.ipynb',
    'Lecture 4/3_running_analysis_at_scale.ipynb',
]
CHECKS = {
    CORE[0]: """
assert np.isclose(portfolio['position_value'].sum(), sum(prices[t] * shares[t] for t in prices.index))
assert len(portfolio) == len(shares)
assert set(answer['ticker']) == {'AMZN', 'TSLA', 'BRK-B'}
""",
    CORE[1]: """
expected = make_prices().sort_values(['ticker', 'date']).reset_index(drop=True)
assert len(clean) == len(expected)
assert clean['close_repaired'].sum() == 4
assert clean['volume_repaired'].sum() == 2
unchanged = ~clean['close_repaired']
np.testing.assert_allclose(clean.loc[unchanged, 'close'], expected.loc[unchanged, 'close'])
assert not raw.equals(clean)
assert (output_dir / 'tech_dashboard.png').stat().st_size > 1000
""",
    CORE[2]: """
assert flags['ohlc_bounds'].sum() == 1
assert flags['missing_or_nonfinite'].sum() == 1
assert flags['negative_volume'].sum() == 1
assert len(quality_report) == 3
assert len(roundtrip) == len(aapl)
# Current value must not influence its own trailing baseline.
probe = pd.Series(np.arange(30, dtype=float))
baseline = rolling_zscore(probe, 5)
probe.iloc[-1] = 1000
assert np.isclose(rolling_zscore(probe, 5).iloc[-1], (1000 - np.mean(np.arange(24, 29))) / np.std(np.arange(24, 29), ddof=1))
assert rolling_zscore(pd.Series([1.] * 30), 5).isna().all()
""",
    CORE[3]: """
assert data_quality_ok
assert stats['trading_days'] == len(df)
assert np.isclose(stats['total_return_pct'], (df['adj_close'].iloc[-1] / df['adj_close'].iloc[0] - 1) * 100)
assert stats['max_drawdown_pct'] <= 0
np.testing.assert_allclose(df['volatility'].dropna(), df['daily_return'].rolling(volatility_window).std().dropna() * np.sqrt(252))
assert np.isfinite(yz_vol.dropna()).all()
""",
    CORE[4]: """
assert not failures, failures
assert len(successful_paths) == 3
assert set(summary['ticker']) == set(tickers)
assert len(json.loads((run_dir / 'manifest.json').read_text())['reports']) == 3
assert (run_dir / 'portfolio_summary.csv').is_file()
assert (run_dir / 'comparison.png').stat().st_size > 1000
# A stale/partial notebook in the folder must never affect manifest-based collection.
(run_dir / 'stale.ipynb').write_text('{}')
pd.testing.assert_frame_equal(collect_reports(successful_paths), summary)
for paths in ([], [first_report, first_report]):
    try:
        collect_reports(paths)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty or duplicate report collection accepted')
if RUN_EXTENSIONS:
    assert not quarterly_failures, quarterly_failures
    assert len(quarterly_paths) == 12
    assert len(quarterly) == 12
    assert set(quarterly['period']) == {'Q1', 'Q2', 'Q3', 'Q4'}
""",
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def notebook_functions(path):
    """Extract actual teaching functions for small numerical regression cases."""
    namespace = {'pd': pd, 'np': np}
    for cell in nbformat.read(path, as_version=4).cells:
        if cell.cell_type != 'code' or '%' in cell.source:
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def check_numerics(template):
    functions = notebook_functions(template)
    flags = functions['quality_flags']
    yz = functions['yang_zhang_volatility']
    dates = pd.bdate_range('2024-01-02', periods=30)
    frame = pd.DataFrame({name: 100. for name in ['open', 'high', 'low', 'close', 'adj_close']}, index=dates)
    frame['volume'] = 1000.
    assert not flags(frame).any().any()
    for field, value, expected_flag in [
        ('close', np.nan, 'missing_or_nonfinite'),
        ('volume', np.inf, 'missing_or_nonfinite'),
        ('adj_close', 0, 'nonpositive_price'),
        ('volume', -1, 'negative_volume'),
        ('close', 101, 'ohlc_bounds'),
    ]:
        bad = frame.copy()
        bad.iloc[3, bad.columns.get_loc(field)] = value
        assert flags(bad).iloc[3][expected_flag], (field, value)
    assert flags(pd.concat([frame, frame.iloc[[0]]]))['duplicate_date'].sum() == 2
    assert np.allclose(yz(frame, 5).dropna(), 0)
    # Deterministic overnight drift has zero sample variance (not uncentered squared returns).
    rising = frame.copy()
    for name in ['open', 'high', 'low', 'close', 'adj_close']:
        rising[name] = 100 * np.exp(np.arange(len(frame)) * 0.01)
    assert np.allclose(yz(rising, 5).dropna(), 0, atol=1e-8)
    # A constant daily range has a known Rogers-Satchell term.
    ranged = frame.copy()
    ranged['high'] = 102.
    ranged['low'] = 98.
    k = 0.34 / (1.34 + 6 / 4)
    expected = np.sqrt((1 - k) * (np.log(1.02)**2 + np.log(0.98)**2) * 252)
    assert np.allclose(yz(ranged, 5).dropna(), expected)
    # Adjusting all OHLC by a split factor restores the same path.
    split = ranged.copy()
    split.loc[dates[15]:, ['open', 'high', 'low', 'close']] /= 2
    np.testing.assert_allclose(yz(split, 5).dropna(), yz(ranged, 5).dropna())
    print('PASS numerical and quality regressions', flush=True)


def validate(workspace, core_only):
    source_db = ROOT / 'Lectures/Lecture 4/data/data.db'
    if not source_db.is_file():
        raise FileNotFoundError(f'Download the bCourses database to {source_db}')
    before = digest(source_db)
    shutil.copy2(ROOT / 'pyproject.toml', workspace / 'pyproject.toml')
    # Deliberate allowlist: never copy .env, local environments, or prior outputs.
    for lesson in ('Lecture 2', 'Lecture 4'):
        src = ROOT / 'Lectures' / lesson
        for path in src.rglob('*'):
            rel = path.relative_to(src)
            if any(part in {'.venv', 'outputs', '.ipynb_checkpoints'} for part in rel.parts):
                continue
            if path.is_file() and path.suffix in {'.ipynb', '.csv', '.json'}:
                dest = workspace / 'Lectures' / lesson / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    copied_db = workspace / 'Lectures/Lecture 4/data/data.db'
    with closing(sqlite3.connect(source_db.resolve().as_uri() + '?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(copied_db)) as target:
            source.backup(target)

    kernel_root = workspace / 'jupyter'
    kernel_dir = kernel_root / 'kernels' / 'course-validation'
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Course validation', 'language': 'python'}))
    env_updates = {'JUPYTER_PATH': str(kernel_root), 'MPLCONFIGDIR': str(workspace / 'matplotlib'),
                   'IPYTHONDIR': str(workspace / 'ipython')}
    old_env = {name: os.environ.get(name) for name in env_updates}
    os.environ.update(env_updates)
    results = []
    try:
        paths = list(CORE)
        if not core_only:
            paths.append('Lecture 2/1_jupyter_tutorial.ipynb')
            paths.extend(str(p.relative_to(workspace / 'Lectures')) for p in sorted((workspace / 'Lectures').glob('Lecture */reference/*.ipynb')))
        executed_dir = workspace / 'executed'
        executed_dir.mkdir()
        for rel in paths:
            path = workspace / 'Lectures' / rel
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            if rel == CORE[4] and not core_only:
                for cell in nb.cells:
                    cell.source = cell.source.replace('RUN_EXTENSIONS = False', 'RUN_EXTENSIONS = True')
            if rel in CHECKS:
                nb.cells.append(nbformat.v4.new_code_cell(CHECKS[rel]))
            started = time.monotonic()
            print(f'RUN {rel}', flush=True)
            # Alternate root/lecture/reference cwd to check path portability.
            cwd = workspace if rel in (CORE[0], CORE[2], CORE[3]) else path.parent
            NotebookClient(nb, timeout=600, kernel_name='course-validation',
                           resources={'metadata': {'path': str(cwd)}}).execute()
            out = executed_dir / rel.replace('/', '__')
            nbformat.write(nb, out)
            elapsed = round(time.monotonic() - started, 2)
            results.append({'notebook': rel, 'seconds': elapsed, 'status': 'passed'})
            print(f'PASS {rel} ({elapsed}s)', flush=True)
        template = workspace / 'Lectures' / CORE[3]
        check_numerics(template)
        # Validate template failure behavior in independent kernels, including a hostile SQL value.
        cases = [
            ({'ticker': "AAPL' OR 1=1 --"}, 'need at least'),
            ({'start_date': '2024-06-30', 'end_date': '2024-01-01'}, 'start_date must'),
            ({'volatility_window': 1}, 'integer >= 2'),
            ({'end_date': '2024-01-05'}, 'need at least'),
            ({'db_path': str(workspace / 'absent.db')}, 'Download the bCourses database'),
        ]
        # Inject one missing price into the TEMPORARY database only.
        dirty_db = workspace / 'dirty.db'
        shutil.copy2(copied_db, dirty_db)
        with closing(sqlite3.connect(dirty_db)) as conn:
            conn.execute("UPDATE ohlc SET close=NULL WHERE ticker='AAPL' AND ts='2024-01-03'")
            conn.commit()
        cases.append(({'db_path': str(dirty_db)}, 'Data quality checks failed'))
        for values, expected in cases:
            nb = nbformat.read(template, as_version=4)
            index = next(i for i, cell in enumerate(nb.cells) if 'parameters' in cell.metadata.get('tags', []))
            nb.cells.insert(index + 1, nbformat.v4.new_code_cell('\n'.join(f'{key} = {value!r}' for key, value in values.items())))
            try:
                NotebookClient(nb, timeout=120, kernel_name='course-validation',
                               resources={'metadata': {'path': str(template.parent)}}).execute()
            except CellExecutionError as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError(f'Expected failure for {values}')
        assert not (workspace / 'absent.db').exists()
        assert digest(source_db) == before, 'Source database changed during validation.'
        print(f'PASS six report failure cases; source database unchanged', flush=True)
        report = {'notebooks': results, 'numerical_checks': 'passed', 'failure_cases': 6,
                  'quarterly_extension': not core_only, 'source_database_unchanged': True}
        (workspace / 'validation_results.json').write_text(json.dumps(report, indent=2))
        print(f'PASS {len(results)} notebooks. Results: {workspace / "validation_results.json"}', flush=True)
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Keep the disposable workspace and executed notebooks here (must not exist).')
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        validate(args.output_dir.resolve(), args.core_only)
    else:
        with tempfile.TemporaryDirectory(prefix='mfe-data-analysis-') as folder:
            validate(Path(folder), args.core_only)


if __name__ == '__main__':
    main()
