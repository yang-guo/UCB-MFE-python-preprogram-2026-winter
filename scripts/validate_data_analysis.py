"""Execute data-analysis modules in fresh kernels using a disposable database copy.

Run from the project environment:
    uv run python scripts/validate_data_analysis.py [--core-only] [--output-dir PATH]

No original database, notebook output, credentials, or user kernel settings are changed.
Lecture examples, end-of-notebook blank exercises, correct learner attempts, and incorrect attempts
are checked separately; executing a blank exercise is not counted as student completion.
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
assert np.isclose(portfolio['position_value'].sum(), 5000)
assert np.isclose(sector_weights.loc['Technology'], 80)
assert len(portfolio) == len(positions)
assert example_selection['ticker'].tolist() == ['MSFT']
assert np.isclose(example_rebalanced['position_value'].sum(), 5500)
assert np.isclose(example_rebalanced.set_index('ticker').loc['AAPL', 'weight_pct'], 2000 / 5500 * 100)
assert positions.set_index('ticker').loc['JPM', 'shares'] == 10
assert example_desk['ticker'].tolist() == ['AAPL']
assert np.isclose(example_financials_weight, 20)
""",
    CORE[1]: """
expected = pd.read_csv(data_path, parse_dates=['date']).sort_values(['ticker', 'date']).reset_index(drop=True)
assert len(clean) == len(expected) == 360
assert clean['close_repaired'].sum() == 4
assert clean['volume_repaired'].sum() == 2
unchanged = ~clean['close_repaired']
np.testing.assert_allclose(clean.loc[unchanged, 'close'], expected.loc[unchanged, 'close'])
assert clean.groupby('ticker')['return'].apply(lambda group: group.isna().sum()).eq(1).all()
assert example_msft['ticker'].eq('MSFT').all() and len(example_msft) == 120
assert example_bad_prices['ticker'].tolist() == ['DEMO_A']
assert example_volumes['volume'].tolist() == [100, 0, 200]
assert example_volumes['volume_repaired'].tolist() == [False, True, False]
assert example_text['ticker'].iloc[:4].tolist() == ['AAPL', 'AAPL', 'MSFT', 'JPM']
assert example_text['exchange'].iloc[:4].tolist() == ['NASDAQ', 'NASDAQ', 'NASDAQ', 'NYSE']
assert example_text['company_name'].iloc[:4].tolist() == ['APPLE INC.', 'APPLE INC.', 'MICROSOFT CORP.', 'JPMORGAN CHASE']
assert example_text.loc[4, ['exchange', 'ticker', 'company_name']].isna().all()
np.testing.assert_allclose(example_text['price'].iloc[:3].to_numpy(dtype=float), [150.25, 151, -5])
assert example_text['price'].iloc[3:].isna().all()
assert example_text['price_needs_review'].tolist() == [False, False, True, True, True]
pd.testing.assert_frame_equal(example_text[vendor_text.columns], vendor_text)
# Whole-field parsing must preserve signs and reject unrelated numbers.
parse_cases = pd.Series([' PRICE : +20 ', 'price: -2.50', 'price:0', 'price: unavailable', 'batch 7 price:120', None], dtype='string')
parsed_cases = pd.to_numeric(parse_cases.str.strip().str.lower().str.extract(price_pattern, expand=False), errors='coerce')
np.testing.assert_allclose(parsed_cases.iloc[:3].to_numpy(dtype=float), [20, -2.5, 0])
assert parsed_cases.iloc[3:].isna().all()
assert (output_dir / 'clean_prices.png').stat().st_size > 1000
""",
    CORE[2]: """
assert len(sql_summary) == 3
assert sql_summary.set_index('ticker').loc['AAPL', 'observations'] == len(aapl)
pd.testing.assert_frame_equal(example_query_result, example_expected_filter)
assert not example_query_result.empty and example_query_result['ticker'].eq('AAPL').all()
assert example_query_result['close'].gt(180).all()
assert len(joined) == len(aapl)
pd.testing.assert_frame_equal(roundtrip, sql_summary)
assert example_extract['ticker'].eq('NVDA').all() and len(example_extract) == 124
""",
    CORE[3]: """
assert data_quality_ok
assert stats['trading_days'] == len(df)
assert np.isclose(stats['total_return_pct'], (df['adj_close'].iloc[-1] / df['adj_close'].iloc[0] - 1) * 100)
assert stats['max_drawdown_pct'] <= 0
np.testing.assert_allclose(df['volatility'].dropna(), df['daily_return'].rolling(volatility_window).std().dropna() * np.sqrt(252))
assert np.isfinite(yz_vol.dropna()).all()
np.testing.assert_allclose(example_returns.dropna(), [0.25, -0.1])
np.testing.assert_allclose(example_drawdown, [0, 0, -0.3, -0.15], atol=1e-12)
""",
    CORE[4]: """
assert len(failures) == 1 and failures[0]['ticker'] == 'NOT_A_TICKER'
assert len(successful_paths) == 3
assert set(summary['ticker']) == {'AAPL', 'MSFT', 'NVDA'}
manifest = json.loads((run_dir / 'manifest.json').read_text())
assert len(manifest['reports']) == 3 and len(manifest['failures']) == 1
assert (run_dir / 'portfolio_summary.csv').is_file()
assert (run_dir / 'comparison.png').stat().st_size > 1000
# Incomplete/stale files must never enter a successful-run summary.
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
    assert len(quarterly_paths) == len(quarterly) == 12
    assert set(quarterly['period']) == {'Q1', 'Q2', 'Q3', 'Q4'}
pd.testing.assert_frame_equal(example_drawdown_comparison,
    summary[['ticker', 'trading_days', 'max_drawdown_pct']].sort_values('max_drawdown_pct'))
""",
}

BLANK_PRACTICE = {
    CORE[0]: 'assert your_selection is None and your_rebalanced is None and your_desk_table is None',
    CORE[1]: 'assert your_nvda is None and your_bad_prices is None and your_delivery is None and your_text_clean is None',
    CORE[2]: 'assert your_query_result is None and your_extract is None',
    CORE[3]: 'assert your_returns is None and your_drawdown is None',
    CORE[4]: 'assert your_comparison is None',
}

# Exercise answers are tested separately from default Run All. They do not overwrite source notebooks.
PRACTICE = {
    CORE[0]: """
your_selection = positions.loc[positions['shares'] >= 10, ['ticker', 'shares']]
your_rebalanced = positions.copy()
your_rebalanced.loc[your_rebalanced['ticker'] == 'MSFT', 'shares'] = 10
your_rebalanced['position_value'] = your_rebalanced['shares'] * your_rebalanced['price']
your_rebalanced['weight_pct'] = your_rebalanced['position_value'] / your_rebalanced['position_value'].sum() * 100
your_desk_table = portfolio.loc[(portfolio['sector'] != 'Technology') & (portfolio['position_value'] >= 1000), ['ticker', 'position_value']]
your_sector_weight = sector_weights.max()
your_conclusion = 'Technology accounts for 80% of portfolio value.'
""",
    CORE[1]: """
your_nvda = prices.loc[prices['ticker'] == 'NVDA'].copy()
your_bad_prices = raw.loc[raw['close'] > raw['high'], ['date', 'ticker', 'close', 'high']]
your_delivery = new_delivery.copy()
bad_volume = ~np.isfinite(your_delivery['volume']) | your_delivery['volume'].lt(0)
your_delivery.loc[bad_volume, 'volume_repaired'] = True
your_delivery.loc[bad_volume, 'volume'] = 0
your_quality_note = 'Zero is a flagged placeholder; observed volume is still unknown.'
your_text_clean = practice_text.copy()
symbols = your_text_clean['raw_symbol'].astype('string').str.replace(r'<[^>]+>', '', regex=True).str.strip().str.upper().str.split(':', n=1, expand=True)
your_text_clean['exchange'] = symbols[0].str.strip()
your_text_clean['ticker'] = symbols[1].str.strip()
your_text_clean['company_name'] = your_text_clean['raw_name'].astype('string').str.strip().str.upper().str.replace(r'\\s+', ' ', regex=True)
quotes = your_text_clean['raw_quote'].astype('string').str.strip().str.lower()
your_text_clean['price'] = pd.to_numeric(quotes.str.extract(price_pattern, expand=False), errors='coerce')
your_text_clean['price_needs_review'] = your_text_clean['price'].isna() | your_text_clean['price'].le(0)
your_text_note = 'An unrelated batch number is not a price; the zero quote is numeric but not a valid equity price.'
""",
    CORE[2]: """
your_query = 'SELECT ts, ticker, close FROM ohlc WHERE ticker = ? AND ts BETWEEN ? AND ? AND close > ? ORDER BY ts'
your_query_result = pd.read_sql_query(your_query, conn, params=('MSFT', start_date, end_date, 400), parse_dates=['ts'])
your_extract = pd.read_sql_query(price_query, conn, params=('JPM', start_date, end_date), parse_dates=['ts'])
your_extract_note = 'The date order, keys and closes passed; calendar coverage needs a separate check.'
""",
    CORE[3]: """
your_returns = calculate_returns(pd.Series([50.0, 55.0, 44.0]))
your_drawdown = calculate_drawdown(pd.Series([100.0, 120.0, 90.0, 108.0]))
your_worst_drawdown = your_drawdown.min()
your_drawdown_explanation = 'The final price recovered, but it is still below the prior peak.'
""",
    CORE[4]: """
your_comparison = summary[['ticker', 'total_return_pct', 'avg_volatility']].sort_values('avg_volatility', ascending=False)
your_comparison_note = 'NVDA had the greatest mean rolling variability here; this historical sample does not predict future risk.'
""",
}
BAD_PRACTICE = {
    CORE[0]: "your_selection = positions[['ticker', 'shares']]",
    CORE[1]: "your_nvda = prices.loc[prices['ticker'] == 'AAPL']",
    CORE[2]: "your_query_result = expected_filter.iloc[:1]",
    CORE[3]: "your_returns = pd.Series([np.nan, 10.0, -20.0])",
    CORE[4]: "your_comparison = summary[['ticker', 'total_return_pct', 'avg_volatility']].sort_values('avg_volatility')",
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
            if path.is_file() and (path.suffix in {'.ipynb', '.csv', '.json'} or path.name == 'report_runtime.py'):
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
                boundaries = [i for i, cell in enumerate(nb.cells)
                              if 'exercises-section' in cell.metadata.get('tags', [])]
                assert len(boundaries) == 1, f'Missing final practice section: {rel}'
                boundary = boundaries[0]
                for cell in nb.cells[:boundary]:
                    assert not {'exercise', 'feedback', 'exercise-prompt'} & set(cell.metadata.get('tags', [])), rel
                    if cell.cell_type == 'code':
                        assert 'your_' not in cell.source, f'Lecture depends on practice: {rel}'
                assert not any('worked-example' in cell.metadata.get('tags', []) for cell in nb.cells[boundary:]), rel
                assert not any('solution' in cell.metadata.get('tags', []) for cell in nb.cells), rel
                # Check the complete lecture before any practice variable has been initialized.
                nb.cells.insert(boundary, nbformat.v4.new_code_cell(CHECKS[rel]))
                nb.cells.append(nbformat.v4.new_code_cell(BLANK_PRACTICE[rel]))
                feedback = [cell.source for cell in nb.cells if 'feedback' in cell.metadata.get('tags', [])]
                assert len(feedback) >= 1, f'Missing learner feedback: {rel}'
                assert not any(cell.metadata.get('jupyter', {}).get('source_hidden') for cell in nb.cells), f'Hidden live code: {rel}'
                for i, cell in enumerate(nb.cells):
                    if 'exercise' in cell.metadata.get('tags', []):
                        assert 'exercise-prompt' in nb.cells[i - 1].metadata.get('tags', []), rel
                        assert 'feedback' in nb.cells[i + 1].metadata.get('tags', []), rel
                assert sum('exercise' in cell.metadata.get('tags', []) for cell in nb.cells) == len(feedback)
                nb.cells.append(nbformat.v4.new_code_cell(PRACTICE[rel]))
                nb.cells.extend(nbformat.v4.new_code_cell(source) for source in feedback)
                # Prove that the first check rejects an actually wrong learner answer.
                negative_check = BAD_PRACTICE[rel] + '\ntry:\n' + '\n'.join('    ' + line for line in feedback[0].splitlines())
                negative_check += "\nexcept AssertionError:\n    print('Incorrect practice answer rejected as expected.')\nelse:\n    raise AssertionError('Feedback accepted an incorrect answer.')"
                nb.cells.append(nbformat.v4.new_code_cell(negative_check))
                if rel == CORE[1]:
                    text_feedback = next(source for source in feedback if 'your_text_clean' in source)
                    # Reject common parsing mistakes without changing the valid learner answer permanently.
                    for wrong_edit in ("your_text_clean.loc[3, 'price'] = 7.0",
                                       "your_text_clean.loc[4, 'ticker'] = 'NONE'",
                                       "your_text_clean.loc[2, 'price_needs_review'] = False"):
                        text_check = "valid_text_answer = your_text_clean.copy()\n" + wrong_edit
                        text_check += '\ntry:\n' + '\n'.join('    ' + line for line in text_feedback.splitlines())
                        text_check += "\nexcept AssertionError:\n    pass\nelse:\n    raise AssertionError('Text feedback accepted an incorrect answer.')\nfinally:\n    your_text_clean = valid_text_answer"
                        nb.cells.append(nbformat.v4.new_code_cell(text_check))
                if rel == CORE[2]:
                    nb.cells.append(nbformat.v4.new_code_cell('conn.close()'))
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
                  'quarterly_extension': not core_only, 'source_database_unchanged': True, 'exercise_feedback': 'correct and incorrect answers checked'}
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
