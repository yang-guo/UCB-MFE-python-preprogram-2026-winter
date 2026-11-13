"""Clean full histories in atomic groups of 50 stocks. No models or backtests.

Run from the repository root:
  .venv/Scripts/python.exe Homeworks/HW4/Mauricio_Gutierrez/clean_full_database.py
Run the same command to resume. Completed groups are never appended twice.
The source, cleaning code, and group plan must match the saved checkpoint.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import sys
import time
import ctypes
from ctypes import wintypes

import duckdb
import numpy as np
import pandas as pd
import data_cleaning as dc

HW = Path(__file__).resolve().parent
ROOT = HW.parents[2]
SOURCE = ROOT / 'Lectures/modeling/data/equities_top2000.duckdb'
CLASS_DB = HW / 'outputs/data_cleaning/equities_class/equities_class_cleaned.duckdb'
OUT = HW / 'outputs/data_cleaning/equities_top2000'
DEST = OUT / 'equities_top2000_cleaned.duckdb'
END = pd.Timestamp('2026-09-30')
GROUP_SIZE = 50


def peak_memory_mib():
    """Read the operating system's peak working set for this process."""
    if sys.platform != 'win32':
        import resource
        scale = 1024 ** 2 if sys.platform == 'darwin' else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 1024 ** 2


def require(checks):
    """Stop at the first failed validation. Never adjust a cleaning rule."""
    failed = [name for name, passed in dict(checks).items() if not passed]
    if failed:
        raise AssertionError('STOP: failed checks: ' + '; '.join(failed))


def append_frame(con, table, frame):
    con.register('incoming', frame)
    try:
        con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM incoming')
    finally:
        con.unregister('incoming')


def clean_group(raw, calendar, known, numeric, source_hash):
    """Apply the shared rules and all notebook data checks to one complete group."""
    before = {key: frame.copy(deep=True) for key, frame in raw.items()}
    p, pq, pdp, pi, pa = dc.clean_prices(raw['prices'], calendar, known)
    p, gaps = dc.add_past_only_price_flags(p, calendar)
    f, fq, fdp, fi, fa = dc.clean_fundamentals(raw['fundamentals'], known, numeric)
    parts = [pi, fi]
    for flag, explanation in {
        'flag_return_outlier': 'Return exceeds the prior-data standard-deviation threshold; price retained.',
        'flag_large_move': 'Absolute daily return exceeds 30%; price retained for investigation.',
        'flag_factor_change': 'Adjustment factor changed by more than the configured multiple; may be a corporate action.',
        'flag_gap_before_quote': 'Prior market-session quote is missing; daily return remains missing.',
    }.items():
        parts.append(dc.make_issues(p.loc[p[flag]], 'prices', 'date', flag.removeprefix('flag_'), explanation, 'medium', 'keep_flagged'))
    if not gaps.empty:
        parts.append(dc.make_issues(gaps.assign(source_row=pd.NA), 'prices', 'date', 'missing_usable_quote',
                                   'No usable quote inside the observed span; no price inserted.', 'medium', 'report_gap'))
    quality = dc.concat_issues(parts)
    quality['source_row'] = pd.to_numeric(quality['source_row'], errors='coerce').astype('Int64')
    quality['issue_date'] = pd.to_datetime(quality['issue_date'], errors='coerce')
    quality = quality.sort_values(['source_table', 'symbol', 'issue_date', 'issue_type'], na_position='last').reset_index(drop=True)
    accounting = pd.DataFrame([pa, fa])
    accounting['reconciles'] = accounting.raw_rows.eq(accounting[['kept_rows', 'duplicate_rows', 'quarantined_rows']].sum(axis=1))
    checks = dc.price_validation(p, calendar, known)
    original = raw['prices'].iloc[p.source_row.to_numpy()][dc.PRICE_COLUMNS].to_numpy(float)
    checks.update({
        'prices: retained observed prices are unchanged': np.array_equal(p[dc.PRICE_COLUMNS].to_numpy(float), original, equal_nan=True),
        'financials: unique report keys': not f.duplicated(['symbol', 'fiscal_date', 'period']).any(),
        'financials: quarterly reports': f.period.eq('quarter').all(),
        'financials: valid known stocks': f.symbol.isin(known).all(),
        'financials: publication follows fiscal end': f.filing_date.gt(f.fiscal_date).all(),
        'financials: next-day availability': f.available_date.eq(f.filing_date + pd.Timedelta(days=1)).all(),
        'financials: numeric values are finite or missing': all((f[c].isna() | np.isfinite(f[c])).all() for c in numeric),
        'accounting: every source row has one destination': accounting.reconciles.all(),
        'scope: no prices beyond the load boundary': p.date.le(END).all(),
        'source: database file is unchanged': dc.file_sha256(SOURCE) == source_hash,
    })
    for key in raw:
        pd.testing.assert_frame_equal(raw[key], before[key])
    checks['source: raw in-memory tables are unchanged'] = True
    require(checks)
    return {
        'prices': p, 'prices_quarantine': pq, 'prices_duplicates': pdp,
        'fundamentals': f, 'fundamentals_quarantine': fq, 'fundamentals_duplicates': fdp,
        'quality_report': quality, 'row_accounting': accounting, 'missing_sessions': gaps,
        'validation': pd.Series(checks, name='passed').rename_axis('check').reset_index(),
    }


def class_regression():
    """Run notebook checks without saving or plotting; compare the saved class rows.

The same class input window must give identical flags. Full-history flags can
have extra pre-2008 observations. Source-row identifiers are load-local audit IDs.
"""
    import io
    from contextlib import redirect_stdout
    notebook = json.loads((HW / '01_data_preparation.ipynb').read_text(encoding='utf-8'))
    namespace = {'display': lambda *args, **kwargs: None}
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code' or i in (22, 30):
            continue
        with redirect_stdout(io.StringIO()):
            exec(compile(''.join(cell['source']), f'notebook01:cell{i}', 'exec'), namespace)
        namespace['display'] = lambda *args, **kwargs: None
    comparisons = []
    with duckdb.connect(str(CLASS_DB), read_only=True) as con:
        for table, variable, keys in [
            ('prices', 'prices_cleaned', ['symbol', 'date']),
            ('fundamentals', 'fundamentals_cleaned', ['symbol', 'fiscal_date', 'period']),
            ('prices_quarantine', 'prices_quarantine', ['source_row']),
            ('fundamentals_quarantine', 'fundamentals_quarantine', ['source_row']),
            ('prices_duplicates', 'prices_duplicates', ['source_row']),
            ('fundamentals_duplicates', 'fundamentals_duplicates', ['source_row']),
        ]:
            expected = con.execute(f'SELECT * FROM "{table}"').fetchdf()
            actual = namespace[variable].reindex(columns=expected.columns)
            pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                          expected.sort_values(keys).reset_index(drop=True), check_dtype=False)
            comparisons.append({'table': table, 'rows': len(actual), 'same_class_window_matches': True})
    pd.DataFrame(comparisons).to_csv(OUT / 'class_regression.csv', index=False)
    print('Class regression and notebook synthetic checks passed.', flush=True)


def final_checks(src, dst, refs, numeric, source_hash):
    """Re-read saved data in groups. Check values, keys, partitions, and counts."""
    results = {}
    for table in ['prices', 'fundamentals']:
        keys = 'symbol,date' if table == 'prices' else 'symbol,fiscal_date,period'
        results[f'{table}: global unique keys'] = dst.execute(f'SELECT count(*) FROM (SELECT {keys} FROM {table} GROUP BY {keys} HAVING count(*)>1)').fetchone()[0] == 0
        total, distinct = dst.execute(f'SELECT count(*),count(distinct source_row) FROM (SELECT source_row FROM {table} UNION ALL SELECT source_row FROM {table}_quarantine UNION ALL SELECT source_row FROM {table}_duplicates)').fetchone()
        results[f'{table}: all source rows saved exactly once'] = total == distinct == src.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
    calendar = pd.DatetimeIndex(refs['calendar'].date)
    known = set(refs['universe'].symbol)
    for group_id, symbols_json, counts_json in dst.execute('SELECT group_id,symbols,saved_counts FROM batch_checkpoints ORDER BY group_id').fetchall():
        symbols = json.loads(symbols_json)
        counts = json.loads(counts_json)
        p = dst.execute('SELECT * FROM prices WHERE raw_symbol IN (SELECT unnest(?))', [symbols]).fetchdf()
        f = dst.execute('SELECT * FROM fundamentals WHERE raw_symbol IN (SELECT unnest(?))', [symbols]).fetchdf()
        require(dc.price_validation(p, calendar, known))
        require({
            'saved financial keys': not f.duplicated(['symbol', 'fiscal_date', 'period']).any(),
            'saved financial symbols': f.symbol.isin(known).all(),
            'saved quarterly reports': f.period.eq('quarter').all(),
            'saved filing lag': f.filing_date.gt(f.fiscal_date).all(),
            'saved availability': f.available_date.eq(f.filing_date + pd.Timedelta(days=1)).all(),
            'saved finite financial values': all((f[c].isna() | np.isfinite(f[c])).all() for c in numeric),
            'saved scope': p.date.le(END).all(),
        })
        for table in ['prices','prices_quarantine','prices_duplicates','fundamentals','fundamentals_quarantine','fundamentals_duplicates']:
            actual = dst.execute(f'SELECT count(*) FROM {table} WHERE raw_symbol IN (SELECT unnest(?))', [symbols]).fetchone()[0]
            require({f'group {group_id} saved {table} count': actual == counts[table]})
        del p, f
        gc.collect()
    results['saved groups: existing price and financial checks'] = True
    for table, frame in refs.items():
        actual = dst.execute(f'SELECT * FROM {table}').fetchdf()
        pd.testing.assert_frame_equal(actual, frame, check_dtype=False)
        results[f'{table}: unchanged reference rows'] = True
    results['source checksum unchanged'] = dc.file_sha256(SOURCE) == source_hash
    results['every group passed'] = dst.execute('SELECT bool_and(passed) FROM validation').fetchone()[0]
    require(results)
    validation = pd.Series(results, name='passed').rename_axis('check').reset_index()
    append_frame(dst, 'validation', validation)
    validation.to_csv(OUT / 'final_validation.csv', index=False)


def reports(dst):
    """Export only data-quality counts. Do not calculate strategy statistics."""
    dst.execute('DELETE FROM quality_summary')
    dst.execute('INSERT INTO quality_summary SELECT source_table,issue_type,action,count(*) AS issues FROM quality_report GROUP BY ALL')
    queries = {
        'row_accounting': 'SELECT "table",sum(raw_rows) raw_rows,sum(kept_rows) kept_rows,sum(duplicate_rows) duplicate_rows,sum(quarantined_rows) quarantined_rows,bool_and(reconciles) reconciles FROM row_accounting GROUP BY "table"',
        'quality_summary': 'SELECT * FROM quality_summary ORDER BY source_table,issue_type',
        'missing_sessions_by_stock': 'SELECT symbol,count(*) missing_sessions FROM missing_sessions GROUP BY symbol ORDER BY missing_sessions DESC,symbol',
        'missing_sessions_top20': 'SELECT symbol,count(*) missing_sessions FROM missing_sessions GROUP BY symbol ORDER BY missing_sessions DESC,symbol LIMIT 20',
        'histories_ending_early': "SELECT u.symbol,max(p.date) last_date FROM universe u LEFT JOIN prices p USING(symbol) GROUP BY u.symbol HAVING max(p.date)<DATE '2026-09-30' OR max(p.date) IS NULL ORDER BY last_date,symbol",
        'large_moves_by_year': 'SELECT year(date) AS "year",count(*) FILTER (WHERE flag_large_move) flagged_large_moves FROM prices GROUP BY year(date) ORDER BY year',
        'validation': 'SELECT * FROM validation',
    }
    for table in ['prices', 'fundamentals']:
        queries[table + '_disposition'] = f"SELECT 'retained' disposition,'passed cleaning rules' reason,count(*) AS \"rows\" FROM {table} UNION ALL SELECT 'quarantined',quarantine_reason,count(*) FROM {table}_quarantine GROUP BY quarantine_reason UNION ALL SELECT 'duplicate',removal_reason,count(*) FROM {table}_duplicates GROUP BY removal_reason"
        queries[table + '_quarantine_reasons'] = f"SELECT issue_type reason,count(*) AS \"rows\" FROM quality_report WHERE source_table='{table}' AND action='quarantine' GROUP BY issue_type ORDER BY rows DESC"
    for name, query in queries.items():
        dst.execute(query).fetchdf().to_csv(OUT / f'{name}.csv', index=False)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = dc.file_sha256(SOURCE)
    class_hash = dc.file_sha256(CLASS_DB)
    code_hash = hashlib.sha256(Path(dc.__file__).read_bytes() + Path(__file__).read_bytes()).hexdigest()
    class_regression()
    with duckdb.connect(str(SOURCE), read_only=True) as src, duckdb.connect(str(DEST)) as dst:
        dst.execute("SET memory_limit='512MB'")
        dst.execute('SET threads=2')
        refs = {t: src.execute('SELECT * FROM ' + t + (' ORDER BY date' if t in ['calendar','benchmark'] else ' ORDER BY symbol' if t == 'universe' else ' ORDER BY quarter_start,universe_rank')).fetchdf() for t in ['calendar','benchmark','universe','universe_membership']}
        require(dc.reference_validation(refs))
        require({'source calendar range': refs['calendar'].date.min() == pd.Timestamp('2000-01-03') and refs['calendar'].date.max() == END})
        calendar = pd.DatetimeIndex(refs['calendar'].date)
        known = set(refs['universe'].symbol)
        numeric = src.execute("SELECT column_name FROM information_schema.columns WHERE table_name='fundamentals' AND regexp_matches(data_type,'DOUBLE|FLOAT|DECIMAL|INTEGER|BIGINT|SMALLINT') ORDER BY ordinal_position").fetchnumpy()['column_name'].tolist()
        symbols = src.execute('SELECT symbol FROM prices UNION SELECT symbol FROM fundamentals UNION SELECT symbol FROM universe ORDER BY symbol').fetchnumpy()['symbol'].tolist()
        require({'raw stock labels present and canonical': pd.Series(symbols).notna().all() and dc.normalize_symbol(pd.Series(symbols)).eq(pd.Series(symbols)).all()})
        identity = json.dumps({'source_sha256':source_hash, 'code_sha256':code_hash, 'class_sha256':class_hash, 'symbols':symbols, 'group_size':GROUP_SIZE}, sort_keys=True)
        tables = {r[0] for r in dst.execute('SHOW TABLES').fetchall()}
        if not tables:
            dst.execute('BEGIN')
            try:
                # Use the class output schema, including empty quarantine/duplicate tables.
                with duckdb.connect(str(CLASS_DB), read_only=True) as cls:
                    for (table,) in cls.execute('SHOW TABLES').fetchall():
                        fields = cls.execute(f'DESCRIBE "{table}"').fetchall()
                        # Empty class audit frames infer INTEGER for object columns.
                        # Keep names; use source types for their actual full-data values.
                        base = table.split('_')[0]
                        source_types = dict((r[0], r[1]) for r in src.execute(f'DESCRIBE {base}').fetchall()) if base in ('prices','fundamentals') else {}
                        text_fields = {'raw_symbol','quarantine_reason','removal_reason'}
                        ddl = ','.join(f'"{r[0]}" {"VARCHAR" if r[0] in text_fields else source_types.get(r[0], r[1]) if table.endswith(("_quarantine","_duplicates")) else r[1]}' for r in fields)
                        dst.execute(f'CREATE TABLE "{table}" ({ddl})')
                dst.execute('CREATE TABLE batch_checkpoints(group_id INTEGER PRIMARY KEY,symbols VARCHAR,saved_counts VARCHAR,price_offset BIGINT,financial_offset BIGINT,seconds DOUBLE,peak_memory_mib DOUBLE)')
                dst.execute('CREATE TABLE batch_identity(identity VARCHAR)')
                dst.execute('INSERT INTO batch_identity VALUES (?)', [identity])
                for table, frame in refs.items():
                    append_frame(dst, table, frame)
                dst.execute('COMMIT')
            except Exception:
                dst.execute('ROLLBACK')
                raise
        else:
            require({'resume source/code/group plan unchanged': dst.execute('SELECT identity FROM batch_identity').fetchone()[0] == identity})
        done = {r[0] for r in dst.execute('SELECT group_id FROM batch_checkpoints').fetchall()}
        for group_id, start in enumerate(range(0, len(symbols), GROUP_SIZE)):
            if group_id in done:
                continue
            tick = time.perf_counter()
            group = symbols[start:start+GROUP_SIZE]
            raw = {table: src.execute(f'SELECT * FROM {table} WHERE symbol IN (SELECT unnest(?)) ORDER BY symbol,{order}', [group]).fetchdf() for table, order in [('prices','date'),('fundamentals','fiscal_date,period')]}
            frames = clean_group(raw, calendar, known, numeric, source_hash)
            # Give each source table a global audit ID. This changes no cleaning decision.
            offsets = {t: dst.execute('SELECT coalesce(sum(raw_rows),0) FROM row_accounting WHERE "table"=?', [t]).fetchone()[0] for t in ['prices','fundamentals']}
            for t in ['prices','fundamentals']:
                for name in [t, t+'_quarantine', t+'_duplicates']:
                    frames[name]['source_row'] += offsets[t]
                mask = frames['quality_report'].source_table.eq(t)
                frames['quality_report'].loc[mask, 'source_row'] += offsets[t]
            counts = {t:len(f) for t,f in frames.items()}
            dst.execute('BEGIN')
            try:
                for table, frame in frames.items():
                    prior = dst.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                    append_frame(dst, table, frame)
                    require({f'{table}: saved count': dst.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == prior + len(frame)})
                dst.execute('INSERT INTO batch_checkpoints VALUES (?,?,?,?,?,?,?)', [group_id,json.dumps(group),json.dumps(counts),offsets['prices'],offsets['fundamentals'],time.perf_counter()-tick,peak_memory_mib()])
                dst.execute('COMMIT')
            except Exception:
                dst.execute('ROLLBACK')
                raise
            print(f'Group {group_id+1}/{(len(symbols)+49)//50}: {len(raw["prices"]):,} price rows; saved and checked.', flush=True)
            del raw, frames
            gc.collect()
        print('All groups saved. Checking final database.', flush=True)
        final_checks(src, dst, refs, numeric, source_hash)
        # Match retained class source values in the full cleaned DB. Extra pre-2008
        # history affects rolling flags, so compare these separately above.
        dst.execute(f"ATTACH '{CLASS_DB.as_posix()}' AS cls (READ_ONLY)")
        full_comparisons = []
        for table, keys in [('prices',['symbol','date']),('fundamentals',['symbol','fiscal_date','period'])]:
            columns = [r[0] for r in src.execute(f'DESCRIBE {table}').fetchall()]
            cols = ','.join('"'+c+'"' for c in columns)
            key_sql = ','.join(keys)
            mismatch = dst.execute(f'SELECT count(*) FROM (SELECT {cols} FROM cls.{table} EXCEPT SELECT {cols} FROM main.{table})').fetchone()[0]
            extra = dst.execute(f'SELECT count(*) FROM (SELECT {key_sql} FROM main.{table} WHERE symbol IN (SELECT symbol FROM cls.universe) AND {keys[1]} BETWEEN (SELECT min({keys[1]}) FROM cls.{table}) AND (SELECT max({keys[1]}) FROM cls.{table}) EXCEPT SELECT {key_sql} FROM cls.{table})').fetchone()[0]
            require({f'class retained {table} source values match full output': mismatch == 0})
            full_comparisons.append({'table':table,'class_rows_missing_or_changed':mismatch,'additional_full_rows_in_class_global_span':extra})
        pd.DataFrame(full_comparisons).to_csv(OUT/'class_full_comparison.csv',index=False)
        dst.execute('DETACH cls')
        reports(dst)
        require({'class output unchanged':dc.file_sha256(CLASS_DB)==class_hash})
        summary = {
            'source_sha256':source_hash, 'source_checksum_unchanged':dc.file_sha256(SOURCE)==source_hash,
            'class_output_unchanged':True, 'completed_utc':datetime.now(timezone.utc).isoformat(),
            'groups':len(symbols)//50 + bool(len(symbols)%50), 'stocks':len(symbols),
            'elapsed_seconds_this_invocation':time.perf_counter()-started,
            'committed_group_seconds_all_invocations':dst.execute('SELECT sum(seconds) FROM batch_checkpoints').fetchone()[0],
            'peak_process_memory_mib':max(peak_memory_mib(),dst.execute('SELECT max(peak_memory_mib) FROM batch_checkpoints').fetchone()[0]),
            'missing_sessions':dst.execute('SELECT count(*) FROM missing_sessions').fetchone()[0],
            'stocks_ending_early':len(pd.read_csv(OUT/'histories_ending_early.csv')),
        }
        require({'final checksum':summary['source_checksum_unchanged']})
        dst.execute('CHECKPOINT')
    (OUT/'run_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame([summary]).to_csv(OUT/'run_summary.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
