"""Development: refit model D after applying the suspect-price rule.

A one-day adjusted-price ratio outside [0.2, 5] is a suspect jump. A row is
dropped when its feature lookback contains one; a target is blanked when its
90-day holding window contains one. Nothing after 2015-12-31 is read and no
backtest is run. The forecasts saved here feed run_portfolio_round.py.

Requires the feature cache written by 06_full_data_development.ipynb.

Run from the repository root:
    .venv/Scripts/python.exe Homeworks/HW4/Mauricio_Gutierrez/run_suspect_rule.py
"""
from pathlib import Path
import gc

import duckdb
import pandas as pd

from feature_engineering import (suspect_price_flags, fit_cached_development,
    forecast_decile_report, PRICE_FEATURES, D_RATIOS)

HW = Path(__file__).resolve().parent
BASE = HW / 'outputs/full_data_development'
OUT = BASE / 'suspect_rule'
SOURCE = HW / 'outputs/data_cleaning/equities_top2000/equities_top2000_cleaned.duckdb'
CACHE = BASE / 'development_features.duckdb'
FILTERED = OUT / 'filtered_features.duckdb'
BEFORE = BASE / 'after_fix/run_d_price_financial'
OUT.mkdir(parents=True, exist_ok=True)
source_stat = (SOURCE.stat().st_size, SOURCE.stat().st_mtime_ns)

with duckdb.connect(str(SOURCE),read_only=True) as src, duckdb.connect(str(CACHE),read_only=True) as cache, duckdb.connect(str(FILTERED)) as dest:
    dest.execute("SET memory_limit='256MB'")
    dest.execute('SET threads=2')
    calendar = pd.DatetimeIndex(src.execute("SELECT date FROM calendar WHERE date BETWEEN '2000-01-01' AND '2015-12-31' ORDER BY date").fetchdf().date).astype('datetime64[ns]')
    symbols = cache.execute("SELECT DISTINCT symbol FROM D WHERE date<='2015-12-31' ORDER BY symbol").fetchnumpy()['symbol'].tolist()
    dest.execute('CREATE OR REPLACE TABLE dropped_audit(year INTEGER, training_rows_before BIGINT, training_rows_dropped BIGINT, stocks_affected BIGINT, lookback_rows BIGINT, label_rows BIGINT)')
    created = False
    for start in range(0,len(symbols),50):
        group = symbols[start:start+50]
        prices = src.execute("SELECT symbol,date,adj_close FROM prices WHERE date BETWEEN '2000-01-01' AND '2015-12-31' AND symbol IN (SELECT unnest(?)) ORDER BY symbol,date",[group]).fetchdf()
        rows = cache.execute("SELECT * FROM D WHERE date BETWEEN '2010-01-01' AND '2015-12-31' AND symbol IN (SELECT unnest(?)) ORDER BY date,symbol",[group]).fetchdf()
        flags = suspect_price_flags(prices,calendar,rows)
        assert rows[['date','symbol']].equals(flags[['date','symbol']])
        bad = flags.suspect_lookback | flags.suspect_label
        for year in [2013,2014,2015]:
            cutoff = cache.execute('SELECT min(date) FROM D WHERE date BETWEEN ? AND ?',[f'{year}-01-01',f'{year}-12-31']).fetchone()[0]
            train = rows.date.lt(cutoff) & rows.label_end.lt(cutoff) & rows.target.notna()
            dropped = train & bad
            dest.execute('INSERT INTO dropped_audit VALUES (?,?,?,?,?,?)',[year,int(train.sum()),int(dropped.sum()),
                int(rows.loc[dropped,'symbol'].nunique()),int((train & flags.suspect_lookback).sum()),int((train & flags.suspect_label).sum())])
        # Future label contamination NEVER removes a live forecast candidate.
        rows.loc[flags.suspect_label,'target'] = float('nan')
        filtered = rows.loc[~flags.suspect_lookback].copy()
        dest.register('part',filtered)
        dest.execute(('INSERT INTO D BY NAME ' if created else 'CREATE OR REPLACE TABLE D AS ')+'SELECT * FROM part')
        dest.unregister('part')
        created = True
        del prices,rows,flags,filtered
        gc.collect()
    dropped = dest.execute('SELECT year,sum(training_rows_before) training_rows_before,sum(training_rows_dropped) training_rows_dropped,sum(stocks_affected) stocks_affected,sum(lookback_rows) lookback_rows,sum(label_rows) label_rows FROM dropped_audit GROUP BY year ORDER BY year').fetchdf()
    dest.execute('CHECKPOINT')

forecasts,scores,splits,coefficients = fit_cached_development(FILTERED,'D',PRICE_FEATURES+D_RATIOS+['filing_age_days'],clip_target=True)
forecasts.to_csv(OUT/'forecasts.csv',index=False)
splits.to_csv(OUT/'training_splits.csv',index=False)
dropped.astype('int64').to_csv(OUT/'rows_dropped.csv',index=False)
old_coef = pd.read_csv(BEFORE/'ridge_coefficients.csv').set_index(['model','year','term']).coefficient
new_coef = coefficients.set_index(['model','year','term']).coefficient
comparison = pd.concat({'before':old_coef,'after':new_coef},axis=1)
comparison['change'] = comparison.after-comparison.before
comparison.to_csv(OUT/'coefficients_before_after.csv')
old_scores = pd.read_csv(BEFORE/'model_scores.csv').set_index('year')
score_columns = ['mean_daily_rank_IC','R2','scored_rows','forecast_rows']
pd.concat({'before':old_scores[score_columns],'after':scores.set_index('year')[score_columns]},axis=1).to_csv(OUT/'scores_before_after.csv')
deciles,spreads = forecast_decile_report(forecasts,screened=True)
old_spreads = pd.read_csv(BEFORE/'screened_candidates_top_minus_bottom.csv').set_index('year')
pd.concat({'before':old_spreads,'after':spreads.set_index('year')},axis=1).to_csv(OUT/'decile_spreads_before_after.csv')
assert source_stat == (SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)
assert forecasts.date.max()<=pd.Timestamp('2015-12-31')
print(dropped.to_string(index=False))
print(pd.concat({'before':old_scores[score_columns[:2]],'after':scores.set_index('year')[score_columns[:2]]},axis=1).to_string())
print('Coefficient changes:',comparison.groupby('year')['change'].agg(['min','max']).to_string())
print('PASS: source metadata unchanged; no backtest run.')
