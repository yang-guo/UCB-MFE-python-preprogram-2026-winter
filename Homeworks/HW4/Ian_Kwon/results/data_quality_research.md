# Data quality evidence (research period, generated)

Database: {'database': 'equities_top2000.duckdb', 'sha256': 'a028641d96c12e6f29d5dd34e3a6ebc059074dd38351648bf0d970b1dd4095b9', 'manifest_match': True}

## One-day price breaks (|move| > x1.8) by classification

| kind | events | symbols | in_universe |
|---|---|---|---|
| bad_print | 936 | 107 | 381 |
| level_shift | 18 | 16 | 7 |
| split | 61 | 39 | 37 |
| unclassified | 1080 | 310 | 519 |

Examples among universe members:

| kind | symbol | date | gross | ratio | dv_day_ratio | universe_rank |
|---|---|---|---|---|---|---|
| unclassified | OBCI | 2007-01-10 | 2.792 |  | 456.7 | 1,685 |
| unclassified | SMTI | 2007-01-11 | 3 | 3 |  | 1,720 |
| unclassified | NBSE | 2007-01-18 | 2.222 |  |  | 1,713 |
| unclassified | CYRX | 2007-01-19 | 3.6 |  | 5.554 | 1,701 |
| unclassified | SMTI | 2007-01-24 | 0.3333 | 0.3333 |  | 1,720 |
| unclassified | CENT | 2007-02-06 | 2.174 |  | 3.735 | 536 |
| bad_print | SMTI | 2007-02-07 | 2 |  |  | 1,720 |
| bad_print | PEGY | 2007-02-09 | 2.359 |  | 0 | 8 |
| bad_print | DCOM | 2007-02-09 | 0.5129 |  | 58.33 | 1,140 |
| bad_print | DCOM | 2007-03-06 | 0.4964 |  | 120 | 1,140 |
| bad_print | DCOM | 2007-03-09 | 0.5026 |  | 55.59 | 1,140 |
| bad_print | DCOM | 2007-03-23 | 0.5346 |  | 19.72 | 1,140 |
| level_shift | STAR | 2007-06-06 | 0.0787 | 0.0787 | 1.689 | 172 |
| split | NCR | 2007-10-01 | 2.018 | 2 | 1.874 | 420 |
| split | SSNT | 2007-11-16 | 0.5008 | 0.5 | 0.4406 | 1,797 |
| split | CYTH | 2007-12-11 | 0.5 | 0.5 | 0.7859 | 1,795 |
| split | SSNT | 2007-12-19 | 1.997 | 2 | 0.4382 | 1,797 |
| split | DCOM | 2007-12-31 | 1.956 | 2 | 0.2482 | 1,176 |
| split | ISDR | 2008-05-13 | 0.4887 | 0.5 | 0.8462 | 1,799 |
| level_shift | VBIV | 2010-04-26 | 835.1 | 835.1 | 0 | 1,901 |
| level_shift | FBC | 2010-05-28 | 10.82 | 10.82 | 1.621 | 1,884 |
| level_shift | AAM | 2011-01-31 | 0 | 0 | 0 | 1 |
| level_shift | AAM | 2011-02-08 | 3.777e+04 | 3.777e+04 | 0.4202 | 1 |
| level_shift | SSNT | 2012-04-17 | 7.265 | 7.265 | 1.76 | 1,960 |

## Load-time quarantine (non-positive adj_close, off-calendar rows)

0 rows in the research window (BHRB zero prices are in 2000-2001, before the loaded window).

## Eligibility screen (each row can fail several screens)

| index | rows |
|---|---|
| member rows (signal date x member) | 809060 |
| no quote at signal close | 1138 |
| fewer than 57 of 63 daily returns | 5300 |
| median dollar volume < $1M | 204081 |
| price break in last 6 sessions (cool-off) | 839 |
| artifact/unreliable history in last 252 sessions | 8329 |
| eligible | 602903 |

## Financial statements

| index | rows |
|---|---|
| quarterly statements loaded | 144772 |
| filing_date <= fiscal_date (placeholder; imputed +120d) | 14898 |
| eligible rows using an imputed-availability statement | 2785 |
| eligible rows with no usable statement | 0 |

## Extreme values kept (not winsorized): largest eligible 63-day returns

| date | symbol | ret_63 | vol_63 | target |
|---|---|---|---|---|
| 2012-12-14 | FBC | 18.63 | 19.5 | -0.276 |
| 2012-11-30 | FBC | 18.51 | 19.5 | -0.288 |
| 2012-12-07 | FBC | 18.47 | 19.5 | -0.186 |
| 2013-01-04 | FBC | 17.64 | 19.5 | -0.225 |
| 2012-12-28 | FBC | 17.48 | 19.5 | -0.288 |
