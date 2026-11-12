# HW4 submission, Emil Blaignan

I built a long only stock strategy and my own backtester, then compared it with the S&P 500 total return index. The explanation, results and findings are in [WRITEUP.md](WRITEUP.md). This file is just how to run things.

## What you need

The data file `equities_top2000.duckdb` from bCourses goes in `Lectures/modeling/data/`. I did not commit it.

Python packages are duckdb, pandas, numpy, scipy, scikit-learn, pyarrow, matplotlib and cvxpy.

```
pip install duckdb pandas numpy scipy scikit-learn pyarrow matplotlib cvxpy
```

## How to run it

Run everything from the repo root.

```
python Homeworks/HW4/Emil_Blaignan/preprocess.py
python Homeworks/HW4/Emil_Blaignan/build_weekly.py
python Homeworks/HW4/Emil_Blaignan/benchrel.py dev
python Homeworks/HW4/Emil_Blaignan/benchrel.py test
```

1. `preprocess.py` builds the panel. It takes about 6 minutes and writes large files to `outputs/`, which git ignores.
2. `build_weekly.py` builds the weekly dataset with the extra features. It takes under a minute.
3. `benchrel.py dev` tries the settings on 2011 to 2015 only and saves the one my rule picks. It takes about 2 minutes.
4. `benchrel.py test` runs the chosen setting on April 2016 to September 2026 and writes the tables and chart to `results/`. It takes about 10 minutes. I only ran this stage on the final setting.

To check the backtester on its own, run the tests.

```
python Homeworks/HW4/Emil_Blaignan/test_backtest.py
```

`audit_panel.py` prints some checks on the panel if you want to look at the data.

## What is in the folder

| File or folder | What it is |
|---|---|
| `preprocess.py` | Builds the panel using the lecture helpers in `Lectures/modeling/course_data.py` |
| `clean_prices.py` | Drops impossible prices that flip between two scales |
| `build_weekly.py` | Weekly dataset with beta, residual momentum and path smoothness features |
| `helpers.py` | Dates, metrics and a wrapper around the backtester |
| `backtest.py` | My backtester |
| `test_backtest.py` | Hand checked tests for the backtester |
| `benchrel.py` | The final strategy |
| `results/` | Tables, chart and accounting checks from the final run |
| `exploratory/` | Earlier attempts that did not make the cut. They are not needed to reproduce the final result. |

The files in `results/` are small and committed. Everything in `outputs/` is generated and ignored by git.
