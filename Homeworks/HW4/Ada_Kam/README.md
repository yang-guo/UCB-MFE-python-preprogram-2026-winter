# HW4: Stock Selection Model and Backtest (Ada Kam)

## Run Instructions
1. Download equities_top2000.duckdb from bCourses into Lectures/modeling/data
2. From repo root: `uv sync --locked`
3. Run notebooks in order:
   - 01_explore.ipynb: data inspection
   - 02_model_selection.ipynb: builds and cache data then compare models
   - 03_backtest.ipynb: fits final model, runs backtest and write charts to results/
4. Accounting tests: `uv run --with pytest pytest Homeworks/HW4/Ada_Kam/test_backtester.py -q` (4 tests)

## Results + Evaluation
| | Strategy | ^SP500TR |
|---|---|---|
| Final value | ⟨$2,508,689⟩ | $4,402,747 |
| CAGR | ⟨9.16%⟩ | 15.17% |
| Max drawdown | ⟨−27.85%⟩ | −33.79% |
| Annual volatility | ⟨14.51%⟩ | 17.82% |
| Sharpe (2% rf) | ⟨0.54⟩ | 0.77 |

- Boosted tree ranks top-2000 universe each month, strategy buys top 30 and hold each for 90days
- From Apr 2016 to Sep 2026, turned $1M int $2.51 vs $4.40M for SPR500TR
- Did not beat benchmark 
- But lower volatility (14.5% vs 17.8%) and smaller max dd (-27.9% vs -33.8%)

## Issues
| Issue | Details | Remedy |
|---|---|---|
| Universe < 2,000 stocks | 2000-Q1 to 2011-Q2 below 2,000 names | Used membership as given; dev starts 2010 |
| Bad filing dates | 23,664 rows with filing date ≤ fiscal date | Excluded |
| Missing price dates | Stocks lack quotes on some sessions | Reindexed to the benchmark calendar with NaN gaps |
| Extreme returns/ features | AAM $0.19 → $20,461 in 90 days (target 106,847); VBIV > 600% | Training target filtered to the training fold's 1–99% quantiles |
| Equity ≤ 0 | ROE and D/E meaningless | Set to NaN |
| Weak technical signal | Technical-only models: no model beats the mean on RMSE; negative IC | Added fundamentals and the rank target |
| No delisting proceeds | Data guide forbids inferring payouts | Lots kept at last price and reported with a $0 lower bound |


## Structure
| File | Purpose |
|---|---|
| 01_explore.ipynb | Tables, dtypes, date ranges, universe size, known data issues |
| features.py | Builds the month-end feature/target, reuse course_data.py: load_data, load_membership, price_features, financial_events, join_financials, apply_membership, make_labels. Added new technical features, TTM fundamentals and ratios |
| 02_model_selection.ipynb | Walk fwd validation of mean/OLS/Ridge/Lasso/boosted trees and comparison of models |
| backtester.py | Portfolio (cash, lots, fills, fees), TopNStrategy, run_backtest daily loop, performance metrics. Design adapted from Lectures/modeling/backtest.py |
| test_backtester.py | Hand-calculated accounting tests on a 2-stock market |
| 03_backtest.ipynb | Final fit, test-period scoring, backtest, accounting checks, metrics, charts |

## Training Features
- All features use only data available at signal close (adj_close), px features computed per stock on trading calendar w NaN gaps, fundamentals available after 'filing=date' and dropped if fiscal period > 550 days

|Feature|Description|
|---|---|
|Momentum_252_21|12m return, excluding last month|
|Volatility_63|63-session std of daily returns, annualized|
|RSI_21|21 day RSI from log returns|
|Dma_dist_20, 50, 100|Distance from moving avgs|
|ROE|Trailing Twelse month net income / equity|
|Debt to Equity|Total debt / equity|
|Book to px|Equity / mkt cap|
|Earnings yield|TTM net income / mkt cap|
|FCF yield|TTM fcf / mkt cap|
|EBITDA to EV|TTM EBITDA / EV|

## Model Selection
| Step | Finding |
|---|---|
| 1. Technical only, raw target | No model beats training mean on RMSE (all R² < 0), Linear models: IC −0.026, negative all 4 years (short-term reversal); trees −0.010 |
| 2. + fundamentals, raw target | IC improves but stays ≈ 0 (trees ⟨+0.002⟩) |
| 3. Rank target | Fit on each date's cross-sectional rank instead of raw return. IC turns positive: Ridge ⟨+0.044⟩, trees ⟨+0.036⟩, positive 3 of 4 years |
| Tried, not used | Sector one-hots lowered IC (⟨+0.044 → +0.025⟩): they learn past sector returns that didn't persist. |


## Backtest Design
| Rule | Choice |
|---|---|
| Capital | $1,000,000, long only, no leverage; 0% interest on cash |
| Signal | Each month-end close; orders fill at the next session's close |
| Candidates | Universe members for the entry quarter, eligible, close ≥ $5, common stock only |
| Selection | Top 30 by model score, equal weight, skip names already held |
| Sizing | Budget = min(cash, NAV/3), giving 3 overlapping 90-day cohorts, no orders under $1,000 |
| Exit | Each lot sold at the close on entry + 90 days, or the next session with a quote |
| Costs | 10 bps per side; fractional units of adj close |
| Missing entry quote | Order skipped, cash kept |
| Missing exit quote | Lot stays open, marked at its last price |
| Benchmark | $1M in ^SP500TR at the 2016-04-01 close, held to 2026-09-30 |

## Next Steps
- Retrain yearly with newly completed labels
- Sector-specific features, e.g. ROE vs peers
- More dynamic weighing of stock picks

## AI Assisted
I used Claude (Claude Code) as a tutor and coding assistant. 
- Suggested and drafted: `build_input` batching and the chart code.
- Found during review: the target outliers (AAM/VBIV), the leakage rules for folds (purging label_end cutoff), and the stuck-lot issue.
- The common-stock screen is based on Nasdaq's 5th-letter symbol convention (not lecture material)