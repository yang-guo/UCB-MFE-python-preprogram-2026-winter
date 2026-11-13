# Research study summary (development data only; final test untouched)

Plan: `experiments/research/PLAN.md`.  Development backtest window 2012-01-03 to 2015-12-31; walk-forward IC on 2012 to 2015 signals.

## Stage 0: random 30-name portfolios

- 20 random portfolios: mean CAGR 12.15%, standard deviation 0.80%, range 9.78% to 13.52%.
- S&P 500 total return over the same window: CAGR 14.89%, max drawdown -12.04%, Sharpe 1.00.
- Random portfolios: mean Sharpe 0.77, mean max drawdown -14.26%.
- Universe effect (random mean minus S&P 500): -2.75% per year.

## Stage 1: single-feature rank IC by signal year (descriptive)

| feature | group | plan_sign | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | mean_IC | tstat_13wk_blocks | years_same_sign_as_mean | coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| operating_roa | financial | + | 0.054 | 0.117 | 0.004 | -0.021 | 0.054 | 0.084 | 0.049 | 3.048 | 5 | 0.999 |
| operating_roa_sr | sector_relative |  | 0.020 | 0.111 | 0.015 | -0.032 | 0.069 | 0.100 | 0.047 | 2.955 | 5 | 0.999 |
| gross_profitability | financial | + | 0.053 | 0.080 | -0.009 | 0.050 | 0.036 | 0.058 | 0.045 | 3.603 | 5 | 0.999 |
| vol_63 | price | - | 0.044 | -0.085 | -0.029 | 0.077 | -0.114 | -0.159 | -0.045 | -1.264 | 4 | 1.000 |
| vol_63_sr | sector_relative |  | 0.011 | -0.077 | -0.010 | 0.045 | -0.102 | -0.128 | -0.044 | -1.720 | 4 | 1.000 |
| fcf_to_assets | financial |  | 0.018 | 0.087 | -0.012 | -0.015 | 0.063 | 0.087 | 0.038 | 3.144 | 4 | 0.997 |
| drawdown_252 | price |  | -0.045 | 0.039 | 0.031 | -0.016 | 0.060 | 0.138 | 0.035 | 1.288 | 4 | 0.981 |
| net_issuance_yoy | financial_trend | - | -0.002 | -0.075 | -0.024 | -0.015 | -0.037 | -0.048 | -0.034 | -3.057 | 6 | 0.993 |
| gross_profitability_sr | sector_relative |  | 0.020 | 0.078 | 0.002 | 0.007 | 0.025 | 0.066 | 0.033 | 2.885 | 6 | 0.999 |
| mom_252_21_sr | sector_relative |  | -0.008 | 0.016 | 0.065 | 0.028 | -0.003 | 0.063 | 0.027 | 1.941 | 4 | 0.973 |
| cash_to_assets | financial |  | -0.004 | -0.040 | -0.078 | 0.065 | -0.059 | -0.036 | -0.025 | -1.454 | 5 | 0.999 |
| mom_252_21 | price | + | -0.008 | -0.006 | 0.071 | 0.040 | -0.027 | 0.071 | 0.024 | 1.358 | 3 | 0.973 |
| debt_to_assets | financial |  | 0.021 | 0.028 | 0.096 | -0.054 | 0.067 | -0.029 | 0.022 | 1.416 | 4 | 0.999 |
| log_mcap_proxy | size |  | -0.011 | 0.037 | 0.007 | -0.054 | 0.092 | 0.055 | 0.021 | 1.154 | 4 | 1.000 |
| roa_yoy_change | financial_trend | + | 0.031 | 0.010 | 0.021 | 0.015 | 0.013 | 0.034 | 0.020 | 2.333 | 6 | 0.983 |
| net_margin | financial |  | -0.015 | 0.066 | -0.004 | -0.060 | 0.030 | 0.102 | 0.020 | 1.030 | 3 | 0.988 |
| revenue_growth_yoy | financial_trend | + | 0.073 | -0.016 | 0.012 | 0.024 | -0.000 | 0.026 | 0.019 | 2.020 | 4 | 0.977 |
| earnings_yield_ttm | financial_trend | + | 0.015 | 0.059 | -0.005 | -0.029 | 0.035 | 0.033 | 0.018 | 1.306 | 4 | 0.933 |
| book_to_market | financial | + | -0.025 | -0.044 | 0.028 | -0.042 | -0.009 | -0.013 | -0.018 | -1.382 | 5 | 0.999 |
| book_to_market_sr | sector_relative |  | -0.013 | -0.052 | 0.024 | -0.019 | -0.010 | -0.027 | -0.016 | -1.310 | 5 | 0.999 |
| filing_age_days | financial |  | -0.003 | 0.024 | 0.015 | -0.000 | 0.020 | 0.026 | 0.014 | 2.473 | 4 | 1.000 |
| log_dollar_volume_63 | liquidity |  | -0.003 | 0.025 | 0.015 | -0.039 | 0.078 | 0.008 | 0.014 | 1.054 | 4 | 1.000 |
| accruals_to_assets | financial | - | 0.000 | -0.019 | 0.038 | -0.011 | -0.019 | 0.029 | 0.003 | 0.439 | 3 | 0.997 |
| ret_21_sr | sector_relative |  | -0.019 | 0.005 | -0.015 | -0.002 | -0.007 | 0.021 | -0.003 | -0.283 | 4 | 1.000 |
| ret_63 | price |  | -0.048 | -0.002 | -0.029 | 0.012 | 0.013 | 0.040 | -0.002 | -0.145 | 3 | 0.999 |
| asset_growth_yoy | financial_trend | - | 0.027 | 0.005 | -0.039 | 0.006 | 0.003 | 0.007 | 0.001 | 0.146 | 5 | 0.983 |
| ret_21 | price | - | -0.012 | 0.003 | -0.018 | -0.002 | 0.011 | 0.013 | -0.001 | -0.079 | 3 | 1.000 |
| ret_63_sr | sector_relative |  | -0.042 | 0.001 | -0.016 | 0.010 | 0.008 | 0.042 | 0.001 | 0.042 | 4 | 0.999 |

## Stage 2: alpha candidates

| experiment | validation_mean_IC | walk_forward_IC | CAGR | benchmark_CAGR | Sharpe | max_drawdown | universe_effect | selection_effect | selection_in_random_sd | turnover_x_avg_nav | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_composite_core | 0.0510 | 0.0501 | 0.1442 | 0.1489 | 0.9949 | -0.0890 | -0.0275 | 0.0228 | 2.8353 | 14.2639 | 17/17 |
| R2_composite_trend | 0.0546 | 0.0538 | 0.1643 | 0.1489 | 1.1352 | -0.0965 | -0.0275 | 0.0429 | 5.3433 | 14.4270 | 17/17 |
| R3_ridge_12 | 0.0385 | 0.0356 | 0.1646 | 0.1489 | 1.0981 | -0.1017 | -0.0275 | 0.0431 | 5.3714 | 14.2563 | 17/17 |
| R4_ridge_21 | 0.0324 | 0.0316 | 0.1895 | 0.1489 | 1.2778 | -0.0948 | -0.0275 | 0.0680 | 8.4779 | 13.4182 | 17/17 |
| R5_ridge_tuned_21 | 0.0323 | 0.0317 | 0.1915 | 0.1489 | 1.2970 | -0.0944 | -0.0275 | 0.0700 | 8.7262 | 13.4099 | 17/17 |

Selection rule applied in order of complexity:

| candidate | complexity | mean_IC_2012_2015 | IC_2012 | IC_2013 | IC_2014 | IC_2015 | years_positive | stable | compared_with | mean_IC_difference | bootstrap_p05_of_difference | bootstrap_share_below_zero | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_composite_core | 0 | 0.0506 | 0.0323 | -0.0125 | 0.0644 | 0.1197 | 3 | True |  |  |  |  | becomes incumbent (first stable candidate) |
| R2_composite_trend | 1 | 0.0543 | 0.0444 | 0.0006 | 0.0638 | 0.1095 | 4 | True | R1_composite_core | 0.0037 | -0.0078 | 0.2962 | rejected: not better than R1_composite_core beyond noise |
| R3_ridge_12 | 2 | 0.0362 | 0.0064 | -0.0042 | 0.0229 | 0.1209 | 3 | True | R1_composite_core | -0.0144 | -0.0317 | 0.9222 | rejected: not better than R1_composite_core beyond noise |
| R4_ridge_21 | 3 | 0.0323 | 0.0571 | -0.0380 | 0.0341 | 0.0768 | 3 | True | R1_composite_core | -0.0183 | -0.0369 | 0.9362 | rejected: not better than R1_composite_core beyond noise |
| R5_ridge_tuned_21 | 4 | 0.0323 | 0.0561 | -0.0380 | 0.0349 | 0.0770 | 3 | True | R1_composite_core | -0.0183 | -0.0368 | 0.9382 | rejected: not better than R1_composite_core beyond noise |

**Selected alpha by the rule: R1_composite_core**

## Stage 3: portfolio variants of the selected alpha

| experiment | validation_mean_IC | walk_forward_IC | CAGR | benchmark_CAGR | Sharpe | max_drawdown | universe_effect | selection_effect | selection_in_random_sd | turnover_x_avg_nav | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_composite_core | 0.0510 | 0.0501 | 0.1442 | 0.1489 | 0.9949 | -0.0890 | -0.0275 | 0.0228 | 2.8353 | 14.2639 | 17/17 |
| P1_top50__R1 | 0.0510 | 0.0501 | 0.1537 | 0.1489 | 1.0648 | -0.0903 | -0.0275 | 0.0322 | 4.0161 | 14.4386 | 17/17 |
| P2_top100__R1 | 0.0510 | 0.0501 | 0.1500 | 0.1489 | 1.0364 | -0.0908 | -0.0275 | 0.0285 | 3.5572 | 14.5407 | 17/17 |
| P3_inverse_vol__R1 | 0.0510 | 0.0501 | 0.1522 | 0.1489 | 1.1108 | -0.0816 | -0.0275 | 0.0308 | 3.8363 | 14.3169 | 17/17 |

| experiment | CAGR | CAGR_minus_base | random_CAGR_sd | beats_base_beyond_noise | max_drawdown | benchmark_max_drawdown | drawdown_no_worse_than_benchmark | drawdown_minus_base | Sharpe | passes_rule |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_top50__R1 | 0.1537 | 0.0095 | 0.0080 | True | -0.0903 | -0.1204 | True | -0.0013 | 1.0648 | True |
| P2_top100__R1 | 0.1500 | 0.0058 | 0.0080 | False | -0.0908 | -0.1204 | True | -0.0019 | 1.0364 | False |
| P3_inverse_vol__R1 | 0.1522 | 0.0080 | 0.0080 | True | -0.0816 | -0.1204 | True | 0.0074 | 1.1108 | True |

**Portfolio variant adopted by the rule: P1_top50__R1**

Run time 5 minutes.  Nothing was frozen.
