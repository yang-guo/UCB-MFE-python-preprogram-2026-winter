# HW4 code map

**Setup (fresh clone):** run `uv sync --locked` at the repo root, then download `equities_top2000.duckdb` and `equities_class.duckdb` from bCourses into `Lectures/modeling/data/`. Run every command from the repo root on Windows (on macOS/Linux use `.venv/bin/python`). Generated files go to `Homeworks/HW4/Ryan_Beauchamp/outputs/`, which git ignores. The submitted results are in `results/`. Times and memory are from this machine (8 cores, 16 GB); "n/m" means not measured.

**Already saved, so you can skip the step:** the frozen final run and its diagnostics are in `results/`: metrics, chart, checks, fit log, stale-quote positions, jump flags, config, tables D1–D8 and the test output. **Don't rerun the final test** (step 15). It was run once, from the frozen config; its config hash is in `results/accounting_checks.json`. Steps 1–14 exist to reproduce the development work. Their outputs aren't in `results/`, but their numbers are quoted in `outputs/writeup_notes.md`. **Run B** (step 15b) is exploratory, run after seeing Run A, and is not the headline. It was also run once, and its summary files are in `results/run_b/`; you can skip it too.

Prefix every command with `.venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/`.

| # | Script (+ args) | What it does, and why | Needs | Writes (under `outputs/` unless noted) | Time / peak mem |
|---|---|---|---|---|---|
| 0 | `test_backtester.py` | 19 hand-checked tests of the backtester's accounting and timing; run these before trusting any backtest. | nothing | console (saved copy: `results/test_output.txt`) | seconds |
| 1 | `stage1_panel.py` | Development panel: features, quarterly membership and 90-day targets, built in ticker batches, reading no data after 2016-03-31. Checks AAPL and compares against the class database. | top2000 + class `.duckdb` | `panel_stage1/`, `stage1_*.csv` | ~6 min / ~2 GB |
| 2 | `stage2_validation.py` | First chronological folds (2012–2015) and the unscreened Ridge baseline. Exposed the broken price series. | 1 | `stage2_*.csv` | ~3.5 min / n/m |
| 3 | `stage2b_clean_flag.py` | Causal clean-flag measures (largest daily move over 252 sessions, price floor), exclusion counts, prefix tests. | 1, top2000 | `clean_measures/`, `stage2b_*.csv` | ~1.5 min / n/m |
| 4 | `stage2c_clean_validation.py` | Refits the folds on clean rows: clipped-return Ridge vs rank-target Ridge vs training mean. | 1, 3 | `stage2c_*.csv` | ~13 min / n/m |
| 5 | `stage3_dry_run.py` | Smoke test of the backtester on 2015 with rank-target scores. | 1, 3 | `stage3_*` | ~1.2 min / n/m |
| 6 | `stage4_dev_grid.py` | Walk-forward scores (once, annual, quarterly) and the 8-setting strategy grid on 2012–2015. Logs every run. | 1, 3, 5 | `stage4_*` | ~10 min / 3.1 GB |
| 7 | `stage4d_hgb.py` | Optional: gradient boosting vs Ridge, time-boxed. | 6 | `stage4d_*` | ~8.5 min / n/m |
| 8 | `garp_features.py` | GARP features, keyed on (date, symbol). Checks AAPL by hand, compares EV with the vendor's, runs a prefix test. | 1, 3, both `.duckdb` | `garp_dev/`, `garp_dev_*.csv` | ~6 min / 6.1 GB |
| 9 | `stage5_garp_experiment.py` | Pre-registered GARP adoption test (result: not adopted) and fee sensitivity. | 6, 8 | `stage5_*` | ~4.5 min / 4.4 GB |
| 10 | `stage1_panel.py --end 2026-09-30 --name panel_full` | Same panel code through 2026-09-30 (data preparation only). | top2000 + class | `panel_full/`, `panel_full_*.csv` | ~11 min / 2.1 GB |
| 11 | `stage2b_clean_flag.py --full` | Clean-flag measures for the full period. Checks they equal the development measures. | 3, 10 | `clean_measures_full/`, `full_*.csv` | ~9 min / 5.2 GB |
| 12 | `garp_features.py --full` | Full-period GARP file. Run A doesn't use it (GARP not adopted); **Run B (15b) needs it**. | 8, 10, 11 | `garp_full/` | ~9 min / 10.7 GB |
| 13 | `final_run.py --smoke` | The final-run code on development data only (2013-06-28 to 2015-12-31). Allowed while the config is DRAFT. | 1, 3, 8 | `final_smoke/` | ~1.5 min / 3.8 GB |
| 14 | — | `frozen_config.json` (not a script) holds every frozen choice. `final_run.py` refuses to start unless its status is FROZEN and `--i-confirm-frozen` is passed. | | | |
| 15 | `run_final_with_log.py` | **Run A, the frozen headline. Done once; don't rerun.** Runs `final_run.py --i-confirm-frozen`, showing output live and saving it to `final_run.log`. | 10, 11, `frozen_config.json` (FROZEN) | `final/`, `final_run.log` | 7.5 min / 10.2 GB |
| 15b | `run_b.py` | **Run B: exploratory, run after seeing Run A. Not the headline. Done once; don't rerun.** Same frozen settings as Run A, but the model uses only 6 GARP columns (fcf_yield, ebitda_yield, earnings_yield, rev_growth_ttm, eps_change_yield, op_margin_ttm), with no sector and no baseline features. Writes only to `run_b/`; also reads Run A's `final/history.csv` and `final/metrics.csv` (read-only) for the comparison table. `run_b.py --smoke` checks the code on development data first (~1 min / 4.1 GB, writes `run_b_smoke/`). | 10, 11, 12, `config_run_b.json`, 15 | `run_b/` (copied by hand to `results/run_b/`); log `run_b.log` | ~6 min / 9.7 GB |
| 16 | `final_diagnostics.py` | Post-hoc, read-only diagnostics D1–D8 of the frozen run. Not used to change anything. | 6, 10, 15 | `final_diagnostics/` | ~1–2 min / n/m |

`results/` was filled by copying the small files from `outputs/final/` and `outputs/final_diagnostics/`, plus `frozen_config.json`. `results/run_b/` was filled the same way, from `outputs/run_b/` plus `config_run_b.json`. No script does these copies.

**Shared modules (not run directly):**
- `backtester.py`: the strategy, account, daily loop and analytics.
- `walkforward.py`: the clean-row loader, refit schedule and walk-forward scoring used by both development (6, 9) and the final runs (13, 15, 15b), so the test used exactly the development procedure. Run B reuses both modules unchanged.

**Lecture code reused or adapted:**
- **`Lectures/modeling/course_data.py`**, imported unmodified:
  - `stage1_panel.py` uses `load_data`, `price_features`, `financial_events`, `join_financials`, `load_membership`, `apply_membership` and `make_labels`. It also uses `prepare_panel` for the class-database check.
  - `stage2_validation.py` and `stage2c_clean_validation.py` use `date_folds`, `make_pipeline`, `prediction_metrics` and `FEATURES`.
  - `walkforward.py` uses `make_pipeline`, `FEATURES` and `NUMERIC_FEATURES`.
  - `garp_features.py` follows the same availability rules (filing date + 1 day, as-of join, 550-day staleness), but its code is new.
- **`Lectures/modeling/backtest.py`**: `backtester.py` follows its split into strategy, order book, loop and analytics. No code was copied.
- Every script's header credits this reuse and notes AI assistance (Claude Code). Nothing under `Lectures/` was modified.
