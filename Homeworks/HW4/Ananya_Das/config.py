"""Every date, path, and setting for HW4, in one place.

Other modules import from here instead of typing dates or numbers inline, so a
change (for example, switching to the full database) happens in one spot.
"""

from pathlib import Path

import pandas as pd

# --- Paths -------------------------------------------------------------------
HW_DIR = Path(__file__).resolve().parent
REPO_ROOT = HW_DIR.parents[2]
DATA_DIR = REPO_ROOT / "Lectures" / "modeling" / "data"
CLASS_DB = DATA_DIR / "equities_class.duckdb"      # 30 stocks, 2008-2015: for debugging
FULL_DB = DATA_DIR / "equities_top2000.duckdb"     # 3,849 stocks, 2000-Sep 2026: final run
DATABASE = CLASS_DB                                # Switch to FULL_DB for the real experiment.
OUTPUT_DIR = HW_DIR / "outputs"                    # Ignored by Git.

# --- Date splits (see hw4.md section 2) --------------------------------------
WARMUP_START = pd.Timestamp("2000-01-01")  # Earlier data: feature lookbacks only.
DEV_START = pd.Timestamp("2010-01-01")     # Development: train, validate, choose.
DEV_END = pd.Timestamp("2015-12-31")       # Last allowed development signal.
BUFFER_END = pd.Timestamp("2016-03-31")    # Lets 90-day targets from 2015 signals finish.
TEST_START = pd.Timestamp("2016-04-01")    # Final test: only when Ananya says so.
TEST_END = pd.Timestamp("2026-09-30")      # No prices after this date, ever.

# --- Final-test switch -------------------------------------------------------
# While False, data.py refuses any request that reaches past BUFFER_END.
# Do NOT change this until the strategy is frozen and Ananya explicitly
# approves the final test run.
RUN_FINAL_TEST = False

# --- Strategy and accounting settings (agreed design; see CLAUDE.md) ---------
HOLDING_DAYS = 90            # Calendar days per purchase.
INITIAL_CAPITAL = 1_000_000  # Dollars; long only, no leverage.
FEE_BPS = 10                 # Cost per side, in basis points (0.10%).
FEE_SENSITIVITY_BPS = (5, 20)
TOP_N = 20                   # Stocks bought per monthly cohort.
COHORT_FRACTION = 1 / 3      # Each cohort spends min(cash, NAV / 3).
NAME_CAP = 0.05              # Max 5% of NAV in one stock.
SECTOR_CAP = 0.25            # Max 25% of NAV in one sector.
MIN_PRICE = 5.0              # Raw close on the signal date must be at least $5.
MAX_STALE_SESSIONS = 5       # Longest a held stock may go without a quote before it is frozen.
ANNUAL_RISK_FREE = 0.02      # For the Sharpe ratio.
CASH_INTEREST = 0.0          # Idle cash earns nothing (conservative).
