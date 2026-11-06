"""Source-supported mandatory stock exchanges; no invented quotes or exits."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def load_actions(database, definitions=None):
    actions = (json.loads(Path(__file__).with_suffix(".json").read_text())
               if definitions is None else [dict(action) for action in definitions])
    with duckdb.connect(str(database), read_only=True) as con:
        for action in actions:
            old = con.execute("""SELECT date,close,adj_close FROM prices
                WHERE symbol=? AND date<? ORDER BY date DESC LIMIT 1""",
                [action["old_symbol"], action["effective_date"]]).fetchone()
            if old is None or not all(np.isfinite(x) and x > 0 for x in [old[1], old[2]]):
                raise ValueError(f"Invalid old adjusted/raw basis for {action['old_symbol']}")
            action["old_quote_date"] = str(old[0])
            action["old_adjustment_factor"] = old[2] / old[1]
            if action["kind"] == "cash":
                action["cash_per_adjusted_unit"] = action["old_adjustment_factor"] * action["cash_per_share"]
                continue
            new = con.execute("""SELECT date,close,adj_close FROM prices
                WHERE symbol=? AND date=?""",
                [action["new_symbol"], action["effective_date"]]).fetchone()
            if old is None or new is None:
                raise ValueError(f"Missing merger conversion basis for {action['old_symbol']}")
            if not all(np.isfinite(x) and x > 0 for x in [old[1], old[2], new[1], new[2]]):
                raise ValueError("Invalid raw/adjusted price basis for mandatory conversion")
            # u adjusted units = u * (adjusted/raw) actual-share equivalents.
            # Apply the official share exchange, then convert into successor
            # adjusted units. These are accounting factors, not fabricated fills.
            action["new_adjustment_factor"] = new[2] / new[1]
            action["adjusted_unit_ratio"] = (
                action["old_adjustment_factor"] * action["share_exchange_ratio"]
                / action["new_adjustment_factor"]
            )
            action["effective_date"] = str(pd.Timestamp(action["effective_date"]).date())
    return actions
