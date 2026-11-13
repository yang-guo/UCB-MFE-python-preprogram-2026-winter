"""Build the feature/label panel once and cache it as Parquet in outputs/ (gitignored).

Usage (from the repo root):
    uv run python Homeworks/HW4/Antonio_Rivera/hw4_build_cache.py dev     # top2000, data cut 2015-12-31
    uv run python Homeworks/HW4/Antonio_Rivera/hw4_build_cache.py final   # top2000, data cut 2026-09-30
    uv run python Homeworks/HW4/Antonio_Rivera/hw4_build_cache.py class   # class DB, 2015-12-31

The `end` cut is applied in SQL, so the dev cache cannot contain any test-period data.
"""

import sys
import time
from pathlib import Path

import hw4_data as hd
import hw4_features as hf

OUT = Path(__file__).parent / "outputs"
TARGETS = {
    "dev": (hd.FULL_DB, "2015-12-31"),
    "final": (hd.FULL_DB, "2026-09-30"),
    "class": (hd.CLASS_DB, "2015-12-31"),
}


def cache_path(name):
    return OUT / f"panel_{name}.parquet"


def build(name):
    database, end = TARGETS[name]
    t0 = time.time()
    panel, prices, calendar = hf.build_feature_panel(database, end=end)
    t1 = time.time()
    hd.audit_label_timing(panel, prices, calendar)        # module-1 timing audit on the real build
    assert panel.date.max() <= hd.pd.Timestamp(end) and panel.label_end.max() <= hd.pd.Timestamp(end)
    OUT.mkdir(exist_ok=True)
    panel.to_parquet(cache_path(name), index=False)
    print(f"{name}: {len(panel):,} rows, {panel.symbol.nunique():,} symbols, "
          f"eligible {int(panel.eligible.sum()):,}; build {t1 - t0:.0f}s, "
          f"audit+save {time.time() - t1:.0f}s -> {cache_path(name)}")


# Final run needs only the frozen model's features (F1) and the strategy's columns.
FINAL_COLS = (["date", "symbol", "adj_close", "volume", "eligible", "mcap_suspect", "sector",
               "target", "label_end", "entry_date"] + hd.cd.NUMERIC_FEATURES)
N_BATCHES = 4


def build_batched(name, n_batches=N_BATCHES, columns=FINAL_COLS):
    """Per-ticker batches -> keep needed columns -> concatenate (bounded memory).

    Every step kept here is per ticker (no same-date peer step), so the result is
    row-for-row identical to an unbatched build (see test_hw4_cache.py).
    """
    import duckdb
    database, end = TARGETS[name]
    with duckdb.connect(str(database), read_only=True) as con:
        symbols = [s for (s,) in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    t0, parts = time.time(), []
    for i, batch in enumerate(hd.np.array_split(symbols, n_batches)):
        panel, prices, calendar = hf.build_feature_panel(database, end=end, symbols=list(batch),
                                                         cross_sectional=False)
        hd.audit_label_timing(panel, prices, calendar)    # timing audit per batch
        parts.append(panel[columns])
        print(f"  batch {i + 1}/{n_batches}: {len(panel):,} rows ({time.time() - t0:.0f}s)", flush=True)
        del panel, prices
    panel = hd.pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"], ignore_index=True)
    assert panel.date.max() <= hd.pd.Timestamp(end) and panel.label_end.max() <= hd.pd.Timestamp(end)
    assert not panel.duplicated(["date", "symbol"]).any()
    OUT.mkdir(exist_ok=True)
    panel.to_parquet(cache_path(name), index=False)
    print(f"{name}: {len(panel):,} rows, {panel.symbol.nunique():,} symbols, "
          f"eligible {int(panel.eligible.sum()):,}; {time.time() - t0:.0f}s -> {cache_path(name)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    build_batched(which) if which == "final" else build(which)
