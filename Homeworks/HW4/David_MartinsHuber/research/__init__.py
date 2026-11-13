"""Research code for the HW4 HMM regime study.

The package is deliberately separate from ``backtesting`` and ``strategies``:
it builds causal research features, runs walk-forward fits, and writes
diagnostics.  Strategy code can import the saved state series later.

The volume-bar study and the FFD study are independent samples; see
``pipeline.py``.
"""

from .bars import VolumeBars, volume_bars
from .fracdiff import adf_table, ffd, ffd_weights, select_d
from .hmm import (
    HMMFit,
    WalkForwardResult,
    fit_hmm,
    walk_forward_ffd_states,
    walk_forward_states,
)
from .market import MarketSeries, build_market_series, load_market_cap_frame
from .pairs import PairsConfig, PairsResearch, PairsResult
from .pipeline import (
    FFDResearch,
    MarketResearch,
    ResearchConfig,
    VolumeResearch,
    forward_return,
    run_market,
    state_summary,
    transition_matrix,
)

__all__ = [
    "MarketSeries",
    "build_market_series",
    "load_market_cap_frame",
    "PairsConfig",
    "PairsResearch",
    "PairsResult",
    "VolumeBars",
    "volume_bars",
    "ffd_weights",
    "ffd",
    "adf_table",
    "select_d",
    "HMMFit",
    "WalkForwardResult",
    "fit_hmm",
    "walk_forward_states",
    "walk_forward_ffd_states",
    "ResearchConfig",
    "VolumeResearch",
    "FFDResearch",
    "MarketResearch",
    "run_market",
    "state_summary",
    "forward_return",
    "transition_matrix",
]
