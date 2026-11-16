"""Regenerate only the two ignored development dependencies needed for replay.

No search is rerun. Existing files are never overwritten. --verify regenerates
the same fixed forecasts in memory and compares them with saved files, allowing
a reproducibility check without deleting any user work.
"""

import argparse

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from development_revision import corrected_data, remove_fluidigm_alias
from hw4.data import build_data
from hw4.modeling import MOMENTUM_SPECS, predict_period
from hw4.settings import DATABASE, PROJECT
from research_models import predict_research
from research_session import load_context
from run_rank_model_stability import FIXED


def persist_or_check(path, generated):
    if path.exists():
        existing = pd.read_csv(path)
        assert list(existing.columns) == list(generated.columns)
        assert len(existing) == len(generated)
        np.testing.assert_allclose(existing.score, generated.score, atol=1e-12, rtol=0)
        assert existing.symbol.tolist() == generated.symbol.tolist()
        assert pd.to_datetime(existing.date).equals(generated.date.reset_index(drop=True))
        print('Reproduced without overwriting:', path.relative_to(PROJECT))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(path, index=False)
        print('Generated missing dependency:', path.relative_to(PROJECT))


def run(verify=False):
    first = PROJECT / 'outputs/sector_caps/continuous_2013_2015/shared_forecasts.csv'
    second = PROJECT / 'outputs/rank_model_stability/V00_fixed_replay/forecasts.csv'
    with threadpool_limits(limits=2):
        if verify or not first.exists():
            data = build_data(DATABASE, end='2015-12-31')
            data, _, _ = corrected_data(data)
            data, _ = remove_fluidigm_alias(data)
            scores, _, _ = predict_period(data.panel, data.quotes.index, MOMENTUM_SPECS[0],
                                          '2013-01-01', '2015-12-31')
            persist_or_check(first, scores)
        if verify or not second.exists():
            context, _ = load_context()
            scores, _, _ = predict_research(context['panel'], context['quotes'].index, FIXED, 1000.0)
            persist_or_check(second, scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args().verify)
