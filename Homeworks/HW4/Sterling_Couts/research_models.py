"""Causal feature and fitting helpers for the bounded model research session.

Adapted from Lectures 7/8 and the student's frozen hw4 modules. All model
selection uses completed pre-fit labels. Portfolio evaluation remains a
separate calculation; a better squared-error forecast need not trade better.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from hw4.modeling import ModelSpec, make_model, training_rows


@dataclass(frozen=True)
class ResearchSpec:
    name: str
    features: tuple[str, ...]
    schedule: str = "fixed"
    feature_clip: bool = True
    estimator: str = "ridge"
    include_sector: bool = False
    target: str = "return"
    rule: str | None = None

    def as_dict(self):
        return asdict(self)


def make_research_model(spec, alpha):
    """Learn every transform within the current fit; optional sector encoding."""
    base = make_model(
        ModelSpec(
            spec.name,
            spec.features,
            alpha=alpha,
            target_quantiles=(0.01, 0.99),
            risk_screen=True,
            estimator=spec.estimator,
        )
    )
    if not spec.feature_clip:
        base.set_params(clip="passthrough")
    if spec.include_sector:
        prep = ColumnTransformer(
            [
                ("numeric", base[:-1], list(spec.features)),
                (
                    "sector",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["sector"],
                ),
            ]
        )
        return Pipeline([("prepare", prep), ("regression", base["regression"])])
    return base


def input_columns(spec):
    return list(spec.features) + (["sector"] if spec.include_sector else [])


def eligible_panel(panel):
    result = panel.copy()
    result["eligible"] &= result.risk_screen_pass
    return result


def model_target(train, spec):
    if spec.target == "return":
        return train.target.to_numpy(), 0.0
    if spec.target == "excess":
        # Benchmark outcome is a label, never an input. Its training mean is a
        # simple ex-ante market-return estimate added back to excess forecasts.
        assert train.benchmark_target.notna().all()
        return (train.target - train.benchmark_target).to_numpy(), float(
            train.benchmark_target.mean()
        )
    raise ValueError("Unknown training target")


def choose_alpha(panel, calendar, spec, selection_date="2012-12-31"):
    """Lecture grid, with 2011/12 entry-year folds known before initial fitting.

    Validation outcomes completed after selection_date cannot select alpha.
    All stocks at a signal date remain in the same fold. The final refit is
    performed separately, after selecting solely from these earlier folds.
    """
    selection_date = pd.Timestamp(selection_date)
    p = eligible_panel(panel)
    rows = []
    for year in [2011, 2012]:
        asof = calendar[calendar.searchsorted(pd.Timestamp(year, 1, 1)) - 1]
        train = training_rows(p, asof)
        valid = p.loc[
            p.eligible
            & p.entry_date.dt.year.eq(year)
            & p.label_end.lt(selection_date)
            & p.target.notna()
        ]
        assert len(train) >= 20 and len(valid) >= 20
        assert train.label_end.max() < asof <= valid.date.min()
        target, offset = model_target(train, spec)
        dummy = float(train.target.mean())
        for alpha in [1.0, 100.0, 1000.0]:
            model = make_research_model(spec, alpha)
            model.fit(train[input_columns(spec)], target)
            pred = model.predict(valid[input_columns(spec)]) + offset
            err = pred - valid.target.to_numpy()
            rows.append(
                {
                    "fold": year,
                    "alpha": alpha,
                    "training_rows": len(train),
                    "validation_rows": len(valid),
                    "fit_date": asof,
                    "last_training_label": train.label_end.max(),
                    "last_validation_label": valid.label_end.max(),
                    "RMSE": float(np.sqrt(np.mean(err**2))),
                    "MAE": float(np.mean(np.abs(err))),
                    "training_mean_RMSE": float(
                        np.sqrt(np.mean((dummy - valid.target) ** 2))
                    ),
                }
            )
    table = pd.DataFrame(rows)
    means = table.groupby("alpha").RMSE.mean()
    best = means.loc[means.eq(means.min())].index.max()
    return float(best), table


def predict_research(
    panel, calendar, spec, alpha, start="2013-01-01", end="2015-12-31"
):
    """Expanding fits and fresh monthly predictions, with exact label maturity."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    p = eligible_panel(panel)
    first = calendar.searchsorted(start)
    fits = [calendar[first - 1]]
    if spec.schedule not in ["fixed", "annual", "quarterly"]:
        raise ValueError("Unknown fitting schedule")
    if spec.schedule != "fixed":
        freq = "Y" if spec.schedule == "annual" else "Q"
        fits += [
            calendar[i - 1]
            for i in range(first + 1, len(calendar))
            if calendar[i] <= end
            and calendar[i].to_period(freq) != calendar[i - 1].to_period(freq)
        ]
    scores, audit, models = [], [], []
    for i, asof in enumerate(fits):
        stop = fits[i + 1] if i + 1 < len(fits) else end + pd.Timedelta(days=1)
        candidates = p.loc[
            p.eligible
            & p.date.ge(asof)
            & p.date.lt(stop)
            & p.entry_date.ge(start)
            & p.entry_date.le(end)
        ]
        if candidates.empty:
            continue
        train = training_rows(p, asof)
        target, offset = model_target(train, spec)
        model = make_research_model(spec, alpha)
        model.fit(train[input_columns(spec)], target)
        pred = model.predict(candidates[input_columns(spec)]) + offset
        observed = candidates[["date", "symbol", "target", "label_end"]].copy()
        observed["score"], observed["fit_date"] = pred, asof
        # A missing future target must not remove a candidate prediction.
        assert len(observed) == len(candidates) and np.isfinite(pred).all()
        reg = model["regression"].estimator_
        audit.append(
            {
                "variant": spec.name,
                "fit_date": asof,
                "alpha": alpha,
                "training_rows": len(train),
                "first_training_signal": train.date.min(),
                "last_training_signal": train.date.max(),
                "last_training_label": train.label_end.max(),
                "first_prediction": candidates.date.min(),
                "last_prediction": candidates.date.max(),
                "prediction_rows": len(candidates),
                "target_lower": model["regression"].target_bounds_[0],
                "target_upper": model["regression"].target_bounds_[1],
                "market_offset": offset,
                "coefficients": str(
                    getattr(reg, "coef_", []).tolist() if hasattr(reg, "coef_") else []
                ),
                "intercept": float(getattr(reg, "intercept_", np.nan)),
            }
        )
        assert train.label_end.max() < asof <= candidates.date.min()
        scores.append(observed)
        models.append((asof, model))
    return pd.concat(scores, ignore_index=True), pd.DataFrame(audit), models


def add_market_features(panel, quotes, benchmark, sectors):
    """Trailing signals using only each row's signal close and earlier prices.

    Six-month and peer-relative features follow Lecture 7 exercises. Near-high,
    downside volatility and simple blends are additional hypotheses. Current
    vendor sectors are static and remain a disclosed data limitation.
    """
    p = panel.copy()
    dates = quotes.index.get_indexer(pd.DatetimeIndex(p.date))
    symbols = quotes.columns.get_indexer(p.symbol)
    assert (dates >= 0).all() and (symbols >= 0).all()

    def add(name, frame):
        p[name] = frame.to_numpy()[dates, symbols]

    ret = quotes.pct_change(fill_method=None)
    add("momentum_21", quotes / quotes.shift(21) - 1)
    add("momentum_126_21", quotes.shift(21) / quotes.shift(126) - 1)
    add("drawdown_252", quotes / quotes.rolling(252, min_periods=252).max() - 1)
    add("volatility_21", ret.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252))
    add(
        "downside_volatility_63",
        np.sqrt(ret.clip(upper=0).pow(2).rolling(63, min_periods=63).mean() * 252),
    )
    p["volatility_acceleration"] = p.volatility_21 / p.volatility_63
    p["sector"] = p.symbol.map(sectors)
    assert p.sector.notna().all()
    # Cross-sectional features use eligible members known on the SAME date.
    mask = p.eligible & p.risk_screen_pass
    for feature in [
        "momentum_21",
        "momentum_63",
        "momentum_126_21",
        "momentum_252_21",
        "volatility_63",
        "drawdown_252",
        "operating_roa",
        "fcf_to_assets",
    ]:
        value = p[feature].where(mask)
        p[f"rank_{feature}"] = value.groupby(p.date).rank(pct=True)
    for feature in ["momentum_63", "momentum_252_21", "operating_roa"]:
        value = p[feature].where(mask)
        group = value.groupby([p.date, p.sector])
        median, count = group.transform("median"), group.transform("count")
        p[f"sector_relative_{feature}"] = (value - median).where(count.ge(10))
    # Extra target, not a predictor: calculate benchmark over each observed
    # stock label's exact entry and realized exit dates.
    entry = p.entry_date.map(benchmark)
    exit_price = p.label_end.map(benchmark)
    p["benchmark_target"] = exit_price / entry - 1
    new = [c for c in p if c not in panel]
    numeric = p[new].select_dtypes(include="number")
    assert not np.isinf(numeric.to_numpy()).any()
    return p


def score_diagnostics(scores):
    rows = []
    for date, d in scores.groupby("date"):
        ranked = d.sort_values(["score", "symbol"], ascending=[False, True])
        top = ranked.head(20)
        value = top.score.iloc[-1]
        rows.append(
            {
                "date": date,
                "candidate_count": len(d),
                "max_score_ties": int(d.score.eq(d.score.max()).sum()),
                "top20_distinct_scores": top.score.nunique(),
                "alphabetical_cutoff": int(d.score.eq(value).sum())
                > int(top.score.eq(value).sum()),
            }
        )
    return pd.DataFrame(rows)
