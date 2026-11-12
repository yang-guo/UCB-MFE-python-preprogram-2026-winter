"""Select regularization on development years; freeze before the final test."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .data import course


def training_rows(panel, fit_date):
    return panel.loc[panel.date.ge("2010-01-01") & panel.date.lt(fit_date)
                     & panel.label_end.lt(fit_date) & panel.target.notna()
                     & panel.date.le("2015-12-31")].copy()


def fit(panel, fit_date, alpha, features):
    train = training_rows(panel, fit_date)
    if train.empty:
        raise ValueError("No completed training outcomes")
    assert train.label_end.max() < pd.Timestamp(fit_date)
    model = course.make_pipeline(Ridge(alpha=alpha), features, include_sector=False)
    # Clip extreme TRAINING targets only, using thresholds learned from training.
    # Raw target and prices remain intact in the audit data and backtest.
    target = train.target.clip(*train.target.quantile([.01, .99]).to_list())
    model.fit(train[features], target)
    audit = {"fit_date": str(fit_date), "alpha": float(alpha), "training_rows": len(train),
             "latest_training_signal": str(train.date.max()), "latest_label_end": str(train.label_end.max())}
    return model, audit


def select(panel, config):
    rows, fits = [], []
    for alpha in config["ridge_alphas"]:
        for year in (2013, 2014, 2015):
            cutoff = pd.Timestamp(year, 1, 1)
            model, audit = fit(panel, cutoff, alpha, config["features"])
            fits.append(audit)
            valid = panel.loc[panel.date.dt.year.eq(year) & panel.label_end.le("2015-12-31") & panel.target.notna()]
            if valid.empty:
                raise ValueError(f"No completed validation outcomes in {year}")
            valid = valid.assign(prediction=model.predict(valid[config["features"]]))
            ic = []
            top = []
            for _, group in valid.groupby("date"):
                if group.prediction.nunique() > 1 and group.target.nunique() > 1:
                    ic.append(group.prediction.corr(group.target, method="spearman"))
                top.append(group.nlargest(config["top_n"], "prediction").target.mean())
            rows.append({"alpha": alpha, "year": year, "rank_IC": np.mean(ic),
                         "top_picks_mean_gross_return": np.mean(top), "rows": len(valid)})
    validation = pd.DataFrame(rows)
    averages = validation.groupby("alpha").rank_IC.mean()
    winner = float(averages.idxmax())
    return winner, validation, fits
