"""Selection-bias analysis. Only 854 of 4,273 movies have a usable label. Does the labelled set differ from the rest in ways that matter?

Method: a logistic model for P(labelled | pre-release covariates) among mature movies. Inverse-probability weights (stabilised, capped) are then
used as a *sensitivity analysis* of model performance. Honest limit: weights can only correct selection explained by the covariates we observe.
If a movie's budget/revenue is reported *because* it did well, no covariate can undo that.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

COVARIATES_NUM = ["runtime", "year", "month_sin", "month_cos", "log_pv30", "n_genres", "n_cast", "n_tech", "has_wiki", "in_yt_file"]
COVARIATES_CAT = ["language"]


def selection_frame(merged, lab):
    cutoff = merged.release_date.max() - pd.Timedelta(days=C.MATURITY_DAYS)
    m = merged[merged.release_date <= cutoff].copy()
    m["is_labelled"] = m.tmdb_id.isin(lab.tmdb_id).astype(int)
    month = m.release_date.dt.month
    m["month_sin"], m["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    m["log_pv30"] = np.log1p(m.pv_30.fillna(0))
    m["n_genres"] = m.genres.fillna("").map(lambda s: len([g for g in s.split(",") if g.strip()]))
    m["n_cast"] = m["cast"].fillna("").map(lambda s: len([g for g in s.split(",") if g.strip()]))
    m["n_tech"] = m.technical_team.fillna("").map(lambda s: len([g for g in s.split(",") if g.strip()]))
    m["has_wiki"] = m.wiki_title.notna().astype(int)
    m["runtime"] = m.runtime.fillna(m.runtime.median())
    return m


def selection_model(m, seed=C.SEED):
    X = pd.concat([m[COVARIATES_NUM], pd.get_dummies(m[COVARIATES_CAT], drop_first=True).astype(float)], axis=1)
    y = m.is_labelled.values
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    p = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    pipe.fit(X, y)
    coefs = pd.Series(pipe[-1].coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
    return p, auc, coefs


def ipw_weights(m, p, cap=10.0):
    """Stabilised inverse-probability weights for the labelled movies: P(labelled) / P(labelled | x), capped. Mean ~1."""
    lab_mask = m.is_labelled.values == 1
    w = np.where(lab_mask, y_share(m) / np.clip(p, 1e-3, 1), np.nan)
    w = np.minimum(w, cap)
    out = pd.Series(w[lab_mask], index=m.tmdb_id[lab_mask].values)
    return out / out.mean()


def y_share(m):
    return m.is_labelled.mean()


def weighted_hit_rate(lab, w, by=None):
    d = lab.assign(w=lab.tmdb_id.map(w))
    if by is None:
        return {"unweighted": float(d.hit.mean()), "weighted": float(np.average(d.hit, weights=d.w))}
    g = d.groupby(by).apply(lambda x: pd.Series({"n": len(x), "unweighted": x.hit.mean(), "weighted": np.average(x.hit, weights=x.w)}), include_groups=False)
    return g


def effective_sample_size(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def weighted_auc(y, p, w):
    return float(roc_auc_score(y, p, sample_weight=w))
