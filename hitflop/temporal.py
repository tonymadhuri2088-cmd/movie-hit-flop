"""Upgrade Phase 3: training-window choices for a market that changes.

expanding   the original study: train on every earlier film
rolling-N   train only on the N most recent earlier films (forgets the distant past completely)
decay-Hy    train on every earlier film but give a film released t years before the end of the window the weight 0.5 ** (t / H)
            (a film H years old counts half as much; weights come from release dates only, never from labels)

Validation blocks, inner tuning grids and scoring are the same as in the original study.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config as C
from . import models as MD

DAYS_PER_YEAR = 365.25


def decay_weights(dates, half_life_years):
    """Weight 0.5 ** (age / half-life), age measured back from the newest training film; rescaled to mean 1 so regularisation strength is comparable."""
    d = np.asarray(dates, "datetime64[D]")
    age = (d.max() - d).astype(float) / DAYS_PER_YEAR
    w = 0.5 ** (age / half_life_years)
    return w / w.mean()


def parse_spec(spec):
    """'expanding' | 'rolling-150' | 'decay-2y' -> (window, half_life_years)"""
    if spec == "expanding":
        return None, None
    kind, val = spec.split("-")
    return (int(val), None) if kind == "rolling" else (None, float(val.rstrip("y")))


def _job(df, y, tr, va, model_key, set_name, window, half_life, tune):
    if window:
        tr = tr[-window:]
    cols = MD.feature_columns(set_name)
    fit = {}
    if half_life:
        fit["model__sample_weight"] = decay_weights(df.release_date.values[tr], half_life)
    Xtr, Xva = df.iloc[tr][cols], df.iloc[va][cols]
    if tune:
        gs = GridSearchCV(MD.make_pipe(model_key, set_name), MD.MODELS[model_key][1], scoring="roc_auc", cv=TimeSeriesSplit(n_splits=C.N_INNER), n_jobs=1, refit=True).fit(Xtr, y[tr], **fit)
        return va, gs.predict_proba(Xva)[:, 1]
    return va, MD.make_pipe(model_key, set_name).fit(Xtr, y[tr], **fit).predict_proba(Xva)[:, 1]


def walk_forward_temporal(df, y, model_key, set_name, spec="expanding", n_outer=C.N_OUTER, tune=True, n_jobs=2):
    """Same contract as `validation.walk_forward`. Returns (out-of-fold probabilities, validation index arrays)."""
    assert df.release_date.is_monotonic_increasing
    y = np.asarray(y)
    window, half_life = parse_spec(spec)
    folds = list(TimeSeriesSplit(n_splits=n_outer).split(df))
    res = Parallel(n_jobs=n_jobs)(delayed(_job)(df, y, tr, va, model_key, set_name, window, half_life, tune) for tr, va in folds)
    oof = np.full(len(df), np.nan)
    for va, p in res:
        oof[va] = p
    return oof, [va for _, va in folds]
