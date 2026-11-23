"""Upgrade Phase 6 helpers: how does the model behave inside each language, and what happens for a language it has never seen?

language_prior_oof        a baseline that knows only the language: smoothed Hit rate of that language among the training films of each fold
walk_forward_without      expanding-window walk-forward where the training films of one language are removed (the model never sees it);
                         only the films of that language are scored
Both use the original folds, so every prediction is chronological and out-of-fold.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config as C
from . import models as MD

MIN_FILMS = 30          # validated films needed before a per-language verdict is reported
MIN_PER_CLASS = 8       # and at least this many Hits and this many Flops


def language_prior_oof(dev, y, folds, strength=10.0):
    """Out-of-fold probability = (hits + strength * overall) / (n + strength) for the film's language among the fold's TRAINING films."""
    y = np.asarray(y)
    oof = np.full(len(dev), np.nan)
    lang = dev.language.values
    for tr, va in folds:
        overall = y[tr].mean()
        for lg in np.unique(lang[va]):
            m = lang[tr] == lg
            prior = (y[tr][m].sum() + strength * overall) / (m.sum() + strength)
            oof[va[lang[va] == lg]] = prior
    return oof


def reportable(y_block):
    """True when a language has enough validated films, and enough of each class, for an AUC to mean something."""
    y_block = np.asarray(y_block)
    return len(y_block) >= MIN_FILMS and y_block.sum() >= MIN_PER_CLASS and (len(y_block) - y_block.sum()) >= MIN_PER_CLASS


def walk_forward_without(dev, y, model_key, set_name, language, folds, tune=True):
    """Train on earlier films of every language EXCEPT `language`; predict that language's films in each validation block.
    Blocks with no film of the language, or with no training film, stay NaN."""
    y = np.asarray(y)
    cols = MD.feature_columns(set_name)
    lang = dev.language.values
    oof = np.full(len(dev), np.nan)
    for tr, va in folds:
        tr2 = tr[lang[tr] != language]
        va2 = va[lang[va] == language]
        if len(va2) == 0 or len(tr2) < 60 or y[tr2].min() == y[tr2].max():
            continue
        pipe = MD.make_pipe(model_key, set_name)
        if tune:
            fit = GridSearchCV(pipe, MD.MODELS[model_key][1], scoring="roc_auc", cv=TimeSeriesSplit(n_splits=C.N_INNER), n_jobs=1, refit=True).fit(dev.iloc[tr2][cols], y[tr2])
            oof[va2] = fit.predict_proba(dev.iloc[va2][cols])[:, 1]
        else:
            oof[va2] = pipe.fit(dev.iloc[tr2][cols], y[tr2]).predict_proba(dev.iloc[va2][cols])[:, 1]
    return oof
