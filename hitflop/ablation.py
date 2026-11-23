"""Upgrade Phase 5: which kinds of information carry the prediction?

Ten feature groups. Six are in the base set (budget, runtime, release year, release timing, language, genres) and four are optional extras
(Wikipedia buzz, cast/crew/studio track records, festival calendar, competing releases).
Three designs, all with the original expanding-window folds and inner tuning:
  leave-one-group-out  start from every group, remove one: how much does ranking quality drop?
  group alone          use one group only: does it beat chance by itself?
  add-one to base      start from the base set, add one extra group: does it help?
The preprocessing is the original one (median imputation, one-hot, top-12 genres, missing-value indicators for extra blocks), so a model with all base groups
reproduces the original set-A predictions exactly.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import features as F
from . import models as MD

GROUPS = {
    "budget": ["log_budget"],
    "runtime": ["runtime"],
    "release_year": ["year"],
    "release_timing": ["month_sin", "month_cos", "season"],
    "language": ["language"],
    "genres": ["genres"],
    "buzz_wikipedia": list(MD.WIKI3),
    "cast_crew_studio_history": list(MD.HIST),
    "festival_calendar": list(F.CALENDAR),
    "competing_releases": list(F.COMPETITION),
}
BASE_GROUPS = ["budget", "runtime", "release_year", "release_timing", "language", "genres"]
EXTRA_GROUPS = ["buzz_wikipedia", "cast_crew_studio_history", "festival_calendar", "competing_releases"]
ALL_GROUPS = BASE_GROUPS + EXTRA_GROUPS


def columns_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def make_preprocessor_groups(groups, scale):
    """The original ColumnTransformer restricted to the chosen groups (column order of the original base features is kept)."""
    inc = {c for g in groups if g in BASE_GROUPS for c in GROUPS[g]}

    def num_pipe(indicator):
        steps = [("imp", SimpleImputer(strategy="median", add_indicator=indicator, keep_empty_features=True))]
        return Pipeline(steps + ([("sc", StandardScaler())] if scale else []))

    parts = []
    num = [c for c in MD.BASE_NUM if c in inc]
    cat = [c for c in MD.BASE_CAT if c in inc]
    if num:
        parts.append(("num", num_pipe(False), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if "genres" in inc:
        parts.append(("genre", MD.GenreTopK(12), MD.GENRE))
    for i, g in enumerate([g for g in groups if g in EXTRA_GROUPS]):
        parts.append((f"blk{i}", num_pipe(True), list(GROUPS[g])))
    if not parts:
        raise ValueError("choose at least one feature group")
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)


def make_pipe_groups(model_key, groups):
    est, _, scale = MD.MODELS[model_key]
    return Pipeline([("prep", make_preprocessor_groups(groups, scale)), ("model", est())])


def _job(df, y, tr, va, model_key, groups, tune):
    cols = columns_for(groups)
    pipe = make_pipe_groups(model_key, groups)
    if tune:
        gs = GridSearchCV(pipe, MD.MODELS[model_key][1], scoring="roc_auc", cv=TimeSeriesSplit(n_splits=C.N_INNER), n_jobs=1, refit=True).fit(df.iloc[tr][cols], y[tr])
        return va, gs.predict_proba(df.iloc[va][cols])[:, 1]
    return va, pipe.fit(df.iloc[tr][cols], y[tr]).predict_proba(df.iloc[va][cols])[:, 1]


def walk_forward_groups(df, y, model_key, groups, n_outer=C.N_OUTER, tune=True, n_jobs=2):
    """Same contract as `validation.walk_forward` for an arbitrary set of feature groups. Returns (out-of-fold probabilities, validation index arrays)."""
    assert df.release_date.is_monotonic_increasing
    y = np.asarray(y)
    folds = list(TimeSeriesSplit(n_splits=n_outer).split(df))
    res = Parallel(n_jobs=n_jobs)(delayed(_job)(df, y, tr, va, model_key, list(groups), tune) for tr, va in folds)
    oof = np.full(len(df), np.nan)
    for va, p in res:
        oof[va] = p
    return oof, [va for _, va in folds]
