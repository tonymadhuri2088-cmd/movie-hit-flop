"""Upgrade Phase 2: ways of handling a minority Hit class, built on the original models and the original chronological validation.

variants
  base          the original model, untouched (reproduces the stored out-of-fold predictions exactly)
  class_weight  the loss counts a Hit more than a Flop in proportion to how rare Hits are in the TRAINING rows
  balanced_rf   random forest where every tree is grown on a balanced bootstrap (imbalanced-learn); random forest only
  smote         synthetic Hit rows are added to the TRAINING rows after preprocessing (SMOTE-NC: yes/no columns such as genre flags,
                one-hot language and missing-value indicators stay yes/no instead of becoming fractions)

Resampling lives inside an imbalanced-learn Pipeline, so it runs only on the rows a model is trained on:
it never touches validation rows, and inside the inner grid search it runs on the inner training part only.
"""
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config as C
from . import models as MD

VARIANTS = ["base", "class_weight", "balanced_rf", "smote"]
SUPPORTED = {"LR": ["base", "class_weight", "smote"], "RF": ["base", "class_weight", "balanced_rf", "smote"],
             "HGB": ["base", "class_weight", "smote"], "XGB": ["base", "class_weight", "smote"]}


class BinaryAwareSMOTE(BaseEstimator):
    """SMOTE that knows which preprocessed columns are yes/no. A column with at most two distinct values in the training rows is treated as
    categorical: its synthetic value is the majority among the neighbours instead of an interpolated fraction. With no such column it is plain SMOTE."""

    def __init__(self, k_neighbors=5, random_state=C.SEED):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n_min = int(min((y == 0).sum(), (y == 1).sum()))
        if n_min <= 1 or (y == 0).sum() == (y == 1).sum():
            return X, y                                                       # nothing to balance, or too few rows to interpolate
        k = max(1, min(self.k_neighbors, n_min - 1))
        cat = np.array([len(np.unique(X[:, j])) <= 2 for j in range(X.shape[1])])
        if cat.all():
            self.mask_ = cat
            return SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(X, y)
        if cat.any():
            self.mask_ = cat
            return SMOTENC(categorical_features=[int(j) for j in np.flatnonzero(cat)], k_neighbors=k, random_state=self.random_state).fit_resample(X, y)
        self.mask_ = cat
        return SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(X, y)

    def get_params(self, deep=True):
        return {"k_neighbors": self.k_neighbors, "random_state": self.random_state}


def _xgb_balanced():
    from xgboost import XGBClassifier

    class BalancedXGB(XGBClassifier):
        """XGBoost whose positive-class weight is set at fit time from the rows it is given (negatives / positives)."""

        def fit(self, X, y, **kw):
            y = np.asarray(y)
            self.set_params(scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())))
            return super().fit(X, y, **kw)

    return BalancedXGB(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=C.SEED, n_jobs=1, verbosity=0, eval_metric="logloss")


def _estimator(model_key, variant):
    est = MD.MODELS[model_key][0]()
    if variant == "class_weight":
        if model_key == "XGB":
            return _xgb_balanced()
        est.set_params(class_weight={"RF": "balanced_subsample"}.get(model_key, "balanced"))
    elif variant == "balanced_rf":
        assert model_key == "RF", "balanced_rf is a random-forest variant"
        return BalancedRandomForestClassifier(n_estimators=200, random_state=C.SEED, n_jobs=1, sampling_strategy="all", replacement=True, bootstrap=False)
    return est


def make_variant(model_key, set_name, variant="base", params=None):
    """Pipeline for (model, feature set, variant). Grid parameter names (`model__...`) are the same as in `models.MODELS` for every variant."""
    if variant not in SUPPORTED.get(model_key, []):
        raise ValueError(f"{variant} is not defined for {model_key}")
    scale = MD.MODELS[model_key][2]
    steps = [("prep", MD.make_preprocessor(set_name, scale))]
    if variant == "smote":
        steps.append(("sampler", BinaryAwareSMOTE()))
    steps.append(("model", _estimator(model_key, variant)))
    pipe = ImbPipeline(steps)
    return pipe.set_params(**params) if params else pipe


def _fold_job(df, y, tr, va, model_key, set_name, variant, tune):
    cols = MD.feature_columns(set_name)
    pipe = make_variant(model_key, set_name, variant)
    if tune:
        gs = GridSearchCV(pipe, MD.MODELS[model_key][1], scoring="roc_auc", cv=TimeSeriesSplit(n_splits=C.N_INNER), n_jobs=1, refit=True).fit(df.iloc[tr][cols], y[tr])
        return va, gs.predict_proba(df.iloc[va][cols])[:, 1], gs.best_params_
    pipe.fit(df.iloc[tr][cols], y[tr])
    return va, pipe.predict_proba(df.iloc[va][cols])[:, 1], {}


def walk_forward_variant(df, y, model_key, set_name, variant="base", n_outer=C.N_OUTER, tune=True, n_jobs=2):
    """Same contract as `validation.walk_forward` (expanding windows, nested inner chronological search), for any variant.
    Returns (out-of-fold probabilities with NaN for rows that only ever train, validation index arrays, best params per fold)."""
    assert df.release_date.is_monotonic_increasing
    y = np.asarray(y)
    folds = list(TimeSeriesSplit(n_splits=n_outer).split(df))
    res = Parallel(n_jobs=n_jobs)(delayed(_fold_job)(df, y, tr, va, model_key, set_name, variant, tune) for tr, va in folds)
    oof = np.full(len(df), np.nan)
    for va, p, _ in res:
        oof[va] = p
    return oof, [va for _, va in folds], [prm for _, _, prm in res]


def simulate_minority(df, hit_share=0.30, seed=C.SEED):
    """A controlled minority-class setting: keep every Flop and randomly drop Hits (any release date) until Hits are `hit_share` of the rows.
    Order by release date is preserved. Used only to see how a method behaves when Hits really are rare, since the development data is about 47% Hits."""
    rg = np.random.default_rng(seed)
    y = df.hit.values
    n_flop = int((y == 0).sum())
    n_hit = int(round(n_flop * hit_share / (1 - hit_share)))
    keep_hit = rg.choice(np.flatnonzero(y == 1), size=min(n_hit, int((y == 1).sum())), replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(y == 0), keep_hit]))
    return df.iloc[keep].reset_index(drop=True)
