"""Model registry and leak-free preprocessing. Every learned step (imputation, scaling, one-hot, top-k genres) lives inside the sklearn Pipeline,
so it is fitted on the training rows of one fold only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import features as F
from . import history as H

BASE_NUM = ["log_budget", "runtime", "year", "month_sin", "month_cos"]
BASE_CAT = ["language", "season"]
GENRE = ["genres"]
WIKI = ["l_pv_90_31", "l_pv_30_8", "l_pv_7_1", "l_pv_peak_pre", "l_buzz_growth", "buzz_per_budget"]
YT = ["trailer_found", "n_promos", "l_trailer_lead", "l_campaign_lead", "l_cm_pre_n", "l_cm_pre_likes"]
WIKI3 = ["l_pv_90_31", "l_pv_30_8", "l_pv_7_1", "buzz_per_budget"]     # only the three documented page-view windows (what a user can type into the app)
HIST = H.CREW + H.TECH + H.STUDIO

# name -> list of extra column blocks added to the base features. Blocks get median imputation + missing indicators.
FEATURE_SETS = {
    "A": [],
    "B": [WIKI],
    "C": [YT],
    "D": [WIKI, YT],
    "B+": [WIKI, F.WIKI_RATE],
    "C+": [YT, F.YT_RATIO],
    "D+": [WIKI, YT, F.WIKI_RATE, F.YT_RATIO],
    "A+CAL": [F.CALENDAR],
    "A+COMP": [F.COMPETITION],
    "A+CAL+COMP": [F.CALENDAR, F.COMPETITION],
    "A+HIST": [HIST],
    "B+CAL+COMP": [WIKI, F.CALENDAR, F.COMPETITION],
    "D+CAL+COMP": [WIKI, YT, F.CALENDAR, F.COMPETITION],
    "B3": [WIKI3],
    "B3+": [WIKI3, F.WIKI_RATE],
    "B3+CAL+COMP": [WIKI3, F.WIKI_RATE, F.CALENDAR, F.COMPETITION],
    "FULL": [WIKI, YT, F.WIKI_RATE, F.YT_RATIO, F.CALENDAR, F.COMPETITION, HIST],
}


def feature_columns(set_name):
    return BASE_NUM + BASE_CAT + GENRE + [c for blk in FEATURE_SETS[set_name] for c in blk]


class GenreTopK(BaseEstimator, TransformerMixin):
    """Encode the k most frequent genres of the TRAINING rows; unseen genres are ignored at prediction time."""

    def __init__(self, k=12):
        self.k = k

    @staticmethod
    def _split(X):
        s = pd.Series(np.asarray(X, dtype=object).ravel()).fillna("")
        return [[g.strip() for g in str(t).split(",") if g.strip()] for t in s]

    def fit(self, X, y=None):
        counts = pd.Series([g for L in self._split(X) for g in L]).value_counts()
        self.genres_ = list(counts.index[: self.k])
        return self

    def transform(self, X):
        return np.array([[float(g in L) for g in self.genres_] for L in self._split(X)])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"genre_{g}" for g in self.genres_])


def make_preprocessor(set_name, scale):
    def num_pipe(indicator):
        steps = [("imp", SimpleImputer(strategy="median", add_indicator=indicator, keep_empty_features=True))]
        return Pipeline(steps + ([("sc", StandardScaler())] if scale else []))

    parts = [("num", num_pipe(False), BASE_NUM), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CAT), ("genre", GenreTopK(12), GENRE)]
    for i, blk in enumerate(FEATURE_SETS[set_name]):
        parts.append((f"blk{i}", num_pipe(True), list(blk)))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=C.SEED, n_jobs=1, verbosity=0, eval_metric="logloss")


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, n_jobs=1, verbose=-1)


def _cat():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=250, learning_rate=0.05, depth=4, random_seed=C.SEED, verbose=0, thread_count=1, allow_writing_files=False)


# name -> (factory, inner grid or None, needs scaling)
MODELS = {
    "LR": (lambda: LogisticRegression(max_iter=5000), {"model__C": [0.03, 0.1, 0.3, 1.0]}, True),
    "RF": (lambda: RandomForestClassifier(n_estimators=200, random_state=C.SEED, n_jobs=1), {"model__min_samples_leaf": [3, 8, 15], "model__max_features": ["sqrt", 0.5]}, False),
    "HGB": (lambda: HistGradientBoostingClassifier(random_state=C.SEED, max_iter=150), {"model__learning_rate": [0.03, 0.08], "model__max_depth": [2, 3], "model__min_samples_leaf": [15, 30]}, False),
    "XGB": (_xgb, {"model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.08]}, False),
    "LGBM": (_lgbm, {"model__num_leaves": [4, 8], "model__learning_rate": [0.03, 0.08]}, False),
    "CAT": (_cat, {"model__depth": [3, 5]}, False),
}


def make_pipe(model_key, set_name, params=None):
    est, _, scale = MODELS[model_key]
    pipe = Pipeline([("prep", make_preprocessor(set_name, scale)), ("model", est())])
    return pipe.set_params(**params) if params else pipe


def tune_and_fit(X, y, model_key, set_name, n_inner=C.N_INNER):
    """Inner chronological grid search (scored by ROC-AUC) on the training window only."""
    grid = MODELS[model_key][1]
    gs = GridSearchCV(make_pipe(model_key, set_name), grid, scoring="roc_auc", cv=TimeSeriesSplit(n_splits=n_inner), n_jobs=1, refit=True)
    return gs.fit(X, y)
