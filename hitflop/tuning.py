"""Bayesian (TPE) hyper-parameter search with Optuna, nested correctly: for every OUTER fold the search sees only that fold's training window and scores
trials with an INNER expanding-window split of it. The outer validation block is touched exactly once, to predict with the chosen parameters."""
import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit

from . import config as C, metrics as M, models as MD

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial, model_key):
    if model_key == "XGB":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 80, 400), "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "model__max_depth": trial.suggest_int("max_depth", 2, 5), "model__min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                "model__subsample": trial.suggest_float("subsample", 0.6, 1.0), "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "model__reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
    if model_key == "LGBM":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 80, 400), "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "model__num_leaves": trial.suggest_int("num_leaves", 3, 16), "model__min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                "model__subsample": trial.suggest_float("subsample", 0.6, 1.0), "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "model__reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
    if model_key == "CAT":
        return {"model__iterations": trial.suggest_int("iterations", 100, 400), "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "model__depth": trial.suggest_int("depth", 3, 6), "model__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True)}
    raise ValueError(model_key)


def tune_window(X, y, model_key, set_name, n_trials=25, seed=C.SEED, n_inner=C.N_INNER):
    """Search on one training window. Returns (best sklearn-style params, best inner mean AUC)."""
    inner = list(TimeSeriesSplit(n_splits=n_inner).split(X))

    def objective(trial):
        params = suggest(trial, model_key)
        scores = []
        for tr, va in inner:
            pipe = MD.make_pipe(model_key, set_name, params).fit(X.iloc[tr], y[tr])
            scores.append(M.auc(y[va], pipe.predict_proba(X.iloc[va])[:, 1]))
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(_default_trial(model_key))            # the untuned defaults are always among the candidates, so tuning cannot do worse on inner data
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_trial.params
    return {f"model__{k}": v for k, v in bp.items()}, float(study.best_value), len(study.trials)


def _default_trial(model_key):
    return {"XGB": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 3.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            "LGBM": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            "CAT": {"iterations": 250, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3.0}}[model_key]


def _outer_job(df, y, tr, va, model_key, set_name, n_trials):
    cols = MD.feature_columns(set_name)
    X = df[cols]
    best, inner_score, n = tune_window(X.iloc[tr].reset_index(drop=True), np.asarray(y)[tr], model_key, set_name, n_trials)
    pipe = MD.make_pipe(model_key, set_name, best).fit(X.iloc[tr], np.asarray(y)[tr])
    return va, pipe.predict_proba(X.iloc[va])[:, 1], best, inner_score


def nested_optuna_oof(df, y, model_key, set_name, n_trials=25, n_outer=C.N_OUTER, n_jobs=2):
    assert df.release_date.is_monotonic_increasing
    folds = list(TimeSeriesSplit(n_splits=n_outer).split(df))
    res = Parallel(n_jobs=n_jobs)(delayed(_outer_job)(df, y, tr, va, model_key, set_name, n_trials) for tr, va in folds)
    oof = np.full(len(df), np.nan)
    for va, p, _, _ in res:
        oof[va] = p
    return oof, [va for _, va in folds], [{"params": b, "inner_auc": s} for _, _, b, s in res]
