"""Expanding-window (walk-forward) validation with nested hyper-parameter search, common-cohort comparison and subgroup robustness."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit

from . import config as C
from . import metrics as M
from . import models as MD


def _fold_job(df, y, tr, va, model_key, set_name, tune):
    cols = MD.feature_columns(set_name)
    if tune:
        gs = MD.tune_and_fit(df.iloc[tr][cols], y[tr], model_key, set_name)
        return va, gs.predict_proba(df.iloc[va][cols])[:, 1], gs.best_params_
    pipe = MD.make_pipe(model_key, set_name).fit(df.iloc[tr][cols], y[tr])
    return va, pipe.predict_proba(df.iloc[va][cols])[:, 1], {}


def walk_forward(df, y, model_key, set_name, n_outer=C.N_OUTER, tune=True, n_jobs=2):
    """Out-of-fold predictions for every row that is ever in a validation block. `df` must be sorted by release date.
    Returns (oof array with NaN for rows that only ever train, list of validation index arrays, list of best params per fold)."""
    assert df.release_date.is_monotonic_increasing
    folds = list(TimeSeriesSplit(n_splits=n_outer).split(df))
    res = Parallel(n_jobs=n_jobs)(delayed(_fold_job)(df, np.asarray(y), tr, va, model_key, set_name, tune) for tr, va in folds)
    oof = np.full(len(df), np.nan)
    for va, p, _ in res:
        oof[va] = p
    return oof, [va for _, va in folds], [prm for _, _, prm in res]


def common_cohort(df):
    """Movies that have BOTH Wikipedia page views and a YouTube record. Comparing A/B/C/D on these removes the 'different movies' confound."""
    return df[df.pv_30.notna() & (df.in_yt_file == 1)].reset_index(drop=True)


def subgroup_auc(df, y, oof, by, min_n=40, n_boot=1000):
    """OOF AUC per subgroup with bootstrap CI; groups with fewer than `min_n` validated movies are reported but flagged."""
    ok = ~np.isnan(oof)
    rows = []
    for g, idx in df[ok].groupby(by).groups.items():
        pos = df.index.get_indexer(idx)
        yy, pp = np.asarray(y)[pos], oof[pos]
        a = M.auc(yy, pp)
        lo, hi = M.bootstrap_ci(yy, pp, n=n_boot) if (len(yy) >= 10 and 0 < yy.sum() < len(yy)) else (np.nan, np.nan)
        rows.append({"group": g, "n": len(yy), "hit_rate": float(yy.mean()), "auc": float(a), "ci_low": lo, "ci_high": hi, "reliable": len(yy) >= min_n})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def compare_feature_sets(df, y, results, baseline, others, metric=M.auc, n_boot=2000):
    """Paired bootstrap of each set in `others` versus `baseline` on identical validated movies, with Benjamini-Hochberg across comparisons.
    `results[name]` is the OOF array for that set. Also reports in how many folds the set wins."""
    ok = ~np.isnan(results[baseline])
    for o in others:
        ok &= ~np.isnan(results[o])
    yy = np.asarray(y)[ok]
    rows = []
    for o in others:
        d, lo, hi, p = M.paired_bootstrap(yy, results[o][ok], results[baseline][ok], metric=metric, n=n_boot)
        rows.append({"set": o, "vs": baseline, "n": int(ok.sum()), "auc_set": M.auc(yy, results[o][ok]), "auc_baseline": M.auc(yy, results[baseline][ok]), "diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    out = pd.DataFrame(rows)
    out["q_bh"] = M.benjamini_hochberg(out.p.values)
    out["convincing"] = (out.ci_low > 0) & (out.q_bh < 0.05)
    return out
