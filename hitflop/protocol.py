"""Upgrade Phase 0: the rules every new experiment must follow, written as code so they cannot be skipped by accident.

1. Development data only. New experiments never see the frozen test set (`ensure_dev_only`, `dev_only_world`).
2. Chronological validation. The same 5 expanding-window folds as the original study (`data.outer_folds`).
3. Thresholds come from earlier data only (`expanding_threshold`).
4. One comparator, one claim rule. A new result is called better (or worse) ONLY if the paired bootstrap 95% interval of the difference
   excludes zero AND the Benjamini-Hochberg adjusted p-value is below 0.05 (`compare_to_baseline`). Anything else is "no clear difference".
5. Every run is written to the append-only experiment log with its full configuration (`log_run`).
"""
import platform

import numpy as np
import pandas as pd
import sklearn

from . import config as C
from . import metrics as M
from .experiment_log import log_experiment
from .world import build_world

ALPHA = 0.05
BASELINE = ("LR", "A")           # logistic regression on the base features: the comparator named in the original study


def ensure_dev_only(df):
    """Raise if any row is on or after the frozen-test start date. Call this at the top of every experiment."""
    if len(df) and pd.Timestamp(df.release_date.max()) >= pd.Timestamp(C.SPLIT_DATE):
        raise ValueError(f"Rows released on or after {C.SPLIT_DATE} found: the frozen test set must not be used for experiments.")
    return df


def dev_only_world():
    """(all movies, labelled movies before the split, development table). The test table is built and immediately discarded, never returned."""
    merged, lab, dev, _ = build_world()
    return merged, lab[lab.release_date < pd.Timestamp(C.SPLIT_DATE)].reset_index(drop=True), ensure_dev_only(dev)


def expanding_threshold(y, oof, folds_va, rule="youden"):
    """Threshold for each validation block chosen from the out-of-fold predictions of EARLIER blocks only.
    Returns (threshold per row, NaN where none exists; boolean mask of rows that can be scored). The first block has no earlier data and is not scored."""
    y, oof = np.asarray(y), np.asarray(oof, float)
    thr = np.full(len(oof), np.nan)
    for k in range(1, len(folds_va)):
        past = np.concatenate(folds_va[:k])
        t = M.youden_threshold(y[past], oof[past]) if rule == "youden" else float(rule)
        thr[folds_va[k]] = t if np.isfinite(t) else 0.5        # a constant predictor has no meaningful Youden point; fall back to 0.5
    return thr, ~np.isnan(thr)


def verdict(diff, lo, hi, q, alpha=ALPHA):
    """The only wording allowed for a comparison."""
    if q < alpha and lo > 0:
        return "better (supported)"
    if q < alpha and hi < 0:
        return "worse (supported)"
    return "no clear difference"


def compare_to_baseline(y, preds, baseline, metric=M.auc, n_boot=2000, seed=C.SEED):
    """Paired bootstrap of every prediction vector in `preds` versus `preds[baseline]` on identical rows (NaN rows dropped for all),
    with Benjamini-Hochberg correction over the whole family of comparisons and a `verdict` column that applies the claim rule."""
    ok = np.ones(len(y), bool)
    for v in preds.values():
        ok &= ~np.isnan(np.asarray(v, float))
    yy = np.asarray(y)[ok]
    rows = []
    for name, p in preds.items():
        if name == baseline:
            continue
        d, lo, hi, pv = M.paired_bootstrap(yy, np.asarray(p)[ok], np.asarray(preds[baseline])[ok], metric=metric, n=n_boot, seed=seed)
        rows.append({"model": name, "vs": baseline, "n": int(ok.sum()), "metric_model": float(metric(yy, np.asarray(p)[ok])), "metric_baseline": float(metric(yy, np.asarray(preds[baseline])[ok])),
                     "diff": d, "ci_low": lo, "ci_high": hi, "p": pv})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = M.benjamini_hochberg(out.p.values)
        out["verdict"] = [verdict(r["diff"], r["ci_low"], r["ci_high"], r["q_bh"]) for _, r in out.iterrows()]
    return out


def correct_family(table):
    """Recompute Benjamini-Hochberg q-values and verdicts over a whole family of comparisons (for example several metrics in one run).
    Use this when one run reports more than one `compare_to_baseline` table, so that looking at several metrics does not inflate false positives."""
    out = table.copy()
    out["q_bh"] = M.benjamini_hochberg(out.p.values)
    out["verdict"] = [verdict(r["diff"], r["ci_low"], r["ci_high"], r["q_bh"]) for _, r in out.iterrows()]
    return out


def run_config(**extra):
    """The configuration block stored with every logged experiment."""
    cfg = {"seed": C.SEED, "split_date": C.SPLIT_DATE, "hit_threshold_roi": C.HIT_THRESHOLD, "n_outer_folds": C.N_OUTER, "n_inner_folds": C.N_INNER,
           "validation": "expanding-window chronological, development data only", "comparator": "%s / set %s" % BASELINE,
           "claim_rule": "paired bootstrap 95% CI excludes 0 and Benjamini-Hochberg q < 0.05",
           "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    cfg.update(extra)
    return cfg


def log_run(name, phase, config, results, notes="", uses_test_set=False, path=None):
    """Append an experiment record. Only the one-time final scoring may declare that it uses the test set."""
    if uses_test_set and phase != "final":
        raise ValueError("Only the 'final' phase may use the frozen test set.")
    return log_experiment(name, phase, run_config(uses_test_set=uses_test_set, **config), results, notes, path=path)
