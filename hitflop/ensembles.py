"""Ensembles with correct nested validation.

Base-model predictions for the rows of fold k are out-of-fold (made by models that never saw fold k). To combine them we learn weights or a meta-model
ONLY from the out-of-fold predictions of folds before k, so the combiner has never seen fold k's labels either. Fold 0 has no earlier fold and is
scored with equal weights (reported separately; headline numbers use folds >= 1).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import metrics as M


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def auc_weights(P, y):
    """Weights proportional to how far each model's AUC is above chance on the given (earlier-fold) rows; equal weights if none is above chance."""
    a = np.array([M.auc(y, P[:, j]) for j in range(P.shape[1])])
    w = np.clip(a - 0.5, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def walk_forward_ensembles(oof_by_model, y, folds_va):
    """oof_by_model: dict name -> array of OOF probabilities (NaN where never validated). Returns dict of ensemble OOF arrays and the per-fold weights."""
    names = list(oof_by_model)
    P = np.column_stack([oof_by_model[n] for n in names])
    y = np.asarray(y)
    avg = np.full(len(y), np.nan)
    wv = np.full(len(y), np.nan)
    st = np.full(len(y), np.nan)
    weights = []
    for k, va in enumerate(folds_va):
        avg[va] = P[va].mean(axis=1)
        if k == 0:
            wv[va] = avg[va]
            st[va] = avg[va]
            weights.append(np.full(len(names), 1 / len(names)))
            continue
        past = np.concatenate(folds_va[:k])
        w = auc_weights(P[past], y[past])
        wv[va] = P[va] @ w
        meta = LogisticRegression(C=1.0, max_iter=2000).fit(_logit(P[past]), y[past])
        st[va] = meta.predict_proba(_logit(P[va]))[:, 1]
        weights.append(w)
    return {"average": avg, "weighted_vote": wv, "stacking": st}, np.array(weights), names


def fit_final_combiners(P_oof, y_oof, P_test):
    """Final ensemble predictions on the test set; combiners are fitted on ALL development out-of-fold predictions only."""
    w = auc_weights(P_oof, y_oof)
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(_logit(P_oof), y_oof)
    return {"average": P_test.mean(axis=1), "weighted_vote": P_test @ w, "stacking": meta.predict_proba(_logit(P_test))[:, 1]}, w
