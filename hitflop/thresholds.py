"""Decision-threshold rules. Each rule looks at (labels, predicted probabilities) of data the caller is allowed to use and returns one threshold.
In this project the caller always passes EARLIER validation blocks only (see `protocol.expanding_threshold`), never the data being scored.

youden        maximises recall + specificity - 1 (the original study's rule; equals maximising balanced accuracy)
max_f1        maximises the F1 score of the Hit class
max_mcc       maximises the Matthews correlation coefficient
prior_match   predicts Hit for the share of films equal to the Hit rate of the data used (keeps the predicted Hit share stable)
recall_60     the highest threshold that still finds at least 60% of the Hits (a Hit-friendly operating point)
"""
import numpy as np

from . import metrics as M


def _candidates(p, cap=300):
    u = np.unique(np.asarray(p, float))
    if len(u) > cap:
        u = np.quantile(u, np.linspace(0, 1, cap))
    return u


def _counts(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return tn, fp, fn, tp


def _best(y, p, score):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    cand = _candidates(p)
    vals = np.array([score(*_counts(y, p, t)) for t in cand])
    return float(cand[np.flatnonzero(vals == vals.max())[len(np.flatnonzero(vals == vals.max())) // 2]])       # middle of any plateau of ties


def _f1(tn, fp, fn, tp):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _mcc(tn, fp, fn, tp):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def youden(y, p):
    t = M.youden_threshold(y, p)
    return t if np.isfinite(t) else 0.5


def max_f1(y, p):
    return _best(y, p, _f1)


def max_mcc(y, p):
    return _best(y, p, _mcc)


def prior_match(y, p):
    return float(np.quantile(np.asarray(p, float), 1 - float(np.mean(y))))


def recall_60(y, p, target=0.6):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    ok = [t for t in _candidates(p) if _counts(y, p, t)[3] / max(1, y.sum()) >= target]
    return float(max(ok)) if ok else float(p.min())


RULES = {"youden": youden, "max_f1": max_f1, "max_mcc": max_mcc, "prior_match": prior_match, "recall_60": recall_60}


def choose(rule, y, p):
    """`rule` is a name in RULES, a fixed number, or a callable (y, p) -> threshold."""
    if callable(rule):
        return float(rule(y, p))
    if isinstance(rule, str):
        return float(RULES[rule](y, p))
    return float(rule)
