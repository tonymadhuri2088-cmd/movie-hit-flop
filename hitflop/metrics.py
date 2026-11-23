"""Metrics, bootstrap intervals and paired comparisons. Everything works on plain numpy arrays and is deterministic given a seed."""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_curve

from . import config as C


def auc(y, p):
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(y, p):
    return float(average_precision_score(y, p))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def ece(y, p, bins=10):
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def classification_metrics(y, p, threshold):
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    return {"threshold": float(threshold), "accuracy": float((pred == y).mean()), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred))}


def full_metrics(y, p, threshold=None):
    """All headline metrics. If no threshold is given, 0.5 is used (never tuned on the data being scored)."""
    y, p = np.asarray(y), np.asarray(p)
    m = {"n": int(len(y)), "hit_rate": float(y.mean()), "auc": float(auc(y, p)), "pr_auc": pr_auc(y, p), "brier": float(brier_score_loss(y, p)),
         "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "ece": ece(y, p)}
    m.update(classification_metrics(y, p, 0.5 if threshold is None else threshold))
    return m


def bootstrap_ci(y, p, metric=auc, n=2000, seed=C.SEED, alpha=0.05):
    """Percentile bootstrap CI over movies. Note: movies are treated as exchangeable, so intervals are somewhat optimistic for time-ordered data."""
    y, p = np.asarray(y), np.asarray(p)
    rg = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i = rg.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(metric(y[i], p[i]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(y, p_new, p_old, metric=auc, n=2000, seed=C.SEED):
    """Difference in `metric` (new minus old) on identical resamples. Returns (diff, ci_low, ci_high, two-sided p)."""
    y, p_new, p_old = np.asarray(y), np.asarray(p_new), np.asarray(p_old)
    rg = np.random.default_rng(seed)
    d = []
    while len(d) < n:
        i = rg.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(metric(y[i], p_new[i]) - metric(y[i], p_old[i]))
    d = np.array(d)
    p = max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / n)
    return float(metric(y, p_new) - metric(y, p_old)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (q-values), same order as input."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[idx] * n / k)
        adj[idx] = prev
    return adj


def fold_aucs(y, p, folds_va):
    return [float(auc(np.asarray(y)[va], np.asarray(p)[va])) for va in folds_va]


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar test on paired correct/incorrect decisions (two-sided). Returns (b, c, p): b = A right & B wrong, c = A wrong & B right."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    ra, rb = a == y, b == y
    n_b, n_c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    if n_b + n_c == 0:
        return n_b, n_c, 1.0
    return n_b, n_c, float(stats.binomtest(min(n_b, n_c), n_b + n_c, 0.5).pvalue)
