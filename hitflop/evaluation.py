"""Upgrade Phase 1: one complete evaluation for any set of predictions.

Reports, with bootstrap 95% intervals:
  ranking / probability quality   ROC-AUC, PR-AUC, Brier score, log-loss, expected calibration error
  decision quality at a threshold accuracy, balanced accuracy, precision, recall (Hit recall), specificity (Flop recall),
                                  F1 of the Hit class, macro-F1, Matthews correlation (MCC) and the confusion matrix

Nothing here fits a model or chooses a threshold from the data it scores; the caller passes the threshold.
Thresholds must come from data BEFORE the scored data (see `protocol.expanding_threshold`) or be fixed (0.5).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from . import config as C
from . import metrics as M

PROB_METRICS = ["auc", "pr_auc", "brier", "log_loss", "ece"]
THR_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1_hit", "macro_f1", "mcc"]
ALL_METRICS = PROB_METRICS + THR_METRICS


def _div(a, b):
    return float(a / b) if b else 0.0                     # same convention as scikit-learn's zero_division=0


def counts_to_metrics(tn, fp, fn, tp):
    """Every threshold-based metric from the four confusion-matrix cells. Hit is the positive class."""
    tn, fp, fn, tp = (float(v) for v in (tn, fp, fn, tp))
    n = tn + fp + fn + tp
    recall, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec_hit, prec_flop = _div(tp, tp + fp), _div(tn, tn + fn)
    f1_hit = _div(2 * prec_hit * recall, prec_hit + recall)
    f1_flop = _div(2 * prec_flop * spec, prec_flop + spec)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": _div(tp + tn, n), "balanced_accuracy": (recall + spec) / 2, "precision": prec_hit, "recall": recall, "specificity": spec,
            "f1_hit": f1_hit, "macro_f1": (f1_hit + f1_flop) / 2, "mcc": _div(tp * tn - fp * fn, den)}


def confusion_counts(y, p, threshold):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    return int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())


def prob_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    return {"auc": float(M.auc(y, p)), "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), "ece": M.ece(y, p)}


def point_metrics(y, p, threshold):
    """All metrics at one threshold, no intervals."""
    out = prob_metrics(y, p)
    out.update(counts_to_metrics(*confusion_counts(y, p, threshold)))
    return out


def evaluate(y, p, threshold, n_boot=1000, seed=C.SEED, ci=True):
    """Point estimates plus percentile bootstrap 95% intervals (same resamples for every metric). `threshold` may be one number or an array with one value per row.
    A constant predictor has no ranking ability: its AUC is reported as 0.5 and its PR-AUC as the Hit rate, with no interval for the ranking metrics."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    thr = np.broadcast_to(np.asarray(threshold, float), y.shape)     # one threshold, or one per row (e.g. chosen per validation block from earlier blocks)
    constant = np.ptp(p) == 0
    est = point_metrics(y, p, thr) if not constant else {**point_metrics(y, p, thr), "auc": 0.5, "pr_auc": float(y.mean())}
    tn, fp, fn, tp = confusion_counts(y, p, thr)
    row = {"n": int(len(y)), "hit_rate": float(y.mean()), "threshold": float(np.mean(thr)), "TN": tn, "FP": fp, "FN": fn, "TP": tp}
    for k in ALL_METRICS:
        row[k] = est[k]
    if ci:
        rg = np.random.default_rng(seed)
        draws = {k: [] for k in ALL_METRICS}
        done = 0
        while done < n_boot:
            i = rg.integers(0, len(y), len(y))
            if not 0 < y[i].sum() < len(i):
                continue
            m = point_metrics(y[i], p[i], thr[i]) if not constant else {**point_metrics(y[i], p[i], thr[i]), "auc": 0.5, "pr_auc": float(y[i].mean())}
            for k in ALL_METRICS:
                draws[k].append(m[k])
            done += 1
        for k in ALL_METRICS:
            lo, hi = np.percentile(draws[k], [2.5, 97.5])
            row[k + "_lo"], row[k + "_hi"] = float(lo), float(hi)
    return row


def evaluate_table(y, preds, thresholds, n_boot=1000, seed=C.SEED, ci=True):
    """One row per named prediction vector. `thresholds` maps name -> threshold (or a single float for all)."""
    rows = []
    for name, p in preds.items():
        thr = thresholds[name] if isinstance(thresholds, dict) else thresholds
        rows.append({"model": name, **evaluate(y, p, thr, n_boot=n_boot, seed=seed, ci=ci)})
    return pd.DataFrame(rows)


def confusion_frame(table):
    """Tidy confusion matrices (rows = true class, columns = predicted class) from an `evaluate_table` result."""
    return table[["model", "threshold", "TN", "FP", "FN", "TP"]].rename(columns={"TN": "true_flop_pred_flop", "FP": "true_flop_pred_hit", "FN": "true_hit_pred_flop", "TP": "true_hit_pred_hit"})


def format_ci(row, metric, digits=3):
    return f"{row[metric]:.{digits}f} [{row[metric + '_lo']:.{digits}f}, {row[metric + '_hi']:.{digits}f}]"
