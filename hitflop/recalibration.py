"""Upgrade Phase 4a: is the Hit probability honest, and can it be repaired without peeking?

A recalibration for validation block k is fitted ONLY on the out-of-fold predictions of earlier blocks (never on block k).
methods
  platt_all      sigmoid (slope and intercept) on all earlier blocks
  platt_recent   the same on the two most recent earlier blocks only
  isotonic_all   monotone step function on all earlier blocks
  shift_all      intercept only (moves every probability up or down by the same amount on the log-odds scale), all earlier blocks
  shift_recent   intercept only, the two most recent earlier blocks
A monotone map cannot change the ranking of films inside one block, so ROC-AUC per block is unchanged (pooled AUC can move a little because each block gets its own map). The question here is about probabilities, so the primary metric is the Brier score.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .calibration import Iso, Platt

METHODS = ["platt_all", "platt_recent", "isotonic_all", "shift_all", "shift_recent"]
RECENT_BLOCKS = 2


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_shift(p, y, iters=50):
    """The constant a that makes the mean of sigmoid(logit(p) + a) equal to the observed Hit rate (maximum likelihood for an intercept-only update)."""
    z, y = _logit(p), np.asarray(y, float)
    a = 0.0
    for _ in range(iters):
        q = _sigmoid(z + a)
        g, h = (y - q).sum(), (q * (1 - q)).sum()
        if h < 1e-9:
            break
        step = g / h
        a += step
        if abs(step) < 1e-9:
            break
    return float(a)


class Shift:
    def fit(self, p, y):
        self.a_ = fit_shift(p, y)
        return self

    def predict(self, p):
        return _sigmoid(_logit(p) + self.a_)


def calibration_slope_intercept(y, p):
    """Slope of a logistic regression of the outcome on logit(p) (1 is ideal; below 1 means probabilities are too extreme) and the intercept when the slope is fixed at 1
    (0 is ideal; negative means probabilities are too high on average)."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y).coef_[0][0])
    return slope, fit_shift(p, y)


def expanding_recalibrate(oof, y, folds_va, method):
    """Recalibrated predictions for blocks 2..K (block 1 has nothing earlier to learn from). Returns (array aligned to `oof` with NaN elsewhere, mask of rows recalibrated)."""
    oof, y = np.asarray(oof, float), np.asarray(y)
    out = np.full(len(oof), np.nan)
    for k in range(1, len(folds_va)):
        recent = method.endswith("recent")
        first = max(0, k - RECENT_BLOCKS) if recent else 0
        past = np.concatenate(folds_va[first:k])
        base = method.split("_")[0]
        cal = {"platt": Platt, "isotonic": Iso, "shift": Shift}[base]().fit(oof[past], y[past])
        out[folds_va[k]] = np.clip(cal.predict(oof[folds_va[k]]), 1e-4, 1 - 1e-4)
    return out, ~np.isnan(out)


def neg_brier(y, p):
    return -float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))


def neg_log_loss(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_row(y, p):
    slope, icpt = calibration_slope_intercept(y, p)
    return {"n": int(len(y)), "hit_rate": float(np.mean(y)), "mean_predicted": float(np.mean(p)), "brier": float(-neg_brier(y, p)), "log_loss": float(-neg_log_loss(y, p)),
            "ece": M.ece(y, p), "calibration_slope": slope, "calibration_intercept": icpt}


def reliability(y, p, bins=5):
    y, p = np.asarray(y, float), np.asarray(p, float)
    chunks = np.array_split(np.argsort(p), bins)
    return pd.DataFrame([{"bin": i + 1, "n": len(c), "mean_predicted": float(p[c].mean()), "observed_hit_rate": float(y[c].mean())} for i, c in enumerate(chunks)])
