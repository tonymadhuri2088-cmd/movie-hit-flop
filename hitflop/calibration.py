"""Probability calibration that respects time: the calibrator for fold k is fitted ONLY on out-of-fold predictions of folds before k."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Platt:
    """Sigmoid calibration: logistic regression on the logit of the raw probability."""

    def fit(self, p, y):
        z = self._logit(p).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))


class Iso:
    def fit(self, p, y):
        self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p, y)
        return self

    def predict(self, p):
        return self.m_.predict(p)


def expanding_calibration(oof, y, folds_va, method="platt"):
    """Calibrate each fold k >= 1 with a calibrator fitted on folds 0..k-1 only. Returns (calibrated predictions aligned to `oof`, mask of rows that were calibrated)."""
    oof, y = np.asarray(oof, float), np.asarray(y)
    out = np.full(len(oof), np.nan)
    done = np.zeros(len(oof), bool)
    for k in range(1, len(folds_va)):
        past = np.concatenate(folds_va[:k])
        cal = (Platt() if method == "platt" else Iso()).fit(oof[past], y[past])
        out[folds_va[k]] = cal.predict(oof[folds_va[k]])
        done[folds_va[k]] = True
    return out, done


def reliability_table(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    chunks = np.array_split(np.argsort(p), bins)
    import pandas as pd
    return pd.DataFrame([{"bin": i + 1, "n": len(c), "mean_predicted": p[c].mean(), "observed_hit_rate": y[c].mean()} for i, c in enumerate(chunks) if len(c)])
