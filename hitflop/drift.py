"""Upgrade Phase 3: measuring how the film market changes over time.

Everything here is descriptive. It never fits a model and never chooses anything from the frozen test set.
`unlabelled_test_features` returns the test films' FEATURES only (no revenue, ROI or Hit label) so that the shift of the inputs can be measured
before Phase 4 builds an out-of-distribution check. That is covariate drift; it cannot tune a model because it has no labels.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import models as MD
from .world import build_world

LABEL_COLUMNS = {"hit", "roi", "revenue", "revenue_source", "vote_count", "rating"}


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def hit_rate_by(df, key):
    rows = []
    for g, s in df.groupby(key):
        lo, hi = wilson(int(s.hit.sum()), len(s))
        rows.append({"period": g, "n": len(s), "hits": int(s.hit.sum()), "hit_rate": float(s.hit.mean()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def hit_rate_trend(df, n_boot=2000, seed=C.SEED):
    """Is the Hit rate moving with release date? Kendall's tau between release order and the Hit label, plus the Hit-rate difference
    between the later half and the earlier half of the films (split by count, not by a hand-picked year), with a bootstrap interval."""
    d = df.sort_values("release_date")
    y = d.hit.values
    tau, p = stats.kendalltau(np.arange(len(y)), y)
    h = len(y) // 2
    rg = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a, b = rg.choice(y[:h], h), rg.choice(y[h:], len(y) - h)
        diffs.append(b.mean() - a.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": len(y), "hit_rate_earlier_half": float(y[:h].mean()), "hit_rate_later_half": float(y[h:].mean()), "difference": float(y[h:].mean() - y[:h].mean()),
            "diff_ci_low": float(lo), "diff_ci_high": float(hi), "kendall_tau": float(tau), "kendall_p": float(p)}


def psi(ref, cur, bins=10, eps=1e-4):
    """Population stability index of a numeric feature, with bin edges from the reference quantiles. Rule of thumb: <0.1 small, 0.1 to 0.25 moderate, >0.25 large."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + eps
    b = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((b - a) * np.log(b / a)))


def psi_categorical(ref, cur, eps=1e-4):
    ref, cur = pd.Series(ref).astype(str), pd.Series(cur).astype(str)
    cats = sorted(set(ref) | set(cur))
    a = ref.value_counts(normalize=True).reindex(cats).fillna(0).values + eps
    b = cur.value_counts(normalize=True).reindex(cats).fillna(0).values + eps
    return float(np.sum((b - a) * np.log(b / a)))


def _band(v):
    return "large" if v >= 0.25 else "moderate" if v >= 0.10 else "small"


NUMERIC_FOR_DRIFT = ["log_budget", "runtime", "month_sin", "month_cos", "l_pv_90_31", "l_pv_30_8", "l_pv_7_1", "buzz_per_budget"]
CATEGORICAL_FOR_DRIFT = ["language", "season"]


def feature_drift(ref, cur):
    """One row per input feature: PSI, Kolmogorov-Smirnov statistic and standardised mean shift between two groups of films (no labels used)."""
    rows = []
    for c in NUMERIC_FOR_DRIFT:
        a, b = ref[c].astype(float).dropna().values, cur[c].astype(float).dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        ks = stats.ks_2samp(a, b)
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1.0
        v = psi(a, b)
        rows.append({"feature": c, "kind": "numeric", "n_ref": len(a), "n_cur": len(b), "psi": v, "psi_band": _band(v), "ks": float(ks.statistic), "ks_p": float(ks.pvalue), "std_mean_shift": float((b.mean() - a.mean()) / sd)})
    for c in CATEGORICAL_FOR_DRIFT:
        v = psi_categorical(ref[c], cur[c])
        rows.append({"feature": c, "kind": "categorical", "n_ref": len(ref), "n_cur": len(cur), "psi": v, "psi_band": _band(v), "ks": np.nan, "ks_p": np.nan, "std_mean_shift": np.nan})
    g_ref = pd.Series([x.strip() for t in ref.genres.fillna("") for x in str(t).split(",") if x.strip()]).value_counts(normalize=True)
    g_cur = pd.Series([x.strip() for t in cur.genres.fillna("") for x in str(t).split(",") if x.strip()]).value_counts(normalize=True)
    cats = sorted(set(g_ref.index) | set(g_cur.index))
    a, b = g_ref.reindex(cats).fillna(0).values + 1e-4, g_cur.reindex(cats).fillna(0).values + 1e-4
    v = float(np.sum((b / b.sum() - a / a.sum()) * np.log((b / b.sum()) / (a / a.sum()))))
    rows.append({"feature": "genres", "kind": "categorical", "n_ref": len(ref), "n_cur": len(cur), "psi": v, "psi_band": _band(v), "ks": np.nan, "ks_p": np.nan, "std_mean_shift": np.nan})
    return pd.DataFrame(rows)


def unlabelled_test_features():
    """FEATURES of the frozen-test films with every label-bearing column removed. For measuring input shift only."""
    _, _, _, test = build_world()
    cols = sorted(set(MD.feature_columns("B3") + NUMERIC_FOR_DRIFT + CATEGORICAL_FOR_DRIFT + ["release_date", "year"]))
    out = test[cols].copy()
    assert not (set(out.columns) & LABEL_COLUMNS)
    return out
