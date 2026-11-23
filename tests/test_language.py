import numpy as np

from hitflop import data
from hitflop import language as LG
from hitflop import protocol as P


def test_language_prior_uses_only_training_films():
    dev = P.dev_only_world()[2]
    y = dev.hit.values.copy()
    folds = data.outer_folds(dev)
    a = LG.language_prior_oof(dev, y, folds)
    y2 = y.copy()
    va_last = folds[-1][1]
    y2[va_last] = 1 - y2[va_last]                       # flipping the labels of the last block must not change any prediction
    assert np.allclose(a, LG.language_prior_oof(dev, y2, folds), equal_nan=True)


def test_language_prior_is_a_probability():
    dev = P.dev_only_world()[2]
    p = LG.language_prior_oof(dev, dev.hit.values, data.outer_folds(dev))
    ok = ~np.isnan(p)
    assert ok.sum() == sum(len(v) for _, v in data.outer_folds(dev)) and ((p[ok] > 0) & (p[ok] < 1)).all()


def test_reportable_needs_both_classes():
    assert not LG.reportable(np.ones(100))
    assert not LG.reportable(np.array([1] * 10 + [0] * 10))
    assert LG.reportable(np.array([1] * 20 + [0] * 20))


def test_held_out_language_is_never_trained_on():
    dev = P.dev_only_world()[2]
    y = dev.hit.values
    folds = data.outer_folds(dev)
    a = LG.walk_forward_without(dev, y, "LR", "A", "te", folds, tune=False)
    y2 = y.copy()
    y2[dev.language.values == "te"] = 1 - y2[dev.language.values == "te"]       # the held-out language's own labels must not matter
    b = LG.walk_forward_without(dev, y2, "LR", "A", "te", folds, tune=False)
    assert np.allclose(a, b, equal_nan=True)
    scored = ~np.isnan(a)
    assert (dev.language.values[scored] == "te").all() and scored.sum() > 0
