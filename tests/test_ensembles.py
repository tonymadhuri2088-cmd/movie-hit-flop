import numpy as np
import pytest

from hitflop import ensembles as E, metrics as M


def _toy(seed=0, n=600):
    rg = np.random.default_rng(seed)
    y = rg.integers(0, 2, n)
    good = np.clip(0.5 + 0.25 * (y - .5) + rg.normal(0, .15, n), .01, .99)
    noise = rg.random(n)
    folds = [np.arange(i * 100, (i + 1) * 100) for i in range(1, 6)] if False else [np.arange(i * 100, (i + 1) * 100) for i in range(6)]
    return y, {"good": good, "noise": noise}, folds


def test_weights_favour_the_better_model_and_sum_to_one():
    y, d, _ = _toy()
    P = np.column_stack([d["good"], d["noise"]])
    w = E.auc_weights(P, y)
    assert w.sum() == pytest.approx(1) and w[0] > 0.9


def test_ensembles_never_use_the_validated_folds_labels():
    y, d, folds = _toy()
    rg = np.random.default_rng(9)
    d = {"good": d["good"], "mid": np.clip(0.5 + 0.08 * (y - .5) + rg.normal(0, .15, len(y)), .01, .99)}   # two informative models, so weights vary continuously
    base, w, names = E.walk_forward_ensembles(d, y, folds)
    y2 = y.copy(); y2[folds[3]] = 1 - y2[folds[3]]                 # flip labels of fold 3
    alt, w2, _ = E.walk_forward_ensembles(d, y2, folds)
    for key in ("weighted_vote", "stacking"):
        assert np.allclose(base[key][np.concatenate(folds[:4])], alt[key][np.concatenate(folds[:4])]), key   # folds 0-3 unchanged
        assert not np.allclose(base[key][folds[4]], alt[key][folds[4]]), key                               # fold 4 uses fold 3's labels, so it moves
    assert np.allclose(base["average"], alt["average"])


def test_stacking_and_vote_beat_the_noise_model():
    y, d, folds = _toy(seed=3)
    out, _, _ = E.walk_forward_ensembles(d, y, folds)
    rows = np.concatenate(folds[1:])
    assert M.auc(y[rows], out["weighted_vote"][rows]) > M.auc(y[rows], d["noise"][rows]) + 0.2
    assert M.auc(y[rows], out["stacking"][rows]) > M.auc(y[rows], d["noise"][rows]) + 0.2


def test_final_combiners_shapes():
    y, d, _ = _toy()
    P = np.column_stack([d["good"], d["noise"]])
    out, w = E.fit_final_combiners(P[:500], y[:500], P[500:])
    assert all(len(v) == 100 for v in out.values()) and w.sum() == pytest.approx(1)
