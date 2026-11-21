import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from hitflop import calibration as CAL, metrics as M, models as MD, validation as V
from hitflop import features as F, history as H


@pytest.fixture(scope="module")
def dev(world):
    lab = F.add_research_features(world["lab"], world["merged"])
    lab = lab.join(H.add_track_records(lab, world["merged"]))
    return lab[lab.release_date < "2023-01-01"].reset_index(drop=True)


def test_auc_matches_sklearn():
    rg = np.random.default_rng(0)
    y = rg.integers(0, 2, 200); p = rg.random(200) + 0.3 * y
    assert M.auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_metrics_hand_example():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.4, 0.6, 0.1])
    m = M.classification_metrics(y, p, 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5 and m["mcc"] == 0.0
    assert M.full_metrics(y, p)["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_ece_perfect_and_bad():
    rg = np.random.default_rng(5); p = np.linspace(0.02, 0.98, 20000); y = (rg.random(20000) < p).astype(int)
    assert M.ece(y, p) < 0.02                                    # calibrated probabilities
    assert M.ece(np.ones(100), np.zeros(100) + np.linspace(0, 0.01, 100)) == pytest.approx(1.0, abs=0.01)   # confidently wrong


def test_bootstrap_is_deterministic_and_brackets_estimate():
    rg = np.random.default_rng(1); y = rg.integers(0, 2, 300); p = rg.random(300) + 0.4 * y
    a, b = M.bootstrap_ci(y, p, n=300), M.bootstrap_ci(y, p, n=300)
    assert a == b and a[0] < M.auc(y, p) < a[1]


def test_paired_bootstrap_detects_real_gain_and_not_noise():
    rg = np.random.default_rng(2); y = rg.integers(0, 2, 600)
    good = rg.normal(size=600) + 1.5 * y; bad = rg.normal(size=600)
    d, lo, hi, p = M.paired_bootstrap(y, good, bad, n=500)
    assert d > 0.2 and lo > 0 and p < 0.01
    d2, lo2, hi2, p2 = M.paired_bootstrap(y, bad, rg.normal(size=600), n=500)
    assert lo2 < 0 < hi2


def test_benjamini_hochberg_known_values():
    q = M.benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.04, 0.04, 0.5])


def test_mcnemar():
    y = np.array([1] * 10); a = np.array([1] * 10); b = np.array([0] * 8 + [1] * 2)
    nb, nc, p = M.mcnemar_exact(y, a, b)
    assert (nb, nc) == (8, 0) and p < 0.01


def test_expanding_calibration_uses_only_earlier_folds():
    rg = np.random.default_rng(3); n = 500
    y = rg.integers(0, 2, n); p = np.clip(0.5 + 0.2 * (y - 0.5) + rg.normal(0, .15, n), 0.01, 0.99)
    folds = [np.arange(i * 100, (i + 1) * 100) for i in range(5)]
    cal, done = CAL.expanding_calibration(p, y, folds)
    assert not done[:100].any() and done[100:].all()
    # flipping labels of fold 3 must not change calibrated predictions of folds 1-3 (they never saw those labels) but must change fold 4
    y2 = y.copy(); y2[300:400] = 1 - y2[300:400]
    cal2, _ = CAL.expanding_calibration(p, y2, folds)
    assert np.allclose(cal[100:400], cal2[100:400]) and not np.allclose(cal[400:], cal2[400:])


def test_walk_forward_folds_and_no_leak(dev):
    y = dev.hit.values
    oof, folds, _ = V.walk_forward(dev, y, "LR", "A", tune=False)
    assert np.isnan(oof).sum() == len(dev) - 480 and np.isnan(oof[: len(dev) - 480]).all()
    a = M.auc(y[~np.isnan(oof)], oof[~np.isnan(oof)])
    assert 0.55 < a < 0.8
    # flipping the labels of the final fold's VALIDATION movies must not change any earlier fold's predictions
    y2 = y.copy(); y2[folds[-1]] = 1 - y2[folds[-1]]
    oof2, _, _ = V.walk_forward(dev, y2, "LR", "A", tune=False)
    early = np.concatenate(folds[:-1])
    assert np.allclose(oof[early], oof2[early])


def test_walk_forward_is_deterministic(dev):
    y = dev.hit.values
    o1, _, _ = V.walk_forward(dev, y, "RF", "B", tune=False)
    o2, _, _ = V.walk_forward(dev, y, "RF", "B", tune=False)
    assert np.allclose(o1, o2, equal_nan=True)


def test_common_cohort_has_both_sources(dev):
    c = V.common_cohort(dev)
    assert (c.pv_30.notna() & (c.in_yt_file == 1)).all() and 100 < len(c) < len(dev)


def test_feature_sets_have_unique_valid_columns(dev):
    for s in MD.FEATURE_SETS:
        cols = MD.feature_columns(s)
        assert len(cols) == len(set(cols)) and set(cols) <= set(dev.columns), s
    forbidden = {"rating", "vote_count", "revenue", "roi", "hit", "budget_source", "revenue_source", "name", "tmdb_id"}
    for s in MD.FEATURE_SETS:
        assert not (set(MD.feature_columns(s)) & forbidden)
