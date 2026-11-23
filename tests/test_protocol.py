import pickle

import numpy as np
import pandas as pd
import pytest

from hitflop import config as C, metrics as M, protocol as P
from hitflop.experiment_log import read_log


def test_test_rows_are_refused(world):
    with pytest.raises(ValueError):
        P.ensure_dev_only(world["test"])
    assert P.ensure_dev_only(world["dev"]) is world["dev"]


def test_dev_only_world_never_contains_test_films():
    _, lab, dev = P.dev_only_world()
    assert pd.Timestamp(dev.release_date.max()) < pd.Timestamp(C.SPLIT_DATE)
    assert pd.Timestamp(lab.release_date.max()) < pd.Timestamp(C.SPLIT_DATE)
    assert len(dev) == 578


def _folds(n=500, k=5):
    idx = np.arange(n)
    first = n // (k + 1)
    return [idx[first + i * ((n - first) // k): first + (i + 1) * ((n - first) // k)] for i in range(k)]


def test_expanding_threshold_uses_only_earlier_blocks():
    rg = np.random.default_rng(3)
    n = 500
    y = (rg.random(n) < 0.4).astype(int)
    p = np.clip(0.4 + 0.2 * (y - 0.4) + rg.normal(0, 0.15, n), 0.01, 0.99)
    folds = _folds(n)
    thr, scored = P.expanding_threshold(y, p, folds)
    assert not scored[folds[0]].any() and scored[folds[1]].all()          # the first block has no earlier data
    # leakage test: flipping labels and predictions of block k and later must not move the threshold used for block k
    y2, p2 = y.copy(), p.copy()
    late = np.concatenate(folds[2:])
    y2[late], p2[late] = 1 - y2[late], 1 - p2[late]
    thr2, _ = P.expanding_threshold(y2, p2, folds)
    assert np.allclose(thr[folds[1]], thr2[folds[1]]) and np.allclose(thr[folds[2]], thr2[folds[2]])
    assert not np.allclose(thr[folds[3]], thr2[folds[3]])               # block 3 does depend on blocks 1 to 2, which changed


def test_constant_predictor_threshold_falls_back_to_half():
    y = np.array([0, 1] * 120)
    thr, _ = P.expanding_threshold(y, np.full(240, 0.4), _folds(240))
    assert np.isfinite(thr[~np.isnan(thr)]).all()


def test_claim_rule_wording():
    assert P.verdict(0.03, 0.01, 0.05, 0.01) == "better (supported)"
    assert P.verdict(0.03, -0.01, 0.07, 0.01) == "no clear difference"       # interval includes zero
    assert P.verdict(0.03, 0.01, 0.05, 0.20) == "no clear difference"        # not significant after correction
    assert P.verdict(-0.03, -0.05, -0.01, 0.01) == "worse (supported)"


def test_comparison_finds_a_real_gain_and_ignores_noise():
    rg = np.random.default_rng(5)
    n = 600
    y = (rg.random(n) < 0.4).astype(int)
    base = rg.normal(size=n) + 0.8 * y
    better = rg.normal(size=n) * 0.6 + 1.6 * y
    same = base + rg.normal(0, 0.01, n)
    c = P.compare_to_baseline(y, {"base": base, "better": better, "same": same}, "base", n_boot=500)
    v = dict(zip(c.model, c.verdict))
    assert v["better"] == "better (supported)" and v["same"] == "no clear difference"
    assert (c.vs == "base").all()


def test_family_correction_can_only_make_claims_harder():
    c = pd.DataFrame({"model": list("abcd"), "diff": [.03] * 4, "ci_low": [.001] * 4, "ci_high": [.06] * 4, "p": [.02, .03, .04, .5]})
    c["q_bh"], c["verdict"] = M.benjamini_hochberg(c.p.values[:1].repeat(4)), "x"
    fixed = P.correct_family(c)
    assert (fixed.q_bh >= c.p - 1e-12).all()


def test_log_run_refuses_test_use_and_records_configuration(tmp_path):
    path = tmp_path / "log.jsonl"
    with pytest.raises(ValueError):
        P.log_run("x", "upgrade-p9", {}, {}, uses_test_set=True, path=path)
    P.log_run("demo", "upgrade-p0", {"note": "a"}, {"auc": 0.7}, path=path)
    rec = read_log(path)[0]
    assert rec["settings"]["seed"] == C.SEED and rec["settings"]["uses_test_set"] is False and "claim_rule" in rec["settings"]


def test_original_dev_predictions_still_reproduce(world):
    """Reproducibility guard: the stored out-of-fold predictions still give the AUC recorded in the original study."""
    from hitflop.world import build_world
    _, _, dev, _ = build_world()
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    assert (p3["dev_ids"] == dev.tmdb_id.values).all()
    o, y = p3["oof"][("LR", "A")], dev.hit.values
    ok = ~np.isnan(o)
    assert ok.sum() == 480
    recorded = pd.read_csv(C.OUT_DIR / "phase3" / "3a_comparisons.csv")
    assert M.auc(y[ok], o[ok]) == pytest.approx(recorded[(recorded.model == "LR") & (recorded.vs == "A")].auc_baseline.iloc[0], abs=1e-9)


def test_frozen_test_lock_is_present():
    assert (C.OUT_DIR / "final" / "TEST_SCORED.lock").exists()
