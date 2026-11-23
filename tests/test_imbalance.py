import numpy as np
import pytest

from hitflop import imbalance as IM, metrics as M, models as MD, protocol as P, thresholds as TH
from hitflop import upgrade_phase2 as U2


def test_plain_variant_is_the_original_pipeline(world):
    dev = world["dev"].iloc[:200]
    cols = MD.feature_columns("A")
    y = dev.hit.values
    a = MD.make_pipe("LR", "A").fit(dev[cols], y).predict_proba(dev[cols])[:, 1]
    b = IM.make_variant("LR", "A", "base").fit(dev[cols], y).predict_proba(dev[cols])[:, 1]
    assert np.allclose(a, b)


def test_every_variant_runs_and_grid_names_match(world):
    dev = world["dev"].iloc[:150]
    cols, y = MD.feature_columns("B3"), world["dev"].hit.values[:150]
    for m, variants in IM.SUPPORTED.items():
        for v in variants:
            pipe = IM.make_variant(m, "B3", v)
            for key in MD.MODELS[m][1]:
                assert key.split("__")[0] in pipe.named_steps                       # same tuning grid applies to every variant
            p = pipe.fit(dev[cols], y).predict_proba(dev[cols])[:, 1]
            assert p.shape == (150,) and np.isfinite(p).all()


def test_unsupported_variant_is_rejected():
    with pytest.raises(ValueError):
        IM.make_variant("LR", "A", "balanced_rf")


def test_smote_only_resamples_training_rows():
    rg = np.random.default_rng(0)
    X = np.c_[rg.normal(size=(120, 3)), (rg.random(120) < .4).astype(float)]
    y = np.r_[np.ones(20), np.zeros(100)].astype(int)
    Xr, yr = IM.BinaryAwareSMOTE().fit_resample(X, y)
    assert len(yr) > len(y) and (yr == 1).sum() == (yr == 0).sum() == 100
    assert np.allclose(Xr[:120], X)                                              # original rows are untouched, synthetic rows are appended
    assert set(np.unique(Xr[:, 3])) <= {0.0, 1.0}                                 # a yes/no column stays yes/no (no fractions)


def test_smote_inside_pipeline_does_not_change_prediction_row_count(world):
    dev = world["dev"].iloc[:180]
    cols, y = MD.feature_columns("A"), dev.hit.values
    pipe = IM.make_variant("LR", "A", "smote").fit(dev[cols].iloc[:120], y[:120])
    assert len(pipe.predict_proba(dev[cols].iloc[120:])) == 60                    # resampling never applies at prediction time


def test_smote_is_a_no_op_when_classes_are_already_balanced():
    X = np.random.default_rng(1).normal(size=(40, 3))
    y = np.r_[np.ones(20), np.zeros(20)].astype(int)
    Xr, yr = IM.BinaryAwareSMOTE().fit_resample(X, y)
    assert len(yr) == 40


def test_simulated_minority_keeps_every_flop_and_order(world):
    dev = world["dev"]
    sim = IM.simulate_minority(dev, 0.30, 7)
    assert abs(sim.hit.mean() - 0.30) < 0.01
    assert (sim.hit == 0).sum() == (dev.hit == 0).sum()
    assert sim.release_date.is_monotonic_increasing and sim.release_date.max() < np.datetime64("2023-01-01")


def test_walk_forward_variant_reproduces_stored_predictions(world):
    dev = world["dev"]
    p3 = __import__("pickle").load(open(__import__("hitflop").config.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    oof, folds, _ = IM.walk_forward_variant(dev, dev.hit.values, "LR", "A", "base")
    assert np.allclose(oof, p3["oof"][("LR", "A")], equal_nan=True)
    assert len(folds) == 5


# ---- threshold rules ----
def _scores(n=400, seed=5):
    rg = np.random.default_rng(seed)
    y = (rg.random(n) < 0.35).astype(int)
    p = np.clip(0.35 + 0.25 * (y - 0.35) + rg.normal(0, 0.15, n), 0.01, 0.99)
    return y, p


def test_each_rule_returns_a_threshold_inside_the_score_range():
    y, p = _scores()
    for name in TH.RULES:
        t = TH.choose(name, y, p)
        assert p.min() - 1e-9 <= t <= p.max() + 1e-9, name


def test_prior_match_predicts_the_earlier_hit_share():
    y, p = _scores()
    t = TH.choose("prior_match", y, p)
    assert abs((p >= t).mean() - y.mean()) < 0.02


def test_recall_rule_reaches_its_target():
    y, p = _scores()
    t = TH.choose("recall_60", y, p)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.6


def test_f1_rule_beats_a_fixed_cutoff_on_its_own_data():
    y, p = _scores()
    def f1(t):
        pr = p >= t
        return 2 * (pr & (y == 1)).sum() / (pr.sum() + y.sum())
    assert f1(TH.choose("max_f1", y, p)) >= f1(0.5) - 1e-12


def test_threshold_rules_never_see_the_block_they_score():
    y, p = _scores(600)
    idx = np.arange(600)
    folds = [idx[100 + i * 100: 200 + i * 100] for i in range(5)]
    for rule in TH.RULES:
        base, _ = P.expanding_threshold(y, p, folds, rule)
        y2, p2 = y.copy(), p.copy()
        late = np.concatenate(folds[2:])
        y2[late], p2[late] = 1 - y[late], 1 - p[late]                             # corrupt block 3 onward
        moved, _ = P.expanding_threshold(y2, p2, folds, rule)
        assert np.allclose(base[folds[1]], moved[folds[1]]) and np.allclose(base[folds[2]], moved[folds[2]]), rule


def test_balanced_accuracy_helper_matches_scikit_learn():
    from sklearn.metrics import balanced_accuracy_score
    y, p = _scores()
    pred = (p >= .5).astype(int)
    assert abs(U2.balanced_accuracy_binary(y, pred) - balanced_accuracy_score(y, pred)) < 1e-12
