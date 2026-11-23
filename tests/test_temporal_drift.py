import numpy as np
import pandas as pd

from hitflop import drift as D
from hitflop import temporal as T


def test_decay_weights_halve_at_the_half_life_and_ignore_labels():
    dates = np.array(["2020-01-01", "2021-01-01", "2022-01-01"], dtype="datetime64[D]")
    w = T.decay_weights(dates, 1.0)
    raw = w / w[-1]
    assert abs(raw[1] - 0.5) < 0.01 and abs(raw[0] - 0.25) < 0.01              # one year older = half the weight
    assert abs(w.mean() - 1) < 1e-12 and (np.diff(w) > 0).all()


def test_spec_parser():
    assert T.parse_spec("expanding") == (None, None)
    assert T.parse_spec("rolling-150") == (150, None)
    assert T.parse_spec("decay-2y") == (None, 2.0)


def test_rolling_window_trains_only_on_recent_films(world):
    dev = world["dev"]
    tr, va = world["folds"][3]
    seen = {}
    orig = T.MD.make_pipe

    def spy(model_key, set_name, params=None):
        pipe = orig(model_key, set_name, params)
        real_fit = pipe.fit
        def fit(X, y, **kw):
            seen["rows"] = len(X); seen["last"] = X.index.max()
            return real_fit(X, y, **kw)
        pipe.fit = fit
        return pipe
    T.MD.make_pipe = spy
    try:
        T._job(dev, dev.hit.values, tr, va, "LR", "A", 150, None, False)
    finally:
        T.MD.make_pipe = orig
    assert seen["rows"] == 150 and seen["last"] < va.min()


def test_decay_does_not_change_prediction_shape_or_leak_future(world):
    dev = world["dev"]
    tr, va = world["folds"][2]
    v1, p1 = T._job(dev, dev.hit.values, tr, va, "LR", "A", None, 2.0, False)
    y2 = dev.hit.values.copy()
    y2[va] = 1 - y2[va]                                                       # flipping validation labels must not move predictions
    v2, p2 = T._job(dev, y2, tr, va, "LR", "A", None, 2.0, False)
    assert np.allclose(p1, p2) and len(p1) == len(va)


def test_weighted_fit_differs_from_unweighted(world):
    dev = world["dev"]
    tr, va = world["folds"][3]
    _, a = T._job(dev, dev.hit.values, tr, va, "LR", "A", None, None, False)
    _, b = T._job(dev, dev.hit.values, tr, va, "LR", "A", None, 1.0, False)
    assert not np.allclose(a, b)


def test_wilson_interval_contains_the_estimate_and_is_inside_0_1():
    lo, hi = D.wilson(30, 100)
    assert 0 <= lo < 0.3 < hi <= 1
    assert D.wilson(0, 10)[0] == 0 or D.wilson(0, 10)[0] < 1e-12


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rg = np.random.default_rng(0)
    a = rg.normal(0, 1, 2000)
    assert D.psi(a, a) < 1e-6
    assert D.psi(a, rg.normal(0, 1, 2000)) < 0.1
    assert D.psi(a, rg.normal(1.5, 1, 2000)) > 0.25


def test_trend_detects_a_falling_hit_rate_and_ignores_a_flat_one():
    rg = np.random.default_rng(1)
    n = 400
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")
    falling = pd.DataFrame({"release_date": dates, "hit": (rg.random(n) < np.linspace(0.6, 0.25, n)).astype(int)})
    flat = pd.DataFrame({"release_date": dates, "hit": (rg.random(n) < 0.45).astype(int)})
    f, g = D.hit_rate_trend(falling), D.hit_rate_trend(flat)
    assert f["diff_ci_high"] < 0 and f["kendall_p"] < 0.01
    assert g["diff_ci_low"] < 0 < g["diff_ci_high"]


def test_feature_drift_flags_shifted_budget(world):
    dev = world["dev"]
    cur = dev.copy()
    cur["log_budget"] = cur.log_budget + 2
    t = D.feature_drift(dev, cur).set_index("feature")
    assert t.loc["log_budget", "psi_band"] == "large" and t.loc["runtime", "psi_band"] == "small"


def test_unlabelled_test_features_carry_no_label_columns():
    x = D.unlabelled_test_features()
    assert len(x) == 276
    assert not (set(x.columns) & D.LABEL_COLUMNS)
