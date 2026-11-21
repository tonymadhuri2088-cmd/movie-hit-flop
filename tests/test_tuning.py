import numpy as np
import pytest

from hitflop import features as F, history as H, tuning as T


@pytest.fixture(scope="module")
def dev(world):
    lab = F.add_research_features(world["lab"], world["merged"])
    lab = lab.join(H.add_track_records(lab, world["merged"]))
    return lab[lab.release_date < "2023-01-01"].reset_index(drop=True)


def test_tuning_is_deterministic_and_default_is_candidate(dev):
    X = dev.iloc[:200][__import__("hitflop").models.feature_columns("A")].reset_index(drop=True)
    y = dev.hit.values[:200]
    a = T.tune_window(X, y, "XGB", "A", n_trials=4)
    b = T.tune_window(X, y, "XGB", "A", n_trials=4)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == 4


def test_nested_oof_never_uses_outer_validation_labels(dev):
    y = dev.hit.values
    small = dev.iloc[:300].reset_index(drop=True)
    ys = y[:300]
    oof, folds, info = T.nested_optuna_oof(small, ys, "LGBM", "A", n_trials=3, n_outer=3, n_jobs=1)
    y2 = ys.copy(); y2[folds[-1]] = 1 - y2[folds[-1]]
    oof2, _, _ = T.nested_optuna_oof(small, y2, "LGBM", "A", n_trials=3, n_outer=3, n_jobs=1)
    early = np.concatenate(folds[:-1])
    assert np.allclose(oof[early], oof2[early]) and len(info) == 3
