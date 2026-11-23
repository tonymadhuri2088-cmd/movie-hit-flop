import numpy as np
import pytest

from hitflop import ablation as AB
from hitflop import models as MD
from hitflop import protocol as P
from hitflop import upgrade_phase5 as U5


@pytest.fixture(scope="module")
def full_dev():
    return P.dev_only_world()[2]            # every feature block (history, calendar, competition) is present here


def test_groups_partition_the_extra_and_base_columns():
    assert set(AB.ALL_GROUPS) == set(AB.GROUPS)
    cols = AB.columns_for(AB.ALL_GROUPS)
    assert len(cols) == len(set(cols))                                   # no column belongs to two groups
    assert set(AB.columns_for(AB.BASE_GROUPS)) == set(MD.feature_columns("A"))


def test_base_groups_reproduce_set_a(full_dev):
    dev = full_dev.iloc[:200]
    y = dev.hit.values
    cols = MD.feature_columns("A")
    a = MD.make_pipe("LR", "A").fit(dev[cols], y).predict_proba(dev[cols])[:, 1]
    b = AB.make_pipe_groups("LR", AB.BASE_GROUPS).fit(dev[AB.columns_for(AB.BASE_GROUPS)], y).predict_proba(dev[AB.columns_for(AB.BASE_GROUPS)])[:, 1]
    assert np.allclose(a, b)


@pytest.mark.parametrize("group", AB.ALL_GROUPS)
def test_every_group_alone_and_removed_runs(full_dev, group):
    dev = full_dev.iloc[:160]
    y = dev.hit.values
    for groups in ([group], [g for g in AB.ALL_GROUPS if g != group]):
        cols = AB.columns_for(groups)
        p = AB.make_pipe_groups("LR", groups).fit(dev[cols], y).predict_proba(dev[cols])[:, 1]
        assert p.shape == (160,) and np.isfinite(p).all()


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError):
        AB.make_preprocessor_groups([], scale=True)


def test_design_has_expected_configurations():
    cfgs = U5.design()
    assert len(cfgs) == 2 + 10 + 10 + 4
    assert cfgs["minus budget"][1] == "leave-one-out" and "budget" not in cfgs["minus budget"][0]
    assert cfgs["only genres"][0] == ["genres"]
    assert cfgs["base plus buzz_wikipedia"][0][-1] == "buzz_wikipedia"


def test_walk_forward_groups_scores_only_later_blocks(full_dev):
    dev = full_dev.iloc[:300]
    oof, folds = AB.walk_forward_groups(dev, dev.hit.values, "LR", ["budget", "language"], n_outer=3, tune=False, n_jobs=1)
    first_validation_start = min(f.min() for f in folds)
    assert np.isnan(oof[:first_validation_start]).all()
    assert np.isfinite(oof[first_validation_start:]).all()
