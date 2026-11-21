import numpy as np
import pandas as pd
import pytest

from hitflop import data_audit, selection_bias as sb


def test_audit_tables_exist_and_consistent(world):
    merged, lab = world["merged"], world["lab"]
    a = data_audit.run_audit(merged, lab)
    assert a["missingness"].loc["budget", "missing_pct_labelled"] == 0
    assert a["duplicates"].loc["duplicate_tmdb_id", "value"] == 0
    s = a["labelled_share_language"]
    assert s.labelled.sum() == 854 and (s.share.between(0, 1)).all()
    q = a["roi_quality"].value
    assert q["hit_rate_roi_ge_1.5"] > q["hit_rate_roi_ge_2.0"] > q["hit_rate_roi_ge_2.5"]


@pytest.fixture(scope="module")
def sel(world):
    m = sb.selection_frame(world["merged"], world["lab"])
    p, auc, coefs = sb.selection_model(m)
    return m, p, auc, sb.ipw_weights(m, p)


def test_selection_frame_only_mature_movies(world, sel):
    m = sel[0]
    assert m.release_date.max() <= world["merged"].release_date.max() - pd.Timedelta(days=120)
    assert int(m.is_labelled.sum()) == 854


def test_selection_model_is_reproducible_and_informative(world, sel):
    m, p, auc, w = sel
    p2, auc2, _ = sb.selection_model(m)
    assert auc == pytest.approx(auc2) and auc > 0.7


def test_weights_positive_capped_mean_one(sel):
    w = sel[3]
    assert (w > 0).all() and w.mean() == pytest.approx(1.0) and len(w) == 854
    assert 1 < sb.effective_sample_size(w) <= len(w)


def test_weighted_hit_rate_equals_unweighted_for_uniform_weights(world):
    lab = world["lab"]
    w = pd.Series(1.0, index=lab.tmdb_id.values)
    r = sb.weighted_hit_rate(lab, w)
    assert r["unweighted"] == pytest.approx(r["weighted"])
