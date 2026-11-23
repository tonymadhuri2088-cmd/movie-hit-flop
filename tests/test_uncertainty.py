import numpy as np
import pytest

from hitflop import recalibration as R
from hitflop import uncertainty as U


# ---------------- recalibration ----------------
def _overconfident(n=600, seed=0):
    rg = np.random.default_rng(seed)
    z = rg.normal(0, 1, n)
    y = (rg.random(n) < 1 / (1 + np.exp(-(0.6 * z - 0.4)))).astype(int)
    p = 1 / (1 + np.exp(-(1.4 * z + 0.3)))                           # too extreme and too high
    return y, p


def _blocks(n=600, k=5):
    first = n // (k + 1)
    step = (n - first) // k
    idx = np.arange(n)
    return [idx[first + i * step: first + (i + 1) * step] for i in range(k)]


def test_shift_makes_mean_prediction_equal_hit_rate():
    y, p = _overconfident()
    a = R.fit_shift(p, y)
    assert abs(R._sigmoid(R._logit(p) + a).mean() - y.mean()) < 1e-6


def test_slope_and_intercept_detect_overconfidence():
    y, p = _overconfident()
    slope, icpt = R.calibration_slope_intercept(y, p)
    assert slope < 0.8 and icpt < -0.3


def test_recalibration_improves_a_miscalibrated_model():
    y, p = _overconfident()
    folds = _blocks()
    for m in R.METHODS:
        cal, ok = R.expanding_recalibrate(p, y, folds, m)
        assert not ok[folds[0]].any() and ok[folds[1]].all()
        assert R.neg_brier(y[ok], cal[ok]) > R.neg_brier(y[ok], p[ok]), m


def test_recalibration_never_uses_the_block_it_scores():
    y, p = _overconfident()
    folds = _blocks()
    for m in R.METHODS:
        base, _ = R.expanding_recalibrate(p, y, folds, m)
        y2 = y.copy()
        late = np.concatenate(folds[2:])
        y2[late] = 1 - y[late]                                                    # flip the labels of block 3 onward
        moved, _ = R.expanding_recalibrate(p, y2, folds, m)
        assert np.allclose(base[folds[1]], moved[folds[1]]) and np.allclose(base[folds[2]], moved[folds[2]]), m


def test_monotone_recalibration_keeps_the_ranking_inside_every_block():
    from hitflop import metrics as M
    y, p = _overconfident()
    folds = _blocks()
    cal, ok = R.expanding_recalibrate(p, y, folds, "platt_all")
    for va in folds[1:]:
        assert abs(M.auc(y[va], cal[va]) - M.auc(y[va], p[va])) < 1e-9


# ---------------- warnings ----------------
@pytest.fixture(scope="module")
def ref_basic(world):
    return U.fit_reference(world["dev"], "basic"), world["dev"]


def test_reference_flags_about_the_chosen_share_of_its_own_films(ref_basic):
    ref, dev = ref_basic
    assert 0.015 < U.flags(ref, dev).mahalanobis_high.mean() < 0.04


def test_a_typical_film_is_normal_and_an_absurd_one_is_high(ref_basic):
    ref, dev = ref_basic
    typical = dev.iloc[[100]].copy()
    absurd = typical.copy()
    absurd["log_budget"] = ref["range"]["log_budget"][3] + 3
    r_ok = U.assess(ref, typical, 0.9, 0.5)
    r_bad = U.assess(ref, absurd, 0.9, 0.5)
    assert r_bad["level"] == "high" and any("budget" in s for s in r_bad["reasons"])
    assert r_ok["level"] in ("normal", "caution", "note") and r_ok["distance"] < ref["limit"]


def test_close_call_uses_the_margin(ref_basic):
    ref, dev = ref_basic
    row = dev.iloc[[50]]
    assert U.assess(ref, row, 0.53, 0.50)["close_call"] and not U.assess(ref, row, 0.70, 0.50)["close_call"]
    r = U.assess(ref, row, 0.52, 0.50)
    assert any("Close call" in s for s in r["reasons"]) and r["level"] in ("caution", "high")


def test_rare_language_and_unseen_genre_are_flagged(ref_basic):
    ref, dev = ref_basic
    row = dev.iloc[[10]].copy()
    row["language"] = "xx"
    row["genres"] = "Underwater Opera"
    f = U.flags(ref, row).iloc[0]
    assert f.language_few_films and f.unseen_genre


def test_reference_is_built_from_the_given_films_only(world):
    early = world["dev"].iloc[:150]
    ref = U.fit_reference(early, "basic")
    assert ref["n"] == 150 and ref["range"]["log_budget"][3] == early.log_budget.max()


def test_buzz_reference_uses_the_three_page_view_columns(world):
    ref = U.fit_reference(world["dev"], "buzz")
    assert ref["cols"][-3:] == ["l_pv_90_31", "l_pv_30_8", "l_pv_7_1"] and len(ref["mu"]) == 7


def test_unusual_inputs_are_notes_not_alarms(ref_basic):
    ref, dev = ref_basic
    row = dev.iloc[[10]].copy()
    row["language"], row["genres"] = "xx", "Underwater Opera"
    r = U.assess(ref, row, 0.9, 0.5)                       # far from the Hit line, inside every seen range
    assert r["level"] == "note" and not r["reasons"] and len(r["notes"]) >= 2


def test_levels_are_ordered_high_over_caution_over_note(ref_basic):
    ref, dev = ref_basic
    row = dev.iloc[[10]].copy()
    row["language"] = "xx"
    assert U.assess(ref, row, 0.9, 0.5)["level"] == "note"
    assert U.assess(ref, row, 0.51, 0.5)["level"] == "caution"
    row["log_budget"] = ref["range"]["log_budget"][3] + 1
    assert U.assess(ref, row, 0.51, 0.5)["level"] == "high"


def test_shipped_artifact_carries_references_for_both_models():
    from hitflop import artifacts as A
    art = A.load()
    for m in ("basic", "buzz"):
        assert art["uncertainty"][m]["n"] == 854 and art["uncertainty"][m]["limit"] > 0
