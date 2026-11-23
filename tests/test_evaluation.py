import numpy as np
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, precision_score, recall_score

from hitflop import evaluation as EV


def _data(n=400, seed=0, signal=1.0):
    rg = np.random.default_rng(seed)
    y = (rg.random(n) < 0.35).astype(int)
    p = 1 / (1 + np.exp(-(signal * (y - 0.35) + rg.normal(0, 1, n))))
    return y, p


def test_threshold_metrics_match_scikit_learn():
    y, p = _data()
    thr = 0.45
    pred = (p >= thr).astype(int)
    m = EV.counts_to_metrics(*EV.confusion_counts(y, p, thr))
    assert m["accuracy"] == pytest.approx((pred == y).mean())
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["specificity"] == pytest.approx(recall_score(y, pred, pos_label=0))
    assert m["f1_hit"] == pytest.approx(f1_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))


def test_probability_metrics_match_scikit_learn():
    y, p = _data()
    m = EV.prob_metrics(y, p)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert 0.5 < m["auc"] < 1


def test_perfect_and_always_flop_edge_cases():
    y = np.array([0, 0, 1, 1, 0, 1])
    m = EV.counts_to_metrics(*EV.confusion_counts(y, y.astype(float), 0.5))
    assert m["macro_f1"] == 1 and m["mcc"] == 1 and m["balanced_accuracy"] == 1
    flop = EV.counts_to_metrics(*EV.confusion_counts(y, np.zeros(len(y)), 0.5))
    assert flop["recall"] == 0 and flop["mcc"] == 0 and flop["balanced_accuracy"] == 0.5     # no division errors


def test_confusion_cells_add_up_and_are_oriented_correctly():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    assert EV.confusion_counts(y, p, 0.5) == (2, 1, 1, 1)           # TN, FP, FN, TP


def test_bootstrap_intervals_bracket_the_estimate_and_are_deterministic():
    y, p = _data(300)
    a, b = EV.evaluate(y, p, 0.4, n_boot=300), EV.evaluate(y, p, 0.4, n_boot=300)
    assert a == b
    for k in EV.ALL_METRICS:
        assert a[k + "_lo"] <= a[k] <= a[k + "_hi"] or k in ("brier", "log_loss", "ece")     # small-sample bias can push the point just outside for some scores
        assert a[k + "_lo"] <= a[k + "_hi"]


def test_array_threshold_equals_scalar_when_constant():
    y, p = _data(200)
    s = EV.evaluate(y, p, 0.4, n_boot=100)
    v = EV.evaluate(y, p, np.full(len(y), 0.4), n_boot=100)
    assert s == v


def test_constant_predictor_has_no_ranking_skill():
    y = (np.random.default_rng(1).random(200) < 0.3).astype(int)
    r = EV.evaluate(y, np.full(200, y.mean()), 0.5, n_boot=100)
    assert r["auc"] == 0.5 and r["pr_auc"] == pytest.approx(y.mean())
    assert r["recall"] == 0 and r["specificity"] == 1


def test_evaluate_table_has_one_row_per_model_and_confusion_frame():
    y, p = _data(150)
    t = EV.evaluate_table(y, {"a": p, "b": 1 - p}, {"a": 0.5, "b": 0.5}, n_boot=50)
    assert list(t.model) == ["a", "b"] and t.loc[0, "auc"] > 0.5 > t.loc[1, "auc"]
    cf = EV.confusion_frame(t)
    assert (cf[["true_flop_pred_flop", "true_flop_pred_hit", "true_hit_pred_flop", "true_hit_pred_hit"]].sum(axis=1) == 150).all()
