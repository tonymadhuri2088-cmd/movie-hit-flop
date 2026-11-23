"""Upgrade Phase 1 runner: complete evaluation (all metrics, intervals, confusion matrices) for the deployed and reference models.

Development part: out-of-fold predictions that were already produced in the original study (nothing is refitted), scored two ways:
  * fixed threshold 0.5 on all 480 validated films
  * a threshold chosen from EARLIER validation blocks only (so blocks 2 to 5, 384 films)
Test part (REPORTING ONLY): the stored predictions of the one-time final scoring are re-expressed with the extra metrics.
Nothing is refitted, no threshold is tuned on test, and no model is selected with it.
Part A part (REPORTING ONLY): test predictions of the 19 course-notebook models, if `part_a_test_predictions.pkl` is supplied.
Run:  python -m hitflop.upgrade_phase1 [path/to/part_a_test_predictions.pkl]
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from . import config as C, evaluation as EV, metrics as M, protocol as P

OUT = C.OUT_DIR / "upgrade" / "phase1"
LOCKED_TEST_PREDICTIONS = C.OUT_DIR / "final" / "test_predictions.pkl"


def dev_predictions(dev):
    """The out-of-fold predictions of the original study for the four named specs, plus a no-skill reference."""
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    p45 = pickle.load(open(C.OUT_DIR / "phase45" / "oof_all.pkl", "rb"))
    assert (p3["dev_ids"] == dev.tmdb_id.values).all(), "development ordering differs from the stored predictions"
    folds = p3["folds"]
    y = dev.hit.values
    preds = {"reference: LR, set A": p3["oof"][("LR", "A")], "basic: HGB, set A": p3["oof"][("HGB", "A")],
             "buzz: XGB, set B3": p45["oof"][("default_XGB", "B3")], "earlier headline: XGB, set D": p3["oof"][("XGB", "D")]}
    dummy = np.full(len(dev), np.nan)
    for va in folds:
        dummy[va] = y[: va.min()].mean()                              # the Hit rate of all films before this block
    preds["no skill: earlier Hit rate"] = dummy
    return y, preds, folds


def main(part_a_path=None):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, _, dev = P.dev_only_world()
    y, preds, folds = dev_predictions(dev)
    ok = ~np.isnan(preds["reference: LR, set A"])
    yv = y[ok]
    pv = {k: v[ok] for k, v in preds.items()}

    # ---- development, fixed threshold 0.5 ----
    t05 = EV.evaluate_table(yv, pv, 0.5)
    t05.to_csv(OUT / "dev_metrics_threshold_0.5.csv", index=False)

    # ---- development, threshold from earlier blocks only ----
    thr_rows, scored = {}, None
    sub = {}
    for name, v in preds.items():
        thr, scored = P.expanding_threshold(y, v, folds)
        sub[name] = (y[scored], v[scored], thr[scored])
    texp = pd.DataFrame([{"model": n, **EV.evaluate(yy, pp, tt)} for n, (yy, pp, tt) in sub.items()])
    texp.to_csv(OUT / "dev_metrics_expanding_youden.csv", index=False)
    EV.confusion_frame(texp).to_csv(OUT / "dev_confusion_matrices.csv", index=False)

    # ---- development comparisons with the claim rule (all models versus the comparator) ----
    comps = []
    for label, metric in (("auc", M.auc), ("pr_auc", M.pr_auc)):
        c = P.compare_to_baseline(yv, {k: v for k, v in pv.items()}, "reference: LR, set A", metric=metric)
        c.insert(0, "metric", label)
        comps.append(c)
    comps = P.correct_family(pd.concat(comps))          # one family: every model and both metrics, so looking at two metrics does not inflate false positives
    comps.to_csv(OUT / "dev_comparisons_vs_logistic_regression.csv", index=False)

    # ---- test: reporting only, from the stored one-time predictions ----
    t = pickle.load(open(LOCKED_TEST_PREDICTIONS, "rb"))
    ytest = t["y"]
    test_rows = []
    for name, p in t["pred"].items():
        for rule, thr in (("dev Youden threshold", t["thr"][name]), ("fixed 0.5", 0.5)):
            test_rows.append({"model": name, "threshold_rule": rule, **EV.evaluate(ytest, p, thr)})
    test_tab = pd.DataFrame(test_rows)
    test_tab.to_csv(OUT / "test_metrics_reporting_only.csv", index=False)
    plot_confusions(test_tab[test_tab.threshold_rule == "dev Youden threshold"], OUT / "test_confusion_matrices.png")

    # ---- Part A: 19 notebook models, reporting only ----
    a_rows = None
    if part_a_path:
        pa = pickle.load(open(part_a_path, "rb"))
        a_rows = pd.DataFrame([{"model": m, "feature_group": v["group"], "threshold_rule": "validation Youden threshold", **EV.evaluate(pa["y_test"], v["p"], v["thr"])} for m, v in pa["preds"].items()])
        a_rows = a_rows.sort_values("auc", ascending=False)
        a_rows.to_csv(OUT / "part_a_19_models_test_metrics_reporting_only.csv", index=False)

    P.log_run("upgrade_phase1_extended_evaluation", "upgrade-p1",
              {"dev_specs": list(preds), "dev_scored_films_fixed_threshold": int(ok.sum()), "dev_scored_films_expanding_threshold": int(scored.sum()),
               "threshold_rules": ["fixed 0.5", "Youden point of earlier validation blocks only"], "bootstrap_resamples": 1000,
               "test_set_access": "read the stored predictions of the one-time final scoring for extra metrics; nothing refitted, tuned or selected",
               "part_a_models": None if a_rows is None else int(len(a_rows))},
              {"dev_expanding": {r["model"]: {k: round(r[k], 4) for k in ("auc", "pr_auc", "brier", "balanced_accuracy", "macro_f1", "mcc")} for _, r in texp.iterrows()},
               "comparisons_supported_better": int((comps.verdict == "better (supported)").sum()), "comparisons_total": int(len(comps))},
              "Phase 1 adds metrics only; it does not change any model. Reporting on test uses stored predictions.")
    print(f"done in {time.time() - t0:.0f}s")
    return {"t05": t05, "texp": texp, "comps": comps, "test": test_tab, "part_a": a_rows}


def plot_confusions(tab, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = tab[tab.model.str.startswith(("basic", "buzz", "reference: LR"))]
    fig, axes = plt.subplots(1, len(tab), figsize=(4.2 * len(tab), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, tab.iterrows()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
        ax.set_xticks([0, 1], ["pred Flop", "pred Hit"])
        ax.set_yticks([0, 1], ["true Flop", "true Hit"])
        ax.set_title(f"{r.model}\nthreshold {r.threshold:.2f}", fontsize=9)
    fig.suptitle("Frozen test set (276 films): confusion matrices, rows normalised", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
