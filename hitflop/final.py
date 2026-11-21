"""Final, one-time evaluation on the frozen 2023+ test set. Specs were declared in the experiment log BEFORE this ran and nothing is selected on test.
A lock file stops accidental re-scoring: the test set is scored exactly once.
Procedure per spec: tune on ALL development data with the same inner chronological grid used in validation, predict the test movies,
threshold = Youden point of the development out-of-fold predictions (never fitted on test), no calibration (it hurt under drift in validation)."""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C, metrics as M, models as MD, selection_bias as sb, validation as V
from .experiment_log import log_experiment
from .world import build_world

OUT = C.OUT_DIR / "final"
LOCK = OUT / "TEST_SCORED.lock"
SPECS = {                                   # name -> (model, feature set)
    "basic (HGB, set A)": ("HGB", "A"),
    "buzz (XGB, set B3)": ("XGB", "B3"),
    "reference: LR, set A": ("LR", "A"),
    "reference: XGB, set D (earlier headline)": ("XGB", "D"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and not os.environ.get("ALLOW_TEST_RESCORE"):
        raise SystemExit("The frozen test set was already scored once (see outputs/final). Refusing to score it again.")
    t0 = time.time()
    merged, lab, dev, test = build_world()
    ydev, ytest = dev.hit.values, test.hit.values
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    folds = p3["folds"]
    dev_ok = ~np.isnan(p3["oof"][("LR", "A")])
    pred, thr, fitted = {}, {}, {}
    for name, (mk, s) in SPECS.items():
        cols = MD.feature_columns(s)
        gs = MD.tune_and_fit(dev[cols], ydev, mk, s)
        fitted[name] = gs.best_estimator_
        pred[name] = gs.predict_proba(test[cols])[:, 1]
        oof = p3["oof"].get((mk, s))
        if oof is None:
            oof, _, _ = V.walk_forward(dev, ydev, mk, s)
        thr[name] = M.youden_threshold(ydev[dev_ok], oof[dev_ok])
        print(f"{name} fitted {time.time() - t0:.0f}s", flush=True)
    base_rate_dev = float(ydev.mean())
    pred["dummy (always the development Hit rate)"] = np.full(len(test), base_rate_dev)
    thr["dummy (always the development Hit rate)"] = 0.5

    rows = []
    for name, p in pred.items():
        m = M.full_metrics(ytest, p, threshold=0.5)
        lo, hi = M.bootstrap_ci(ytest, p, n=2000) if np.ptp(p) > 0 else (0.5, 0.5)
        plo, phi = M.bootstrap_ci(ytest, p, metric=M.pr_auc, n=2000) if np.ptp(p) > 0 else (np.nan, np.nan)
        mt = M.classification_metrics(ytest, p, thr[name])
        rows.append({"model": name, "auc": m["auc"], "auc_ci_low": lo, "auc_ci_high": hi, "pr_auc": m["pr_auc"], "pr_ci_low": plo, "pr_ci_high": phi, "brier": m["brier"], "log_loss": m["log_loss"], "ece": m["ece"],
                     "mean_predicted": float(np.mean(p)), "observed_hit_rate": float(ytest.mean()),
                     "acc@0.5": m["accuracy"], "precision@0.5": m["precision"], "recall@0.5": m["recall"], "f1@0.5": m["f1"], "mcc@0.5": m["mcc"],
                     "dev_youden_threshold": thr[name], "acc@dev_thr": mt["accuracy"], "precision@dev_thr": mt["precision"], "recall@dev_thr": mt["recall"], "f1@dev_thr": mt["f1"], "mcc@dev_thr": mt["mcc"]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "test_results.csv", index=False)

    comps = []
    ref = "reference: LR, set A"
    for name in [n for n in SPECS if n != ref]:
        d, lo, hi, p = M.paired_bootstrap(ytest, pred[name], pred[ref], n=2000)
        comps.append({"model": name, "vs": ref, "auc_diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    d, lo, hi, p = M.paired_bootstrap(ytest, pred["buzz (XGB, set B3)"], pred["basic (HGB, set A)"], n=2000)
    comps.append({"model": "buzz (XGB, set B3)", "vs": "basic (HGB, set A)", "auc_diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    comps = pd.DataFrame(comps)
    comps["q_bh"] = M.benjamini_hochberg(comps.p.values)
    comps.to_csv(OUT / "test_comparisons.csv", index=False)

    # language robustness and selection-bias-weighted AUC on the test set
    lang = []
    for name in ["basic (HGB, set A)", "buzz (XGB, set B3)"]:
        d = test.assign(p=pred[name])
        for g, x in d.groupby("language"):
            lo, hi = M.bootstrap_ci(x.hit.values, x.p.values, n=1000) if 0 < x.hit.sum() < len(x) and len(x) >= 10 else (np.nan, np.nan)
            lang.append({"model": name, "language": g, "n": len(x), "hit_rate": float(x.hit.mean()), "auc": M.auc(x.hit.values, x.p.values), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(lang).to_csv(OUT / "test_by_language.csv", index=False)
    w = pd.read_csv(C.OUT_DIR / "phase1" / "ipw_weights.csv").set_index("tmdb_id").ipw_weight
    wt = test.tmdb_id.map(w).values
    ipw = [{"model": n, "auc_unweighted": M.auc(ytest, p), "auc_ipw_weighted": sb.weighted_auc(ytest, p, wt)} for n, p in pred.items() if np.ptp(p) > 0]
    pd.DataFrame(ipw).to_csv(OUT / "test_ipw_sensitivity.csv", index=False)
    pickle.dump({"pred": pred, "test_ids": test.tmdb_id.values, "y": ytest, "thr": thr}, open(OUT / "test_predictions.pkl", "wb"))
    LOCK.write_text("scored once on " + time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()))
    log_experiment("final_test_scoring", "final", {"specs": {k: list(v) for k, v in SPECS.items()}, "threshold": "dev out-of-fold Youden", "calibration": "none", "test_n": int(len(test))},
                   {r["model"]: {"auc": round(r["auc"], 4), "pr_auc": round(r["pr_auc"], 4), "brier": round(r["brier"], 4)} for r in rows},
                   "Scored once. Nothing selected on test. Test Hit rate is lower than development (distribution shift).")
    print(res.round(3).to_string())
    print(comps.round(3).to_string())
    print("ALL DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
