"""Phase 3 runner: feature-set comparison, common cohort, ROI thresholds, language/year robustness. DEVELOPMENT DATA ONLY (before 2023-01-01).
The frozen test set is never loaded here."""
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from . import validation as V
from .experiment_log import log_experiment
from .world import build_world

OUT = C.OUT_DIR / "phase3"
MODEL_KEYS = ["LR", "HGB", "XGB", "LGBM"]
SETS_MAIN = ["A", "B", "C", "D", "B+", "C+", "D+", "A+CAL", "A+COMP", "A+CAL+COMP", "A+HIST", "D+CAL+COMP", "FULL"]
COMPARISONS = [("B+", "B"), ("C+", "C"), ("D+", "D"), ("A+CAL", "A"), ("A+COMP", "A"), ("A+CAL+COMP", "A"), ("A+HIST", "A"), ("D+CAL+COMP", "D"), ("FULL", "D"), ("B", "A"), ("C", "A"), ("D", "A")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, lab, dev, _test = build_world()          # test is returned by the builder but deliberately ignored in this module
    y = dev.hit.values
    # ---- 3a: every feature set x model on the same 480 validated movies -------------------------------------------------
    oof = {}
    for mk in MODEL_KEYS:
        for s in SETS_MAIN:
            oof[(mk, s)], folds, _ = V.walk_forward(dev, y, mk, s)
        print(f"{mk} done {time.time() - t0:.0f}s", flush=True)
    ok = ~np.isnan(next(iter(oof.values())))
    rows = []
    for (mk, s), p in oof.items():
        lo, hi = M.bootstrap_ci(y[ok], p[ok], n=1000)
        rows.append({"model": mk, "set": s, "auc": M.auc(y[ok], p[ok]), "ci_low": lo, "ci_high": hi, "pr_auc": M.pr_auc(y[ok], p[ok]), "brier": M.full_metrics(y[ok], p[ok])["brier"],
                     "fold_auc_mean": float(np.mean(M.fold_aucs(y, p, folds))), "fold_auc_sd": float(np.std(M.fold_aucs(y, p, folds), ddof=1))})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "3a_feature_sets_by_model.csv", index=False)
    comp = []
    for mk in MODEL_KEYS:
        res = {s: oof[(mk, s)] for s in SETS_MAIN}
        for new, base in COMPARISONS:
            c = V.compare_feature_sets(dev, y, res, base, [new], n_boot=2000)
            c.insert(0, "model", mk)
            comp.append(c)
    comp = pd.concat(comp, ignore_index=True)
    comp["q_bh_all"] = M.benjamini_hochberg(comp.p.values)                      # one correction across every comparison in this table
    comp["convincing"] = (comp.ci_low > 0) & (comp.q_bh_all < 0.05)
    comp["fold_wins"] = [int(np.sum(np.array(M.fold_aucs(y, oof[(r.model, r.set)], folds)) > np.array(M.fold_aucs(y, oof[(r.model, r.vs)], folds)))) for r in comp.itertuples()]
    comp.to_csv(OUT / "3a_comparisons.csv", index=False)
    pickle.dump({"oof": oof, "folds": folds, "dev_ids": dev.tmdb_id.values}, open(OUT / "oof_3a.pkl", "wb"))
    print(f"3a done {time.time() - t0:.0f}s; convincing comparisons: {int(comp.convincing.sum())} of {len(comp)}", flush=True)

    # ---- 3b: identical common cohort (both Wikipedia and YouTube) ---------------------------------------------------------
    coh = V.common_cohort(dev)
    yc = coh.hit.values
    coh_rows, coh_oof = [], {}
    for mk in ["LR", "HGB"]:
        for s in ["A", "B", "C", "D", "B+", "C+", "D+"]:
            p, f, _ = V.walk_forward(coh, yc, mk, s)
            coh_oof[(mk, s)] = p
        okc = ~np.isnan(coh_oof[(mk, "A")])
        for s in ["A", "B", "C", "D", "B+", "C+", "D+"]:
            p = coh_oof[(mk, s)]
            lo, hi = M.bootstrap_ci(yc[okc], p[okc], n=1000)
            d, dlo, dhi, pv = M.paired_bootstrap(yc[okc], p[okc], coh_oof[(mk, "A")][okc], n=2000) if s != "A" else (0, 0, 0, 1)
            coh_rows.append({"model": mk, "set": s, "n_validated": int(okc.sum()), "auc": M.auc(yc[okc], p[okc]), "ci_low": lo, "ci_high": hi, "diff_vs_A": d, "diff_ci_low": dlo, "diff_ci_high": dhi, "p": pv})
    cohort_tab = pd.DataFrame(coh_rows)
    cohort_tab["q_bh"] = M.benjamini_hochberg(np.where(cohort_tab.set == "A", 1.0, cohort_tab.p))
    cohort_tab.to_csv(OUT / "3b_common_cohort.csv", index=False)
    print(f"3b done {time.time() - t0:.0f}s (cohort dev n={len(coh)})", flush=True)

    # ---- 3c: ROI thresholds 1.5 / 2.0 / 2.5 --------------------------------------------------------------------------------
    thr_rows = []
    for thr in C.ROI_THRESHOLDS:
        yt = (dev.roi >= thr).astype(int).values
        for mk in ["LR", "HGB"]:
            for s in ["A", "B", "D", "D+", "FULL"]:
                p, f, _ = V.walk_forward(dev, yt, mk, s)
                okt = ~np.isnan(p)
                lo, hi = M.bootstrap_ci(yt[okt], p[okt], n=1000)
                thr_rows.append({"threshold": thr, "model": mk, "set": s, "hit_rate_validated": float(yt[okt].mean()), "auc": M.auc(yt[okt], p[okt]), "ci_low": lo, "ci_high": hi, "pr_auc": M.pr_auc(yt[okt], p[okt])})
    pd.DataFrame(thr_rows).to_csv(OUT / "3c_roi_thresholds.csv", index=False)
    print(f"3c done {time.time() - t0:.0f}s", flush=True)

    # ---- 3d: language and year robustness (best dev model by mean fold AUC among HGB/XGB/LGBM on sets A, B, D, FULL) -----------------
    best = tab[tab.set.isin(["A", "B", "D", "FULL"])].sort_values("fold_auc_mean", ascending=False).iloc[0]
    p_best = oof[(best.model, best.set)]
    dv = dev.assign(val_year=dev.release_date.dt.year)
    sl = V.subgroup_auc(dv, y, p_best, "language")
    sy = V.subgroup_auc(dv, y, p_best, "val_year", min_n=40)
    sl.to_csv(OUT / "3d_language_auc.csv", index=False)
    sy.to_csv(OUT / "3d_year_auc.csv", index=False)
    fa = pd.DataFrame({"fold": range(1, len(folds) + 1), "n": [len(v) for v in folds], "auc": M.fold_aucs(y, p_best, folds), "hit_rate": [float(y[v].mean()) for v in folds]})
    fa.to_csv(OUT / "3d_fold_auc.csv", index=False)

    log_experiment("phase3_feature_sets_cohort_thresholds_robustness", "phase3",
                   {"models": MODEL_KEYS, "sets": SETS_MAIN, "n_outer": C.N_OUTER, "data": "development only (before 2023-01-01)", "seed": C.SEED},
                   {"n_validated": int(ok.sum()), "n_comparisons": len(comp), "n_convincing": int(comp.convincing.sum()), "best_dev_model_set": f"{best.model}/{best.set}",
                    "best_dev_fold_auc": float(best.fold_auc_mean), "runtime_s": round(time.time() - t0)},
                   "Test set not touched. 'convincing' = paired-bootstrap CI above 0 AND BH q < 0.05 across all comparisons.")
    print("ALL DONE", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
