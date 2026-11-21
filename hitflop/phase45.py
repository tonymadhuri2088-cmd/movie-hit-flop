"""Phase 4+5 runner (development data only): nested Optuna tuning of XGBoost / LightGBM / CatBoost, tuned-vs-default comparison,
ensembles (average, weighted soft voting, stacking) with correct nested validation, expanding-window calibration, and the full metric table."""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from . import calibration as CAL, config as C, ensembles as E, metrics as M, tuning as T, validation as V
from .experiment_log import log_experiment
from .world import build_world

OUT = C.OUT_DIR / "phase45"
TUNE_PLAN = {"A": ["XGB", "LGBM", "CAT"], "B": ["XGB", "LGBM", "CAT"], "B3": ["XGB", "LGBM"], "D": ["XGB", "LGBM"]}
N_TRIALS = {"XGB": 20, "LGBM": 20, "CAT": 12}
ENS_SETS = ["A", "B", "B3", "D"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, lab, dev, _ = build_world()                 # test split ignored here
    y = dev.hit.values
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    assert (p3["dev_ids"] == dev.tmdb_id.values).all()
    folds = p3["folds"]
    ok = ~np.isnan(p3["oof"][("LR", "A")])
    oof = {}
    info = {}
    saved = OUT / "oof_tuned.pkl"
    if saved.exists():                                    # tuned results are deterministic (seeded); reuse them after a crash
        sv = pickle.load(open(saved, "rb"))
        oof, info = sv["oof"], sv["info"]
    for s, mks in TUNE_PLAN.items():
        for mk in mks:
            if ("tuned_" + mk, s) in oof:
                continue
            o, f, inf = T.nested_optuna_oof(dev, y, mk, s, n_trials=N_TRIALS[mk])
            oof[("tuned_" + mk, s)] = o
            info[("tuned_" + mk, s)] = inf
            print(f"tuned {mk}/{s} auc={M.auc(y[ok], o[ok]):.3f} {time.time() - t0:.0f}s", flush=True)
            pickle.dump({"oof": oof, "info": info, "folds": folds}, open(OUT / "oof_tuned.pkl", "wb"))
    # default (grid-tuned in phase 3) OOF for comparison
    for (mk, s), o in p3["oof"].items():
        oof[("default_" + mk, s)] = o
    # defaults that phase 3 did not compute (CatBoost, and the app-friendly set B3)
    need = [(mk, s) for s, mks in TUNE_PLAN.items() for mk in mks] + [(mk, s) for s in ENS_SETS for mk in ("LR", "HGB")]
    for mk, s in need:
        if ("default_" + mk, s) not in oof:
            oof[("default_" + mk, s)], _, _ = V.walk_forward(dev, y, mk, s)
    pickle.dump({"oof": oof, "info": info, "folds": folds}, open(OUT / "oof_all.pkl", "wb"))

    # ---- tuned vs default (paired bootstrap, same movies) ----
    rows = []
    for (name, s), o in oof.items():
        if not name.startswith("tuned_"):
            continue
        mk = name[6:]
        base = oof[("default_" + mk, s)]
        d, lo, hi, p = M.paired_bootstrap(y[ok], o[ok], base[ok], n=2000)
        rows.append({"model": mk, "set": s, "auc_default": M.auc(y[ok], base[ok]), "auc_tuned": M.auc(y[ok], o[ok]), "diff": d, "ci_low": lo, "ci_high": hi, "p": p,
                     "mean_inner_auc": float(np.mean([i["inner_auc"] for i in info[(name, s)]]))})
    tv = pd.DataFrame(rows)
    tv["q_bh"] = M.benjamini_hochberg(tv.p.values)
    tv.to_csv(OUT / "4_tuned_vs_default.csv", index=False)

    # ---- ensembles per feature set, on folds >= 1 ----
    rows, ens_oof = [], {}
    later = np.concatenate(folds[1:])
    for s in ENS_SETS:
        members = {"LR": oof[("default_LR", s)], "HGB": oof[("default_HGB", s)]}
        for mk in ["XGB", "LGBM", "CAT"]:
            if ("tuned_" + mk, s) in oof:
                members["tuned_" + mk] = oof[("tuned_" + mk, s)]
        out, w, names = E.walk_forward_ensembles(members, y, folds)
        ens_oof[s] = {**members, **out}
        best_single = max(members, key=lambda n: M.auc(y[later], members[n][later]))
        for n, p in {**members, **out}.items():
            d, lo, hi, pv = M.paired_bootstrap(y[later], p[later], members[best_single][later], n=1000) if n != best_single else (0, 0, 0, 1)
            m = M.full_metrics(y[later], p[later])
            rows.append({"set": s, "predictor": n, "kind": "ensemble" if n in out else "single", "auc": m["auc"], "pr_auc": m["pr_auc"], "brier": m["brier"], "diff_vs_best_single": d,
                         "ci_low": lo, "ci_high": hi, "p": pv, "best_single": best_single})
    et = pd.DataFrame(rows)
    et.to_csv(OUT / "5_ensembles.csv", index=False)
    pickle.dump({"ens_oof": ens_oof, "folds": folds}, open(OUT / "oof_ensembles.pkl", "wb"))

    # ---- calibration + full metrics for the best dev predictor (by AUC on folds >= 1; chosen on development data only) ----
    cand = et[et.kind.isin(["single", "ensemble"])].sort_values("auc", ascending=False).iloc[0]
    p_best = ens_oof[cand.set][cand.predictor]
    rows = []
    for meth in ["none", "platt", "isotonic"]:
        if meth == "none":
            pc, done = p_best, np.zeros(len(p_best), bool); done[later] = True
            pc = np.where(done, p_best, np.nan)
        else:
            pc, done = CAL.expanding_calibration(p_best, y, folds, "platt" if meth == "platt" else "iso")
        r = np.concatenate(folds[1:])        # fold 0 is dropped for every method so all rows compared are identical
        m = M.full_metrics(y[r], pc[r], threshold=None)
        rows.append({"calibration": meth, "predictor": f"{cand.predictor}/{cand.set}", **{k: v for k, v in m.items() if k in ("auc", "pr_auc", "brier", "log_loss", "ece")}})
    pd.DataFrame(rows).to_csv(OUT / "5_calibration.csv", index=False)

    best_row = {"predictor": f"{cand.predictor}/{cand.set}", "auc_folds_ge1": float(cand.auc)}
    log_experiment("phase45_optuna_ensembles_calibration", "phase4-5",
                   {"tune_plan": TUNE_PLAN, "n_trials": N_TRIALS, "ensembles": ["average", "weighted_vote", "stacking"], "data": "development only", "seed": C.SEED},
                   {**best_row, "runtime_s": round(time.time() - t0), "tuned_vs_default_convincing": int(((tv.ci_low > 0) & (tv.q_bh < 0.05)).sum()), "n_tuned": len(tv)},
                   "Test set not touched.")
    print("ALL DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
