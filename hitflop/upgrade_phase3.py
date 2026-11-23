"""Upgrade Phase 3 runner: temporal drift.  Development data only; test features are used label-free for input-shift measurement only.

Part 1  drift description   Hit rate by year, trend test, input drift (earlier vs later development films; development vs test, no labels)
Part 2  per-block behaviour  how the four reference models do in each chronological validation block (AUC, mean prediction versus actual Hit rate)
Part 3  training windows     expanding (original) versus rolling windows and time-decay weights, each compared with the expanding version of the SAME model

Claim rule: paired-bootstrap 95% interval excludes 0 and Benjamini-Hochberg q < 0.05 over the whole family (AUC and Brier together).
Run:  python -m hitflop.upgrade_phase3
"""
import itertools
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C, data, drift as D, metrics as M, protocol as P, temporal as T

OUT = C.OUT_DIR / "upgrade" / "phase3"
SPECS = [("LR", "A"), ("HGB", "A"), ("XGB", "B3"), ("RF", "B3")]
WINDOWS = ["expanding", "rolling-150", "rolling-250", "decay-1y", "decay-2y", "decay-4y"]
STORED = {"LR | set A": ("oof_3a", ("LR", "A")), "HGB | set A": ("oof_3a", ("HGB", "A"))}


def neg_brier(y, p):
    return -float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))          # higher is better, so a positive difference is an improvement


def block_table(y, preds, folds, dates):
    rows = []
    for name, p in preds.items():
        for k, va in enumerate(folds, start=1):
            yy, pp = y[va], p[va]
            lo, hi = M.bootstrap_ci(yy, pp, n=1000)
            rows.append({"model": name, "block": k, "n": len(va), "from": str(dates[va].min())[:10], "to": str(dates[va].max())[:10], "hit_rate": float(yy.mean()), "mean_predicted": float(pp.mean()),
                         "calibration_in_the_large": float(pp.mean() - yy.mean()), "auc": float(M.auc(yy, pp)), "auc_lo": lo, "auc_hi": hi, "brier": float(np.mean((pp - yy) ** 2))})
    return pd.DataFrame(rows)


def part1(dev):
    by_year = D.hit_rate_by(dev, dev.release_date.dt.year).rename(columns={"period": "year"})
    t = pickle.load(open(C.OUT_DIR / "final" / "test_predictions.pkl", "rb"))["y"]                  # already published; one line, reporting only
    lo, hi = D.wilson(int(t.sum()), len(t))
    by_year = pd.concat([by_year, pd.DataFrame([{"year": "2023+ (frozen test, reporting only)", "n": len(t), "hits": int(t.sum()), "hit_rate": float(t.mean()), "ci_low": lo, "ci_high": hi}])], ignore_index=True)
    by_year.to_csv(OUT / "hit_rate_by_year.csv", index=False)
    trend = pd.DataFrame([D.hit_rate_trend(dev)])
    trend.to_csv(OUT / "hit_rate_trend_development.csv", index=False)
    h = len(dev) // 2
    early, late = dev.iloc[:h], dev.iloc[h:]
    fd1 = D.feature_drift(early, late)
    fd1.insert(0, "comparison", "development: earlier half vs later half")
    test_x = D.unlabelled_test_features()
    fd2 = D.feature_drift(dev, test_x)
    fd2.insert(0, "comparison", "development vs frozen test (features only, no labels)")
    pd.concat([fd1, fd2]).to_csv(OUT / "feature_drift.csv", index=False)
    return by_year, trend, fd1, fd2


def part3(dev, y, folds):
    path = OUT / "oof_phase3.pkl"
    store = pickle.load(open(path, "rb")) if path.exists() else {}
    for (m, s), w in itertools.product(SPECS, WINDOWS):
        if (m, s, w) not in store:
            t0 = time.time()
            store[(m, s, w)] = T.walk_forward_temporal(dev, y, m, s, w, tune=True)[0]
            pickle.dump(store, open(path, "wb"))
            print(f"  {m} | set {s} | {w}  {time.time() - t0:.0f}s", flush=True)
    return store


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, _, dev = P.dev_only_world()
    y, dates = dev.hit.values, dev.release_date.values
    folds = [va for _, va in data.outer_folds(dev)]
    print("part 1: drift description", flush=True)
    by_year, trend, fd1, fd2 = part1(dev)
    print("part 3: training windows", flush=True)
    store = part3(dev, y, folds)
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    for m in ("LR", "HGB"):
        assert np.allclose(store[(m, "A", "expanding")], p3["oof"][(m, "A")], equal_nan=True), f"{m} expanding does not reproduce the stored predictions"
    ok = ~np.isnan(store[("LR", "A", "expanding")])
    yv = y[ok]

    print("part 2: per-block behaviour", flush=True)
    blocks = block_table(y, {f"{m} | set {s}": store[(m, s, "expanding")] for m, s in SPECS}, folds, dates)
    blocks.to_csv(OUT / "per_block_performance.csv", index=False)

    rows = []
    for (m, s) in SPECS:
        for w in WINDOWS[1:]:
            for label, metric in (("auc", M.auc), ("brier (negated, higher is better)", neg_brier)):
                d, lo, hi, p = M.paired_bootstrap(yv, store[(m, s, w)][ok], store[(m, s, "expanding")][ok], metric=metric, n=2000)
                rows.append({"model": m, "set": s, "training_window": w, "metric": label, "n": int(ok.sum()), "metric_value": float(metric(yv, store[(m, s, w)][ok])),
                             "metric_expanding": float(metric(yv, store[(m, s, "expanding")][ok])), "diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    vs_exp = P.correct_family(pd.DataFrame(rows))
    vs_exp.to_csv(OUT / "windows_vs_expanding.csv", index=False)

    summ = []
    for (m, s), w in itertools.product(SPECS, WINDOWS):
        p = store[(m, s, w)]
        per = [float(p[va].mean() - y[va].mean()) for va in folds]
        summ.append({"model": m, "set": s, "training_window": w, "auc": float(M.auc(yv, p[ok])), "brier": float(np.mean((p[ok] - yv) ** 2)),
                     "mean_abs_calibration_in_the_large": float(np.mean(np.abs(per))), "calibration_in_the_large_last_block": per[-1]})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "windows_summary.csv", index=False)

    pv = {f"{m} | set {s} | {w}": store[(m, s, w)][ok] for (m, s), w in itertools.product(SPECS, WINDOWS)}
    vs_lr = P.compare_to_baseline(yv, pv, "LR | set A | expanding", metric=M.auc)
    vs_lr.to_csv(OUT / "windows_vs_logistic_regression.csv", index=False)
    forest_plot(by_year, vs_exp)

    P.log_run("phase3-temporal-drift", "upgrade-p3",
              {"training_windows": WINDOWS, "specs": [f"{m}/{s}" for m, s in SPECS], "decay": "weight 0.5 ** (age_years / half_life), from release dates only", "validation_rows": int(ok.sum()),
               "test_use": "features only, no labels, for input-shift measurement; published test Hit rate quoted for context"},
              {"hit_rate_earlier_half": float(trend.hit_rate_earlier_half[0]), "hit_rate_later_half": float(trend.hit_rate_later_half[0]), "kendall_p": float(trend.kendall_p[0]),
               "window_vs_expanding_supported_better": int((vs_exp.verdict == "better (supported)").sum()), "window_vs_expanding_supported_worse": int((vs_exp.verdict == "worse (supported)").sum()), "window_vs_expanding_total": int(len(vs_exp)),
               "vs_logistic_regression_supported_better": int((vs_lr.verdict == "better (supported)").sum()), "vs_logistic_regression_supported_worse": int((vs_lr.verdict == "worse (supported)").sum())},
              notes="Development data only for every label-based result. Expanding LR and HGB reproduce the stored out-of-fold predictions exactly.", uses_test_set=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def forest_plot(by_year, vs_exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(by_year))
    cols = ["#34506B"] * (len(by_year) - 1) + ["#B8512A"]
    a.bar(x, by_year.hit_rate, color=cols, width=0.65)
    a.errorbar(x, by_year.hit_rate, yerr=[by_year.hit_rate - by_year.ci_low, by_year.ci_high - by_year.hit_rate], fmt="none", ecolor="#222", lw=1)
    a.set_xticks(x)
    a.set_xticklabels([str(v).split(" ")[0] for v in by_year.year], rotation=45, ha="right", fontsize=8)
    a.set_ylabel("Share of films that are Hits")
    a.set_ylim(0, 0.8)
    a.set_title("Hit rate by release year (orange: frozen test, reporting only)", fontsize=9)
    d = vs_exp[vs_exp.metric == "auc"].reset_index(drop=True)
    for i, r in d.iterrows():
        b.plot([r.ci_low, r.ci_high], [i, i], color="#34506B", lw=1.5)
        b.plot(r["diff"], i, "o", color="#B8512A" if r.verdict != "no clear difference" else "#34506B", ms=4)
    b.axvline(0, color="#777", ls="--", lw=1)
    b.set_yticks(range(len(d)))
    b.set_yticklabels([f"{r.model} {r.set} {r.training_window}" for _, r in d.iterrows()], fontsize=7)
    b.invert_yaxis()
    b.set_xlabel("Change in ROC-AUC versus the expanding window of the same model (95% interval)")
    b.set_title("Rolling windows and time-decay weights", fontsize=9)
    fig.tight_layout()
    (C.ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(C.ROOT / "docs" / "figures" / "phase3_drift_and_windows.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
