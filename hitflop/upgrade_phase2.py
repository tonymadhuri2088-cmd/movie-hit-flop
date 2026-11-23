"""Upgrade Phase 2 runner: minority-class Hit detection.  Development data only; the frozen test set is never loaded.

Part 1  natural setting   every method x model x feature set (A, B3) with the original expanding-window validation and tuning.
                          Each variant is compared with its OWN plain version (clean test of the method) and with the project comparator (LR, set A).
Part 2  threshold rules   six rules for choosing the Hit/Flop cut-off, each chosen from EARLIER validation blocks only.
Part 3  simulated minority  Hits are randomly dropped until they are 30% of the films, several seeds, so the methods meet a real minority.
                          (The development data is ~47% Hits, so in the natural setting class weights and SMOTE have little to correct.)

Claim rule everywhere: better/worse only if the paired-bootstrap 95% interval excludes zero AND the Benjamini-Hochberg q < 0.05 over the whole family.
Run:  python -m hitflop.upgrade_phase2
"""
import itertools
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from . import evaluation as EV
from . import imbalance as IM
from . import metrics as M
from . import protocol as P

OUT = C.OUT_DIR / "upgrade" / "phase2"
SETS = ["A", "B3"]
NATURAL = [(m, v) for m, vs in IM.SUPPORTED.items() for v in vs]
RULE_SPECS = [("LR", "A", "base"), ("HGB", "A", "base"), ("XGB", "B3", "base"), ("RF", "A", "base"), ("RF", "A", "balanced_rf"), ("HGB", "A", "class_weight")]
RULES = ["fixed 0.5", "youden", "max_f1", "max_mcc", "prior_match", "recall_60"]
SIM_SEEDS = [42, 43, 44, 45, 46]
SIM_SHARE = 0.30


def name(m, s, v):
    return f"{m} | set {s} | {v}"


def balanced_accuracy_binary(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 1)).sum()
    tn, fp = ((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum()
    return (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2


def _cached(path, key, fn):
    store = pickle.load(open(path, "rb")) if path.exists() else {}
    if key not in store:
        store[key] = fn()
        pickle.dump(store, open(path, "wb"))
    return store[key]


def run_natural(dev, y):
    path = OUT / "oof_phase2.pkl"
    oof = {}
    for (m, v), s in itertools.product(NATURAL, SETS):
        t = time.time()
        oof[(m, s, v)] = _cached(path, (m, s, v), lambda m=m, s=s, v=v: IM.walk_forward_variant(dev, y, m, s, v, tune=True)[0])
        print(f"  natural {name(m, s, v)}  {time.time() - t:.0f}s", flush=True)
    folds = IM.walk_forward_variant(dev, y, "LR", "A", "base", tune=False)[1]
    return oof, folds


def natural_tables(y, oof, folds):
    ok = ~np.isnan(oof[("LR", "A", "base")])
    yv = y[ok]
    pv = {name(*k): v[ok] for k, v in oof.items()}
    metrics = EV.evaluate_table(yv, pv, 0.5)
    metrics.to_csv(OUT / "natural_metrics_threshold_0.5.csv", index=False)

    rows = []                                                                                  # each variant versus its own plain version
    for (m, s, v) in oof:
        if v == "base":
            continue
        for label, metric, pa, pb in (("auc", M.auc, oof[(m, s, v)][ok], oof[(m, s, "base")][ok]),
                                      ("balanced_accuracy_at_0.5", balanced_accuracy_binary, (oof[(m, s, v)][ok] >= .5).astype(int), (oof[(m, s, "base")][ok] >= .5).astype(int))):
            d, lo, hi, p = M.paired_bootstrap(yv, pa, pb, metric=metric, n=2000)
            rows.append({"model": m, "set": s, "variant": v, "metric": label, "n": int(ok.sum()), "metric_variant": float(metric(yv, pa)), "metric_plain": float(metric(yv, pb)),
                         "diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    vb = P.correct_family(pd.DataFrame(rows))
    vb.to_csv(OUT / "natural_variant_vs_plain.csv", index=False)

    vs_lr = []
    for label, metric in (("auc", M.auc), ("pr_auc", M.pr_auc)):
        c = P.compare_to_baseline(yv, pv, name("LR", "A", "base"), metric=metric)
        c.insert(0, "metric", label)
        vs_lr.append(c)
    vs_lr = P.correct_family(pd.concat(vs_lr))
    vs_lr.to_csv(OUT / "natural_vs_logistic_regression.csv", index=False)
    return metrics, vb, vs_lr


def threshold_tables(y, oof, folds):
    rows, preds = [], {}
    for spec in RULE_SPECS:
        for rule in RULES:
            r = 0.5 if rule == "fixed 0.5" else rule
            thr, scored = P.expanding_threshold(y, oof[spec], folds, r)
            yy, pp, tt = y[scored], oof[spec][scored], thr[scored]
            preds[(spec, rule)] = ((pp >= tt).astype(int), yy)
            rows.append({"spec": name(*spec), "rule": rule, **EV.evaluate(yy, pp, tt, n_boot=1000)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "threshold_rules_metrics.csv", index=False)
    comps = []
    for spec in RULE_SPECS:
        for rule in RULES:
            if rule == "youden":
                continue
            new, yy = preds[(spec, rule)]
            old, _ = preds[(spec, "youden")]
            d, lo, hi, p = M.paired_bootstrap(yy, new, old, metric=balanced_accuracy_binary, n=2000)
            comps.append({"spec": name(*spec), "rule": rule, "vs": "youden", "n": len(yy), "ba_rule": float(balanced_accuracy_binary(yy, new)), "ba_youden": float(balanced_accuracy_binary(yy, old)), "diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    comps = P.correct_family(pd.DataFrame(comps))
    comps.to_csv(OUT / "threshold_rules_vs_youden.csv", index=False)
    return tab, comps


def run_simulated(dev):
    rows = []
    for seed in SIM_SEEDS:
        sim = IM.simulate_minority(dev, SIM_SHARE, seed)
        ys = sim.hit.values
        res = {}
        for (m, v), s in itertools.product(NATURAL, SETS):
            res[(m, s, v)] = IM.walk_forward_variant(sim, ys, m, s, v, tune=False)[0]
        ok = ~np.isnan(res[("LR", "A", "base")])
        yv = ys[ok]
        for (m, s, v), p in res.items():
            pv, pb = p[ok], res[(m, s, "base")][ok]
            met = EV.point_metrics(yv, pv, 0.5)
            row = {"seed": seed, "model": m, "set": s, "variant": v, "n": int(ok.sum()), "hit_rate": float(yv.mean()),
                   "auc": met["auc"], "balanced_accuracy": met["balanced_accuracy"], "macro_f1": met["macro_f1"], "mcc": met["mcc"], "hit_recall": met["recall"], "specificity": met["specificity"]}
            if v != "base":
                d, lo, hi, pa = M.paired_bootstrap(yv, pv, pb, metric=M.auc, n=1000, seed=seed)
                row.update({"auc_diff_vs_plain": d, "auc_lo": lo, "auc_hi": hi, "auc_p": pa})
                d, lo, hi, pbv = M.paired_bootstrap(yv, (pv >= .5).astype(int), (pb >= .5).astype(int), metric=balanced_accuracy_binary, n=1000, seed=seed)
                row.update({"ba_diff_vs_plain": d, "ba_lo": lo, "ba_hi": hi, "ba_p": pbv})
            rows.append(row)
        print(f"  simulated minority seed {seed} done", flush=True)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(OUT / "simulated_minority_by_seed.csv", index=False)
    return by_seed


def summarise_simulated(by_seed):
    """One row per (model, set, variant): mean over seeds, how many seeds were individually supported, and a verdict that needs ALL seeds to agree.
    All per-seed tests form one Benjamini-Hochberg family. Seeds share the same Flops, so they are not independent evidence; requiring every seed to agree is the conservative answer."""
    v = by_seed[by_seed.variant != "base"].copy()
    out = []
    fam = pd.concat([v[["seed", "model", "set", "variant"]].assign(metric="auc", diff=v.auc_diff_vs_plain, ci_low=v.auc_lo, ci_high=v.auc_hi, p=v.auc_p),
                     v[["seed", "model", "set", "variant"]].assign(metric="balanced_accuracy_at_0.5", diff=v.ba_diff_vs_plain, ci_low=v.ba_lo, ci_high=v.ba_hi, p=v.ba_p)], ignore_index=True)
    fam = P.correct_family(fam)
    for (m, s, var, metric), g in fam.groupby(["model", "set", "variant", "metric"]):
        n_pos = int((g.verdict == "better (supported)").sum())
        n_neg = int((g.verdict == "worse (supported)").sum())
        verdict = "better (supported)" if n_pos == len(g) else "worse (supported)" if n_neg == len(g) else "no clear difference"
        out.append({"model": m, "set": s, "variant": var, "metric": metric, "mean_diff": float(g["diff"].mean()), "min_diff": float(g["diff"].min()), "max_diff": float(g["diff"].max()),
                    "seeds_supported_better": n_pos, "seeds_supported_worse": n_neg, "seeds": len(g), "verdict": verdict})
    summ = pd.DataFrame(out)
    summ.to_csv(OUT / "simulated_minority_summary.csv", index=False)
    fam.to_csv(OUT / "simulated_minority_per_seed_tests.csv", index=False)
    return summ


def forest_plot(vb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = vb[vb.metric == "auc"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.4))
    for i, r in d.iterrows():
        ax.plot([r.ci_low, r.ci_high], [i, i], color="#34506B", lw=1.6)
        ax.plot(r["diff"], i, "o", color="#B8512A" if r.verdict != "no clear difference" else "#34506B", ms=5)
    ax.axvline(0, color="#777", lw=1, ls="--")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.model} {r.set} {r.variant}" for _, r in d.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in ROC-AUC versus the plain version of the same model (95% interval)")
    ax.set_title("Phase 2: does handling the minority class help ranking?", fontsize=10)
    fig.tight_layout()
    (C.ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(C.ROOT / "docs" / "figures" / "phase2_variant_effects.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, _, dev = P.dev_only_world()
    y = dev.hit.values
    print("natural setting", flush=True)
    oof, folds = run_natural(dev, y)
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    for m in ("LR", "HGB"):                                          # the plain versions must reproduce the original study exactly
        assert np.allclose(oof[(m, "A", "base")], p3["oof"][(m, "A")], equal_nan=True), f"{m} base does not reproduce the stored predictions"
    metrics, vb, vs_lr = natural_tables(y, oof, folds)
    forest_plot(vb)
    print("threshold rules", flush=True)
    th, thc = threshold_tables(y, oof, folds)
    print("simulated minority", flush=True)
    by_seed = run_simulated(dev)
    summ = summarise_simulated(by_seed)
    P.log_run("phase2-minority-class", "upgrade-p2",
              {"variants": IM.VARIANTS, "feature_sets": SETS, "models": list(IM.SUPPORTED), "threshold_rules": RULES, "simulated_hit_share": SIM_SHARE, "simulated_seeds": SIM_SEEDS,
               "validation_rows_natural": int((~np.isnan(oof[("LR", "A", "base")])).sum())},
              {"variant_vs_plain_supported_better": int((vb.verdict == "better (supported)").sum()), "variant_vs_plain_supported_worse": int((vb.verdict == "worse (supported)").sum()), "variant_vs_plain_total": int(len(vb)),
               "vs_logistic_regression_supported_better": int((vs_lr.verdict == "better (supported)").sum()), "vs_logistic_regression_supported_worse": int((vs_lr.verdict == "worse (supported)").sum()),
               "threshold_rule_vs_youden_supported": int((thc.verdict != "no clear difference").sum()), "threshold_rule_comparisons": int(len(thc)),
               "simulated_supported_better": int((summ.verdict == "better (supported)").sum()), "simulated_supported_worse": int((summ.verdict == "worse (supported)").sum()), "simulated_total": int(len(summ))},
              notes="Development data only. Plain LR and HGB reproduce the stored out-of-fold predictions exactly.", uses_test_set=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
