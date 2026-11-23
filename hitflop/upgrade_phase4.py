"""Upgrade Phase 4 runner: calibration, out-of-distribution detection, uncertainty warnings.  Decisions use development data only.

Part A  calibration   five recalibration methods fitted on EARLIER validation blocks only, scored on blocks 2 to 5; primary metric Brier score (declared up front).
Part B  warnings      for every validation block, the reference is the films the model had been trained on; do flagged films get worse predictions?
Part C  test, reporting only   recalibrators fitted on development data and warnings built from development films, applied to the stored test predictions.
                       Nothing is selected on it; a result there can never make a claim.
Run:  python -m hitflop.upgrade_phase4
"""
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C, data, drift as D, metrics as M, protocol as P, recalibration as R, uncertainty as U
from .upgrade_phase1 import dev_predictions

OUT = C.OUT_DIR / "upgrade" / "phase4"
SPECS = {"reference: LR, set A": "basic", "basic: HGB, set A": "basic", "buzz: XGB, set B3": "buzz", "earlier headline: XGB, set D": "buzz"}
FLAGS = ["mahalanobis_high", "input_shift", "language_few_films", "budget_or_runtime_unusual", "any_input_flag", "close_call", "any_warning"]


def part_a(y, preds, folds):
    rows, rel, comps = [], [], []
    for name in SPECS:
        raw = preds[name]
        cal = {m: R.expanding_recalibrate(raw, y, folds, m)[0] for m in R.METHODS}
        ok = ~np.isnan(cal[R.METHODS[0]])
        yy = y[ok]
        rows.append({"model": name, "method": "raw (as in the original study)", **R.calibration_row(yy, raw[ok])})
        for m in R.METHODS:
            rows.append({"model": name, "method": m, **R.calibration_row(yy, cal[m][ok])})
        rel.append(R.reliability(yy, raw[ok]).assign(model=name))
        for m in R.METHODS:
            for label, metric in (("brier (negated, higher is better)", R.neg_brier), ("log-loss (negated, higher is better)", R.neg_log_loss)):
                d, lo, hi, p = M.paired_bootstrap(yy, cal[m][ok], raw[ok], metric=metric, n=2000)
                comps.append({"model": name, "method": m, "metric": label, "n": int(ok.sum()), "diff": d, "ci_low": lo, "ci_high": hi, "p": p})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "calibration_dev_blocks_2_to_5.csv", index=False)
    pd.concat(rel).to_csv(OUT / "reliability_dev_raw.csv", index=False)
    comps = P.correct_family(pd.DataFrame(comps))
    comps.to_csv(OUT / "recalibration_vs_raw.csv", index=False)
    return tab, comps


def group_diff(quality, flag, n=2000, seed=C.SEED, min_group=10):
    """Mean quality of flagged films minus unflagged films, with a bootstrap interval (films resampled; draws with a tiny group are skipped)."""
    quality, flag = np.asarray(quality, float), np.asarray(flag, bool)
    if flag.sum() < min_group or (~flag).sum() < min_group:
        return np.nan, np.nan, np.nan, np.nan
    rg = np.random.default_rng(seed)
    d = []
    while len(d) < n:
        i = rg.integers(0, len(quality), len(quality))
        a, b = flag[i], ~flag[i]
        if a.sum() >= 5 and b.sum() >= 5:
            d.append(quality[i][a].mean() - quality[i][b].mean())
    d = np.array(d)
    p = max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / n)
    return float(quality[flag].mean() - quality[~flag].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def film_flags(dev, y, preds, folds_tr_va):
    """Per validation film: input flags from a reference built on that fold's training films, plus the distance."""
    out = {}
    for name, mode in SPECS.items():
        rows = []
        for tr, va in folds_tr_va:
            ref = U.fit_reference(dev.iloc[tr], mode)
            f = U.flags(ref, dev.iloc[va])
            f["distance"] = U.distance(ref, dev.iloc[va])
            f["block_rows"] = va
            rows.append(f)
        out[name] = pd.concat(rows).set_index("block_rows").sort_index()
    return out


def add_decision_flags(f, p, thr):
    g = f.copy()
    g["budget_or_runtime_unusual"] = g.budget_unusual | g.runtime_unusual | g.budget_outside_seen | g.runtime_outside_seen
    g["close_call"] = np.abs(p - thr) <= U.CLOSE_CALL_MARGIN
    g["any_warning"] = g.any_input_flag | g.close_call
    return g


def part_b(dev, y, preds, folds):
    fl = film_flags(dev, y, preds, data.outer_folds(dev))
    rows, shares, levels = [], [], []
    for name in SPECS:
        thr, scored = P.expanding_threshold(y, preds[name], folds)               # blocks 2 to 5
        idx = np.flatnonzero(scored)
        p, yy, t = preds[name][idx], y[idx], thr[idx]
        g = add_decision_flags(fl[name].loc[idx], p, t)
        correct = ((p >= t).astype(int) == yy).astype(float)
        quality = {"accuracy": correct, "brier (negated, higher is better)": -(p - yy) ** 2}
        for fg in FLAGS:
            for mname, q in quality.items():
                d, lo, hi, pv = group_diff(q, g[fg].values)
                rows.append({"model": name, "flag": fg, "metric": mname, "n_flagged": int(g[fg].sum()), "n_total": len(g), "share_flagged": float(g[fg].mean()), "quality_flagged": float(q[g[fg].values].mean()) if g[fg].any() else np.nan,
                             "quality_unflagged": float(q[~g[fg].values].mean()), "diff": d, "ci_low": lo, "ci_high": hi, "p": pv})
        rho, rp = stats.spearmanr(fl[name].loc[idx, "distance"].values, np.abs(p - yy))
        levels.append({"model": name, "spearman_distance_vs_abs_error": float(rho), "p": float(rp)})
        for k, va in enumerate(folds, start=1):                                   # share of validation films flagged as input shift, per block (drift readout)
            f_blk = fl[name].loc[va]
            shares.append({"model": name, "block": k, "mahalanobis_high": float(f_blk.mahalanobis_high.mean()), "input_shift": float(f_blk.input_shift.mean()), "any_input_flag": float(f_blk.any_input_flag.mean())})
    res = pd.DataFrame(rows)
    ok = res.p.notna()
    fam = P.correct_family(res[ok].copy())
    fam["verdict"] = fam.verdict.replace({"worse (supported)": "flagged films do worse (supported)", "better (supported)": "flagged films do better (supported)"})
    res = res.merge(fam[["model", "flag", "metric", "q_bh", "verdict"]], on=["model", "flag", "metric"], how="left")
    res["verdict"] = res.verdict.fillna("too few flagged films to test")
    res.to_csv(OUT / "warnings_vs_error_dev.csv", index=False)
    pd.DataFrame(shares).to_csv(OUT / "input_shift_share_by_block.csv", index=False)
    sp = pd.DataFrame(levels)
    sp["q_bh"] = M.benjamini_hochberg(sp.p.values)
    sp.to_csv(OUT / "distance_vs_error_spearman.csv", index=False)
    return res, pd.DataFrame(shares), sp


def part_c(dev, y, preds, folds):
    """Reporting only. Recalibrators from development out-of-fold predictions; warnings from development films; stored test predictions."""
    t = pickle.load(open(C.OUT_DIR / "final" / "test_predictions.pkl", "rb"))
    xt = D.unlabelled_test_features().set_index("tmdb_id").loc[t["test_ids"]]
    yt = np.asarray(t["y"])
    names = {"basic: HGB, set A": "basic (HGB, set A)", "buzz: XGB, set B3": "buzz (XGB, set B3)", "reference: LR, set A": "reference: LR, set A", "earlier headline: XGB, set D": "reference: XGB, set D (earlier headline)"}
    cal_rows, warn_rows, share_rows = [], [], []
    for name, key in names.items():
        raw_dev, p_test, thr = preds[name], np.asarray(t["pred"][key]), float(t["thr"][key])
        cal_rows.append({"model": name, "method": "raw", **R.calibration_row(yt, p_test)})
        for m in R.METHODS:
            first = max(0, len(folds) - R.RECENT_BLOCKS) if m.endswith("recent") else 0
            past = np.concatenate(folds[first:])
            cal = {"platt": R.Platt, "isotonic": R.Iso, "shift": R.Shift}[m.split("_")[0]]().fit(raw_dev[past], y[past])
            pc = np.clip(cal.predict(p_test), 1e-4, 1 - 1e-4)
            d, lo, hi, _ = M.paired_bootstrap(yt, pc, p_test, metric=R.neg_brier, n=2000)
            cal_rows.append({"model": name, "method": m, **R.calibration_row(yt, pc), "brier_gain_vs_raw": d, "brier_gain_ci_low": lo, "brier_gain_ci_high": hi})
        ref = U.fit_reference(dev, SPECS[name])
        g = add_decision_flags(U.flags(ref, xt), p_test, thr)
        correct = ((p_test >= thr).astype(int) == yt).astype(float)
        for fg in FLAGS:
            share_rows.append({"model": name, "flag": fg, "n_flagged": int(g[fg].sum()), "share_flagged": float(g[fg].mean())})
            if g[fg].sum() >= 10 and (~g[fg]).sum() >= 10:
                d, lo, hi, pv = group_diff(correct, g[fg].values)
                warn_rows.append({"model": name, "flag": fg, "n_flagged": int(g[fg].sum()), "accuracy_flagged": float(correct[g[fg].values].mean()), "accuracy_unflagged": float(correct[~g[fg].values].mean()), "diff": d, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(cal_rows).to_csv(OUT / "test_calibration_reporting_only.csv", index=False)
    pd.DataFrame(share_rows).to_csv(OUT / "test_flag_shares_reporting_only.csv", index=False)
    pd.DataFrame(warn_rows).to_csv(OUT / "test_warning_accuracy_reporting_only.csv", index=False)
    return pd.DataFrame(cal_rows), pd.DataFrame(share_rows), pd.DataFrame(warn_rows)


def figure(rel, comps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    a.plot([0, 1], [0, 1], "--", color="#777")
    for (name, g), c in zip(rel.groupby("model", sort=False), ["#34506B", "#B8512A", "#2E7D6B", "#8A6FB0"]):
        a.plot(g.mean_predicted, g.observed_hit_rate, "o-", color=c, label=name, ms=4)
    a.set_xlabel("Mean predicted Hit chance (5 equal-size groups)")
    a.set_ylabel("Observed Hit rate")
    a.set_title("Reliability on development blocks 2 to 5 (384 films)", fontsize=9)
    a.legend(fontsize=7)
    d = comps[comps.metric.str.startswith("brier")].reset_index(drop=True)
    for i, r in d.iterrows():
        b.plot([r.ci_low, r.ci_high], [i, i], color="#34506B", lw=1.4)
        b.plot(r["diff"], i, "o", color="#B8512A" if r.verdict != "no clear difference" else "#34506B", ms=4)
    b.axvline(0, color="#777", ls="--", lw=1)
    b.set_yticks(range(len(d)))
    b.set_yticklabels([f"{r.model.split(':')[0]} | {r.method}" for _, r in d.iterrows()], fontsize=6)
    b.invert_yaxis()
    b.set_xlabel("Change in Brier score (positive = better) versus raw probabilities, 95% interval")
    b.set_title("Recalibration fitted on earlier blocks only", fontsize=9)
    fig.tight_layout()
    (C.ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(C.ROOT / "docs" / "figures" / "phase4_calibration.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, _, dev = P.dev_only_world()
    y, preds, folds = dev_predictions(dev)
    print("A calibration", flush=True)
    tab, comps = part_a(y, preds, folds)
    figure(pd.read_csv(OUT / "reliability_dev_raw.csv"), comps)
    print("B warnings", flush=True)
    res, shares, sp = part_b(dev, y, preds, folds)
    print("C test (reporting only)", flush=True)
    cal_t, share_t, warn_t = part_c(dev, y, preds, folds)
    P.log_run("phase4-calibration-uncertainty", "upgrade-p4",
              {"recalibration_methods": R.METHODS, "recent_blocks": R.RECENT_BLOCKS, "primary_metric": "Brier score (calibration question)", "ood": f"Mahalanobis limit = {U.LIMIT_PERCENTILE}th percentile of reference distances",
               "close_call_margin": U.CLOSE_CALL_MARGIN, "min_language_films": U.MIN_LANGUAGE_FILMS, "validation_rows": 384},
              {"recalibration_supported_better": int((comps.verdict == "better (supported)").sum()), "recalibration_supported_worse": int((comps.verdict == "worse (supported)").sum()), "recalibration_total": int(len(comps)),
               "warning_tests_supported": int(res.verdict.isin(["flagged films do worse (supported)", "flagged films do better (supported)"]).sum()), "warning_tests_total": int(res.p.notna().sum())},
              notes="Development data decides everything. Test results (calibration of dev-fitted recalibrators, warning accuracy) are reporting only.", uses_test_set=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
