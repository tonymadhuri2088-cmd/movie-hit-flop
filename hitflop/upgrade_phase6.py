"""Upgrade Phase 6 runner: language-wise performance and robustness.  Development data only.

Part A  per-language behaviour   AUC with a 95% interval, Brier, calibration-in-the-large for each language, and whether the AUC interval clears chance (0.5).
        A language-only predictor is constant inside a language, so inside a language it can only score chance by construction; it is shown for calibration and for the pooled AUC only.
        Languages with too few validated films are shown but never judged.
Part B  unseen-language test     leave-one-language-out: the model is trained without a language and scored on it, versus the normal model on the same films.
Claim rule: paired-bootstrap 95% interval excludes 0 and Benjamini-Hochberg q < 0.05 over the whole family.
Run:  python -m hitflop.upgrade_phase6
"""
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data
from . import language as LG
from . import metrics as M
from . import protocol as P

OUT = C.OUT_DIR / "upgrade" / "phase6"
SPECS = [("LR", "A"), ("HGB", "A"), ("XGB", "B3")]
LOLO_LANGS = ["hi", "te", "ml", "ta"]          # kn has too few films to hold out; it is still reported in part A
NAMES = {"hi": "Hindi", "te": "Telugu", "ml": "Malayalam", "ta": "Tamil", "kn": "Kannada"}


def stored_oof():
    s = pickle.load(open(C.OUT_DIR / "upgrade" / "phase3" / "oof_phase3.pkl", "rb"))
    return {f"{m} | set {st}": s[(m, st, "expanding")] for m, st in SPECS}


def part_a(dev, y, folds, oofs):
    prior = LG.language_prior_oof(dev, y, [(tr, va) for tr, va in folds])
    ok = ~np.isnan(prior)
    langs = dev.language.values
    rows = []
    for lg in sorted(set(langs), key=lambda g: -(langs == g).sum()):
        m = ok & (langs == lg)
        yy = y[m]
        rep = LG.reportable(yy)
        for name, p in {**oofs, "language-only baseline": prior}.items():
            pp = p[m]
            base = name == "language-only baseline"
            lo, hi = M.bootstrap_ci(yy, pp, n=2000) if (rep and not base) else (np.nan, np.nan)
            rows.append({"language": NAMES.get(lg, lg), "model": name, "validated_films": int(m.sum()), "hits": int(yy.sum()), "hit_rate": float(yy.mean()),
                         "reportable": rep, "auc": float(M.auc(yy, pp)) if (rep and not base) else np.nan, "auc_lo": lo, "auc_hi": hi,
                         "beats_chance": bool(rep and not base and lo > 0.5), "brier": float(np.mean((pp - yy) ** 2)), "calibration_in_the_large": float(pp.mean() - yy.mean())})
    table = pd.DataFrame(rows)
    # how much of the pooled ranking is "which language"?  weighted mean of within-language AUCs versus the pooled AUC (the language-only predictor is 0.5 inside a language by construction)
    pooled = []
    for name, p in {**oofs, "language-only baseline": prior}.items():
        wa = [(M.auc(y[ok & (langs == g)], p[ok & (langs == g)]) if name != "language-only baseline" else 0.5, (ok & (langs == g)).sum())
              for g in set(langs) if LG.reportable(y[ok & (langs == g)])]
        pooled.append({"model": name, "pooled_auc": float(M.auc(y[ok], p[ok])), "mean_within_language_auc": float(np.average([a for a, _ in wa], weights=[n for _, n in wa])),
                       "languages_averaged": len(wa)})
    return table, pd.DataFrame(pooled)


def part_b(dev, y, folds, oofs):
    path = OUT / "oof_lolo.pkl"
    store = pickle.load(open(path, "rb")) if path.exists() else {}
    rows = []
    for (m, st), lg in ((s, l) for s in SPECS for l in LOLO_LANGS):
        if (m, st, lg) not in store:
            t0 = time.time()
            store[(m, st, lg)] = LG.walk_forward_without(dev, y, m, st, lg, folds, tune=True)
            pickle.dump(store, open(path, "wb"))
            print(f"  without {lg} | {m} set {st}  {time.time() - t0:.0f}s", flush=True)
        un = store[(m, st, lg)]
        normal = oofs[f"{m} | set {st}"]
        k = ~np.isnan(un) & ~np.isnan(normal)
        yy = y[k]
        if not LG.reportable(yy):
            continue
        lo, hi = M.bootstrap_ci(yy, un[k], n=2000)
        d, dlo, dhi, pv = M.paired_bootstrap(yy, un[k], normal[k], n=2000)
        rows.append({"language": NAMES[lg], "model": f"{m} | set {st}", "n": int(k.sum()), "auc_never_seen_language": float(M.auc(yy, un[k])), "auc_lo": lo, "auc_hi": hi,
                     "auc_normal_model": float(M.auc(yy, normal[k])), "diff": d, "ci_low": dlo, "ci_high": dhi, "p": pv,
                     "brier_never_seen": float(np.mean((un[k] - yy) ** 2)), "brier_normal": float(np.mean((normal[k] - yy) ** 2)),
                     "mean_pred_never_seen": float(un[k].mean()), "hit_rate": float(yy.mean())})
    return P.correct_family(pd.DataFrame(rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, _, dev = P.dev_only_world()
    y = dev.hit.values
    folds = data.outer_folds(dev)
    oofs = stored_oof()
    table, pooled = part_a(dev, y, folds, oofs)
    table.to_csv(OUT / "per_language_performance.csv", index=False)
    pooled.to_csv(OUT / "pooled_vs_within_language_auc.csv", index=False)
    lolo = part_b(dev, y, folds, oofs)
    lolo.to_csv(OUT / "leave_one_language_out.csv", index=False)
    forest_plot(table, lolo)
    P.log_run("phase6-language-robustness", "upgrade-p6",
              {"models": [f"{m}/{s}" for m, s in SPECS], "min_validated_films": LG.MIN_FILMS, "min_per_class": LG.MIN_PER_CLASS, "held_out_languages": LOLO_LANGS,
               "language_only_baseline": "Hit rate of the language among each fold's training films, smoothed with strength 10 (constant inside a language, so chance there by construction)", "within_language_test": "bootstrap 95% interval of the AUC excludes 0.5"},
              {"auc_clears_chance": {f"{r.model} | {r.language}": bool(r.beats_chance) for _, r in table[table.reportable & (table.model != "language-only baseline")].iterrows()},
               "unseen_language": {f"{r.model} | {r.language}": [round(r["diff"], 4), r.verdict] for _, r in lolo.iterrows()},
               "pooled": pooled.round(4).to_dict("records")},
              notes="Development data only. Kannada (25 films) is shown but never judged.", uses_test_set=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def forest_plot(table, lolo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cols = {"LR | set A": "#6B7A8F", "HGB | set A": "#34506B", "XGB | set B3": "#B8512A"}
    langs = [NAMES[g] for g in ["hi", "te", "ml", "ta"]][::-1]
    for j, (name, c) in enumerate(cols.items()):
        for i, lg in enumerate(langs):
            r = table[(table.language == lg) & (table.model == name)].iloc[0]
            yy = i + (j - 1) * 0.2
            a.plot([r.auc_lo, r.auc_hi], [yy, yy], color=c, lw=2)
            a.plot(r.auc, yy, "o", color=c, ms=4.5, label=name if i == 0 else None)
    a.axvline(0.5, color="#222", lw=0.8, ls="--")
    a.set_yticks(range(len(langs)))
    a.set_yticklabels([f"{lg}  (n={int(table[(table.language == lg)].validated_films.iloc[0])})" for lg in langs])
    a.set_xlabel("AUC inside the language (95% interval; dashed = chance)")
    a.set_title("Does the model rank films within a language?", fontsize=9)
    a.legend(fontsize=7, loc="lower right")
    for j, (name, c) in enumerate(cols.items()):
        for i, lg in enumerate(langs):
            r = lolo[(lolo.language == lg) & (lolo.model == name)]
            if len(r):
                r = r.iloc[0]
                yy = i + (j - 1) * 0.2
                b.plot([r.auc_lo, r.auc_hi], [yy, yy], color=c, lw=2)
                b.plot(r.auc_never_seen_language, yy, "o", color=c, ms=4.5)
                b.plot(r.auc_normal_model, yy, "|", color="#222", ms=9)
    b.axvline(0.5, color="#222", lw=0.8, ls="--")
    b.set_yticks(range(len(langs)))
    b.set_yticklabels([])
    b.set_xlabel("AUC when the language was never seen in training (black tick = normal model)")
    b.set_title("Leave-one-language-out", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.ROOT / "docs" / "figures" / "phase6_language_robustness.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
