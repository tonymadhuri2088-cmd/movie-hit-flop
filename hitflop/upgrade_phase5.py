"""Upgrade Phase 5 runner: which kinds of information carry the prediction?  Development data only (the frozen test set is never touched).

Three designs, each with the original expanding-window folds and inner tuning, for LR, HGB and XGB:
  leave-one-group-out   all ten feature groups, remove one            -> its AUC / Brier change versus "all groups"
  group alone           one group by itself                            -> AUC with 95% interval versus chance (0.5)
  add-one to base       the six base groups, add one extra group       -> change versus "base"
Claim rule: paired-bootstrap 95% interval excludes 0 and Benjamini-Hochberg q < 0.05 over the whole family (AUC and Brier together).
Run:  python -m hitflop.upgrade_phase5
"""
import pickle
import time

import numpy as np
import pandas as pd

from . import ablation as AB
from . import config as C
from . import metrics as M
from . import protocol as P

OUT = C.OUT_DIR / "upgrade" / "phase5"
MODELS = ["LR", "HGB", "XGB"]
LABEL = {"budget": "Budget", "runtime": "Runtime", "release_year": "Release year", "release_timing": "Release timing (month, season)", "language": "Language", "genres": "Genres",
         "buzz_wikipedia": "Wikipedia buzz", "cast_crew_studio_history": "Cast, crew and studio track records", "festival_calendar": "Festival calendar", "competing_releases": "Competing releases"}


def neg_brier(y, p):
    return -float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))


def design():
    """name -> list of groups, plus the role of each configuration."""
    cfgs = {"all": (AB.ALL_GROUPS, "reference"), "base": (AB.BASE_GROUPS, "reference")}
    for g in AB.ALL_GROUPS:
        cfgs[f"minus {g}"] = ([x for x in AB.ALL_GROUPS if x != g], "leave-one-out")
        cfgs[f"only {g}"] = ([g], "alone")
    for g in AB.EXTRA_GROUPS:
        cfgs[f"base plus {g}"] = (AB.BASE_GROUPS + [g], "add-one")
    return cfgs


def run_all(dev, y, cfgs):
    path = OUT / "oof_phase5.pkl"
    store = pickle.load(open(path, "rb")) if path.exists() else {}
    for m in MODELS:
        for name, (groups, _) in cfgs.items():
            if (m, name) in store:
                continue
            t0 = time.time()
            store[(m, name)] = AB.walk_forward_groups(dev, y, m, groups, tune=True, n_jobs=2)[0]
            pickle.dump(store, open(path, "wb"))
            print(f"  {m} | {name}  {time.time() - t0:.0f}s", flush=True)
    return store


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, _, dev = P.dev_only_world()
    y = dev.hit.values
    cfgs = design()
    store = run_all(dev, y, cfgs)
    ok = ~np.isnan(store[("LR", "all")])
    yv = y[ok]

    # reproduction check: the base configuration is the original set A
    p3 = pickle.load(open(C.OUT_DIR / "phase3" / "oof_3a.pkl", "rb"))
    for m in ("LR", "HGB"):
        assert np.allclose(store[(m, "base")], p3["oof"][(m, "A")], equal_nan=True), f"{m} base does not reproduce set A"

    summary, rows = [], []
    for m in MODELS:
        for name, (groups, role) in cfgs.items():
            p = store[(m, name)][ok]
            lo, hi = M.bootstrap_ci(yv, p, n=2000)
            summary.append({"model": m, "configuration": name, "role": role, "groups": "+".join(groups), "n_groups": len(groups), "auc": float(M.auc(yv, p)), "auc_lo": lo, "auc_hi": hi,
                            "brier": float(np.mean((p - yv) ** 2)), "beats_chance": bool(lo > 0.5)})
            ref = "all" if role == "leave-one-out" else "base" if role == "add-one" else None
            if ref:
                for label, metric in (("auc", M.auc), ("brier (negated, higher is better)", neg_brier)):
                    d, dlo, dhi, pv = M.paired_bootstrap(yv, p, store[(m, ref)][ok], metric=metric, n=2000)
                    rows.append({"model": m, "configuration": name, "design": role, "reference": ref, "metric": label, "n": int(ok.sum()), "diff": d, "ci_low": dlo, "ci_high": dhi, "p": pv})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "ablation_summary.csv", index=False)
    comp = P.correct_family(pd.DataFrame(rows))
    comp.to_csv(OUT / "ablation_vs_reference.csv", index=False)
    forest_plot(comp, summary)

    auc = comp[comp.metric == "auc"]
    P.log_run("phase5-feature-ablation", "upgrade-p5",
              {"models": MODELS, "groups": AB.GROUPS, "designs": ["leave-one-group-out", "group alone", "add-one to base"], "validation_rows": int(ok.sum()),
               "reproduction": "the base configuration equals set A exactly (checked for LR and HGB)"},
              {"supported_changes_auc": {f"{r.model} | {r.configuration}": [round(r["diff"], 4), r.verdict] for _, r in auc.iterrows() if r.verdict != "no clear difference"},
               "comparisons": int(len(comp)), "supported_better": int((comp.verdict == "better (supported)").sum()), "supported_worse": int((comp.verdict == "worse (supported)").sum()),
               "alone_beats_chance": {f"{r.model} | {r.configuration}": bool(r.beats_chance) for _, r in summary[summary.role == "alone"].iterrows()}},
              notes="Development data only. Extra-group blocks keep median imputation with missing-value indicators, exactly as in the original sets.", uses_test_set=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def forest_plot(comp, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6))
    cols = {"LR": "#6B7A8F", "HGB": "#34506B", "XGB": "#B8512A"}
    d = comp[(comp.metric == "auc") & (comp.design == "leave-one-out")]
    order = [g for g in AB.ALL_GROUPS][::-1]
    for j, m in enumerate(MODELS):
        for i, g in enumerate(order):
            r = d[(d.model == m) & (d.configuration == f"minus {g}")].iloc[0]
            yy = i + (j - 1) * 0.22
            a.plot([r.ci_low, r.ci_high], [yy, yy], color=cols[m], lw=2)
            a.plot(r["diff"], yy, "o", color=cols[m], ms=5, label=m if i == 0 else None)
    a.axvline(0, color="#222", lw=0.8)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([LABEL[g] for g in order], fontsize=8)
    a.set_xlabel("Change in AUC when the group is removed (95% interval)")
    a.set_title("Leave-one-group-out (left of 0 = the group was helping)", fontsize=9)
    a.legend(fontsize=8, loc="lower left")
    s = summary[summary.role == "alone"]
    for j, m in enumerate(MODELS):
        for i, g in enumerate(order):
            r = s[(s.model == m) & (s.configuration == f"only {g}")].iloc[0]
            yy = i + (j - 1) * 0.22
            b.plot([r.auc_lo, r.auc_hi], [yy, yy], color=cols[m], lw=2)
            b.plot(r.auc, yy, "o", color=cols[m], ms=5)
    b.axvline(0.5, color="#222", lw=0.8, ls="--")
    b.set_yticks(range(len(order)))
    b.set_yticklabels([])
    b.set_xlabel("AUC of the group by itself (95% interval; dashed = chance)")
    b.set_title("Each group alone", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.ROOT / "docs" / "figures" / "phase5_feature_ablation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
