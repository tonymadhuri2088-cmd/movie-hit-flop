"""Phase 6: SHAP global, dependence and per-movie explanations for both deployed models. Predictive attribution only; see explain.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from . import artifacts as A
from . import config as C
from . import explain as X
from .experiment_log import log_experiment

OUT = C.OUT_DIR / "phase6"
GROUPS = {"budget": ["log_budget", "buzz_per_budget"], "language": ["language_"], "genres": ["genre_"], "calendar (month/season/year)": ["month_", "season_", "year"], "runtime": ["runtime"],
          "Wikipedia page views": ["l_pv_"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    art = A.load()
    from .world import build_world
    _, labw, dev, test = build_world()
    summary = {}
    for mode in ("basic", "buzz"):
        a = art[mode]
        X_all = labw[a["columns"]].sample(300, random_state=C.SEED).reset_index(drop=True)
        vals, Z, names = X.shap_values(a["pipeline"], a["shap_background"], X_all)
        imp = X.global_importance(vals, names)
        imp.to_csv(OUT / f"{mode}_global_importance.csv", index=False)
        X.group_importance(imp, GROUPS).rename("mean_abs_shap").to_csv(OUT / f"{mode}_group_importance.csv")
        plt.figure(figsize=(8, 6))
        shap.summary_plot(vals, Z, feature_names=names, show=False, max_display=15)
        plt.title(f"{mode} model: SHAP (log-odds of Hit)")
        plt.tight_layout(); plt.savefig(OUT / f"{mode}_shap_summary.png", dpi=130); plt.close()
        top = imp.feature.head(3).tolist()
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for k, f in enumerate(top):
            d = X.dependence_table(vals, Z, names, f)
            ax[k].scatter(d.value, d.shap, s=8, alpha=.5)
            ax[k].axhline(0, color="grey", lw=.7)
            ax[k].set_xlabel(f); ax[k].set_ylabel("SHAP")
        fig.suptitle(f"{mode} model: dependence of the top 3 features"); fig.tight_layout(); fig.savefig(OUT / f"{mode}_shap_dependence.png", dpi=130); plt.close(fig)
        summary[mode] = imp.head(5).to_dict("records")

    # per-movie examples from the TEST set using the buzz model (a confident Hit, a confident Flop, and one miss)
    a = art["buzz"]
    tp = test[test.pv_30.notna()].copy()
    tp["p"] = a["pipeline"].predict_proba(tp[a["columns"]])[:, 1]
    picks = {"confident correct Hit": tp[tp.hit == 1].sort_values("p").iloc[-1], "confident correct Flop": tp[tp.hit == 0].sort_values("p").iloc[0], "confident miss (predicted Hit, was Flop)": tp[tp.hit == 0].sort_values("p").iloc[-1]}
    rows = []
    for label, r in picks.items():
        v, Z, names = X.shap_values(a["pipeline"], a["shap_background"], pd.DataFrame([r[a["columns"]]]))
        le = X.local_explanation(v[0], Z[0], names, top=6)
        le.insert(0, "case", label); le.insert(1, "movie", r["name"]); le.insert(2, "predicted_probability", float(r.p)); le.insert(3, "actual", "Hit" if r.hit == 1 else "Flop")
        rows.append(le)
    pd.concat(rows).to_csv(OUT / "per_movie_examples.csv", index=False)
    log_experiment("phase6_shap", "phase6", {"models": ["basic HGB/A", "buzz XGB/B3"], "sample": 300}, {k: [x["feature"] for x in v] for k, v in summary.items()},
                   "Predictive attribution, not causal. Models explained are the deployed refits on all 854 labelled movies.")
    print({k: [(x["feature"], round(x["mean_abs_shap"], 3)) for x in v] for k, v in summary.items()})


if __name__ == "__main__":
    main()
