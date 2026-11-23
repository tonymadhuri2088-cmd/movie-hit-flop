"""Small dev-only check of the app-friendly Wikipedia sets (B3 etc.) against B and A, so the app's 'buzz' mode is justified by evidence."""
import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from . import validation as V
from .experiment_log import log_experiment
from .world import build_world


def main():
    _, lab, dev, _ = build_world()
    y = dev.hit.values
    sets = ["A", "B", "B3", "B3+", "B3+CAL+COMP"]
    res, rows = {}, []
    for mk in ["LR", "HGB", "XGB"]:
        for s in sets:
            res[(mk, s)], folds, _ = V.walk_forward(dev, y, mk, s)
    ok = ~np.isnan(res[("LR", "A")])
    for (mk, s), p in res.items():
        d, lo, hi, pv = M.paired_bootstrap(y[ok], p[ok], res[(mk, "A")][ok], n=2000) if s != "A" else (0, 0, 0, 1)
        rows.append({"model": mk, "set": s, "auc": M.auc(y[ok], p[ok]), "diff_vs_A": d, "ci_low": lo, "ci_high": hi, "p": pv})
    t = pd.DataFrame(rows)
    t["q_bh"] = M.benjamini_hochberg(np.where(t.set == "A", 1.0, t.p))
    (C.OUT_DIR / "phase3").mkdir(parents=True, exist_ok=True)
    t.to_csv(C.OUT_DIR / "phase3" / "3e_app_sets.csv", index=False)
    log_experiment("phase3e_app_friendly_wikipedia_sets", "phase3", {"sets": sets, "data": "development only"}, {"best": t.sort_values("auc").iloc[-1][["model", "set", "auc"]].to_dict()}, "Supports app buzz-mode design.")
    print(t.round(3).to_string())

if __name__ == "__main__":
    main()
