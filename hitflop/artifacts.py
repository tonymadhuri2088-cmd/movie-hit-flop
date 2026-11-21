"""Build the model artifact used by the app. The evaluation numbers stored in it come from the dev-only fit scored once on the frozen test set;
the deployed models are then REFIT on all 854 labelled movies (development + 2023+), so the app uses every labelled film. This is disclosed in the app."""
import joblib
import numpy as np
import pandas as pd

from . import config as C, models as MD, metrics as M
from .world import build_world
from .predict import SPEC

PATH = C.ROOT / "artifacts" / "hitflop_models.joblib"


def build_artifacts(path=PATH):
    merged, lab, dev, test = build_world()
    y = lab.hit.values
    art = {"ref_merged": merged, "ref_lab": lab[["tmdb_id", "release_date", "hit", "director", "writer", "cast", "technical_team", "art_director", "studio", "producer", "revenue"]].copy()}
    res = pd.read_csv(C.OUT_DIR / "final" / "test_results.csv").set_index("model")
    names = {"basic": "basic (HGB, set A)", "buzz": "buzz (XGB, set B3)"}
    for mode, (mk, s) in SPEC.items():
        cols = MD.feature_columns(s)
        gs = MD.tune_and_fit(lab[cols], y, mk, s)
        r = res.loc[names[mode]]
        art[mode] = {"pipeline": gs.best_estimator_, "columns": cols, "model": mk, "feature_set": s, "threshold": float(r.dev_youden_threshold),
                     "trained_on": int(len(lab)), "date_range": [str(lab.release_date.min().date()), str(lab.release_date.max().date())],
                     "test_auc": float(r.auc), "test_auc_ci": [float(r.auc_ci_low), float(r.auc_ci_high)], "test_mcc": float(r["mcc@dev_thr"]), "test_mean_predicted": float(r.mean_predicted),
                     "test_observed_hit_rate": float(r.observed_hit_rate), "shap_background": lab[cols].sample(150, random_state=C.SEED).reset_index(drop=True)}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path, compress=3)
    return art


def load(path=PATH):
    return joblib.load(path)


if __name__ == "__main__":
    a = build_artifacts()
    print({m: (a[m]["model"], a[m]["feature_set"], a[m]["threshold"]) for m in ("basic", "buzz")})
