"""Phase 1 runner: data-quality audit and selection-bias analysis. Writes CSVs to outputs/phase1 and one entry to the experiment log."""
import pandas as pd

from . import config as C, data, history, data_audit, selection_bias as sb
from .experiment_log import log_experiment


def load_world():
    merged = data.load_merged()
    infobox = pd.read_csv(C.DATA_DIR / "wiki_infobox_labelled.csv", keep_default_na=False, encoding="utf-8-sig")
    merged = history.attach_infobox(merged, infobox)
    lab, _ = data.label_movies(merged)
    return merged, lab


def run(out=C.OUT_DIR / "phase1"):
    out.mkdir(parents=True, exist_ok=True)
    merged, lab = load_world()
    aud = data_audit.run_audit(merged, lab)
    for k, v in aud.items():
        v.to_csv(out / f"audit_{k}.csv")
    m = sb.selection_frame(merged, lab)
    p, auc, coefs = sb.selection_model(m)
    w = sb.ipw_weights(m, p)
    coefs.rename("coef").to_csv(out / "selection_model_coefficients.csv")
    pd.Series(w, name="ipw_weight").rename_axis("tmdb_id").to_csv(out / "ipw_weights.csv")
    sb.weighted_hit_rate(lab, w, "language").to_csv(out / "selection_weighted_hit_rate_language.csv")
    res = {"selection_model_cv_auc": auc, "mature_movies": len(m), "labelled": int(m.is_labelled.sum()), "ess": sb.effective_sample_size(w),
           "hit_rate": sb.weighted_hit_rate(lab, w), "roi_quality": aud["roi_quality"].value.to_dict()}
    log_experiment("phase1_data_audit_selection_bias", "phase1", {"seed": C.SEED, "covariates": sb.COVARIATES_NUM + sb.COVARIATES_CAT, "weight_cap": 10}, res,
                   "Labelled movies are predictable from public covariates (selection model AUC well above 0.5), so the 854 are not a random sample. IPW only fixes selection on observed covariates.")
    return res


if __name__ == "__main__":
    print(run())
