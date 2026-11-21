"""Data-quality audit that goes beyond the original notebook: where the labels come from, what is missing, what is suspicious."""
import numpy as np
import pandas as pd

from . import config as C

PRE_RELEASE_COLS = ["budget", "runtime", "genres", "language", "director", "cast", "technical_team", "pv_30", "in_yt_file"]


def missingness(merged, lab):
    cols = ["budget", "revenue", "runtime", "genres", "director", "writer", "screenplay", "art_director", "technical_team", "cast", "wiki_title", "pv_30", "pv_peak_pre",
            "trailer_found", "trailer_lead_days", "cm_pre_n", "cm_pre_likes", "studio", "producer"]
    cols = [c for c in cols if c in merged.columns]
    out = pd.DataFrame({"missing_pct_all": merged[cols].isna().mean().mul(100).round(1), "missing_pct_labelled": lab[cols].isna().mean().mul(100).round(1)})
    out["n_labelled_present"] = lab[cols].notna().sum()
    return out


def labelled_share(merged, lab, by):
    """Share of movies in the file that ended up with a usable label, per group. Mature movies only (too-recent movies cannot be labelled by design)."""
    cutoff = merged.release_date.max() - pd.Timedelta(days=C.MATURITY_DAYS)
    m = merged[merged.release_date <= cutoff].copy()
    m["is_labelled"] = m.tmdb_id.isin(lab.tmdb_id).astype(int)
    g = m.groupby(by).is_labelled.agg(["size", "sum", "mean"]).rename(columns={"size": "movies", "sum": "labelled", "mean": "share"})
    return g


def labelled_vs_unlabelled(merged, lab):
    cutoff = merged.release_date.max() - pd.Timedelta(days=C.MATURITY_DAYS)
    m = merged[merged.release_date <= cutoff].copy()
    m["group"] = np.where(m.tmdb_id.isin(lab.tmdb_id), "labelled", "unlabelled")
    num = ["runtime", "pv_30", "rating", "vote_count", "trailer_found", "cm_pre_n"]
    num = [c for c in num if c in m.columns]
    t = m.groupby("group")[num].median().T
    t["note"] = ["pre-release" if c in ("runtime", "pv_30", "trailer_found", "cm_pre_n") else "post-release (diagnosis only, never a feature)" for c in t.index]
    return t


def roi_quality(lab):
    r = lab.roi
    out = {"n": len(lab), "roi_median": float(r.median()), "roi_p5": float(r.quantile(.05)), "roi_p95": float(r.quantile(.95)),
           "n_roi_below_0.05": int((r < 0.05).sum()), "n_roi_above_50": int((r > 50).sum()),
           "share_tmdb_both": float(lab.tmdb_only.mean()),
           "hit_rate_tmdb_both": float(lab[lab.tmdb_only == 1].hit.mean()), "hit_rate_has_wikipedia_value": float(lab[lab.tmdb_only == 0].hit.mean()),
           "n_has_wikipedia_value": int((lab.tmdb_only == 0).sum())}
    thr = {f"hit_rate_roi_ge_{t}": float((r >= t).mean()) for t in C.ROI_THRESHOLDS}
    out.update(thr)
    near = {f"share_within_10pct_of_{t}": float(((r / t).between(0.9, 1.1)).mean()) for t in C.ROI_THRESHOLDS}
    out.update(near)
    return out


def duplicates_report(merged):
    key = merged.dropna(subset=["wiki_title"]).wiki_title
    dup_title_year_lang = merged.duplicated(["name", "year", "language"], keep=False)
    return {"duplicate_tmdb_id": int(merged.tmdb_id.duplicated().sum()), "duplicate_wiki_title": int(key.duplicated().sum()),
            "same_name_year_language": int(dup_title_year_lang.sum()), "full_row_duplicates": int(merged.duplicated().sum())}


def run_audit(merged, lab):
    return {"missingness": missingness(merged, lab), "labelled_share_language": labelled_share(merged, lab, "language"), "labelled_share_year": labelled_share(merged, lab, "year"),
            "labelled_vs_unlabelled": labelled_vs_unlabelled(merged, lab), "roi_quality": pd.Series(roi_quality(lab)).to_frame("value"), "duplicates": pd.Series(duplicates_report(merged)).to_frame("value")}
