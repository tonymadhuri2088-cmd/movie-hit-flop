"""Load, clean, label and split the data exactly as the original notebook did (section 2-6), so every later number is comparable."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config as C

SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer", 6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "festive", 11: "festive"}


def load_merged(data_dir=None):
    """movies_full.csv left-joined with yt_features.csv on tmdb_id (one row per movie). Runtime 0 is a missing value."""
    d = data_dir or C.DATA_DIR
    movies = pd.read_csv(d / "movies_full.csv")
    yt = pd.read_csv(d / "yt_features.csv")
    merged = movies.merge(yt, on="tmdb_id", how="left", validate="one_to_one", indicator="yt_merge")
    merged["in_yt_file"] = (merged.yt_merge == "both").astype(int)
    merged = merged.drop(columns="yt_merge")
    assert len(merged) == len(movies) and merged.tmdb_id.is_unique
    merged["release_date"] = pd.to_datetime(merged.release_date, errors="coerce")
    merged.loc[merged.runtime <= 0, "runtime"] = np.nan
    return merged


def label_movies(merged, threshold=C.HIT_THRESHOLD, maturity_days=C.MATURITY_DAYS):
    """Apply the three filters and the Hit label. Returns (labelled, filter_log). Sorted by (release_date, tmdb_id)."""
    last = merged.release_date.max()
    cutoff = last - pd.Timedelta(days=maturity_days)
    s1 = merged.dropna(subset=["budget", "revenue"])
    s2 = s1[(s1.budget > C.MIN_BUDGET) & (s1.revenue > C.MIN_REVENUE)]
    lab = s2[s2.release_date <= cutoff].copy()
    log = [("all movies", len(merged)), ("budget and revenue present", len(s1)), ("plausible money values", len(s2)), (f"mature (<= {cutoff.date()})", len(lab))]
    lab["roi"] = lab.revenue / lab.budget
    lab["hit"] = (lab.roi >= threshold).astype(int)
    lab["tmdb_only"] = ((lab.budget_source == "tmdb") & (lab.revenue_source == "tmdb")).astype(int)
    lab = lab.sort_values(["release_date", "tmdb_id"]).reset_index(drop=True)
    assert lab.tmdb_id.is_unique
    return lab, pd.DataFrame(log, columns=["step", "movies_left"])


def build_basic_features(df):
    """Deterministic, row-only features (no learning from other movies, so no leakage). Identical to the original `build_features`."""
    f = df.copy()
    f["log_budget"] = np.log10(f.budget)
    month = f.release_date.dt.month
    f["month"] = month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f["season"] = month.map(SEASON)
    for c in ["pv_90_31", "pv_30_8", "pv_7_1", "pv_peak_pre", "buzz_growth"]:
        f["l_" + c] = np.log1p(f[c])
    f["buzz_per_budget"] = np.log10(1 + f.pv_30) - f.log_budget
    f["l_trailer_lead"] = np.log1p(f.trailer_lead_days.clip(upper=365))
    f["l_campaign_lead"] = np.log1p(f.campaign_lead_days.clip(upper=365))
    f["l_cm_pre_n"] = np.log1p(f.cm_pre_n)
    f["l_cm_pre_likes"] = np.log1p(f.cm_pre_likes)
    return f


def split_dev_test(lab, split_date=C.SPLIT_DATE):
    """Chronological split: development = before split_date, frozen test = from split_date."""
    cut = pd.Timestamp(split_date)
    dev = lab[lab.release_date < cut].reset_index(drop=True)
    test = lab[lab.release_date >= cut].reset_index(drop=True)
    return dev, test


def outer_folds(dev, n=C.N_OUTER):
    """Expanding-window folds: each fold trains on everything before its validation block. `dev` must already be time-sorted."""
    assert dev.release_date.is_monotonic_increasing
    return list(TimeSeriesSplit(n_splits=n).split(dev))


def load_all():
    """Convenience: merged (all movies), labelled, dev, test, folds."""
    merged = load_merged()
    lab, flog = label_movies(merged)
    return merged, lab, flog
