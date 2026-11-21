"""Leak-free track records for director, writer, cast, technical team, art director, studio and producer.

Rule: for a movie released on day t, a person's outcome history uses only films released at least MATURITY_DAYS before t
(their box office was already known), and their experience count uses only films released before t. Nothing from day t or later is used.
`brute_force_track_record` recomputes one value the slow, obviously-correct way and is used by the tests.
"""
import bisect
import re

import numpy as np
import pandas as pd

from . import config as C

_DROP = {"pvt", "ltd", "private", "limited", "films", "film", "studios", "studio", "productions", "production", "pictures", "entertainment", "entertainments", "movies", "movie",
         "motion", "creations", "international", "inc", "llp", "the", "and", "presenters", "presenter", "presents"}

ROLE_COLS = {"dir": ("director", None), "wri": ("writer", None), "cast": ("cast", C.TOP_CAST), "tech": ("technical_team", None), "art": ("art_director", None),
             "studio": ("studio", None), "prod": ("producer", None)}
CREW = ["dir_n_prev", "dir_hit_rate", "dir_log_rev", "wri_n_prev", "wri_hit_rate", "wri_log_rev", "cast_n_prev", "cast_hit_rate", "cast_log_rev", "cast_hit_rate_max"]
TECH = ["tech_n_prev", "tech_hit_rate", "tech_log_rev", "tech_hit_rate_max", "art_n_prev", "art_hit_rate", "art_log_rev"]
STUDIO = ["studio_n_prev", "studio_hit_rate", "studio_log_rev", "studio_hit_rate_max", "prod_n_prev", "prod_hit_rate", "prod_log_rev"]
HISTORY_FEATURES = CREW + TECH + STUDIO


def org_list(v):
    """'T-Series Films ; Dharma Productions' -> 'tseries, dharma' (lower case, company-type words removed, duplicates removed)."""
    if not isinstance(v, str) or not v.strip():
        return np.nan
    v = v.replace("&nbsp;", " ").replace("&amp;", " and ")
    names = []
    for piece in re.split(r"\s*;\s*|\s*\*\s*|\s*,\s*|\s+/\s+", v):
        piece = re.sub(r"\(.*?\)", " ", piece.lower())
        toks = [t for t in re.sub(r"[^a-z0-9 ]", "", piece.replace("-", "")).split() if t not in _DROP]
        name = "".join(toks)
        if name and name not in names:
            names.append(name)
    return ", ".join(names) if names else np.nan


def attach_infobox(df, infobox):
    """Add `studio` and `producer` columns from the Wikipedia infobox table (keyed by wiki_title)."""
    ib = infobox.drop_duplicates("wiki_title").set_index("wiki_title")
    studio, prod = ib.company.map(org_list), ib.producer.map(org_list)
    out = df.copy()
    out["studio"] = out.wiki_title.map(studio)
    out["producer"] = out.wiki_title.map(prod)
    return out


def people(v, top=None):
    if not isinstance(v, str):
        return []
    names = [n.strip().lower() for n in v.split(",") if n.strip()]
    return names[:top] if top else names


def add_track_records(lab_df, all_df, maturity_days=C.MATURITY_DAYS):
    """Track-record feature table (same index as lab_df). `all_df` is every movie in the file (experience counts need unlabelled movies too)."""
    mature = pd.Timedelta(days=maturity_days)
    all_df = all_df[all_df.release_date.notna()]
    hist_all = {r: {} for r in ROLE_COLS}
    hist_hit = {r: {} for r in ROLE_COLS}
    hist_rev = {r: {} for r in ROLE_COLS}
    labd = lab_df[["tmdb_id", "release_date", "hit"]].sort_values("release_date")
    lab_dates, lab_cum = labd.release_date.values.astype("datetime64[ns]"), np.cumsum(labd.hit.values)
    for r, (col, top) in ROLE_COLS.items():
        for d, v in all_df.sort_values("release_date")[["release_date", col]].itertuples(index=False):
            for n in people(v, top):
                hist_all[r].setdefault(n, []).append(d)
        tmp = all_df.merge(lab_df[["tmdb_id", "hit"]], on="tmdb_id", how="inner").sort_values("release_date")
        for d, v, h in tmp[["release_date", col, "hit"]].itertuples(index=False):
            for n in people(v, top):
                hist_hit[r].setdefault(n, ([], []))[0].append(d)
                hist_hit[r][n][1].append(int(h))
        rv = all_df[all_df.revenue > 0].sort_values("release_date")
        for d, v, rev in rv[["release_date", col, "revenue"]].itertuples(index=False):
            for n in people(v, top):
                hist_rev[r].setdefault(n, ([], []))[0].append(d)
                hist_rev[r][n][1].append(float(np.log10(rev)))
    out = []
    for row in lab_df.itertuples(index=False):
        t = row.release_date
        k = np.searchsorted(lab_dates, np.datetime64(t - mature), side="right")
        prior = (lab_cum[k - 1] / k) if k > 0 else C.PRIOR_NO_HISTORY
        feats = {}
        for r, (col, top) in ROLE_COLS.items():
            n_prev, rates, revs = [], [], []
            for n in people(getattr(row, col), top):
                n_prev.append(bisect.bisect_left(hist_all[r].get(n, []), t))
                dts, hits = hist_hit[r].get(n, ([], []))
                j = bisect.bisect_right(dts, t - mature)
                rates.append((sum(hits[:j]) + C.SMOOTH_M * prior) / (j + C.SMOOTH_M))
                dts2, lr = hist_rev[r].get(n, ([], []))
                j2 = bisect.bisect_right(dts2, t - mature)
                if j2:
                    revs.append(np.mean(lr[:j2]))
            feats[f"{r}_n_prev"] = np.log1p(np.mean(n_prev)) if n_prev else np.nan
            feats[f"{r}_hit_rate"] = np.mean(rates) if rates else np.nan
            feats[f"{r}_log_rev"] = np.mean(revs) if revs else np.nan
            if r in ("cast", "tech", "studio"):
                feats[f"{r}_hit_rate_max"] = np.max(rates) if rates else np.nan
        out.append(feats)
    return pd.DataFrame(out, index=lab_df.index)


def brute_force_track_record(movie_row, role, lab_df, all_df, maturity_days=C.MATURITY_DAYS):
    """Slow reference: (hit_rate, n_prev, log_rev) for one movie and one role, by filtering whole DataFrames. Used only by tests."""
    col, top = ROLE_COLS[role]
    t = movie_row.release_date
    mature = pd.Timedelta(days=maturity_days)
    lab_before = lab_df[lab_df.release_date <= t - mature]
    prior = lab_before.hit.mean() if len(lab_before) else C.PRIOR_NO_HISTORY
    rates, n_prev, revs = [], [], []
    for n in people(movie_row[col] if not hasattr(movie_row, col) else getattr(movie_row, col), top):
        has = lambda v: n in people(v, top)
        n_prev.append(int(sum(has(v) for v, d in zip(all_df[col], all_df.release_date) if d < t)))
        past = lab_before[np.array([has(v) for v in lab_before[col]], dtype=bool)]
        rates.append((past.hit.sum() + C.SMOOTH_M * prior) / (len(past) + C.SMOOTH_M))
        rv = all_df[(all_df.revenue > 0) & (all_df.release_date <= t - mature)]
        rv = rv[np.array([has(v) for v in rv[col]], dtype=bool)]
        if len(rv):
            revs.append(float(np.log10(rv.revenue).mean()))
    return (float(np.mean(rates)) if rates else np.nan, float(np.log1p(np.mean(n_prev))) if n_prev else np.nan, float(np.mean(revs)) if revs else np.nan)
