"""Turn a plain dictionary describing an upcoming film into the model's input row, and predict a Hit probability.

Two modes:
  basic  uses only information that exists long before release (budget, language, genres, runtime, release date, calendar, competition, cast/crew track records)
  buzz   additionally uses Wikipedia page views in the three windows before release (days -90..-31, -30..-8, -7..-1)
"""
import numpy as np
import pandas as pd

from . import config as C
from . import data
from . import features as F
from . import history as H

REQUIRED = ["language", "release_date", "budget", "runtime"]
MODE_WIKI_FIELDS = ["pv_90_31", "pv_30_8", "pv_7_1"]
LANGS = ["hi", "ta", "te", "ml", "kn"]


class InputError(ValueError):
    pass


def validate_input(movie, mode):
    """Raise InputError with a readable message for anything that would make a prediction meaningless."""
    for k in REQUIRED:
        if movie.get(k) in (None, "", np.nan):
            raise InputError(f"'{k}' is required")
    if movie["language"] not in LANGS:
        raise InputError(f"language must be one of {LANGS}")
    try:
        rd = pd.Timestamp(movie["release_date"])
    except Exception as e:
        raise InputError("release_date is not a valid date") from e
    if not (float(movie["budget"]) > C.MIN_BUDGET):
        raise InputError(f"budget (US dollars) must be above {C.MIN_BUDGET:,}")
    if not (30 <= float(movie["runtime"]) <= 300):
        raise InputError("runtime must be between 30 and 300 minutes")
    if mode == "buzz":
        for k in MODE_WIKI_FIELDS:
            v = movie.get(k)
            if v is None or v == "" or float(v) < 0:
                raise InputError(f"buzz mode needs a non-negative '{k}' (Wikipedia page views in that window)")
    return rd


def featurize(movie, ref_merged, ref_lab, mode="basic"):
    """One-row DataFrame with every model column. `ref_merged` = all movies in the reference file; `ref_lab` = labelled movies (for track-record priors)."""
    rd = validate_input(movie, mode)
    row = {c: np.nan for c in ref_merged.columns}
    genres = movie.get("genres") or ""
    if isinstance(genres, (list, tuple)):
        genres = ", ".join(genres)
    row.update({"tmdb_id": -1, "name": movie.get("name", "new film"), "language": movie["language"], "release_date": rd, "year": rd.year, "budget": float(movie["budget"]),
                "runtime": float(movie["runtime"]), "genres": genres, "in_yt_file": 0})
    for k in ("director", "writer", "art_director", "technical_team", "cast", "studio", "producer"):
        v = movie.get(k) or np.nan
        row[k] = ", ".join(v) if isinstance(v, (list, tuple)) else v
    if mode == "buzz":
        a, b, c = (float(movie[k]) for k in MODE_WIKI_FIELDS)
        row.update({"pv_90_31": a, "pv_30_8": b, "pv_7_1": c, "pv_30": b + c, "pv_peak_pre": np.nan, "buzz_growth": np.nan})
    new = pd.DataFrame([row])
    new["release_date"] = pd.to_datetime(new.release_date)
    new["hit"] = 0
    new["roi"] = np.nan
    f = data.build_basic_features(new)
    f = F.add_research_features(f, pd.concat([ref_merged, new[ref_merged.columns.intersection(new.columns)]], ignore_index=True))
    allm = pd.concat([ref_merged, new[ref_merged.columns.intersection(new.columns)]], ignore_index=True)
    allm = allm[allm.release_date.notna()]
    lab_new = new.copy()
    lab_ref_cols = ["tmdb_id", "release_date", "hit"] + [c for _, (c, _) in H.ROLE_COLS.items()]
    big_lab = pd.concat([ref_lab[lab_ref_cols], lab_new[lab_ref_cols]], ignore_index=True)
    tr = H.add_track_records(big_lab, allm)
    for c in tr.columns:
        f[c] = tr.iloc[-1][c]
    return f


SPEC = {"basic": ("HGB", "A"), "buzz": ("XGB", "B3")}


def predict(artifact, movie, mode="basic"):
    """Predicted Hit probability for one film. `artifact` is the dict produced by hitflop.artifacts.build_artifacts()."""
    f = featurize(movie, artifact["ref_merged"], artifact["ref_lab"], mode)
    a = artifact[mode]
    p = float(a["pipeline"].predict_proba(f[a["columns"]])[:, 1][0])
    return {"probability": p, "label": "Hit" if p >= a["threshold"] else "Flop", "threshold": a["threshold"], "features": f}
