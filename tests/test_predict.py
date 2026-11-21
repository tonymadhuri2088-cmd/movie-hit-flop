"""The app's featurizer must reproduce the research pipeline's features exactly for a film that is in the data."""
import numpy as np
import pandas as pd
import pytest

from hitflop import features as F, models as MD, predict as P, world as W


@pytest.fixture(scope="module")
def w():
    merged, lab, dev, test = W.build_world()
    return merged, lab, test


def _movie_from_row(r):
    return {"name": r["name"], "language": r["language"], "release_date": str(r.release_date.date()), "budget": r.budget, "runtime": r.runtime, "genres": r.genres,
            "director": r.director, "writer": r.writer, "cast": r["cast"], "technical_team": r.technical_team, "art_director": r.art_director,
            "studio": r.studio, "producer": r.producer, "pv_90_31": r.pv_90_31, "pv_30_8": r.pv_30_8, "pv_7_1": r.pv_7_1}


def test_featurizer_matches_pipeline_for_existing_movies(w):
    merged, lab, test = w
    rows = test[test.pv_30.notna()].sample(6, random_state=1)
    for _, r in rows.iterrows():
        ref_m = merged[merged.tmdb_id != r.tmdb_id]
        ref_l = lab[lab.tmdb_id != r.tmdb_id]
        f = P.featurize(_movie_from_row(r), ref_m, ref_l, mode="buzz")
        for c in MD.feature_columns("B3") + MD.feature_columns("A"):
            a, b = f.iloc[0][c], r[c]
            if isinstance(b, str) or isinstance(a, str):
                assert a == b, c
            elif pd.isna(b):
                assert pd.isna(a), c
            else:
                assert a == pytest.approx(b, rel=1e-9, abs=1e-9), (c, a, b)


def test_competition_and_calendar_match(w):
    merged, lab, test = w
    r = test.iloc[10]
    f = P.featurize(_movie_from_row(r), merged[merged.tmdb_id != r.tmdb_id], lab[lab.tmdb_id != r.tmdb_id], mode="basic")
    for c in F.CALENDAR + F.COMPETITION:
        assert f.iloc[0][c] == pytest.approx(r[c]), c


@pytest.mark.parametrize("bad,msg", [({"budget": 5000}, "budget"), ({"language": "xx"}, "language"), ({"runtime": 5}, "runtime"), ({"release_date": "not a date"}, "release_date"), ({"runtime": None}, "runtime")])
def test_invalid_input_is_rejected(bad, msg):
    m = {"language": "hi", "release_date": "2026-12-25", "budget": 20_000_000, "runtime": 140}
    m.update(bad)
    with pytest.raises(P.InputError, match=msg):
        P.validate_input(m, "basic")


def test_buzz_mode_requires_views():
    m = {"language": "hi", "release_date": "2026-12-25", "budget": 20_000_000, "runtime": 140}
    with pytest.raises(P.InputError, match="pv_90_31"):
        P.validate_input(m, "buzz")
    P.validate_input(m, "basic")
