"""Track records may only use films released at least 120 days earlier; proved against a slow brute-force reference and by perturbation."""
import numpy as np
import pandas as pd
import pytest

from hitflop import history


@pytest.fixture(scope="module")
def table(world):
    return history.add_track_records(world["lab"], world["merged"])


def test_shape_and_columns(world, table):
    assert table.shape == (854, 24)
    assert list(table.columns)[:3] == ["dir_n_prev", "dir_hit_rate", "dir_log_rev"]


def test_matches_brute_force_on_sample(world, table):
    lab, merged = world["lab"], world["merged"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lab), 40, replace=False)
    checked = 0
    for i in idx:
        row = lab.iloc[i]
        for role in ("dir", "cast", "studio"):
            rate, n_prev, rev = history.brute_force_track_record(row, role, lab, merged)
            for got, exp in ((table.loc[i, f"{role}_hit_rate"], rate), (table.loc[i, f"{role}_n_prev"], n_prev), (table.loc[i, f"{role}_log_rev"], rev)):
                if np.isnan(exp):
                    assert np.isnan(got)
                else:
                    assert got == pytest.approx(exp, abs=1e-9)
                    checked += 1
    assert checked > 100


def test_future_outcomes_cannot_change_a_past_movies_features(world, table):
    """Flip the Hit label of every movie released in the last 2 years: features of movies released before that window must not change."""
    lab, merged = world["lab"].copy(), world["merged"]
    cut = lab.release_date.max() - pd.Timedelta(days=730)
    lab2 = lab.copy()
    lab2.loc[lab2.release_date > cut, "hit"] = 1 - lab2.loc[lab2.release_date > cut, "hit"]
    alt = history.add_track_records(lab2, merged)
    early = lab.release_date <= cut
    pd.testing.assert_frame_equal(table[early], alt[early], check_exact=False, atol=1e-12)
    # ...and the late movies' features do change (the test is not vacuous)
    assert not np.allclose(table[~early].fillna(-1).values, alt[~early].fillna(-1).values)


def test_label_inside_120_day_window_is_not_used(world):
    """A movie's own outcome and outcomes of films released in the 120 days before it must not enter its record."""
    lab, merged = world["lab"], world["merged"]
    base = history.add_track_records(lab, merged)
    i = len(lab) // 2
    t = lab.release_date.iloc[i]
    near = (lab.release_date > t - pd.Timedelta(days=120)) & (lab.release_date <= t)
    lab2 = lab.copy()
    lab2.loc[near, "hit"] = 1 - lab2.loc[near, "hit"]
    alt = history.add_track_records(lab2, merged)
    cols = [c for c in base.columns if c.endswith("hit_rate") or c.endswith("hit_rate_max")]
    # the overall running prior only uses films >= 120 days old, so hit-rate features of movie i must be unchanged
    assert np.allclose(base.loc[i, cols].astype(float).fillna(-1), alt.loc[i, cols].astype(float).fillna(-1), atol=1e-12)


def test_no_history_gives_neutral_values(world, table):
    first = table.iloc[0]
    assert first["dir_n_prev"] == 0 or np.isnan(first["dir_n_prev"])


def test_org_name_normalisation():
    assert history.org_list("T-Series Films ; Dharma Productions") == "tseries, dharma"
    assert np.isnan(history.org_list(""))
    assert np.isnan(history.org_list(None))
